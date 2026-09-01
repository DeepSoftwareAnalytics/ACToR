import torch
import torch.nn as nn
import os
from loguru import logger
from typing import Dict

class Classifier(nn.Module):
    """Critical Token classifier"""
    def __init__(self, input_dim: int, weights_path: str = None, train_mode: bool = False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # If not in training mode or no weights path provided, do not load pre-trained weights

        if not train_mode:
            assert weights_path is not None and os.path.exists(weights_path)
            self.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.eval()
        
        self.to(self.device)
    
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return self.model(x)
    
    def get_logits(self, x):
        """Return raw logits, used for CrossEntropyLoss in training"""
        if x.dtype != torch.float32:
            x = x.float()
        return self.model(x)
    
    def get_probs(self, x):
        """Return probability distribution, used for probability output in inference"""
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.model(x)
        return torch.softmax(logits, dim=-1)
            
    def judge(self, x, threshold: float = 0.9):
        with torch.no_grad():
            if x.dtype != torch.float32:
                x = x.float()
            # Ensure model is in evaluation mode
            self.eval()
            probs = self.get_probs(x)
            # logger.info(f"probs: {probs}")
            pred_label = bool(probs[0, 1] > threshold)
            return pred_label

class MultiClassifier:
    """Multi-classifier manager"""
    def __init__(self, base_classifier_path: str, input_dim: int, num_classifiers: int = 5):
        self.num_classifiers = num_classifiers
        self.input_dim = input_dim
        self.classifiers = []
        
        # Load all classifiers
        for i in range(num_classifiers):
            classifier_path = base_classifier_path.replace('.pth', f'_{i+1}.pth')
            try:
                classifier = Classifier(
                    input_dim=self.input_dim,
                    weights_path=classifier_path
                )
                self.classifiers.append(classifier)
                logger.info(f"Successfully loaded classifier {i+1}: {classifier_path}")
            except Exception as e:
                logger.error(f"Failed to load classifier {i+1}: {e}")
        
        logger.info(f"Successfully loaded {len(self.classifiers)} classifiers")
    
    def judge(self, x, threshold: float = 0.7) -> bool:
        """Vote to determine if it is a critical token"""
        votes = []
        
        # Collect all classifier judgments
        for i, classifier in enumerate(self.classifiers):
            try:
                # Ensure classifier is in evaluation mode
                classifier.eval()
                vote = classifier.judge(x, threshold)
                votes.append(vote)
                    
            except Exception as e:
                logger.warning(f"Classifier {i+1} failed: {e}")
                votes.append(False)  # Default to non-critical when failed

        # Count voting results
        critical_votes = sum(votes)
        total_votes = len(votes)        
        is_critical = (critical_votes >= total_votes)
        
        return is_critical
