import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from .data import QueryVector, QueryWindow
from .utils import FilePathBuilder, Tools
from loguru import logger
import json
import os
from pathlib import Path

class UniXcoder(nn.Module):
    def __init__(self, use_position_weighting: bool = False, sigma_ratio: float = 0.4):
        """
        Build UniXcoder model for code understanding and generation.
        
        Args:
            use_position_weighting: Whether to enable position weighting functionality
            sigma_ratio: Sigma ratio for position weighting
        """        
        model_name = "/root/autodl-tmp/models/unixcoder"
        super(UniXcoder, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name, revision="main", local_files_only=True)
        self.config = RobertaConfig.from_pretrained(model_name, revision="main", local_files_only=True)
        self.config.is_decoder = True
        self.model = RobertaModel.from_pretrained(model_name, config=self.config, revision="main", local_files_only=True)
        
        self.use_position_weighting = use_position_weighting
        self.sigma_ratio = sigma_ratio
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"UniXcoder using device: {self.device}")
        logger.info(f"Position weighting: {'enabled' if use_position_weighting else 'disabled'}{f', sigma_ratio: {sigma_ratio}' if use_position_weighting else ''}")
        self._position_weights: Dict[int, torch.Tensor] = {}
        if self.use_position_weighting:
            self._load_all_position_weights()
            logger.info(f"Loaded {len(self._position_weights)} position weight sets")
        self.model = self.model.to(self.device)
        
        self.register_buffer("bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024))
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embeddings.word_embeddings.weight
        self.lm_head = self.lm_head.to(self.device)
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.tokenizer.add_tokens(["<mask0>"],special_tokens=True)
    
    def _load_all_position_weights(self):
        """
        Load all position weights during initialization.
        """
        try:
            filepath = FilePathBuilder.get_position_weight_path(self.sigma_ratio)
            logger.info(f"Loading position weights: {filepath}")
            weight_data = Tools.load_jsonl(filepath)
            for weight in weight_data:
                length = weight['L']
                weights_tensor = torch.tensor(weight['weights'], dtype=torch.float32, device=self.device)
                self._position_weights[length] = weights_tensor
        except Exception as e:
            logger.warning(f"Failed to load position weights: {e}, disabling position weighting")
            self.use_position_weighting = False
          
    def tokenize(self, inputs: list[str], max_length: int = 1023, padding: bool = False) -> list[list[int]]:
        """
        Convert string to token ids.
        
        Args:
            inputs: List of input strings
            max_length: The maximum total source sequence length after tokenization
            padding: Whether to pad source sequence length to max_length
            
        Returns:
            List[List[int]]: List of token id sequences
        """
        
        tokenizer = self.tokenizer
        
        tokens_ids = []
        for x in inputs:
            tokens = tokenizer.tokenize(x)
            tokens = tokens[:max_length-4]
            tokens = [tokenizer.cls_token,tokenizer.sep_token] + tokens + [tokenizer.sep_token]                
            tokens_id = tokenizer.convert_tokens_to_ids(tokens)
            if padding:
                tokens_id = tokens_id + [self.config.pad_token_id] * (max_length-len(tokens_id))
            tokens_ids.append(tokens_id)
        return tokens_ids
            
    def decode(self, source_ids: list[list[int]]) -> list[list[str]]:   
        """
        Convert token ids to string.
        
        Args:
            source_ids: List of token id sequences
            
        Returns:
            List[List[str]]: List of decoded string sequences
        """      
        predictions = []
        for x in source_ids:
            prediction = []
            for y in x:
                t = y.cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = self.tokenizer.decode(t,clean_up_tokenization_spaces=False)
                prediction.append(text)        
            predictions.append(prediction)
        return predictions
    
    def forward(self, source_ids):   
        """
        Obtain token embeddings and sentence embeddings.
        
        Args:
            source_ids: Input token ids
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Token embeddings and sentence embeddings
        """
        mask = source_ids.ne(self.config.pad_token_id)
        token_embeddings = self.model(source_ids,attention_mask = mask.unsqueeze(1) * mask.unsqueeze(2))[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        return token_embeddings, sentence_embeddings
    
    def embedding(self, codes: list[str]) -> torch.Tensor:
        """
        Generate embedding for code snippet.
        
        Args:
            codes: List of code snippets
            
        Returns:
            torch.Tensor: Code embedding tensor
        """
        assert len(codes) == 1
        tokens_ids = self.tokenize(codes, max_length=1023)
        source_ids = torch.tensor(tokens_ids).to(self.device)
        
        if self.use_position_weighting:
            token_embeddings, _ = self.forward(source_ids)
            
            # Create mask to exclude padding tokens
            mask = source_ids.ne(self.config.pad_token_id)  # (1, L)
            
            # Get valid sequence length (excluding padding)
            valid_length = mask.sum().item()
            
            # Get weights for corresponding valid length
            weights = self._position_weights[valid_length].to(device=self.device, dtype=torch.float32)
            
            # Correctly broadcast weights: (valid_length,) -> (1, valid_length, 1) to match (1, valid_length, D)
            weighted_embeddings = token_embeddings[:, :valid_length, :] * weights.view(1, -1, 1)
            
            # Calculate weighted average
            code_embedding = weighted_embeddings.sum(1)  # (1, D)
        else:    
            token_embeddings, _ = self.forward(source_ids)
            
            # When not using position weighting, directly average token embeddings
            mask = source_ids.ne(self.config.pad_token_id)
            masked_embeddings = token_embeddings * mask.unsqueeze(-1)
            code_embedding = masked_embeddings.sum(1) / mask.sum(-1, keepdim=True)  # (1, D)
            
        return code_embedding

class UniXcoderEmbedding:    
    def __init__(self, use_position_weighting: bool = False, sigma_ratio: float = 0.4) -> None:
        self.model: UniXcoder = UniXcoder(
            use_position_weighting=use_position_weighting, 
            sigma_ratio=sigma_ratio
        )

    def build_repos(self, window_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build dense vectors for repositories.
        
        Args:
            window_results: List of window results
            
        Returns:
            List[Dict[str, Any]]: List of results with embeddings
        """
        new_lines = []
        for line in tqdm(window_results, desc=f'building dense vector'):
            embedding = self.model.embedding([line['context']])
            new_lines.append({
                'context': line['context'],
                'metadata': line['metadata'],
                'embedding': embedding.tolist()[0]
            })
        return new_lines
    
    def build_vectors(self, window_results: List[QueryWindow]) -> List[QueryVector]:
        """
        Build vectors for QueryWindow results.
        
        Args:
            window_results: List of QueryWindow objects containing window information
            
        Returns:
            List[QueryVector]: List of vector results containing prompt, metadata and embeddings
        """
        VectorResults = []
        for window_result in window_results:
            vector_embedding_dict = {}
            for key, value in window_result.window_dict.items():
                embedding = self.model.embedding([value])
                vector_embedding_dict[key] = embedding.tolist()[0]
            VectorResults.append(QueryVector.from_window(
                query_window_result=window_result,
                embedding_dict=vector_embedding_dict
            ))
        return VectorResults