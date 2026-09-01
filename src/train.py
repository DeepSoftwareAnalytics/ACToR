import copy
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from data import (
    CONSTANTS, 
    TaskDataProcess, 
    FilePathBuilder, 
    Tools,
)
from dataclasses import asdict, dataclass
from server.classifier import Classifier
from loguru import logger



@dataclass
class TokenTask:
    task_id: str
    token_index: int
    ground_truth_token: str
    prediction_token: str
    ground_truth_token_log_prob: float
    prediction_token_log_prob: float
    hidden_states: List[float]
    uncertainty: float
    attention_score: List[float]
    syntax_type: str  # New: syntax type field

class TrainDataTask:
    """Training task processing class"""
    def __init__(
        self,
        llm=None,
        model_name=None,
        benchmark: str = CONSTANTS.train_dataset,
        repos: List[str] = CONSTANTS.repost_train_repos, 
        sigma_ratio: float = 0.4,
        max_new_tokens: int = 512,
        save_dir: str = "datasets/train",
        ):
        self.tokenizer = Tools.get_tokenizer(model_name)
        self.llm = llm
        self.model_name = model_name
        self.benchmark = benchmark
        self.sigma_ratio = sigma_ratio
        self.max_new_tokens = max_new_tokens
        self.save_dir = save_dir
        self.task_processor = TaskDataProcess(
            benchmark=benchmark,
            repos=repos, 
            tokenizer=self.tokenizer,
            task_type='train',
            window_type='flexible',
            sigma_ratio=sigma_ratio,
        )
    
    def _analyze_syntax_types(self, code: str) -> List[Dict[str, any]]:
        """
        Analyze syntax type of each token in code based on AST
        
        Args:
            code: Code string to analyze
            
        Returns:
            List[Dict]: List containing position and syntax type of each token
        """
        import ast
        import tokenize
        from io import StringIO
        
        token_info = []
        
        try:
            # Use tokenize module to get detailed token information
            tokens = list(tokenize.generate_tokens(StringIO(code).readline))
            
            for token_type, token_string, start, end, line in tokens:
                # Determine syntax type
                syntax_type = self._get_token_syntax_type(token_type, token_string)
                
                token_info.append({
                    'token': token_string,
                    'start_pos': start[1],  # Column position
                    'end_pos': end[1],      # End column position
                    'line': start[0],       # Line number
                    'syntax_type': syntax_type
                })
                
        except Exception as e:
            logger.warning(f"AST analysis failed: {e}")
            # If AST analysis fails, fall back to simple token analysis
            return self._fallback_token_analysis(code)
        
        return token_info
    
    def _get_token_syntax_type(self, token_type: int, token_string: str) -> str:
        """
        根据token类型和字符串确定语法类型
        
        Args:
            token_type: tokenize模块的token类型
            token_string: token字符串
            
        Returns:
            str: 语法类型
        """
        import tokenize
        
        # 关键字
        keywords = {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 
            'finally', 'with', 'as', 'import', 'from', 'return', 'yield', 'break', 
            'continue', 'pass', 'raise', 'assert', 'del', 'global', 'nonlocal', 
            'lambda', 'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None'
        }
        
        if token_type == tokenize.NAME:
            if token_string in keywords:
                return "keyword"
            else:
                return "identifier"
        elif token_type == tokenize.STRING:
            return "string_literal"
        elif token_type == tokenize.NUMBER:
            return "number_literal"
        elif token_type == tokenize.OP:
            return "operator"
        elif token_type == tokenize.COMMENT:
            return "comment"
        elif token_type == tokenize.NL:
            return "newline"
        elif token_type == tokenize.NEWLINE:
            return "newline"
        elif token_type == tokenize.INDENT:
            return "indent"
        elif token_type == tokenize.DEDENT:
            return "dedent"
        else:
            return "other"
    
    def _fallback_token_analysis(self, code: str) -> List[Dict[str, any]]:
        """
        回退的简单token分析
        
        Args:
            code: 要分析的代码字符串
            
        Returns:
            List[Dict]: 简化的token信息列表
        """
        import re
        
        # 简单的正则表达式分割
        tokens = re.findall(r'\b\w+\b|[^\w\s]|\s+', code)
        token_info = []
        
        for i, token in enumerate(tokens):
            syntax_type = self._simple_syntax_analysis(token)
            token_info.append({
                'token': token,
                'position': i,
                'syntax_type': syntax_type
            })
        
        return token_info
    
    def _simple_syntax_analysis(self, token: str) -> str:
        """
        简单的语法类型分析
        
        Args:
            token: token字符串
            
        Returns:
            str: 语法类型
        """
        # 关键字
        keywords = {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 
            'finally', 'with', 'as', 'import', 'from', 'return', 'yield', 'break', 
            'continue', 'pass', 'raise', 'assert', 'del', 'global', 'nonlocal', 
            'lambda', 'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None'
        }
        
        if token in keywords:
            return "keyword"
        
        # 操作符
        operators = {
            '+', '-', '*', '/', '//', '%', '**', '==', '!=', '<', '>', '<=', '>=',
            '=', '+=', '-=', '*=', '/=', '//=', '%=', '**=', '&=', '|=', '^=',
            '<<=', '>>=', '&', '|', '^', '~', '<<', '>>', '->', ':='
        }
        
        if token in operators:
            return "operator"
        
        # 分隔符
        delimiters = {
            '(', ')', '[', ']', '{', '}', ',', ':', ';', '.', '@', '`'
        }
        
        if token in delimiters:
            return "delimiter"
        
        # 字符串字面量
        if (token.startswith('"') and token.endswith('"')) or \
           (token.startswith("'") and token.endswith("'")) or \
           (token.startswith('"""') and token.endswith('"""')) or \
           (token.startswith("'''") and token.endswith("'''")):
            return "string_literal"
        
        # 数字字面量
        if token.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').isdigit():
            return "number_literal"
        
        # 空白字符
        if token.isspace():
            return "whitespace"
        
        # 标识符
        if token[0].isalpha() or token[0] == '_':
            if all(c.isalnum() or c == '_' for c in token):
                return "identifier"
        
        return "other"
    
    def _get_token_syntax_type_by_position(self, token: str, token_index: int, syntax_analysis: List[Dict]) -> str:
        """
        根据token位置获取语法类型
        
        Args:
            token: 当前token (LLM tokenizer生成的)
            token_index: token在序列中的索引
            syntax_analysis: 语法分析结果 (基于Python语法规则)
            
        Returns:
            str: 语法类型
        """
        # 由于LLM tokenizer和语法分析tokenizer的差异，我们需要更智能的匹配策略
        
        # 策略1: 直接匹配
        for info in syntax_analysis:
            if info['token'] == token:
                return info['syntax_type']
        
        # 策略2: 部分匹配（处理子词分割的情况）
        for info in syntax_analysis:
            if token in info['token'] or info['token'] in token:
                return info['syntax_type']
        
        # 策略3: 基于位置的大致匹配（如果token_index在合理范围内）
        if token_index < len(syntax_analysis):
            return syntax_analysis[token_index]['syntax_type']
        
        # 策略4: 基于token的特征进行推断
        return self._infer_syntax_type_from_token(token)
    
    def _infer_syntax_type_from_token(self, token: str) -> str:
        """
        基于token特征推断语法类型
        
        Args:
            token: LLM tokenizer生成的token
            
        Returns:
            str: 推断的语法类型
        """
        # 去除可能的特殊字符
        clean_token = token.strip()
        
        if not clean_token:
            return "whitespace"
        
        # 检查是否是子词的一部分
        if clean_token.startswith('##') or clean_token.startswith('▁'):
            return "subword"
        
        # 检查是否是特殊token
        if clean_token in ['<s>', '</s>', '<pad>', '<unk>', '<mask>']:
            return "special_token"
        
        # 使用简单规则进行推断
        return self._simple_syntax_analysis(clean_token)
    
    def _create_llm_token_syntax_mapping(self, llm_token_ids: List[int], syntax_analysis: List[Dict]) -> Dict[int, str]:
        """
        创建LLM token ID到语法类型的映射
        
        Args:
            llm_token_ids: LLM tokenizer生成的token ID列表
            syntax_analysis: 语法分析结果
            
        Returns:
            Dict[int, str]: token索引到语法类型的映射
        """
        mapping = {}
        
        # 将语法分析结果转换为字符串
        syntax_tokens = [info['token'] for info in syntax_analysis]
        syntax_types = [info['syntax_type'] for info in syntax_analysis]
        
        # 将LLM token IDs转换为字符串
        llm_tokens = []
        for token_id in llm_token_ids:
            try:
                token_str = self.tokenizer.decode([token_id])
                llm_tokens.append(token_str)
            except:
                llm_tokens.append(f"<token_{token_id}>")
        
        # 使用改进的对齐算法
        mapping = self._improved_align_tokens(llm_tokens, syntax_tokens, syntax_types)
        
        return mapping
    
    def _align_tokens(self, llm_tokens: List[str], syntax_tokens: List[str], syntax_types: List[str]) -> Dict[int, str]:
        """
        对齐LLM tokens和语法分析tokens
        
        Args:
            llm_tokens: LLM tokenizer生成的token列表
            syntax_tokens: 语法分析得到的token列表
            syntax_types: 对应的语法类型列表
            
        Returns:
            Dict[int, str]: LLM token索引到语法类型的映射
        """
        mapping = {}
        
        # 简单的贪心对齐策略
        llm_idx = 0
        syntax_idx = 0
        current_llm_text = ""
        current_syntax_text = ""
        
        while llm_idx < len(llm_tokens) and syntax_idx < len(syntax_tokens):
            current_llm_text += llm_tokens[llm_idx]
            current_syntax_text += syntax_tokens[syntax_idx]
            
            # 如果当前累积的文本匹配
            if current_llm_text.strip() == current_syntax_text.strip():
                # 为当前LLM token分配语法类型
                mapping[llm_idx] = syntax_types[syntax_idx]
                llm_idx += 1
                syntax_idx += 1
                current_llm_text = ""
                current_syntax_text = ""
            elif len(current_llm_text) > len(current_syntax_text):
                # LLM token更长，尝试添加更多语法token
                syntax_idx += 1
                if syntax_idx < len(syntax_tokens):
                    current_syntax_text += syntax_tokens[syntax_idx]
            else:
                # 语法token更长，尝试添加更多LLM token
                llm_idx += 1
                if llm_idx < len(llm_tokens):
                    current_llm_text += llm_tokens[llm_idx]
        
        # 为未映射的token分配默认类型
        for i in range(len(llm_tokens)):
            if i not in mapping:
                mapping[i] = self._infer_syntax_type_from_token(llm_tokens[i])
        
        return mapping
    
    def _improved_align_tokens(self, llm_tokens: List[str], syntax_tokens: List[str], syntax_types: List[str]) -> Dict[int, str]:
        """
        改进的token对齐算法
        
        Args:
            llm_tokens: LLM tokenizer生成的token列表
            syntax_tokens: 语法分析得到的token列表
            syntax_types: 对应的语法类型列表
            
        Returns:
            Dict[int, str]: LLM token索引到语法类型的映射
        """
        mapping = {}
        
        # 构建完整的LLM文本和语法分析文本
        llm_text = ''.join(llm_tokens)
        syntax_text = ''.join(syntax_tokens)
        
        # 如果文本长度差异太大，使用简单的位置映射
        if abs(len(llm_text) - len(syntax_text)) > len(syntax_text) * 0.5:
            for i, llm_token in enumerate(llm_tokens):
                # 使用启发式规则推断语法类型
                mapping[i] = self._infer_syntax_type_from_token(llm_token)
            return mapping
        
        # 使用滑动窗口进行对齐
        llm_idx = 0
        syntax_idx = 0
        
        while llm_idx < len(llm_tokens) and syntax_idx < len(syntax_tokens):
            current_llm_token = llm_tokens[llm_idx]
            current_syntax_token = syntax_tokens[syntax_idx]
            
            # 检查当前token是否匹配
            if current_llm_token == current_syntax_token:
                mapping[llm_idx] = syntax_types[syntax_idx]
                llm_idx += 1
                syntax_idx += 1
            elif current_llm_token in current_syntax_token:
                # LLM token是语法token的一部分
                mapping[llm_idx] = syntax_types[syntax_idx]
                llm_idx += 1
            elif current_syntax_token in current_llm_token:
                # 语法token是LLM token的一部分
                mapping[llm_idx] = syntax_types[syntax_idx]
                llm_idx += 1
                syntax_idx += 1
            else:
                # 不匹配，使用启发式规则
                mapping[llm_idx] = self._infer_syntax_type_from_token(current_llm_token)
                llm_idx += 1
        
        # 为剩余的LLM tokens分配语法类型
        for i in range(llm_idx, len(llm_tokens)):
            mapping[i] = self._infer_syntax_type_from_token(llm_tokens[i])
        
        return mapping
    
    def _get_existing_and_missing_repos(self) -> Tuple[List[str], List[str]]:
        """
        获取已存在和缺失的repo列表
        
        Returns:
            Tuple[List[str], List[str]]: (已存在的repo列表, 缺失的repo列表)
        """
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 使用构造函数中提供的repos参数
        repo_names = self.task_processor.repos
        
        if not repo_names:
            logger.warning("没有找到任何repo，将进行训练数据生成")
            return [], []
        
        # 检查每个repo的训练文件是否存在
        missing_repos = []
        existing_repos = []
        
        for repo_name in repo_names:
            repo_save_dir = os.path.join(self.save_dir, repo_name)
            data_path = f"{repo_save_dir}/{self.model_name}_{self.benchmark}.parquet"
            
            if os.path.exists(data_path):
                # 检查文件大小，确保不是空文件
                file_size = os.path.getsize(data_path)
                if file_size > 0:
                    existing_repos.append(repo_name)
                    logger.info(f"发现已存在的训练文件: {data_path} (大小: {file_size} bytes)")
                else:
                    missing_repos.append(repo_name)
                    logger.warning(f"发现空文件，将重新生成: {data_path}")
            else:
                missing_repos.append(repo_name)
        
        if missing_repos:
            logger.info(f"以下repo缺少训练文件，将进行生成: {missing_repos}")
        if existing_repos:
            logger.info(f"以下repo已有训练文件，将跳过: {existing_repos}")
        
        return existing_repos, missing_repos
        
    def run(self):
        """
        Run training data generation task
        First filter data, then process by repo groups, save after each group is completed
        """
        logger.info("Train task started")
        
        # 1. Load task data
        benchmark_path = FilePathBuilder.get_benchmark_path(self.benchmark)
        tasks = Tools.load_jsonl(benchmark_path)
        
        # 2. Get existing and missing repo lists
        _, missing_repos = self._get_existing_and_missing_repos()
        
        # If all repos already exist, skip
        if not missing_repos:
            logger.info("All repo training files already exist, skipping training data generation")
            return
        
        # 3. Filter data: only keep tasks for missing repos (filter before preprocessing)
        filtered_tasks = []
        for task in tasks:
            # Extract repo name from task_id (task is still in dict format)
            repo_name = task['metadata']['task_id'].split('/')[0]
            if repo_name in missing_repos:
                filtered_tasks.append(task)
        
        logger.info(f"Total {len(tasks)} tasks, {len(filtered_tasks)} tasks remaining after filtering")
        
        # 4. Preprocess filtered tasks
        for task in filtered_tasks:
            task['metadata']['prediction'] = [""]
        filtered_tasks = self.task_processor.process_task_data(filtered_tasks, [0] * len(filtered_tasks))
        

        
        logger.info(f"Total {len(tasks)} tasks, {len(filtered_tasks)} tasks remaining after filtering")
        
        # 4. Group tasks by repo
        repo_tasks = {}
        for task in filtered_tasks:
            repo_name = task.metadata.task_id.split('/')[0]
            if repo_name not in repo_tasks:
                repo_tasks[repo_name] = []
            repo_tasks[repo_name].append(task)
        
        logger.info(f"Grouping by repo completed, {len(repo_tasks)} repos need processing")
        
        # 5. Process by repo groups, save after each group is completed
        total_tasks_processed = 0
        total_tokens_generated = 0
        
        for repo_name, repo_task_list in repo_tasks.items():
            logger.info(f"Starting to process repo: {repo_name}, {len(repo_task_list)} tasks")
            
            repo_token_tasks = []  # TokenTask objects for current repo
            
            for i, task in enumerate(tqdm(repo_task_list, desc=f"Processing {repo_name}")):
                try:
                    # Get task prompt and ground_truth
                    prompt = task.prompt
                    ground_truth = task.metadata.ground_truth
                    
                    # Check if ground_truth is empty
                    if not ground_truth or ground_truth.strip() == "":
                        logger.warning(f"Task {task.metadata.task_id} ground_truth is empty, skipping")
                        continue
                    
                    # Use LLM's get_training_data method to get training data
                    training_records = self.llm.get_training_data(prompt, ground_truth)
                    
                    # Check if training data was successfully generated
                    if not training_records:
                        logger.warning(f"Task {task.metadata.task_id} generated no training records")
                        continue
                    
                    # Convert each training record to TokenTask object
                    task_tokens_count = 0
                    
                    # Perform syntax analysis on ground_truth
                    syntax_analysis = self._analyze_syntax_types(ground_truth)
                    
                    # Create mapping from LLM token to syntax type
                    llm_token_ids = [record['gt_token_id'] for record in training_records]
                    token_syntax_mapping = self._create_llm_token_syntax_mapping(llm_token_ids, syntax_analysis)
                    
                    for record in training_records:
                        # Calculate uncertainty (using entropy as uncertainty metric)
                        uncertainty = record['entropy']
                        
                        # Get syntax type of current token
                        gt_token = record['gt_token']
                        token_index = record['step']
                        syntax_type = token_syntax_mapping.get(token_index, self._infer_syntax_type_from_token(gt_token))
                        
                        # Create TokenTask object
                        token_task = TokenTask(
                            task_id=task.metadata.task_id,
                            token_index=record['step'],
                            ground_truth_token=record['gt_token'],
                            prediction_token=record['pred_token'],
                            ground_truth_token_log_prob=record['gt_logprob'],
                            prediction_token_log_prob=record['pred_logprob'],
                            hidden_states=record['hidden_state'],
                            uncertainty=uncertainty,
                            attention_score=record['attention_scores'],
                            syntax_type=syntax_type  # New: syntax type
                        )
                        
                        repo_token_tasks.append(token_task)
                        task_tokens_count += 1
                    
                    total_tasks_processed += 1
                    total_tokens_generated += task_tokens_count
                    
                    if i % 10 == 0:  # Record progress every 10 tasks
                        logger.info(f"Repo {repo_name}: Processed {i+1}/{len(repo_task_list)} tasks, current repo generated {len(repo_token_tasks)} token samples")
                    
                except Exception as e:
                    logger.error(f"Error processing task {task.metadata.task_id}: {e}")
                    continue
            
            # 6. Save current repo data
            if repo_token_tasks:
                logger.info(f"Saving repo {repo_name} data, {len(repo_token_tasks)} token samples")
                
                # Convert to DataFrame
                repo_df = pd.DataFrame([asdict(token_task) for token_task in repo_token_tasks])
                
                # Ensure save directory exists
                os.makedirs(self.save_dir, exist_ok=True)
                repo_save_dir = os.path.join(self.save_dir, repo_name)
                os.makedirs(repo_save_dir, exist_ok=True)
                
                # Save this repo's data
                output_path = f"{repo_save_dir}/{self.model_name}_{self.benchmark}.parquet"
                repo_df.to_parquet(output_path, index=False)
                logger.info(f"Repo {repo_name} TokenTask data saved to: {output_path}")
                logger.info(f"Repo {repo_name} saved {len(repo_df)} TokenTask records")
            else:
                logger.warning(f"Repo {repo_name} generated no TokenTask data")
        
        logger.info(f"Training data generation completed! Processed {total_tasks_processed} tasks, generated {total_tokens_generated} token samples")


class TokenDataset(Dataset):
    """Token dataset class"""
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]





class TrainClassifierTask:
    """Training classifier task processing class"""
    def __init__(
        self,
        model_name: str = "deepseek-coder-1.3b-base",
        benchmark: str = CONSTANTS.train_dataset,
        save_dir: str = "datasets/train",
        classifier_save_dir: str = "models/classifier",
        uncertainty_threshold: float = 0.8,  # uncertainty threshold
        attention_threshold: float = 0.05,  # Top 5 attention average threshold
        top_k_value: int = 5,  # Top K value
        hidden_size: int = 512,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 50,
        num_classifiers: int = 5,  # Number of classifiers
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.benchmark = benchmark
        self.save_dir = save_dir
        self.classifier_save_dir = classifier_save_dir
        self.uncertainty_threshold = uncertainty_threshold
        self.attention_threshold = attention_threshold
        self.top_k_value = top_k_value
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classifiers = num_classifiers
        self.device = device
        
        # Ensure save directory exists
        os.makedirs(self.classifier_save_dir, exist_ok=True)
        
    def _create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Positive examples are the following cases, representing critical tokens:
        1. Token Mismatch: ground truth token and prediction token are different
        2. High Uncertainty: uncertainty value exceeds threshold
        3. High Top 5 Attention: average of top 5 subsequent attention exceeds threshold
        """
        labels = []
        
        for _, row in df.iterrows():
            is_critical = False
            
            # Condition 1: Token Mismatch
            if row['ground_truth_token'] != row['prediction_token']:
                is_critical = True
            else:
                # Condition 2: High Uncertainty
                if row['uncertainty'] >= self.uncertainty_threshold:
                    is_critical = True
                else:
                    # Condition 3: High Top 5 Attention
                    attention_scores = row['attention_score']
                    if hasattr(attention_scores, '__len__') and len(attention_scores) > 0:
                        # Convert to list
                        if hasattr(attention_scores, 'tolist'):
                            scores_list = attention_scores.tolist()
                        else:
                            scores_list = list(attention_scores)
                        
                        # Exclude self-attention (first score)
                        if len(scores_list) > 1:
                            other_attention_scores = scores_list[1:]
                            sorted_scores = sorted(other_attention_scores, reverse=True)
                            top_k_scores = sorted_scores[:min(self.top_k_value, len(sorted_scores))]
                        else:
                            sorted_scores = sorted(scores_list, reverse=True)
                            top_k_scores = sorted_scores[:min(self.top_k_value, len(sorted_scores))]
                        
                        top_k_mean = np.mean(top_k_scores)
                        if top_k_mean >= self.attention_threshold:
                            is_critical = True
            
            labels.append(1 if is_critical else 0)
        
        return np.array(labels)
    def _prepare_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Prepare feature data, only including hidden layer states
        """
        features = []
        
        for _, row in df.iterrows():
            # Only use hidden_states as features
            hidden_states = row['hidden_states']
            if hasattr(hidden_states, '__len__') and len(hidden_states) > 0:
                # Convert to list
                if hasattr(hidden_states, 'tolist'):
                    hidden_list = hidden_states.tolist()
                else:
                    hidden_list = list(hidden_states)
                features.append(hidden_list)
            else:
                logger.warning(f"Task {row['task_id']} hidden_states is empty, skipping")
                continue
        
        return torch.tensor(features, dtype=torch.float32)
    def _balance_dataset(self, features: torch.Tensor, labels: np.ndarray, df: pd.DataFrame, negative_ratio: int = 1, num_classifiers: int = 1) -> List[Tuple[Tuple[torch.Tensor, np.ndarray], Tuple[torch.Tensor, np.ndarray]]]:
        """
        Negative examples are sampled based on Self Information, sampling from large to small until same number as positive examples
        Then, both positive and negative examples are randomly divided into five parts, and each part cuts its own validation set
        
        Returns:
            List[Tuple]: Each element contains ((train_features, train_labels), (val_features, val_labels))
        """
        # Separate positive and negative examples
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]
        
        logger.info(f"Original data distribution: {len(positive_indices)} positive examples, {len(negative_indices)} negative examples")
        
        # Get Self Information values for negative examples (extracted from original data)
        self_info_values = []
        for idx in negative_indices:
            # Get Self Information from original data
            self_information = -df.iloc[idx]['prediction_token_log_prob']
            self_info_values.append(self_information)
        
        # Sort negative example indices by Self Information from large to small
        sorted_negative_indices = [negative_indices[i] for i in np.argsort(self_info_values)[::-1]]
        
        # Select same number of negative examples as positive examples
        selected_negative_indices = sorted_negative_indices[:len(positive_indices)]
        
        logger.info(f"Select negative examples based on self-information value: {len(positive_indices)} positive examples, {len(selected_negative_indices)} negative examples (ratio 1:1)")
        
        # Randomly split positive and negative examples into num_classifiers parts
        positive_indices_array = np.array(positive_indices)
        negative_indices_array = np.array(selected_negative_indices)
        
        # Fix random seed to ensure reproducibility
        np.random.seed(42)
        np.random.shuffle(positive_indices_array)
        np.random.shuffle(negative_indices_array)
        
        # Split positive and negative examples
        positive_chunks = np.array_split(positive_indices_array, num_classifiers)
        negative_chunks = np.array_split(negative_indices_array, num_classifiers)
        
        # Create training and validation sets for each classifier
        datasets = []
        for i in range(num_classifiers):
            logger.info(f"Preparing data for classifier {i+1}...")
            
            # 获取当前分类器的正例和负例
            current_positive_indices = positive_chunks[i]
            current_negative_indices = negative_chunks[i]
            
            # 为当前分类器的正例划分训练集和验证集
            np.random.seed(42 + i)  # 每个分类器使用不同的随机种子
            np.random.shuffle(current_positive_indices)
            positive_train_size = int(0.8 * len(current_positive_indices))
            positive_train_indices = current_positive_indices[:positive_train_size]
            positive_val_indices = current_positive_indices[positive_train_size:]
            
            # 为当前分类器的负例划分训练集和验证集
            np.random.seed(42 + i)  # 使用相同的随机种子确保一致性
            np.random.shuffle(current_negative_indices)
            negative_train_size = int(0.8 * len(current_negative_indices))
            negative_train_indices = current_negative_indices[:negative_train_size]
            negative_val_indices = current_negative_indices[negative_train_size:]
            
            # 获取训练集数据
            train_positive_features = features[positive_train_indices]
            train_positive_labels = labels[positive_train_indices]
            train_negative_features = features[negative_train_indices]
            train_negative_labels = labels[negative_train_indices]
            
            # 获取验证集数据
            val_positive_features = features[positive_val_indices]
            val_positive_labels = labels[positive_val_indices]
            val_negative_features = features[negative_val_indices]
            val_negative_labels = labels[negative_val_indices]
            
            # 合并训练集
            train_features = torch.cat([train_positive_features, train_negative_features], dim=0)
            train_labels = np.concatenate([train_positive_labels, train_negative_labels])
            
            # 合并验证集
            val_features = torch.cat([val_positive_features, val_negative_features], dim=0)
            val_labels = np.concatenate([val_positive_labels, val_negative_labels])
            
            # 打乱训练集和验证集
            train_indices = np.arange(len(train_features))
            val_indices = np.arange(len(val_features))
            np.random.seed(42 + i)
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
            
            train_features = train_features[train_indices]
            train_labels = train_labels[train_indices]
            val_features = val_features[val_indices]
            val_labels = val_labels[val_indices]
            
            # 确保数据类型正确
            train_features = train_features.float()
            val_features = val_features.float()
            
            datasets.append(((train_features, train_labels), (val_features, val_labels)))
            
            logger.info(f"Classifier {i+1} training set: {len(train_positive_features)} positive examples, {len(train_negative_features)} negative examples")
            logger.info(f"Classifier {i+1} validation set: {len(val_positive_features)} positive examples, {len(val_negative_features)} negative examples")
        
        return datasets
    def _prepare_balanced_data_for_multiple_classifiers(self, df: pd.DataFrame) -> List[Tuple[Tuple[torch.Tensor, np.ndarray], Tuple[torch.Tensor, np.ndarray]]]:
        """
        Prepare balanced training data for multiple classifiers
        
        Args:
            df: Original dataframe
            
        Returns:
            List[Tuple]: Each element contains ((train_features, train_labels), (val_features, val_labels))
        """
        # Create labels
        labels = self._create_labels(df)
        df['label'] = labels
        
        # Prepare features
        features = self._prepare_features(df)
        
        # Use _balance_dataset method to complete all data processing
        logger.info("Using _balance_dataset method to complete data processing...")
        datasets = self._balance_dataset(features, labels, df, num_classifiers=self.num_classifiers)
        
        return datasets
    
    def _train_multiple_classifiers(self, balanced_datasets: List[Tuple[Tuple[torch.Tensor, np.ndarray], Tuple[torch.Tensor, np.ndarray]]]):
        """
        Train multiple classifiers
        
        Args:
            balanced_datasets: List of balanced datasets, each element contains ((train_features, train_labels), (val_features, val_labels))
        """
        trained_classifiers = []
        
        for i, ((train_features, train_labels), (val_features, val_labels)) in enumerate(balanced_datasets):
            logger.info(f"Starting to train classifier {i+1}...")
            
            # Create data loaders
            train_dataset = TokenDataset(train_features, torch.tensor(train_labels, dtype=torch.long))
            val_dataset = TokenDataset(val_features, torch.tensor(val_labels, dtype=torch.long))
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Train classifier
            classifier = self._train_classifier(train_loader, val_loader, train_features.shape[1])
            
            # Save classifier
            save_path = os.path.join(
                self.classifier_save_dir,
                f"{self.model_name}_classifier_{i+1}.pth"
            )
            torch.save(classifier.state_dict(), save_path)
            logger.info(f"Classifier {i+1} saved to: {save_path}")
            
            trained_classifiers.append(classifier)
        
        return trained_classifiers
    def _train_classifier(self, train_loader: DataLoader, val_loader: DataLoader, input_size: int) -> Classifier:
        """
        Train classifier
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            input_size: Input feature dimension
            
        Returns:
            Classifier: Trained classifier
        """
        model = Classifier(input_dim=input_size, train_mode=True).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)  # Add L2 regularization
        
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        best_val_acc = 0.0
        best_model = None
        patience_counter = 0
        early_stopping_patience = 20  # Early stopping patience value
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device).float()
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model.get_logits(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device).float()
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = model.get_logits(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            logger.info(f'Epoch [{epoch+1}/{self.num_epochs}] - '
                       f'Train Loss: {train_loss/len(train_loader):.4f}, '
                       f'Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {val_loss/len(val_loader):.4f}, '
                       f'Val Acc: {val_acc:.2f}%, '
                       f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
            
            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_acc)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                logger.info(f"Learning rate adjusted from {old_lr:.2e} to {new_lr:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered, stopped training at epoch {epoch+1}")
                break
        
        return best_model
    
    def run(self):
        """
        Run training classifier task
        """
        logger.info("Starting multi-classifier training task")
        
        # Load training data for all repos
        all_dfs = []
        total_files = 0
        
        # Traverse all repo subdirectories under save_dir
        if os.path.exists(self.save_dir):
            for repo_name in os.listdir(self.save_dir):
                repo_dir = os.path.join(self.save_dir, repo_name)
                if os.path.isdir(repo_dir):
                    data_path = f"{repo_dir}/{self.model_name}_{self.benchmark}.parquet"
                    logger.info(f"加载repo {repo_name} 的训练数据: {data_path}")
                    if os.path.exists(data_path):
                        try:
                            repo_df = pd.read_parquet(data_path)
                            repo_df['repo_name'] = repo_name  # 添加repo_name列以便追踪
                            all_dfs.append(repo_df)
                            logger.info(f"加载了repo {repo_name} 的 {len(repo_df)} 条训练数据")
                            total_files += 1
                        except Exception as e:
                            logger.error(f"加载repo {repo_name} 的数据时出错: {e}")
                            continue
        
        if not all_dfs:
            logger.error(f"在目录 {self.save_dir} 下没有找到任何训练数据文件")
            logger.error("请先运行 'python src/pipeline.py task train' 生成训练数据")
            return
        
        # 合并所有repo的数据
        df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"总共加载了 {total_files} 个repo的 {len(df)} 条训练数据")
        
        # 为多分类器准备平衡数据
        logger.info("为多分类器准备平衡数据...")
        balanced_datasets = self._prepare_balanced_data_for_multiple_classifiers(df)
        
        # 训练多个分类器
        logger.info(f"开始训练 {self.num_classifiers} 个分类器...")
        trained_classifiers = self._train_multiple_classifiers(balanced_datasets)
        
        logger.info(f"多分类器训练完成！共训练了 {len(trained_classifiers)} 个分类器")
        
        # 保存评估结果
        eval_results = {
            'num_classifiers': self.num_classifiers,
            'data_dir': self.save_dir,
            'num_samples': len(df),
            'num_repos': total_files,
            'model_paths': [f"{self.model_name}_classifier_{i+1}.pth" for i in range(self.num_classifiers)]
        }
        
        eval_path = f"{self.classifier_save_dir}/{self.model_name}_{self.benchmark}_multi_eval_results.json"
        import json
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"多分类器评估结果已保存到: {eval_path}")
        logger.info("多分类器训练任务完成")



