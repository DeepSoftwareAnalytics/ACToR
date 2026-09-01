import copy
import os
from typing import List
from tqdm import tqdm
from data import (
    CONSTANTS, 
    TaskDataProcess, 
    FilePathBuilder, 
    Tools,
)
from dataclasses import asdict
from server import Classifier, MultiClassifier
from loguru import logger
import re

def match_end_of_function(prediction):
    """
    Use regular expression to determine if prediction has reached the end of function position
    
    Args:
        prediction: Predicted string
        
    Returns:
        tuple: (bool, str) - (whether function end is matched, string before match position)
    """
    
    # Use regular expression to match pattern of newline followed by non-whitespace non-newline characters
    # \n matches newline
    # [^\s] matches any non-whitespace character (including non-space, non-tab, non-newline, etc.)
    pattern = r'\n[^\s]'
    
    # Use search method to find if there's a match
    match_result = re.search(pattern, prediction)
    
    if match_result is not None:
        # Return string before match position (including newline)
        cut_index = match_result.start() + 1
        matched_prefix = prediction[:cut_index]
        return True, matched_prefix
    else:
        return False, prediction
    
    
class TokenTask:
    """Token task processing class"""
    def __init__(
        self,
        llm=None,
        model_name=None,
        benchmark: str = None,
        repos: List[str] = None,
        classifier_dir="/root/autodl-tmp/CriGen/models/classifier",
        sigma_ratio: float = 0.4,
        max_new_tokens: int = 512,
        n_samples: int = 5,
        ):
        
        classifier_path = os.path.join(classifier_dir, model_name + "_classifier.pth")
        if classifier_path is None:
            raise ValueError("classifier_path is required")
        if benchmark is None:
            benchmark = CONSTANTS.codereval_python_benchmark
        if repos is None:
            repos = CONSTANTS.codereval_python_repos
        self.tokenizer = Tools.get_tokenizer(model_name)
        self.input_dim = Tools.get_input_dim(model_name)
        self.llm = llm
        self.model_name = model_name
        self.classifier_path = classifier_path
        self.benchmark = benchmark
        self.repos = repos
        self.sigma_ratio = sigma_ratio
        self.max_new_tokens = max_new_tokens
        self.n_samples = n_samples
        # Use multi-classifier
        self.classifier = MultiClassifier(
            base_classifier_path=classifier_path,
            num_classifiers=5,
            input_dim=self.input_dim
        )
        self.task_processor = TaskDataProcess(
            benchmark=benchmark,
            repos=repos, 
            tokenizer=self.tokenizer,
            task_type='token',
            sigma_ratio=sigma_ratio,
            window_type='flexible'
        )
        # KV cache management
        self.past_key_values = None
        self.is_cache_initialized = False
        self.current_token_id = None
    
    def _initialize_cache(self, prompt: str):
        """Initialize KV cache"""
        last_hidden_state, current_token, current_token_id, past_key_values = self.llm.generate_token(prompt)
        self.past_key_values = past_key_values
        self.is_cache_initialized = True
        self.current_token_id = current_token_id
        return last_hidden_state, current_token, current_token_id
    
    def _generate_with_cache(self, token_id: int):
        """Generate token using KV cache"""
        last_hidden_state, current_token, current_token_id, new_past_key_values = self.llm.generate_token_with_cache(
            token_id, self.past_key_values
        )
        self.past_key_values = new_past_key_values
        self.current_token_id = current_token_id
        return last_hidden_state, current_token, current_token_id
    
    def _reset_cache(self):
        """Reset KV cache"""
        self.past_key_values = None
        self.is_cache_initialized = False
        self.current_token_id = None
    
    def _generate_tokens_for_task(self, token_task, task_index: int):
        """
        Generate tokens for specified task
        
        Args:
            token_task: Task object to process
            task_index: Task index for logging
            
        Returns:
            tuple: (bool, str, int, int) - (whether generation successful, complete prediction content, critical token count, context changed count)
        """
        critical_count = 0
        context_changed_count = 0
        
        try:
            # Initialize KV cache
            self._reset_cache()
            last_hidden_state, current_token, current_token_id = self._initialize_cache(token_task.prompt)
            
            # Use token_id list instead of string concatenation
            generated_token_ids = [current_token_id]
            
            for j in range(1, self.max_new_tokens):  # Start from 1, as first token is already generated
                # Use KV cache to generate next token
                # Note: Here we pass current_token_id, generate_token_with_cache will generate next token based on this
                last_hidden_state, current_token, current_token_id = self._generate_with_cache(current_token_id)
                
                is_critical = self.classifier.judge(last_hidden_state)
                
                if is_critical:
                    critical_count += 1
                    logger.info(f"Process task {task_index} at token {j}: Token {current_token} is critical.")
            
                    
                    token_task.metadata.prediction[0] = self.tokenizer.decode(generated_token_ids+[current_token_id])
                    # Update task object
                    token_task.prompt = token_task.metadata.target_function_prompt
                    
                    # Check if context has changed
                    token_task_dict = asdict(token_task)
                    processed_tasks = self.task_processor.process_task_data([token_task_dict], [0], log_enable=False)
                    new_token_task = processed_tasks[0]
                    
                    # If context has changed, reset cache and reinitialize
                    if self._check_context_change(token_task, new_token_task):
                        context_changed_count += 1
                        logger.info("Context changed, regenerating token")
                        # Reset cache and reinitialize
                        self._reset_cache()
                        _, current_token, current_token_id = self._initialize_cache(
                            new_token_task.prompt + self.tokenizer.decode(generated_token_ids)
                        )
                        generated_token_ids.append(current_token_id)
                        token_task = new_token_task
                    else:
                        # Context hasn't changed, directly add current token
                        generated_token_ids.append(current_token_id)
                    
                    # Check if function end is reached
                    current_prediction = self.tokenizer.decode(generated_token_ids)
                    is_end, matched_prefix = match_end_of_function(current_prediction)
                    if is_end:
                        # If function end is reached, need to find corresponding token_id position
                        matched_token_ids = self._find_matching_token_ids(generated_token_ids, matched_prefix)
                        generated_token_ids = matched_token_ids
                        break
                else:
                    # Non-critical token, directly add
                    generated_token_ids.append(current_token_id)
                    
                    # Check if function end is reached
                    current_prediction = self.tokenizer.decode(generated_token_ids)
                    is_end, matched_prefix = match_end_of_function(current_prediction)
                    if is_end:
                        # If function end is reached, need to find corresponding token_id position
                        matched_token_ids = self._find_matching_token_ids(generated_token_ids, matched_prefix)
                        generated_token_ids = matched_token_ids
                        break

            # Decode token_ids to final string
            final_prediction = self.tokenizer.decode(generated_token_ids)
            token_task.metadata.prediction[0] = final_prediction
            
            # Check if prediction is empty
            if not final_prediction or final_prediction.strip() == "" or final_prediction.strip() == "pass":
                logger.warning(f"Task {task_index} prediction is empty, regenerating...")
                return False, "", critical_count, context_changed_count
            
            logger.info(f"prediction:\n{final_prediction}")
            return True, final_prediction, critical_count, context_changed_count
            
        except Exception as e:
            logger.error(f"Error generating tokens: {e}")
            return False, "", critical_count, context_changed_count
        
    def _check_context_change(self, original_task, new_task) -> bool:
        """
        Check if context has changed
        
        Args:
            original_task: Original task object
            new_task: New task object
            
        Returns:
            bool: Returns True if context has changed, otherwise False
        """
        # First compare top_k_contexts_selected
        original_selected = original_task.top_k_contexts_selected
        new_selected = new_task.top_k_contexts_selected
        
        if original_selected != new_selected:
            return True
        
        # If top_k_contexts_selected are the same, compare detailed information in top_k_contexts_list
        original_contexts = original_task.top_k_contexts_list
        new_contexts = new_task.top_k_contexts_list
        
        # Compare fpath, start_line_no, end_line_no of each context
        for i in range(original_selected):
            orig_metadata = original_contexts[i][0].metadata  # (WindowData, similarity)
            new_metadata = new_contexts[i][0].metadata
            
            if (orig_metadata.fpath != new_metadata.fpath or
                orig_metadata.start_line_no != new_metadata.start_line_no or
                orig_metadata.end_line_no != new_metadata.end_line_no):
                return True
        
        return False
    
    def _find_matching_token_ids(self, token_ids: List[int], target_text: str) -> List[int]:
        """
        Find corresponding token_id list based on target text
        
        Args:
            token_ids: Complete token_id list
            target_text: Target text
            
        Returns:
            List[int]: Matching token_id list
        """
        # Start from complete token_ids, gradually reduce length until decoded result matches target_text
        for i in range(len(token_ids), 0, -1):
            candidate_ids = token_ids[:i]
            candidate_text = self.tokenizer.decode(candidate_ids)
            if candidate_text == target_text:
                return candidate_ids
        
        # If no exact match found, return closest match
        logger.warning(f"Unable to find exact matching token_ids, returning original list")
        return token_ids
        
    def run(self):
        
        benchmark_path = FilePathBuilder.get_benchmark_path(self.benchmark)
        tasks = Tools.load_jsonl(benchmark_path)
        for task in tasks:
            task['metadata']['prediction'] = [""] * 5
        # tasks = [tasks[20]]
        tasks = self.task_processor.process_task_data(tasks, [0] * len(tasks))
        logger.info("Token task started")
        
        token_results = []
        total_critical_count = 0
        total_context_changed_count = 0
        successful_tasks = 0
        
        for i, task in enumerate(tqdm(tasks, desc="Inference progress")):
            try:
                token_task_result = []
                token_task_list = [copy.deepcopy(task) for _ in range(self.n_samples)]
                
                task_critical_count = 0
                task_context_changed_count = 0
                
                for token_task in token_task_list:
                    # Use function to generate tokens, retry if failed
                    success, full_prediction, critical_count, context_changed_count = self._generate_tokens_for_task(token_task, i)
                    while (not success):
                        success, full_prediction, critical_count, context_changed_count = self._generate_tokens_for_task(token_task, i)
                        continue
                    # Use returned complete prediction to update token_task
                    token_task.metadata.prediction[0] = full_prediction
                    # logger.info(f"token_task.metadata.prediction[0]: {token_task.metadata.prediction[0]}")
                    token_task_result.append(token_task)
                    
                    # Accumulate statistics for each token_task
                    task_critical_count += critical_count
                    task_context_changed_count += context_changed_count
                
                # Process results
                prediction = [token_task.metadata.prediction[0] for token_task in token_task_result]
                # logger.info(f"prediction {i}:\n{prediction}"for i, prediction in enumerate(prediction))
                token_task_result[0].metadata.prediction = prediction
                token_results.append(token_task_result[0])
                
                # Count critical tokens and context changed times for successful tasks
                successful_tasks += 1
                total_critical_count += task_critical_count
                total_context_changed_count += task_context_changed_count

            except Exception as e:
                logger.error(f"Error during inference of sample {i}: {e}")
                token_results.append(task)
        
        
        output_file_path = FilePathBuilder.get_result_save_dir(
            benchmark_type=self.benchmark,
            task_type='token',
            model_name=self.model_name,
            sigma_ratio=self.sigma_ratio,
            window_type='flexible'
        )
                    
        if self.benchmark == CONSTANTS.repoexec_benchmark:
            output_file_path = output_file_path.replace('.jsonl', '.json')
            logger.info(f"Saving RepoExec formatted results to: {output_file_path}")
            
            Tools.format_repoexec_predictions(token_results, output_file_path)
                
        elif self.benchmark == CONSTANTS.codereval_python_benchmark:
            logger.info(f"Saving CoderEval Python formatted results to: {output_file_path}")
            Tools.format_codereval_python_predictions(token_results, output_file_path)
        else:
            logger.info(f"Saving original dataclass results to: {output_file_path}")
            Tools.dump_dataclass(token_results, output_file_path)
        
        # Output statistics
        if successful_tasks > 0:
            # Each task generates 5 times, so need to divide by 5
            n_generations_per_task = self.n_samples
            avg_critical_per_task = total_critical_count / (successful_tasks * n_generations_per_task)
            avg_context_changed_per_task = total_context_changed_count / (successful_tasks * n_generations_per_task)
            logger.info(f"Inference completed! Statistics:")
            logger.info(f"  Successful tasks: {successful_tasks}")
            logger.info(f"  Generations per task: {n_generations_per_task}")
            # logger.info(f"  Total Critical Token count: {total_critical_count}")
            # logger.info(f"  Total Context Changed count: {total_context_changed_count}")
            logger.info(f"  Average Critical Token count per generation: {avg_critical_per_task:.2f}")
            logger.info(f"  Average Context Changed count per generation: {avg_context_changed_per_task:.2f}")
        else:
            logger.info("Inference completed! No successful tasks.")