from typing import List, Tuple
from .utils import DeepSeekTokenizer, CodeLlamaTokenizer
from .data import QuerySearch, QueryPrompt, WindowData

class TaskPromptMaker:
    """Task prompt builder for constructing prompts with relevant context for code generation tasks"""
    
    def __init__(self, tokenizer: DeepSeekTokenizer | CodeLlamaTokenizer, task_type: str) -> None:
        """
        Initialize task prompt builder
        
        Args:
            tokenizer: tokenizer type, supports DeepSeekTokenizer, CodeLlamaTokenizer
        """
        self.tokenizer = tokenizer
        self.max_retrieval_length = 1000
        self.max_examples = 10
        self.task_type = task_type
        assert self.task_type in ['token', 'train']

    def _get_context_block(self, retrieved_context: Tuple[WindowData, float]) -> Tuple[str, int]:
        """
        Get file path and content of retrieved context
        
        Args:
            retrieved_context: Tuple containing window data and similarity score
        Returns:
            Tuple[str, int]: Tuple containing file path and content, and token length
        """
        window_data, _ = retrieved_context
        metadata = window_data.metadata
        assert metadata.fpath.split('/')[0] == metadata.repo
        fpaths = '/'.join(metadata.fpath.split('/')[1:])
        fpaths_context = f'# file path: {fpaths}'
        content_lines = window_data.context.splitlines()
        content_lines_comment = [f'# {line}' for line in content_lines]
        window_context = '\n'.join(content_lines_comment)
        context_block = f'{fpaths_context}\n\n{window_context}'
        total_token_length = len(self.tokenizer.tokenize(context_block))
        return context_block, total_token_length
    
    def _build_prompt(self, prompt: str, top_k_context: List[Tuple[WindowData, float]], fpath_tuple: Tuple[str, ...]) -> Tuple[str, List[Tuple[WindowData, float]]]:
        """
        Build prompt with relevant context
        
        Args:
            prompt: Original prompt text
            top_k_context: List of top-k most relevant contexts
            
        Returns:
            Tuple[str, List[Tuple[WindowData, float]]]: Tuple containing built prompt and selected context list
        """
        current_token_length = 0
        prepend_blocks = []
        chosen_context = []
        for retrieved_context in top_k_context:
            if len(chosen_context) >= self.max_examples:
                break
            context_block, total_token_length = self._get_context_block(retrieved_context)
            if current_token_length + total_token_length < self.max_retrieval_length:
                prepend_blocks.append(context_block) 
                current_token_length += total_token_length
                chosen_context.append(retrieved_context)
            else:
                continue
        fpaths = '/'.join(fpath_tuple[1:])
        fpaths_context = f'\n\n# file path: {fpaths}\n\n'
        prepend_context = '\n\n'.join(prepend_blocks)
        return prepend_context + fpaths_context + prompt, chosen_context

    
    def build_prompts(self, search_results: List[QuerySearch]) -> List[QueryPrompt]:
        """
        Build prompt result list for search results
        
        Args:
            search_results: List of search results containing retrieved relevant contexts
            
        Returns:
            List[QueryPrompt]: List of prompt results, each containing built prompt and selected contexts
        """
        prompt_results = []
        for search_result in search_results:
            prompt, chosen_context = self._build_prompt(search_result.prompt, search_result.top_k_contexts_list, search_result.metadata.fpath_tuple)

            prompt_results.append(QueryPrompt(
                prompt=prompt,
                metadata=search_result.metadata,
                top_k_contexts_list=search_result.top_k_contexts_list,
                top_k_contexts_selected=len(chosen_context)
            ))
        return prompt_results