from loguru import logger
from typing import List, Type, Dict, Tuple
from .process import (
    TaskWindowMaker, 
    UniXcoderEmbedding, DeepSeekTokenizer, 
    QueryVector, QuerySearch, 
    CodeLlamaTokenizer,
    TaskPromptMaker, QueryPrompt, TaskRetrieverMaker, QueryWindow, WindowData
)


class TaskDataProcess:
    def __init__(
        self, 
        benchmark: str,
        repos: List[str], 
        tokenizer,
        task_type: str,
        window_type: str,
        sigma_ratio: float | None = None,
    ):
        """
        Initialize task data processing class
        
        Args:
            benchmark: Benchmark name
            repos: Repository list
            tokenizer: tokenizer instance (one-gram type required)
            task_type: Task type, 'token', 'train'
            window_type: Window type, 'flexible'
            sigma_ratio: Similarity weight, default 0
        """
        assert task_type in ['token', 'train']
        
        self.benchmark = benchmark
        self.repos = repos
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.window_type = window_type
        self.sigma_ratio = sigma_ratio

        assert self.window_type == 'flexible'

        logger.info(f"Initializing {self.task_type} task data processor for {self.benchmark}...")
        logger.info(f"Using {self.window_type} window type and {self.sigma_ratio} sigma ratio...")
        
        self._DataMaker_init()


    def _DataMaker_init(self):
        self.window_maker = TaskWindowMaker(
            self.benchmark, self.repos, self.task_type
        )
        
        self.vector_maker = UniXcoderEmbedding(use_position_weighting=True, sigma_ratio=self.sigma_ratio)
            
        self.retriever_maker = TaskRetrieverMaker(task_type=self.task_type,
                                                  repos=self.repos,
                                                  benchmark=self.benchmark,
                                                  window_type=self.window_type)
        
            
        self.prompt_maker = TaskPromptMaker(self.tokenizer, self.task_type)
        
    def make_window(self, tasks: List[Dict], current_generation: List[int] | None = None) -> List[QueryWindow]:
        """
        Build corresponding type of windows based on task_type
        
        Args:
            tasks: Task list
            current_generation: Current generation number
        Returns:
            window_results: Window results, format is [WindowResult]
        """
        return self.window_maker.build_window(tasks, current_generation)

    def make_vector(self, window_results: List[QueryWindow]) -> List[QueryVector]:
        """
        Build vectors for corresponding type of windows based on task_type
        
        Args:
            window_results: Window results, format is [WindowResult]
        """
        return self.vector_maker.build_vectors(window_results)

    def search_context(self, vector_results: List[QueryVector]) -> List[QuerySearch]:
        """
        Search relevant contexts based on vector results
        
        Args:
            vector_results: List of vector results
            
        Returns:
            List[SearchResult]: Top-k most similar vector results, each element is (vector_result, similarity_score)
        """
        return self.retriever_maker.search_contexts(vector_results)
    
    def build_prompt(self, search_results: List[QuerySearch]) -> List[QueryPrompt]:
        """
        Build prompts
        
        Args:
            search_results: List of search results
            
        Returns:
            prompt_results: List of prompt results
        """
        return self.prompt_maker.build_prompts(search_results)


    def process_task_data(self, tasks: List[Dict], current_generation: List[int] | None = None, log_enable: bool = True) -> List[QueryPrompt]:
        """
        Complete workflow for processing task data
        
        Args:
            tasks: Task list
            current_generation: Current generation number
            log_enable: Whether to print logs
        """
        if log_enable:
            logger.info(f"--- Processing {self.task_type} task data for {self.benchmark} ---")
            logger.info(f"Building {self.task_type} windows...")
        window_results = self.make_window(tasks, current_generation)
        if log_enable:
            logger.info(f"{self.task_type} windows built")
            logger.info(f"Building {self.task_type} vectors...")
        vector_results = self.make_vector(window_results)
        if log_enable:
            logger.info(f"{self.task_type} vectors built")
            logger.info(f"Searching {self.task_type} context...")
        search_results = self.search_context(vector_results)
        if log_enable:
            logger.info(f"{self.task_type} context searched")
            logger.info(f"Building {self.task_type} prompt...")
        prompt_results = self.build_prompt(search_results)
        if log_enable:
            logger.info(f"{self.task_type} prompt built")
            logger.info(f"--- {self.task_type} task data processing completed for {self.benchmark} ---")
        return prompt_results