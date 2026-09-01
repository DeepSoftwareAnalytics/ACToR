import ast
from loguru import logger
from collections import defaultdict
from typing import List, Dict, Tuple, Any
from .utils import Tools, FilePathBuilder
from .data import QueryInfo, QueryWindow

class RepoFlexibleWindowMaker:
    def __init__(
        self, 
        benchmark: str,
        repo: str, 
        window_max_len: int,
    ) -> None:
        """
        Initialize flexible window repository window maker
        
        Args:
            benchmark: Benchmark name
            repo: Repository name
            window_max_len: Maximum window length
        """
        self.benchmark = benchmark
        self.repo = repo
        self.window_max_len = window_max_len
        self.source_code_files = Tools.iterate_repository(FilePathBuilder.get_repo_base_dir(benchmark), repo)
    
    def _merge_windows_with_same_context(self, code_windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged_code_windows = defaultdict(list)
        for code_window in code_windows:
            context = code_window['context']
            metadata = code_window['metadata']
            merged_code_windows[context].append(metadata)
        json_lines = []
        for context, subsequent_context_metadata_list in merged_code_windows.items():
            metadata = subsequent_context_metadata_list[0]
            json_lines.append({
                'context': context,
                'metadata': metadata
            })
        return json_lines
    
    def _build_flexible_windows_for_a_file(self, fpath_tuple: Tuple[str, ...], code: str) -> List[Dict[str, Any]]:
        """
        Build flexible windows for a single file
        
        Args:
            fpath_tuple: File path tuple
            code: File content
            
        Returns:
            List[Dict[str, Any]]: List of windows
        """
        code_windows = []
        code_lines = code.splitlines()
        
        non_empty_line_numbers = []
        for i, line in enumerate(code_lines):
            if line.strip() != '':
                non_empty_line_numbers.append(i + 1)
        
        mini_blocks = []
        mini_block_line_numbers = []
        current_block = []
        current_block_line_numbers = []
        
        for i, line in enumerate(code_lines):
            if line.strip() == '':  
                if current_block: 
                    mini_blocks.append(current_block)
                    mini_block_line_numbers.append(current_block_line_numbers)
                    current_block = []
                    current_block_line_numbers = []
            else:
                current_block.append(line)
                current_block_line_numbers.append(i + 1)
        
        if current_block: 
            mini_blocks.append(current_block)
            mini_block_line_numbers.append(current_block_line_numbers)

        max_len = self.window_max_len
        temp_mini_blocks = []
        temp_mini_block_line_numbers = []
        
        for block, line_numbers in zip(mini_blocks, mini_block_line_numbers):
            if len(block) > max_len:
                for idx in range(0, len(block), max_len):
                    temp_mini_blocks.append(block[idx: idx+max_len])
                    temp_mini_block_line_numbers.append(line_numbers[idx: idx+max_len])
            else:
                temp_mini_blocks.append(block)
                temp_mini_block_line_numbers.append(line_numbers)
        
        mini_blocks = temp_mini_blocks
        mini_block_line_numbers = temp_mini_block_line_numbers

        current_content = []
        current_line_numbers = []
        fpath = '/'.join(fpath_tuple)
        
        for block, block_line_numbers in zip(mini_blocks, mini_block_line_numbers):
            if len(current_content) >= 5000:  
                break  
            if len(current_content) + len(block) <= self.window_max_len:  
                current_content.extend(block)
                current_line_numbers.extend(block_line_numbers)
            else:  
                if current_content:  
                    window_text = '\n'.join(current_content)
                    start_line_no = min(current_line_numbers)
                    end_line_no = max(current_line_numbers)
                    
                    code_windows.append({
                        'context': window_text,
                        'metadata': {
                            'fpath': fpath,
                            'start_line_no': start_line_no,
                            'end_line_no': end_line_no,
                            'repo': self.repo,
                        }
                    })
                current_content = block  
                current_line_numbers = block_line_numbers
        
        if current_content:  
            window_text = '\n'.join(current_content)
            start_line_no = min(current_line_numbers)
            end_line_no = max(current_line_numbers)
            
            code_windows.append({
                'context': window_text,
                'metadata': {
                    'fpath': fpath,
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'repo': self.repo,
                }
            })
        
        return code_windows
    
    def build_windows(self) -> List[Dict[str, Any]]:
        all_code_windows = []
        for fpath_tuple, code in self.source_code_files.items(): 
            all_code_windows += self._build_flexible_windows_for_a_file(fpath_tuple, code)
        merged_code_windows = self._merge_windows_with_same_context(all_code_windows)
        logger.info(f'build {len(merged_code_windows)} flexible windows for {self.repo} with window max len {self.window_max_len}')
        return merged_code_windows


class TaskWindowMaker:
    """Window maker for token and train task types"""
    
    def __init__(
        self, 
        benchmark: str, 
        repos: List[str], 
        task_type: str = 'token', 
    ) -> None:
        """
        Initialize task window maker
        
        Args:
            benchmark: Benchmark name
            repos: List of repository names
            task_type: Task type, 'token', 'train'
        """
        assert task_type in ['token', 'train']
        self.benchmark = benchmark
        self.repos = repos
        self.task_type = task_type
        self.source_code = {}
        self._load_source_code()
    
    def _load_source_code(self) -> None:
        """Load source code for all repositories"""
        repo_base_dir = FilePathBuilder.get_repo_base_dir(self.benchmark)
        
        for repo in self.repos:
            try:
                repo_source_code = Tools.iterate_repository(repo_base_dir, repo)
                self.source_code[repo] = repo_source_code
            except Exception as e:
                logger.warning(f"Failed to load source code for repository {repo}: {e}")
                self.source_code[repo] = {}

    def build_window(self, tasks: List[Dict], current_generation: List[int] | None = None) -> List[QueryWindow]:
        """
        Build windows
        
        Args:
            tasks: List of tasks or single task
            current_generation: Current generation number
        Returns:
            List of built window results
            
        Raises:
            ValueError: When prediction type lacks necessary parameters
        """
        window_results = []
        
        for i, task in enumerate(tasks):
            task_metadata = task['metadata']
            task_repo = task_metadata['task_id'].split('/')[0]

            if task_repo not in self.repos:
                logger.info(f'skip task {task_metadata["task_id"]} because it is not in {self.repos}')
                continue
            
            query_info = QueryInfo(**task_metadata)
            window_dict = {}
            
            assert current_generation[i] >= 0 and current_generation[i] < 5
            clean_function_body = [i for i in task_metadata['prediction'][current_generation[i]].splitlines() if i.strip()]
            clean_function = task_metadata['target_function_prompt'].splitlines() + clean_function_body
            window_dict['function'] = '\n'.join(clean_function)

            query_window = QueryWindow(
                prompt=task['prompt'],
                metadata=query_info,
                window_dict=window_dict,
            )
            window_results.append(query_window)
            
        return window_results
    
