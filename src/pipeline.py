# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
import fire
from multiprocessing import freeze_support
from server import LLM
from data import RepoTask, CONSTANTS, Tools
from critoken import TokenTask
from train import TrainDataTask, TrainClassifierTask
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()   # Hide info/warning, only show error

class QueryTask:
    """Query task processing class"""
    def __init__(
        self,
        task: str = 'token',
        model_path: str = "/root/autodl-tmp/models/codellama-13b-hf",
        benchmark: str = CONSTANTS.repoexec_benchmark,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ):
        assert task in ['token', 'train']
        self.task = task
        self.model_name = model_path.split('/')[-1]
        self.benchmark = benchmark
        
        self.repos = Tools.get_repos(benchmark)
            
        self.max_new_tokens = max_new_tokens
        self.n = 1
        
        self.llm = LLM(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n=self.n
        )
        
    def token(self, **kwargs):
        return TokenTask(llm=self.llm, model_name=self.model_name, benchmark=self.benchmark, repos=self.repos, **kwargs)
    def train(self, **kwargs):
        return TrainDataTask(llm=self.llm, model_name=self.model_name, benchmark=self.benchmark, repos=self.repos, **kwargs)
    
    def run(self, **kwargs):
        """Run specified task"""
        if self.task == 'token':
            return self.token(max_new_tokens=self.max_new_tokens, **kwargs).run()
        elif self.task == 'train':
            return self.train(**kwargs).run()
        else:
            raise ValueError(f"Unsupported task type: {self.task}")


class TrainTask:
    """Training task processing class"""
    def __init__(
        self,
        task: str = 'classifier',
        model_name: str = "deepseek-coder-1.3b-base",
        **kwargs
    ):
        assert task in ['classifier']
        self.task = task
        self.model_name = model_name
        
    def classifier(self, **kwargs):
        return TrainClassifierTask(model_name=self.model_name, **kwargs)
    
    def run(self, **kwargs):
        """Run specified training task"""
        if self.task == 'classifier':
            return self.classifier(**kwargs).run()
        else:
            raise ValueError(f"Unsupported training task type: {self.task}")

class CLI:
    """Main CLI class, contains all subcommands"""
    def repo(self, benchmark: str = CONSTANTS.repoexec_benchmark, **kwargs):
        repos = Tools.get_repos(benchmark)
        return RepoTask()(benchmark=benchmark, repos=repos, **kwargs)
    
    def task(self, task_type='token', **kwargs):
        """Execute specified type of query task"""
        query_task = QueryTask(task=task_type, **kwargs)
        return query_task.run()
    
    def train(self, task_type='data', **kwargs):
        """Execute specified type of training task"""
        train_task = TrainTask(task=task_type, **kwargs)
        return train_task.run()
    

if __name__ == '__main__':
    freeze_support()
    fire.Fire(CLI)
