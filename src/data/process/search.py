from .data import QueryVector, QuerySearch, WindowData, WindowInfo
from typing import List
import chromadb
from loguru import logger

class TaskRetrieverMaker:
    """Task retriever for retrieving relevant contexts for query vectors"""
    def __init__(self,
                 task_type: str,
                 repos: List[str],
                 benchmark: str,
                 window_type: str,
                 top_k: int = 50,
                 max_top_k: int = 10):
        """
        Initialize task retriever
        
        Args:
            task_type: Task type
            repos: Repository list
            benchmark: Benchmark type
            window_type: Window type
            top_k: Return top-k most similar results, default 50
            max_top_k: Return top max_top_k most similar results, default 10
        """
        self.top_k = top_k
        self.max_top_k = max_top_k
        self.task_type = task_type
        self.benchmark = benchmark
        self.window_type = window_type
        self.repos = repos
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = {}
        for repo in self.repos:
            collection_name = f"{self.benchmark}_{repo}_{self.window_type}"
            try:
                self.collection[repo] = self.client.get_collection(name=collection_name)
                logger.info(f"Using existing collection: {collection_name}")
            except:
                logger.error(f"Collection {collection_name} not found. Please run repo data processing first.")
        logger.info(f"Initialized TaskRetriever Database...")

    def search_contexts(self, query_vector_results: List[QueryVector]) -> List[QuerySearch]:
        """
        Retrieve relevant contexts for query vectors
        
        Args:
            query_vector_results: List of query vector results
        Returns:
            List of SearchResult containing retrieved context information
        """
        query_embedding_list = []
        
        for query in query_vector_results:
            query_embedding_dict = query.embedding_dict
            match self.task_type:
                case "token" | "train":
                    query_embedding = query_embedding_dict['function']
                case _:
                    raise ValueError(f"Invalid task type: {self.task_type}")

            query_embedding_list.append(query_embedding)
        
        results = []
        for i, query_vector in enumerate(query_vector_results):
            query_metadata = query_vector.metadata
            repo_name = query_metadata.task_id.split("/")[0]
            query_fpath_str = '/'.join(query_metadata.fpath_tuple)
            where_filter = {
                "$or": [
                    {"fpath": {"$ne": query_fpath_str}},
                    {"$and": [
                        {"fpath": query_fpath_str},
                        {"end_line_no": {"$lt": query_metadata.context_start_lineno}}
                    ]}
                ]
            }
            
            result = self.collection[repo_name].query(
                query_embeddings=[query_embedding_list[i]],
                n_results=self.top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            results.append(result)
        
        query_search_results = []
        for query_vector, result in zip(query_vector_results, results):
            search_results = []
            available_results = len(result['documents'][0])
            actual_top_k = min(self.top_k, available_results)
            query_fpath_str = '/'.join(query_vector.metadata.fpath_tuple)
     
            for i in range(actual_top_k):
                try:
                    if len(search_results) >= self.max_top_k:
                        break
                    metadata = result['metadatas'][0][i]
                    context_fpath_str = metadata['fpath']
                    if query_fpath_str in context_fpath_str and metadata['end_line_no'] >= query_metadata.context_start_lineno:
                        continue
                        
                    window_data = WindowData(
                        context=result['documents'][0][i],
                        metadata=WindowInfo(**(result['metadatas'][0][i]))
                    )
                    similarity = 1 - result['distances'][0][i]
                    search_results.append((window_data, similarity))
                except (IndexError, KeyError, TypeError) as e:
                    logger.warning(f"Error processing search result {i}: {e}")
                    continue
            query_search = QuerySearch(
                prompt=query_vector.prompt,
                metadata=query_vector.metadata,
                top_k_contexts_list=search_results
            )
            query_search_results.append(query_search)
        
        return query_search_results