from typing import Type, List
import chromadb
from .process import RepoFlexibleWindowMaker, DeepSeekTokenizer, CONSTANTS
from .process.vector import UniXcoderEmbedding
from loguru import logger
from tqdm import tqdm

class RepoDataProcess:
    def __init__(self, benchmark, repos, window_type="flexible", rewrite=False):
        assert window_type == "flexible"
        self.benchmark = benchmark
        self.repos = repos
        self.window_type = window_type
        self.window_max_len = 15
        self.rewrite = rewrite
        self.vector_builder = UniXcoderEmbedding()

    def _window_for_repo_files(self, missing_repos=None):
        """
        Create windows for repository files
        
        Args:
            missing_repos: List of missing repositories to process, if None then process all repositories
        """
        window_results = {}
        repos_to_process = missing_repos if missing_repos is not None else self.repos
            
        for repo in repos_to_process:
            repo_window_maker = RepoFlexibleWindowMaker(self.benchmark, repo, self.window_max_len)
            window_results[repo] = repo_window_maker.build_windows()
        return window_results

    def _vectorize_repo_windows(self, window_results: dict):
        """Build vectors for repository windows and create persistent ChromaDB"""
        # Create persistent ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        for repo in window_results.keys():        
            collection_name = f"{self.benchmark}_{repo}_{self.window_type}"
            
            # Check if collection already exists
            try:
                collection = client.get_collection(name=collection_name)
                if self.rewrite:
                    # If rewrite is True, delete existing collection
                    client.delete_collection(name=collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
                    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
                    logger.info(f"Recreated collection: {collection_name}")
                else:
                    logger.info(f"Collection {collection_name} already exists, skipping...")
                    continue
            except:
                # Collection doesn't exist, need to create and process
                collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
                logger.info(f"Created new collection: {collection_name}")
            
            vector_data = self.vector_builder.build_repos(window_results[repo])
            
            batch_size = 4096
            total_docs = len(vector_data)
            
            for batch_start in tqdm(range(0, total_docs, batch_size), desc=f"Adding documents to collection {collection_name}"):
                batch_end = min(batch_start + batch_size, total_docs)
                batch_data = vector_data[batch_start:batch_end]

                documents = [data['context'] for data in batch_data]
                metadatas = [data['metadata'] for data in batch_data]
                embeddings = [data['embedding'] for data in batch_data]
                ids = [f"{repo}_{batch_start + i}" for i in range(len(batch_data))]
                
                # Batch add documents for current batch
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
            
            logger.info(f"Added {total_docs} documents to collection {collection_name} in batches of {batch_size}")

    def process_repo_data(self, build_windows=True, build_vectors=True):
        """
        Complete workflow for processing repository data
        
        Args:
            build_windows: Whether to build windows
            build_vectors: Whether to build vectors
        """
        # Check if persistent ChromaDB already exists
        client = chromadb.PersistentClient(path="./chroma_db")
        existing_collections = set()
        
        for repo in self.repos:
            # Build collection_name based on window_type
            collection_name = f"{self.benchmark}_{repo}_{self.window_type}"
                
            try:
                client.get_collection(name=collection_name)
                existing_collections.add(collection_name)
                logger.info(f"Found existing collection: {collection_name}")
            except:
                logger.info(f"Collection {collection_name} not found")
        
        # If all collections exist and rewrite is False, skip processing
        expected_collections = set()
        for repo in self.repos:
            expected_collections.add(f"{self.benchmark}_{repo}_{self.window_type}")
        
        if expected_collections.issubset(existing_collections) and not self.rewrite:
            logger.info("All collections already exist in persistent ChromaDB. Skipping window creation and vectorization.")
            return
        
        # If rewrite is True, process all repositories; otherwise only process missing ones
        if self.rewrite:
            logger.info("Rewrite mode enabled. Processing all repositories...")
            missing_repos = None  # None means process all repositories
        else:
            # If some collections exist, only process missing ones
            missing_collections = expected_collections - existing_collections
            missing_repos = []
            if missing_collections:
                logger.info(f"Missing collections: {missing_collections}")
                logger.info("Proceeding with window creation and vectorization for missing collections...")
                # Extract missing repository names
                for collection_name in missing_collections:
                    repo_name = collection_name.replace(f"{self.benchmark}_", "").replace(f"_{self.window_type}", "")
                    missing_repos.append(repo_name)
        
        if build_windows:
            logger.info("--- Creating windows for repositories ---")
            window_results = self._window_for_repo_files(missing_repos)
            logger.info("--- Done! Windows for repositories created ---")
        
        if build_vectors:
            logger.info("--- Vectorizing repositories ---")
            self._vectorize_repo_windows(window_results)
            logger.info("--- Done! Vectorized repositories ---")
            
class RepoTask:
    """Repository task processing class"""
    def __call__(
        self, 
        benchmark: str,
        repos: list[str],
        window_type: str = "flexible",
        rewrite: bool = False
    ) -> None:
        """
        Create windows and vectors for repositories
        
        Args:
            benchmark: Benchmark name
            repos: List of repository names
            window_type: Window type
            rewrite: Whether to rewrite existing collection, default is False
        """
        assert window_type == "flexible"
        RepoDataProcess(
            benchmark=benchmark,
            repos=repos,
            window_type=window_type,
            rewrite=rewrite
        ).process_repo_data()
