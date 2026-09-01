from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class WindowInfo:
    """Window info data class"""
    fpath: str
    start_line_no: int
    end_line_no: int
    repo: str

@dataclass
class WindowData:
    """Window data class"""
    context: str
    metadata: WindowInfo
    
    @classmethod
    def from_dict(cls, data: dict) -> "WindowData":
        return cls(
            context=data["context"],
            metadata=WindowInfo(**data["metadata"])
        )

@dataclass
class VectorData:
    """Vector data class"""
    context: str
    metadata: WindowInfo
    embedding: List[float] | List[int] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "VectorData":
        return cls(
            context=data["context"],
            metadata=WindowInfo(**data["metadata"]),
            embedding=data["embedding"]
        )
        
    @classmethod
    def from_window(cls, window_data: WindowData, embedding: List[float] | List[int]) -> "VectorData":
        return cls(
            context=window_data.context,
            metadata=window_data.metadata,
            embedding=embedding
        )


@dataclass
class QueryInfo:
    """Query info data class"""
    task_id: str
    ground_truth: str
    fpath_tuple: Tuple[str, ...]
    context_start_lineno: int
    line_no: int
    target_function_prompt: str
    function_signature: str
    docstring: str
    prediction: List[str] | None = None

@dataclass
class QueryWindow:
    """Query window data class"""
    prompt: str
    metadata: QueryInfo
    window_dict: Dict[str, str]
    
    @classmethod
    def from_dict(cls, data: dict) -> "QueryWindow":
        return cls(
            prompt=data["prompt"],
            metadata=QueryInfo(**data["metadata"]),
            window_dict=data["window_dict"]
        )
        
@dataclass
class QueryVector:
    """Query vector data class"""
    prompt: str
    metadata: QueryInfo
    embedding_dict: Dict[str, List[float] | List[int]]
    
    @classmethod
    def from_dict(cls, data: dict) -> "QueryVector":
        return cls(
            prompt=data["prompt"],
            metadata=QueryInfo(**data["metadata"]),
            embedding_dict=data["embedding_dict"],
        )
        
    @classmethod
    def from_window(cls, query_window_result: QueryWindow, embedding_dict: Dict[str, List[float] | List[int]]) -> "QueryVector":
        return cls(
            prompt=query_window_result.prompt,
            metadata=query_window_result.metadata,
            embedding_dict=embedding_dict
        )
    
@dataclass
class QuerySearch:
    """Query search data class"""
    prompt: str
    metadata: QueryInfo
    top_k_contexts_list: List[List[Tuple[WindowData, float]]]
    
    @classmethod
    def from_vector(cls, query_vector_result: QueryVector, top_k_contexts_list: List[List[Tuple[WindowData, float]]]) -> "QuerySearch":
        return cls(
            prompt=query_vector_result.prompt,
            metadata=query_vector_result.metadata,
            top_k_contexts_list=top_k_contexts_list,
        )

@dataclass
class QueryPrompt:
    """Query prompt data class"""
    prompt: str
    metadata: QueryInfo
    top_k_contexts_list: List[Tuple[WindowData, float]]
    top_k_contexts_selected: int
    
    @classmethod
    def from_dict(cls, data: dict) -> "QueryPrompt":
        return cls(
            prompt=data["prompt"],
            metadata=QueryInfo(**data["metadata"]),
            top_k_contexts_list=data["top_k_contexts_list"],
            top_k_contexts_selected=data["top_k_contexts_selected"],
        )
    
    @classmethod
    def from_search(cls, query_search_result: QuerySearch, top_k_contexts_list: List[Tuple[WindowData, float]], top_k_contexts_selected: int) -> "QueryPrompt":
        return cls(
            prompt=query_search_result.prompt,
            metadata=query_search_result.metadata,
            top_k_contexts_list=top_k_contexts_list,
            top_k_contexts_selected=top_k_contexts_selected,
        )

@dataclass
class QueryResult:
    """Query prompt data class"""
    task_id: str
    function: str
    prediction: List[str]
    
    @classmethod
    def from_dict(cls, data: dict) -> "QueryResult":
        return cls(
            task_id=data["task_id"],
            function=data["function"],
            prediction=data["prediction"],
        )
    
    @classmethod
    def from_prompt(cls, query_prompt: QueryPrompt, prediction: List[str]) -> "QueryResult":
        return cls(
            task_id=query_prompt.metadata.task_id,
            function=query_prompt.metadata.target_function_prompt,
            prediction=prediction,
        )
