"""
情境感知的动态记忆激活系统。

实现基于当前查询上下文的动态激活函数，而非简单的时间衰减。
核心思想：记忆的激活程度取决于它与当前查询上下文的相关性。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from copy import deepcopy

logger = logging.getLogger(__name__)


class ContextAwareMemoryManager:
    """
    管理记忆的情境感知激活。
    
    核心机制：
    1. 查询上下文历史：追踪最近的查询及其embedding
    2. 激活函数：基于当前查询与历史查询的相似性动态激活记忆
    3. 选择性遗忘：持续不相关的记忆逐渐衰退
    """
    
    def __init__(self, 
                 context_window_size: int = 10,
                 recency_weight: float = 0.3,
                 frequency_weight: float = 0.2,
                 relevance_weight: float = 0.5,
                 relevance_threshold: float = 0.3,
                 decay_rate: float = 0.01):
        """
        初始化情境感知管理器。
        
        Args:
            context_window_size: 保持的查询上下文窗口大小
            recency_weight: 最近使用的权重（0-1）
            frequency_weight: 使用频率的权重（0-1）
            relevance_weight: 与当前查询相关性的权重（0-1）
            relevance_threshold: 判定为"相关"的相似度阈值
            decay_rate: 不相关记忆的衰减速率（0-1）
        """
        self.context_window_size = context_window_size
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.relevance_weight = relevance_weight
        self.relevance_threshold = relevance_threshold
        self.decay_rate = decay_rate
        
        # 权重需要归一化
        total_weight = recency_weight + frequency_weight + relevance_weight
        self.recency_weight /= total_weight
        self.frequency_weight /= total_weight
        self.relevance_weight /= total_weight
        
        # 查询历史（FIFO，保持最近的context_window_size个）
        self.query_history: List[Dict] = []
    
    def add_query_context(self, query: str, query_embedding: np.ndarray):
        """添加新的查询到上下文历史"""
        context_event = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'embedding': query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        }
        self.query_history.append(context_event)
        
        # 保持窗口大小
        if len(self.query_history) > self.context_window_size:
            self.query_history.pop(0)
        
        logger.debug(f"Added query context. Window size: {len(self.query_history)}")
    
    def calculate_activation_score(self, 
                                   memory_hash_id: str,
                                   access_history: List[Dict],
                                   current_query_embedding: np.ndarray,
                                   memory_embedding: np.ndarray,
                                   current_time: Optional[datetime] = None) -> Dict:
        """
        计算记忆的动态激活分数。
        
        激活分数 = w_relevance * semantic_relevance +
                   w_recency * recency_bonus +
                   w_frequency * context_frequency
        
        其中：
        - semantic_relevance: 当前查询与记忆的embedding相似度
        - recency_bonus: 最后访问时间越近，分数越高
        - context_frequency: 在相似上下文中被访问的频率
        
        Args:
            memory_hash_id: 记忆ID
            access_history: 记忆的访问历史
            current_query_embedding: 当前查询的embedding
            memory_embedding: 记忆的embedding
            current_time: 当前时间（默认为now）
        
        Returns:
            包含激活分数和各分量的字典
        """
        if current_time is None:
            current_time = datetime.now()
        
        scores = {
            'hash_id': memory_hash_id,
            'semantic_relevance': 0.0,
            'recency_bonus': 0.0,
            'context_frequency': 0.0,
            'total_activation': 0.0,
            'should_retain': True
        }
        
        # 1. 语义相关性：当前查询与记忆的相似度
        if isinstance(current_query_embedding, np.ndarray) and isinstance(memory_embedding, np.ndarray):
            semantic_sim = self._cosine_similarity(current_query_embedding, memory_embedding)
            scores['semantic_relevance'] = max(0, semantic_sim)  # 只取非负值
        
        # 2. 最近使用奖励：时间衰减因子
        if access_history:
            last_access_time = datetime.fromisoformat(access_history[-1]['timestamp'])
            time_delta = (current_time - last_access_time).total_seconds()
            # 使用指数衰减：e^(-decay_rate * days)
            days = time_delta / (24 * 3600)
            recency_bonus = np.exp(-self.decay_rate * days)
            scores['recency_bonus'] = recency_bonus
        
        # 3. 相似上下文频率：在多少个相似的查询中被激活
        if access_history:
            relevant_contexts = self._count_relevant_contexts(
                access_history, 
                threshold=self.relevance_threshold
            )
            context_frequency = min(1.0, relevant_contexts / 5)  # 归一化到[0,1]
            scores['context_frequency'] = context_frequency
        
        # 计算总激活分数
        scores['total_activation'] = (
            self.relevance_weight * scores['semantic_relevance'] +
            self.recency_weight * scores['recency_bonus'] +
            self.frequency_weight * scores['context_frequency']
        )
        
        # 判断是否应该保留：最小激活阈值
        min_activation_threshold = 0.05  # 5%最小阈值
        scores['should_retain'] = (
            scores['total_activation'] >= min_activation_threshold or
            len(access_history) == 0  # 新记忆总是保留
        )
        
        return scores
    
    def calculate_batch_activation(self,
                                   memory_store: 'EmbeddingStore',
                                   current_query_embedding: np.ndarray) -> Dict[str, Dict]:
        """
        为所有记忆计算激活分数。
        
        Args:
            memory_store: EmbeddingStore实例
            current_query_embedding: 当前查询的embedding
        
        Returns:
            {hash_id -> activation_scores} 的映射
        """
        all_activation_scores = {}
        
        for hash_id in memory_store.get_all_ids():
            memory_embedding = memory_store.get_embedding(hash_id)
            access_history = memory_store.get_access_history(hash_id)
            
            scores = self.calculate_activation_score(
                hash_id,
                access_history,
                current_query_embedding,
                memory_embedding
            )
            all_activation_scores[hash_id] = scores
        
        return all_activation_scores
    
    def get_memories_to_forget(self, 
                               memory_store: 'EmbeddingStore',
                               current_query_embedding: np.ndarray,
                               retention_ratio: float = 0.9) -> List[str]:
        """
        获取应该被遗忘的记忆ID列表。
        
        策略：保留激活分数最高的retention_ratio比例的记忆。
        
        Args:
            memory_store: EmbeddingStore实例
            current_query_embedding: 当前查询的embedding
            retention_ratio: 保留比例（0-1），默认保留90%
        
        Returns:
            应该删除的记忆ID列表
        """
        all_scores = self.calculate_batch_activation(memory_store, current_query_embedding)
        
        # 按激活分数排序
        sorted_memories = sorted(
            all_scores.items(),
            key=lambda x: x[1]['total_activation'],
            reverse=True
        )
        
        # 计算保留数量
        total_count = len(sorted_memories)
        retain_count = max(1, int(total_count * retention_ratio))
        
        # 标记应该删除的
        to_forget = [
            hash_id for hash_id, _ in sorted_memories[retain_count:]
            if _['should_retain'] == False  # 只删除明确不应该保留的
        ]
        
        logger.info(f"Memory management: Total={total_count}, "
                   f"Retaining={retain_count}, "
                   f"To forget={len(to_forget)}")
        
        return to_forget
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _count_relevant_contexts(self, access_history: List[Dict], 
                                 threshold: float = 0.3) -> int:
        """
        计算记忆在多少个相似上下文中被激活（基于访问历史中记录的相似度）。
        """
        count = 0
        for event in access_history:
            if 'computed_similarity' in event and event['computed_similarity'] >= threshold:
                count += 1
        return count
    
    def get_context_similarity_matrix(self) -> np.ndarray:
        """
        获取查询上下文的相似度矩阵，用于理解查询之间的关系。
        """
        if len(self.query_history) < 2:
            return np.array([])
        
        embeddings = [
            np.array(event['embedding']) for event in self.query_history
        ]
        
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = self._cosine_similarity(
                    embeddings[i], embeddings[j]
                )
        
        return similarity_matrix
    
    def export_state(self) -> Dict:
        """导出管理器的状态用于持久化"""
        return {
            'context_window_size': self.context_window_size,
            'recency_weight': self.recency_weight,
            'frequency_weight': self.frequency_weight,
            'relevance_weight': self.relevance_weight,
            'relevance_threshold': self.relevance_threshold,
            'decay_rate': self.decay_rate,
            'query_history': deepcopy(self.query_history)
        }
    
    def load_state(self, state: Dict):
        """从导出的状态恢复"""
        self.context_window_size = state.get('context_window_size', 10)
        self.recency_weight = state.get('recency_weight', 0.3)
        self.frequency_weight = state.get('frequency_weight', 0.2)
        self.relevance_weight = state.get('relevance_weight', 0.5)
        self.relevance_threshold = state.get('relevance_threshold', 0.3)
        self.decay_rate = state.get('decay_rate', 0.01)
        self.query_history = state.get('query_history', [])
