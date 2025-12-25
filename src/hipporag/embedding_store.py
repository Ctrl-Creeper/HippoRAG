import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd
import json
from datetime import datetime

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        # 访问历史和元数据存储
        self.access_history_file = os.path.join(
            db_filename, f"access_history_{self.namespace}.json"
        )
        self.access_history = {}  # hash_id -> list of access events
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return  {}# All records already exist.

        # Prepare the texts to encode from the "content" field.
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}
        
        # 加载访问历史
        self._load_access_history()

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        self._save_access_history()
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        logger.info(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            hash_id = self.hash_ids[idx]
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)
            # 删除对应的访问历史
            if hash_id in self.access_history:
                del self.access_history[hash_id]

        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings
    
    def record_access(self, hash_id: str, query: str = None, query_embedding: np.ndarray = None, 
                      ranking_position: int = -1, similarity_score: float = None):
        """
        记录对记忆的访问事件，用于追踪上下文相关性。
        
        Args:
            hash_id: 被访问的记忆ID
            query: 触发访问的查询文本
            query_embedding: 查询的embedding向量
            ranking_position: 在检索结果中的位置（-1表示未被返回）
            similarity_score: 与查询的相似度分数
        """
        if hash_id not in self.access_history:
            self.access_history[hash_id] = []
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'ranking_position': ranking_position,
            'similarity_score': similarity_score
        }
        
        # 不存储embedding向量本身，因为无法JSON序列化
        # 但可以计算query embedding与这条记忆embedding的相似度
        if query_embedding is not None and hash_id in self.hash_id_to_idx:
            idx = self.hash_id_to_idx[hash_id]
            memory_embedding = self.embeddings[idx]
            if isinstance(memory_embedding, np.ndarray):
                similarity = float(np.dot(query_embedding, memory_embedding) / 
                                 (np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding) + 1e-10))
                event['computed_similarity'] = similarity
        
        self.access_history[hash_id].append(event)
    
    def get_access_history(self, hash_id: str) -> List[Dict]:
        """获取指定记忆的访问历史"""
        return self.access_history.get(hash_id, [])
    
    def get_all_access_history(self) -> Dict[str, List[Dict]]:
        """获取所有记忆的访问历史"""
        return deepcopy(self.access_history)
    
    def get_access_count(self, hash_id: str) -> int:
        """获取记忆被访问的总次数"""
        return len(self.access_history.get(hash_id, []))
    
    def get_last_access_time(self, hash_id: str) -> Optional[str]:
        """获取记忆最后一次访问的时间"""
        history = self.access_history.get(hash_id, [])
        if history:
            return history[-1]['timestamp']
        return None
    
    def get_relevant_context_queries(self, hash_id: str, min_similarity: float = 0.5) -> List[Dict]:
        """
        获取与这条记忆高度相关的查询历史（基于相似度）。
        这用于判断记忆在多少种上下文中被激活过。
        """
        history = self.access_history.get(hash_id, [])
        relevant_queries = []
        for event in history:
            similarity = event.get('computed_similarity', 0)
            if similarity >= min_similarity:
                relevant_queries.append(event)
        return relevant_queries
    
    def _load_access_history(self):
        """从文件加载访问历史"""
        if os.path.exists(self.access_history_file):
            try:
                with open(self.access_history_file, 'r', encoding='utf-8') as f:
                    self.access_history = json.load(f)
                logger.info(f"Loaded access history with {len(self.access_history)} entries")
            except Exception as e:
                logger.warning(f"Failed to load access history: {e}. Starting fresh.")
                self.access_history = {}
        else:
            self.access_history = {}
    
    def _save_access_history(self):
        """保存访问历史到文件"""
        try:
            with open(self.access_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.access_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save access history: {e}")