"""
冲突检测与信息替换覆盖系统。

支持：
1. 检测矛盾的事实（相同主谓但不同宾语）
2. 新旧知识替换策略（保留最新/最重要/合并）
3. 访问历史的迁移与合并
4. 冲突审计日志
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ConflictRecord:
    """冲突记录"""
    timestamp: str
    old_fact: Tuple[str, str, str]  # (subject, predicate, object)
    new_fact: Tuple[str, str, str]
    old_hash_id: str
    new_hash_id: str
    old_access_count: int
    new_access_count: int
    resolution_strategy: str  # 'keep_new', 'keep_old', 'merge'
    resolution_result: str  # 最终采用的信息
    notes: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


class ConflictResolver:
    """
    检测和解决记忆中的冲突。
    
    冲突定义：相同主语(subject)和谓语(predicate)，但宾语(object)不同
    
    示例：
    - 旧: ("Erik Hort", "birthplace", "Montebello")
    - 新: ("Erik Hort", "birthplace", "Rockland County")
    => 冲突！需要决定信任哪一个
    """
    
    def __init__(self, 
                 default_strategy: str = 'keep_new',
                 enable_audit_log: bool = True,
                 audit_log_file: Optional[str] = None):
        """
        初始化冲突解决器。
        
        Args:
            default_strategy: 默认冲突解决策略
                - 'keep_new': 新事实覆盖旧事实（默认）
                - 'keep_old': 保留旧事实，拒绝新事实
                - 'merge': 将两个信息合并为"可能是X或Y"
            enable_audit_log: 是否启用审计日志
            audit_log_file: 审计日志文件路径
        """
        self.default_strategy = default_strategy
        self.enable_audit_log = enable_audit_log
        self.audit_log_file = audit_log_file
        self.conflict_history: List[ConflictRecord] = []
    
    def normalize_fact(self, fact: Tuple[str, str, str]) -> Tuple[str, str, str]:
        """
        标准化事实，用于比较。
        将元素转换为小写并去除多余空格。
        """
        return tuple(
            str(element).lower().strip() 
            for element in fact
        )
    
    def detect_conflicts(self, 
                        existing_facts: List[Tuple[str, str, str]],
                        new_facts: List[Tuple[str, str, str]]) -> List[Tuple[int, int]]:
        """
        检测新事实与已有事实的冲突。
        
        Args:
            existing_facts: 已存储的事实列表
            new_facts: 新添加的事实列表
        
        Returns:
            冲突对列表：[(existing_idx, new_idx), ...]
        """
        conflicts = []
        
        for new_idx, new_fact in enumerate(new_facts):
            new_norm = self.normalize_fact(new_fact)
            new_subj, new_pred, new_obj = new_norm
            
            for exist_idx, exist_fact in enumerate(existing_facts):
                exist_norm = self.normalize_fact(exist_fact)
                exist_subj, exist_pred, exist_obj = exist_norm
                
                # 检查是否有冲突：相同主谓但不同宾语
                if (new_subj == exist_subj and 
                    new_pred == exist_pred and 
                    new_obj != exist_obj):
                    
                    logger.warning(
                        f"Conflict detected: "
                        f"({exist_subj}, {exist_pred}) "
                        f"has value '{exist_obj}' (existing) "
                        f"vs '{new_obj}' (new)"
                    )
                    conflicts.append((exist_idx, new_idx))
        
        return conflicts
    
    def resolve_conflict(self,
                        old_fact: Tuple[str, str, str],
                        new_fact: Tuple[str, str, str],
                        old_hash_id: str,
                        new_hash_id: str,
                        old_access_count: int = 0,
                        new_access_count: int = 0,
                        strategy: Optional[str] = None) -> ConflictRecord:
        """
        解决单个冲突，返回冲突记录。
        
        Args:
            old_fact: 旧事实
            new_fact: 新事实
            old_hash_id: 旧记忆的hash_id
            new_hash_id: 新记忆的hash_id
            old_access_count: 旧事实的访问频率
            new_access_count: 新事实的访问频率
            strategy: 解决策略（可选，默认使用self.default_strategy）
        
        Returns:
            ConflictRecord - 冲突解决记录
        """
        if strategy is None:
            strategy = self.default_strategy
        
        record = ConflictRecord(
            timestamp=datetime.now().isoformat(),
            old_fact=old_fact,
            new_fact=new_fact,
            old_hash_id=old_hash_id,
            new_hash_id=new_hash_id,
            old_access_count=old_access_count,
            new_access_count=new_access_count,
            resolution_strategy=strategy,
            resolution_result=""
        )
        
        # 执行解决策略
        if strategy == 'keep_new':
            record.resolution_result = f"{new_fact[2]}"
            record.notes = "New fact replaces old fact (recent information is prioritized)"
            logger.info(f"Conflict resolved with 'keep_new': using new value '{new_fact[2]}'")
        
        elif strategy == 'keep_old':
            record.resolution_result = f"{old_fact[2]}"
            record.notes = "Old fact retained, new fact rejected (stability prioritized)"
            logger.info(f"Conflict resolved with 'keep_old': retaining old value '{old_fact[2]}'")
        
        elif strategy == 'merge':
            merged = f"({old_fact[2]} or {new_fact[2]})"
            record.resolution_result = merged
            record.notes = "Facts merged into uncertain knowledge (both possibilities preserved)"
            logger.info(f"Conflict resolved with 'merge': {merged}")
        
        elif strategy == 'keep_frequent':
            # 基于访问频率选择
            if old_access_count >= new_access_count:
                record.resolution_result = f"{old_fact[2]}"
                record.notes = f"Old fact retained (more frequently accessed: {old_access_count} vs {new_access_count})"
            else:
                record.resolution_result = f"{new_fact[2]}"
                record.notes = f"New fact adopted (more frequently accessed: {new_access_count} vs {old_access_count})"
            logger.info(f"Conflict resolved with 'keep_frequent': result='{record.resolution_result}'")
        
        self.conflict_history.append(record)
        return record
    
    def batch_resolve_conflicts(self,
                               conflicts: List[Tuple[int, int]],
                               existing_facts: List[Tuple[str, str, str]],
                               new_facts: List[Tuple[str, str, str]],
                               fact_to_hash_id: Dict[str, str],
                               access_counts: Dict[str, int],
                               strategy: str = 'keep_new') -> Dict[str, any]:
        """
        批量解决多个冲突。
        
        Args:
            conflicts: 冲突对列表
            existing_facts: 已存储的事实
            new_facts: 新事实
            fact_to_hash_id: 事实到hash_id的映射
            access_counts: hash_id到访问计数的映射
            strategy: 统一应用的解决策略
        
        Returns:
            包含所有冲突记录和统计的字典
        """
        results = {
            'conflicts_detected': len(conflicts),
            'conflict_records': [],
            'facts_to_delete': [],
            'facts_to_merge': []
        }
        
        for exist_idx, new_idx in conflicts:
            old_fact = existing_facts[exist_idx]
            new_fact = new_facts[new_idx]
            
            old_hash = fact_to_hash_id.get(str(old_fact), "unknown")
            new_hash = fact_to_hash_id.get(str(new_fact), "unknown")
            
            old_count = access_counts.get(old_hash, 0)
            new_count = access_counts.get(new_hash, 0)
            
            record = self.resolve_conflict(
                old_fact, new_fact,
                old_hash, new_hash,
                old_count, new_count,
                strategy=strategy
            )
            results['conflict_records'].append(record.to_dict())
            
            # 根据策略记录要删除或合并的事实
            if strategy == 'keep_new':
                results['facts_to_delete'].append(old_hash)
            elif strategy == 'keep_old':
                results['facts_to_delete'].append(new_hash)
            elif strategy == 'merge':
                results['facts_to_merge'].append({
                    'old_hash': old_hash,
                    'new_hash': new_hash,
                    'merged_value': record.resolution_result
                })
        
        return results
    
    def save_audit_log(self, filepath: str = None):
        """保存冲突审计日志"""
        if filepath is None:
            filepath = self.audit_log_file
        
        if not filepath:
            logger.warning("No audit log file specified")
            return
        
        try:
            log_data = {
                'total_conflicts': len(self.conflict_history),
                'conflicts': [record.to_dict() for record in self.conflict_history]
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Conflict audit log saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save audit log: {e}")
    
    def load_audit_log(self, filepath: str):
        """加载历史冲突审计日志"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            self.conflict_history = [
                ConflictRecord(
                    timestamp=record['timestamp'],
                    old_fact=tuple(record['old_fact']),
                    new_fact=tuple(record['new_fact']),
                    old_hash_id=record['old_hash_id'],
                    new_hash_id=record['new_hash_id'],
                    old_access_count=record['old_access_count'],
                    new_access_count=record['new_access_count'],
                    resolution_strategy=record['resolution_strategy'],
                    resolution_result=record['resolution_result'],
                    notes=record.get('notes')
                )
                for record in log_data.get('conflicts', [])
            ]
            logger.info(f"Loaded {len(self.conflict_history)} conflict records from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load audit log: {e}")
    
    def get_conflict_summary(self) -> Dict:
        """获取冲突摘要统计"""
        if not self.conflict_history:
            return {'total_conflicts': 0}
        
        strategies_used = {}
        for record in self.conflict_history:
            strategies_used[record.resolution_strategy] = strategies_used.get(record.resolution_strategy, 0) + 1
        
        return {
            'total_conflicts': len(self.conflict_history),
            'strategies_used': strategies_used,
            'latest_conflict': self.conflict_history[-1].timestamp if self.conflict_history else None
        }
