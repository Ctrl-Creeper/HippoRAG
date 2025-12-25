#!/usr/bin/env python3
"""
演示HippoRAG的情境感知动态记忆激活、手动清除、自动消退和冲突解决功能。
"""

import logging
import sys
import os

# 直接导入模块，避免通过__init__.py导入完整HippoRAG
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_context_aware_memory():
    """演示情境感知的动态记忆激活系统"""
    logger.info("\n" + "="*80)
    logger.info("演示1: 情境感知动态记忆激活")
    logger.info("="*80)
    
    import importlib.util
    import numpy as np
    
    # 直接加载模块
    spec = importlib.util.spec_from_file_location(
        "context_aware_memory",
        os.path.join(os.path.dirname(__file__), 'src/hipporag/context_aware_memory.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ContextAwareMemoryManager = module.ContextAwareMemoryManager
    
    # 创建管理器
    manager = ContextAwareMemoryManager(
        context_window_size=5,
        recency_weight=0.3,
        frequency_weight=0.2,
        relevance_weight=0.5
    )
    
    # 场景：用户进行一系列相关查询
    query_sequence = [
        "Who is Erik Hort?",
        "Where was Erik born?",
        "What is Montebello?",
        "Is Montebello in Rockland County?"
    ]
    
    logger.info("\n📝 查询序列:")
    for i, query in enumerate(query_sequence):
        query_embedding = np.random.randn(384)
        manager.add_query_context(query, query_embedding)
        logger.info(f"  {i+1}. {query}")
    
    # 现在计算某条记忆在当前上下文中的激活分数
    memory_1_history = [
        {'timestamp': '2025-12-25T10:00:00', 'computed_similarity': 0.95},
        {'timestamp': '2025-12-25T10:05:00', 'computed_similarity': 0.88},
        {'timestamp': '2025-12-25T10:10:00', 'computed_similarity': 0.91}
    ]
    
    memory_2_history = [
        {'timestamp': '2025-12-20T10:00:00', 'computed_similarity': 0.15},
        {'timestamp': '2025-12-21T10:00:00', 'computed_similarity': 0.18}
    ]
    
    current_query_embedding = np.random.randn(384)
    memory_embedding = np.random.randn(384)
    
    logger.info("\n🧠 激活分数计算:")
    
    scores_1 = manager.calculate_activation_score(
        'chunk-001',
        memory_1_history,
        current_query_embedding,
        memory_embedding
    )
    logger.info(f"  记忆1 (高度相关): {scores_1['total_activation']:.4f}")
    logger.info(f"    - 语义相关性: {scores_1['semantic_relevance']:.4f}")
    logger.info(f"    - 最近使用奖励: {scores_1['recency_bonus']:.4f}")
    logger.info(f"    - 上下文频率: {scores_1['context_frequency']:.4f}")
    logger.info(f"    - 应保留: {scores_1['should_retain']}")
    
    scores_2 = manager.calculate_activation_score(
        'chunk-002',
        memory_2_history,
        current_query_embedding,
        memory_embedding
    )
    logger.info(f"\n  记忆2 (无关): {scores_2['total_activation']:.4f}")
    logger.info(f"    - 语义相关性: {scores_2['semantic_relevance']:.4f}")
    logger.info(f"    - 最近使用奖励: {scores_2['recency_bonus']:.4f}")
    logger.info(f"    - 上下文频率: {scores_2['context_frequency']:.4f}")
    logger.info(f"    - 应保留: {scores_2['should_retain']}")
    
    logger.info("\n✅ 演示1结论: 与当前查询上下文相关的记忆激活度高，无关的激活度低")


def demo_conflict_resolution():
    """演示冲突检测与解决"""
    logger.info("\n" + "="*80)
    logger.info("演示2: 冲突检测与解决")
    logger.info("="*80)
    
    import importlib.util
    
    # 直接加载冲突解决模块
    spec = importlib.util.spec_from_file_location(
        "conflict_resolution",
        os.path.join(os.path.dirname(__file__), 'src/hipporag/conflict_resolution.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ConflictResolver = module.ConflictResolver
    
    resolver = ConflictResolver(default_strategy='keep_new')
    
    # 场景：新获取的信息与已有信息冲突
    existing_facts = [
        ("Erik Hort", "birthplace", "Montebello"),
        ("Montebello", "location", "New York"),
        ("Oliver Badman", "profession", "politician")
    ]
    
    new_facts = [
        ("Erik Hort", "birthplace", "Rockland County"),  # 冲突！
        ("Sarah Chen", "profession", "scientist"),
        ("Montebello", "location", "Rockland County")  # 冲突！
    ]
    
    logger.info("\n📚 现有知识库:")
    for fact in existing_facts:
        logger.info(f"  {fact}")
    
    logger.info("\n🆕 新添加的事实:")
    for fact in new_facts:
        logger.info(f"  {fact}")
    
    # 检测冲突
    conflicts = resolver.detect_conflicts(existing_facts, new_facts)
    logger.info(f"\n⚠️  检测到 {len(conflicts)} 个冲突:")
    for exist_idx, new_idx in conflicts:
        logger.info(f"  冲突 ({exist_idx}, {new_idx}): {existing_facts[exist_idx]} vs {new_facts[new_idx]}")
    
    # 解决冲突
    logger.info("\n🔧 解决冲突（使用'keep_new'策略）:")
    
    results = resolver.batch_resolve_conflicts(
        conflicts=conflicts,
        existing_facts=existing_facts,
        new_facts=new_facts,
        fact_to_hash_id={
            str(fact): f"fact-{i}" for i, fact in enumerate(existing_facts + new_facts)
        },
        access_counts={f"fact-{i}": i % 3 for i in range(len(existing_facts) + len(new_facts))},
        strategy='keep_new'
    )
    
    logger.info(f"  已解决 {results['conflicts_detected']} 个冲突")
    logger.info(f"  要删除的事实: {len(results['facts_to_delete'])} 个")
    
    for record in results['conflict_records']:
        logger.info(f"  - 采用新值: {record['resolution_result']}")
    
    logger.info("\n✅ 演示2结论: 新事实自动覆盖旧的冲突事实，保证知识的最新性")


def demo_memory_lifecycle():
    """演示记忆的完整生命周期"""
    logger.info("\n" + "="*80)
    logger.info("演示3: 记忆完整生命周期")
    logger.info("="*80)
    
    import json
    from datetime import datetime, timedelta
    
    logger.info("""
    记忆的完整生命周期：
    
    1️⃣  创建阶段 (Creation)
       - 新文档被索引
       - 新实体和事实被提取
       - hash_id生成
    
    2️⃣  激活阶段 (Activation)
       - 检索时与查询上下文匹配
       - 访问历史被记录
       - 激活分数动态计算
    
    3️⃣  维护阶段 (Maintenance)
       - 高激活记忆保持活跃
       - 访问频率被追踪
       - 冲突被检测和解决
    
    4️⃣  衰退阶段 (Decay)
       - 持续无关的记忆激活度下降
       - 用户可手动清除低激活记忆
       - 系统可自动应用消退
    
    5️⃣  替换阶段 (Replacement)
       - 旧信息被新信息替换
       - 访问历史可选择性迁移
       - 审计日志记录所有变化
    """)
    
    # 模拟访问历史
    memory_access_history = {
        'chunk-001': [
            {'timestamp': (datetime.now() - timedelta(days=5)).isoformat(), 'query': 'Query A', 'similarity': 0.85},
            {'timestamp': (datetime.now() - timedelta(days=3)).isoformat(), 'query': 'Query B', 'similarity': 0.78},
            {'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(), 'query': 'Query C', 'similarity': 0.92}
        ],
        'chunk-002': [
            {'timestamp': (datetime.now() - timedelta(days=30)).isoformat(), 'query': 'Query D', 'similarity': 0.45}
        ]
    }
    
    logger.info("\n📊 访问历史示例:")
    for chunk_id, events in memory_access_history.items():
        logger.info(f"  {chunk_id}:")
        for event in events:
            time_ago = (datetime.now() - datetime.fromisoformat(event['timestamp'])).days
            logger.info(f"    - {time_ago}天前: {event['query']} (相似度={event['similarity']})")
    
    logger.info("\n✅ 演示3结论: 记忆通过访问历史追踪其上下文相关性，实现情境感知的生命周期管理")


if __name__ == "__main__":
    try:
        demo_context_aware_memory()
        demo_conflict_resolution()
        demo_memory_lifecycle()
        
        logger.info("\n" + "="*80)
        logger.info("🎉 所有演示完成！")
        logger.info("="*80)
        logger.info("""
核心特性总结：

✨ 情境感知激活: 记忆激活度 = 语义相关性 + 最近使用 + 上下文频率
✨ 动态消退: 持续不相关的记忆逐渐衰退，相关记忆保持活跃
✨ 手动清除: 用户可在检查后手动清除低激活记忆
✨ 自动消退: 系统可在检索后自动应用消退策略
✨ 冲突解决: 新旧知识冲突时自动解决，可选择保留新/旧/合并
✨ 完整审计: 所有冲突和重要操作都记录在审计日志中
        """)
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        sys.exit(1)
