"""
集成测试：使用大模型测试完整的记忆管理系统

展示如何在实际RAG应用中调用三个核心API：
1. apply_context_aware_memory_decay() - 自动消退
2. manual_cleanup_low_activation_memories() - 手动清除
3. detect_and_resolve_fact_conflicts() - 冲突解决
"""

import logging
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMIntegrationTest:
    """展示如何在大模型应用中使用新的记忆管理API"""
    
    def __init__(self, rag_system):
        """
        Args:
            rag_system: HippoRAG实例
        """
        self.rag = rag_system
        self.query_history = []
        self.conflict_records = []
    
    # ============================================================================
    # 第一部分：基础检索流程（自动集成）
    # ============================================================================
    
    def retrieve_with_auto_decay(self, query: str, enable_decay: bool = True) -> List[Dict]:
        """
        标准检索流程 - 自动触发情境感知消退
        
        说明：
            retrieve()方法已被修改，会在返回结果后自动：
            1. 记录访问历史
            2. 更新查询上下文窗口
            3. （可选）触发自动消退
        
        Args:
            query: 用户查询
            enable_decay: 是否启用自动消退
        
        Returns:
            检索结果列表
        
        使用示例：
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"查询 #{len(self.query_history)+1}: {query}")
        logger.info(f"{'='*80}")
        
        # 执行标准检索
        results = self.rag.retrieve(query)
        self.query_history.append(query)
        
        logger.info(f"✅ 检索完成，返回 {len(results)} 条结果")
        logger.info(f"   - 访问历史已自动记录")
        logger.info(f"   - 查询上下文已更新")
        
        # 可选：在检索后立即查看激活状态
        if enable_decay:
            activation_status = self.rag.get_memory_activation_status(query)
            self._print_activation_status(activation_status)
        
        return results
    
    # ============================================================================
    # 第二部分：API #1 - 自动消退（推荐用于后台维护）
    # ============================================================================
    
    def test_auto_memory_decay(self, current_query: str, retention_ratio: float = 0.9):
        """
        API #1: 应用情境感知的记忆消退
        
        这个方法会根据记忆与当前查询的相关性自动删除低激活的记忆。
        
        Args:
            current_query: 当前查询（用于计算激活分数）
            retention_ratio: 保留比例（0.9 = 保留激活度top-90%）
        
        使用场景：
            - 在知识库数量增长时定期清理
            - 在系统检索速度下降时触发
            - 在内存不足时自动调用
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"API #1: 应用自动消退 (retention_ratio={retention_ratio})")
        logger.info(f"{'='*80}\n")
        
        decay_stats = self.rag.apply_context_aware_memory_decay(
            current_query=current_query,
            retention_ratio=retention_ratio,
            auto_forget=True  # 自动执行删除
        )
        
        logger.info(f"消退前的记忆数量：")
        logger.info(f"  - Chunks: {decay_stats['total_chunks']}")
        logger.info(f"  - Entities: {decay_stats['total_entities']}")
        logger.info(f"  - Facts: {decay_stats['total_facts']}")
        
        logger.info(f"\n消退后的记忆数量：")
        logger.info(f"  - Chunks: {decay_stats['total_chunks'] - len(decay_stats['chunks_to_forget'])}")
        logger.info(f"  - Entities: {decay_stats['total_entities'] - len(decay_stats['entities_to_forget'])}")
        logger.info(f"  - Facts: {decay_stats['total_facts'] - len(decay_stats['facts_to_forget'])}")
        
        if 'auto_forgot_chunks' in decay_stats:
            logger.info(f"\n✅ 已自动删除 {decay_stats['auto_forgot_chunks']} 个文档")
        
        return decay_stats
    
    # ============================================================================
    # 第三部分：API #2 - 手动清除（推荐用于交互式审查）
    # ============================================================================
    
    def test_manual_cleanup(self, current_query: str, activation_threshold: float = 0.1):
        """
        API #2: 手动清除低激活记忆
        
        这个方法允许用户查看低激活的记忆，并在审查后手动删除。
        
        Args:
            current_query: 当前查询
            activation_threshold: 激活分数阈值（低于此值的记忆被标记为删除）
        
        使用场景：
            - 用户希望手动审查要删除的记忆
            - 系统管理员清理知识库
            - 在删除前获得用户确认
        
        完整流程：
            1. dry_run=True：查看将要删除的项目
            2. 用户审查和确认
            3. dry_run=False：执行实际删除
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"API #2: 手动清除低激活记忆")
        logger.info(f"{'='*80}\n")
        
        # 第一步：演练模式（不实际删除）
        logger.info(f"📋 演练模式 (dry_run=True)：")
        logger.info(f"   查找激活分数 < {activation_threshold} 的记忆\n")
        
        cleanup_preview = self.rag.manual_cleanup_low_activation_memories(
            current_query=current_query,
            activation_threshold=activation_threshold,
            dry_run=True  # 仅预览
        )
        
        logger.info(f"📊 发现以下低激活记忆：")
        logger.info(f"  - 低激活Chunks数: {len(cleanup_preview['chunks_to_delete'])}")
        logger.info(f"  - 低激活Entities数: {len(cleanup_preview['entities_to_delete'])}")
        logger.info(f"  - 低激活Facts数: {len(cleanup_preview['facts_to_delete'])}")
        
        # 第二步：用户审查和确认（模拟自动确认）
        logger.info(f"\n✅ 用户审查完成，确认删除\n")
        
        # 第三步：执行删除
        logger.info(f"🔧 执行模式 (dry_run=False)：开始实际删除\n")
        
        cleanup_result = self.rag.manual_cleanup_low_activation_memories(
            current_query=current_query,
            activation_threshold=activation_threshold,
            dry_run=False  # 执行删除
        )
        
        if 'error' not in cleanup_result:
            logger.info(f"✅ 清除完成！")
            if 'actually_deleted_count' in cleanup_result:
                logger.info(f"   已删除 {cleanup_result['actually_deleted_count']} 个文档")
        
        return cleanup_result
    
    # ============================================================================
    # 第四部分：API #3 - 冲突检测与解决（推荐用于知识库更新）
    # ============================================================================
    
    def test_conflict_detection_and_resolution(self, 
                                               new_facts: List[Tuple[str, str, str]],
                                               strategy: str = 'keep_new'):
        """
        API #3: 检测并解决事实冲突
        
        当新的事实与现有事实冲突时，自动根据策略解决。
        
        Args:
            new_facts: 新添加的事实列表
                例如: [
                    ("Erik Hort", "birthplace", "Rockland County"),
                    ("Montebello", "location", "Rockland County")
                ]
            strategy: 冲突解决策略
                - 'keep_new': 新值覆盖旧值（默认，推荐）
                - 'keep_old': 保留旧值
                - 'merge': 合并为"可能是X或Y"
                - 'keep_frequent': 基于访问频率选择
        
        使用场景：
            - 导入新的数据源
            - 更新过时的知识
            - 修正错误信息
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"API #3: 冲突检测与解决 (strategy={strategy})")
        logger.info(f"{'='*80}\n")
        
        logger.info(f"📚 要导入的新事实:")
        for i, fact in enumerate(new_facts, 1):
            logger.info(f"   {i}. {fact}")
        
        # 检测并解决冲突
        resolution_result = self.rag.detect_and_resolve_fact_conflicts(
            new_facts=new_facts,
            resolution_strategy=strategy,
            auto_apply=True  # 自动应用解决方案
        )
        
        logger.info(f"\n⚠️  冲突检测结果：")
        if 'conflicts_detected' in resolution_result:
            logger.info(f"   检测到 {resolution_result['conflicts_detected']} 个冲突")
            
            if 'conflict_records' in resolution_result:
                for record in resolution_result['conflict_records']:
                    logger.info(f"\n   冲突：{record}")
        
        logger.info(f"\n🔧 解决方案（使用'{strategy}'策略）：")
        if 'facts_to_delete' in resolution_result:
            logger.info(f"   要删除的事实: {len(resolution_result['facts_to_delete'])}")
        if 'facts_to_merge' in resolution_result:
            logger.info(f"   要合并的事实: {len(resolution_result['facts_to_merge'])}")
        
        logger.info(f"\n✅ 已应用解决方案")
        
        # 记录冲突
        self.conflict_records.append({
            'strategy': strategy,
            'new_facts': new_facts,
            'result': resolution_result
        })
        
        return resolution_result
    
    # ============================================================================
    # 第五部分：完整工作流演示
    # ============================================================================
    
    def complete_workflow_demo(self, queries: List[str]):
        """
        演示完整的工作流：
        1. 多个查询检索
        2. 查看激活状态
        3. 自动消退
        4. 冲突检测
        5. 手动清除
        """
        logger.info("\n" + "="*80)
        logger.info("完整工作流演示")
        logger.info("="*80)
        
        # 步骤1：执行多个查询
        logger.info("\n[步骤1] 执行多个查询并记录访问历史")
        logger.info("-"*80)
        
        all_results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"\n查询 {i}/{len(queries)}")
            results = self.retrieve_with_auto_decay(query, enable_decay=(i==len(queries)))
            all_results.append(results)
        
        # 步骤2：查看最后一个查询的激活状态
        if queries:
            last_query = queries[-1]
            logger.info("\n[步骤2] 检查内存激活状态")
            logger.info("-"*80)
            
            activation = self.rag.get_memory_activation_status(last_query)
            if 'error' not in activation:
                self._print_activation_status(activation)
        
        # 步骤3：自动消退
        if queries:
            logger.info("\n[步骤3] 应用自动消退")
            logger.info("-"*80)
            
            decay_result = self.test_auto_memory_decay(last_query, retention_ratio=0.8)
        
        # 步骤4：冲突检测
        logger.info("\n[步骤4] 冲突检测与解决")
        logger.info("-"*80)
        
        new_facts = [
            ("Erik Hort", "birthplace", "Updated Location"),
            ("New Entity", "property", "value")
        ]
        conflict_result = self.test_conflict_detection_and_resolution(new_facts, strategy='keep_new')
        
        # 步骤5：手动清除
        if queries:
            logger.info("\n[步骤5] 手动清除低激活记忆")
            logger.info("-"*80)
            
            cleanup_result = self.test_manual_cleanup(last_query, activation_threshold=0.15)
        
        logger.info("\n" + "="*80)
        logger.info("✅ 完整工作流演示完成！")
        logger.info("="*80)
    
    # ============================================================================
    # 辅助方法
    # ============================================================================
    
    def _print_activation_status(self, activation):
        """打印内存激活状态"""
        logger.info(f"\n📊 内存激活分析 (查询: {activation.get('current_query', 'N/A')})")
        logger.info(f"   查询窗口大小: {activation.get('context_window_size', 0)}")
        
        for mem_type in ['chunk', 'entity', 'fact']:
            key = f'{mem_type}_activation'
            if key in activation:
                stats = activation[key]
                logger.info(f"\n   {mem_type.upper()}激活度统计:")
                logger.info(f"      - 高激活 (>0.7): {stats.get('high_activation_count', 0)}")
                logger.info(f"      - 中激活 (0.3-0.7): {stats.get('medium_activation_count', 0)}")
                logger.info(f"      - 低激活 (0.05-0.3): {stats.get('low_activation_count', 0)}")
                logger.info(f"      - 非活跃 (≤0.05): {stats.get('inactive_count', 0)}")
                logger.info(f"      - 平均激活度: {stats.get('avg_activation', 0):.3f}")
    
    def print_summary(self):
        """打印测试总结"""
        logger.info("\n" + "="*80)
        logger.info("测试总结")
        logger.info("="*80)
        logger.info(f"执行的查询数: {len(self.query_history)}")
        logger.info(f"检测的冲突数: {len(self.conflict_records)}")
        logger.info(f"\n已执行的查询:")
        for i, q in enumerate(self.query_history, 1):
            logger.info(f"  {i}. {q}")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    """
    使用示例：集成到你的主程序中
    """
    
    logger.info("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║          HippoRAG 记忆管理系统 - 集成测试指南                              ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    
    三个核心API的调用方式：
    
    1️⃣  apply_context_aware_memory_decay()
        ─────────────────────────────────────
        自动消退低激活记忆
        
        decay_stats = rag.apply_context_aware_memory_decay(
            current_query="Who is Erik Hort?",
            retention_ratio=0.9,      # 保留激活度top-90%
            auto_forget=True          # 自动删除标记的记忆
        )
        
        返回: {
            'total_chunks': 100,
            'chunks_to_forget': [hash1, hash2, ...],
            'auto_forgot_chunks': 10
        }
    
    
    2️⃣  manual_cleanup_low_activation_memories()
        ──────────────────────────────────────────
        手动清除低激活记忆（可选择演练模式）
        
        # 第一步：预览（dry_run=True）
        preview = rag.manual_cleanup_low_activation_memories(
            current_query="Who is Erik Hort?",
            activation_threshold=0.1,  # 低于0.1的记忆
            dry_run=True               # 仅预览
        )
        
        # 第二步：执行删除（dry_run=False）
        result = rag.manual_cleanup_low_activation_memories(
            current_query="Who is Erik Hort?",
            activation_threshold=0.1,
            dry_run=False              # 执行删除
        )
        
        返回: {
            'chunks_to_delete': [hash1, hash2, ...],
            'actually_deleted_count': 5
        }
    
    
    3️⃣  detect_and_resolve_fact_conflicts()
        ─────────────────────────────────────
        检测并解决新旧事实的冲突
        
        new_facts = [
            ("Erik Hort", "birthplace", "New Location"),
            ("Montebello", "location", "New County")
        ]
        
        result = rag.detect_and_resolve_fact_conflicts(
            new_facts=new_facts,
            resolution_strategy='keep_new',  # 新值覆盖旧值
            auto_apply=True                  # 自动应用解决方案
        )
        
        返回: {
            'conflicts_detected': 2,
            'facts_to_delete': [...],
            'conflict_records': [...]
        }
    
    
    集成到你的程序中：
    ─────────────────
    
    from src.hipporag import HippoRAG
    
    # 初始化RAG
    rag = HippoRAG(config)
    
    # 创建集成测试对象
    test = LLMIntegrationTest(rag)
    
    # 方式1: 单个API测试
    results = test.retrieve_with_auto_decay("Who is Erik Hort?")
    decay_stats = test.test_auto_memory_decay("Who is Erik Hort?")
    
    # 方式2: 完整工作流
    queries = [
        "Who is Erik Hort?",
        "Where was Erik born?",
        "What is Montebello?",
        "Is Montebello in Rockland County?"
    ]
    test.complete_workflow_demo(queries)
    """)
