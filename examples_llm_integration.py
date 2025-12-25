"""
实际集成示例：展示如何在你的demo.py或main.py中调用三个API

这个脚本展示最常见的集成模式
"""

import logging
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGWithMemoryManagement:
    """
    RAG应用的记忆管理扩展
    
    在你的现有RAG类中添加这些方法即可
    """
    
    def __init__(self, rag_system):
        self.rag = rag_system
        self.config = {
            'auto_decay_enabled': False,           # 是否启用自动消退
            'auto_decay_retention_ratio': 0.9,     # 保留比例
            'manual_cleanup_threshold': 0.1,       # 手动清除的激活阈值
            'conflict_strategy': 'keep_new'        # 冲突解决策略
        }
    
    def answer_question(self, question: str, use_memory_decay: bool = False):
        """
        标准问题回答流程 + 可选的记忆管理
        
        这是最简单的集成方式：在常规检索后调用记忆管理方法
        
        Args:
            question: 用户的问题
            use_memory_decay: 是否在回答后应用消退
        
        Returns:
            答案和关联信息
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"问题：{question}")
        logger.info(f"{'='*80}")
        
        # 步骤1：执行标准检索
        logger.info(f"\n[步骤1] 执行检索...")
        retrieved_docs = self.rag.retrieve(question)
        logger.info(f"✅ 检索完成，获得 {len(retrieved_docs)} 条结果")
        
        # 步骤2：（可选）应用消退
        if use_memory_decay:
            logger.info(f"\n[步骤2] 应用情境感知消退...")
            decay_stats = self.rag.apply_context_aware_memory_decay(
                current_query=question,
                retention_ratio=self.config['auto_decay_retention_ratio'],
                auto_forget=True
            )
            logger.info(f"✅ 消退完成，删除了低激活记忆")
            
            # 打印消退统计
            self._print_decay_summary(decay_stats)
        
        # 步骤3：生成答案（这里只是演示，实际需要调用LLM）
        logger.info(f"\n[步骤3] 调用LLM生成答案...")
        # answer = llm.generate(question, retrieved_docs)
        
        return {
            'question': question,
            'retrieved_docs': retrieved_docs,
            'decay_applied': use_memory_decay
        }
    
    def batch_answer_with_memory_management(self, questions: List[str]):
        """
        批量处理问题，并在最后应用完整的记忆管理
        
        这是适合多轮对话的集成方式
        
        Args:
            questions: 问题列表
        
        Returns:
            所有答案的列表
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"批量问答模式：{len(questions)} 个问题")
        logger.info(f"{'='*80}")
        
        results = []
        
        # 阶段1：处理所有问题
        logger.info(f"\n[阶段1] 处理所有问题")
        logger.info(f"-"*80)
        
        for i, question in enumerate(questions, 1):
            logger.info(f"\n问题 {i}/{len(questions)}: {question}")
            
            # 执行检索（会自动记录访问历史）
            retrieved_docs = self.rag.retrieve(question)
            logger.info(f"  ✅ 检索 {len(retrieved_docs)} 条结果")
            
            results.append({
                'question': question,
                'retrieved_docs': retrieved_docs
            })
        
        # 阶段2：应用消退（基于最后一个问题的上下文）
        if questions:
            last_question = questions[-1]
            
            logger.info(f"\n[阶段2] 应用消退")
            logger.info(f"-"*80)
            
            decay_stats = self.rag.apply_context_aware_memory_decay(
                current_query=last_question,
                retention_ratio=0.85,
                auto_forget=True
            )
            
            logger.info(f"✅ 消退完成")
            self._print_decay_summary(decay_stats)
        
        # 阶段3：手动清除（展示）
        logger.info(f"\n[阶段3] 检查低激活记忆（演练模式）")
        logger.info(f"-"*80)
        
        if questions:
            cleanup_preview = self.rag.manual_cleanup_low_activation_memories(
                current_query=last_question,
                activation_threshold=0.1,
                dry_run=True  # 仅预览
            )
            
            logger.info(f"✅ 预览完成，发现以下低激活项：")
            logger.info(f"  - Chunks: {len(cleanup_preview.get('chunks_to_delete', []))}")
            logger.info(f"  - Entities: {len(cleanup_preview.get('entities_to_delete', []))}")
            logger.info(f"  - Facts: {len(cleanup_preview.get('facts_to_delete', []))}")
            logger.info(f"\n  （可通过 dry_run=False 执行实际删除）")
        
        return results
    
    def add_knowledge_with_conflict_resolution(self, new_documents: List[str]):
        """
        添加新知识并自动处理冲突
        
        这是适合知识库更新的集成方式
        
        Args:
            new_documents: 新文档列表
        
        Returns:
            冲突解决的详细结果
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"添加新知识：{len(new_documents)} 个文档")
        logger.info(f"{'='*80}")
        
        # 步骤1：添加新文档
        logger.info(f"\n[步骤1] 添加新文档...")
        self.rag.add(new_documents)
        logger.info(f"✅ 添加完成")
        
        # 步骤2：从新文档中提取事实（这里需要你的IE模块）
        logger.info(f"\n[步骤2] 从新文档中提取事实...")
        # 这里假设你有一个提取事实的函数
        new_facts = self._extract_facts_from_documents(new_documents)
        logger.info(f"✅ 提取 {len(new_facts)} 个新事实")
        
        # 步骤3：检测并解决冲突
        logger.info(f"\n[步骤3] 检测并解决冲突...")
        
        conflict_result = self.rag.detect_and_resolve_fact_conflicts(
            new_facts=new_facts,
            resolution_strategy='keep_new',  # 新值覆盖旧值
            auto_apply=True
        )
        
        logger.info(f"✅ 冲突处理完成")
        logger.info(f"  - 检测到冲突: {conflict_result.get('conflicts_detected', 0)}")
        logger.info(f"  - 已删除事实: {len(conflict_result.get('facts_to_delete', []))}")
        logger.info(f"  - 已合并事实: {len(conflict_result.get('facts_to_merge', []))}")
        
        return conflict_result
    
    # ============================================================================
    # 辅助方法
    # ============================================================================
    
    def _print_decay_summary(self, decay_stats):
        """打印消退统计摘要"""
        logger.info(f"\n  📊 消退统计：")
        logger.info(f"     - 删除的Chunks: {len(decay_stats.get('chunks_to_forget', []))}")
        logger.info(f"     - 删除的Entities: {len(decay_stats.get('entities_to_forget', []))}")
        logger.info(f"     - 删除的Facts: {len(decay_stats.get('facts_to_forget', []))}")
    
    def _extract_facts_from_documents(self, documents: List[str]) -> List[tuple]:
        """
        从文档中提取事实
        
        这里需要你自己实现，使用你现有的IE模块
        """
        # TODO: 实现你自己的事实提取逻辑
        # 例如：使用OpenAI或本地模型
        # facts = extract_triples_from_text(documents)
        
        # 这里返回示例
        return [
            ("entity1", "relation", "entity2"),
            ("entity3", "property", "value")
        ]


# ============================================================================
# 完整使用示例
# ============================================================================

def example_1_simple_qa():
    """示例1：简单的问答（带可选消退）"""
    logger.info("""
    
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                         示例1: 简单问答                                    ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # 假设你已经初始化了RAG系统
    # from src.hipporag import HippoRAG
    # rag = HippoRAG(config)
    
    # 创建管理对象
    # manager = RAGWithMemoryManagement(rag)
    
    # 方式1：简单问答（不启用消退）
    # answer = manager.answer_question("Who is Erik Hort?")
    
    # 方式2：问答 + 自动消退
    # answer = manager.answer_question(
    #     "Who is Erik Hort?",
    #     use_memory_decay=True
    # )
    
    logger.info("代码示例：")
    logger.info("""
    manager = RAGWithMemoryManagement(rag)
    
    # 简单问答
    result = manager.answer_question("Who is Erik Hort?")
    
    # 或者启用消退
    result = manager.answer_question(
        "Who is Erik Hort?",
        use_memory_decay=True
    )
    """)


def example_2_multi_turn_conversation():
    """示例2：多轮对话"""
    logger.info("""
    
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                       示例2: 多轮对话 + 消退                               ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("代码示例：")
    logger.info("""
    manager = RAGWithMemoryManagement(rag)
    
    questions = [
        "Who is Erik Hort?",
        "Where was Erik born?",
        "What is Montebello?",
        "Is Montebello in Rockland County?"
    ]
    
    results = manager.batch_answer_with_memory_management(questions)
    """)


def example_3_knowledge_update():
    """示例3：知识库更新"""
    logger.info("""
    
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                   示例3: 知识库更新 + 冲突解决                             ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("代码示例：")
    logger.info("""
    manager = RAGWithMemoryManagement(rag)
    
    new_documents = [
        "Updated information about Erik Hort...",
        "New information about Montebello..."
    ]
    
    # 添加知识并自动处理冲突
    conflict_result = manager.add_knowledge_with_conflict_resolution(new_documents)
    """)


def example_4_advanced_workflow():
    """示例4：高级工作流"""
    logger.info("""
    
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                     示例4: 完整高级工作流                                  ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("代码示例：")
    logger.info("""
    manager = RAGWithMemoryManagement(rag)
    
    # 工作流：
    # 1. 多轮对话
    # 2. 添加新知识
    # 3. 处理冲突
    # 4. 手动清除
    
    # 阶段1：对话
    questions = ["Q1", "Q2", "Q3"]
    results = manager.batch_answer_with_memory_management(questions)
    
    # 阶段2：更新知识
    new_docs = ["Updated doc 1", "Updated doc 2"]
    conflict_result = manager.add_knowledge_with_conflict_resolution(new_docs)
    
    # 阶段3：手动清除
    cleanup_result = manager.rag.manual_cleanup_low_activation_memories(
        current_query=questions[-1],
        activation_threshold=0.1,
        dry_run=False  # 执行删除
    )
    """)


if __name__ == "__main__":
    logger.info("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                  HippoRAG 记忆管理API - 集成指南                           ║
    ║                                                                            ║
    ║  这个文件展示4种常见的集成模式，你可以选择最适合你的用途                   ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # 打印示例
    example_1_simple_qa()
    example_2_multi_turn_conversation()
    example_3_knowledge_update()
    example_4_advanced_workflow()
    
    logger.info("""
    
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                           快速开始步骤                                     ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    
    步骤1：导入和初始化
    ─────────────────
    from src.hipporag import HippoRAG
    from integration_test_with_llm import LLMIntegrationTest
    
    rag = HippoRAG(config)
    test = LLMIntegrationTest(rag)
    
    
    步骤2：选择一个使用模式
    ─────────────────
    
    模式A: 自动消退
    ────────────────
    result = rag.apply_context_aware_memory_decay(
        current_query="Your question",
        retention_ratio=0.9,
        auto_forget=True
    )
    
    
    模式B: 手动清除（两步）
    ──────────────────────
    # 第一步：预览
    preview = rag.manual_cleanup_low_activation_memories(
        current_query="Your question",
        activation_threshold=0.1,
        dry_run=True
    )
    
    # 第二步：执行
    result = rag.manual_cleanup_low_activation_memories(
        current_query="Your question",
        activation_threshold=0.1,
        dry_run=False
    )
    
    
    模式C: 冲突解决
    ───────────────
    result = rag.detect_and_resolve_fact_conflicts(
        new_facts=[("Entity", "Relation", "Value")],
        resolution_strategy='keep_new',
        auto_apply=True
    )
    
    
    模式D: 完整工作流
    ─────────────────
    test.complete_workflow_demo([
        "Question 1",
        "Question 2",
        "Question 3"
    ])
    
    
    更多信息
    ──────
    查看 integration_test_with_llm.py 获取完整的API文档
    """)
