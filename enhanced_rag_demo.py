"""
在你现有的 demo.py/main.py 中集成 Ollama Qwen 的示例

这个文件展示如何修改你的主程序以支持新的记忆管理功能
"""

from test_with_local_ollama import OllamaQwenWrapper, LocalOllamaRAGTest
import logging

logger = logging.getLogger(__name__)


class EnhancedRAGDemo:
    """
    增强版 RAG - 集成本地 Qwen LLM 和新的记忆管理功能
    """
    
    def __init__(self, rag_instance, use_llm: bool = True):
        """
        初始化增强版 RAG
        
        Args:
            rag_instance: HippoRAG 实例
            use_llm: 是否使用本地 Qwen LLM
        """
        self.rag = rag_instance
        self.use_llm = use_llm
        
        if use_llm:
            try:
                self.llm = OllamaQwenWrapper()
                logger.info("✅ Qwen LLM 初始化成功")
            except Exception as e:
                logger.warning(f"⚠️  Qwen LLM 初始化失败: {e}")
                self.llm = None
                self.use_llm = False
        else:
            self.llm = None
    
    def demo_1_basic_qa(self):
        """演示1：基础问答"""
        logger.info("\n" + "="*80)
        logger.info("演示1: 基础问答 + 自动消退")
        logger.info("="*80)
        
        question = "Who is Erik Hort?"
        
        logger.info(f"\n👤 问题: {question}")
        
        # 检索
        logger.info("\n[步骤1] 检索相关文档...")
        try:
            results = self.rag.retrieve(question)
            logger.info(f"✅ 检索完成，得到 {len(results)} 条结果")
        except:
            results = []
        
        # 生成答案
        if self.use_llm and self.llm:
            logger.info("\n[步骤2] Qwen 生成答案...")
            context = "\n".join([str(doc)[:200] for doc in results[:2]])
            answer = self.llm.answer_question(question, context=context)
            logger.info(f"🤖 答案: {answer}")
        
        # 自动消退
        logger.info("\n[步骤3] 应用自动消退...")
        try:
            decay_stats = self.rag.apply_context_aware_memory_decay(
                current_query=question,
                retention_ratio=0.85,
                auto_forget=True
            )
            logger.info(f"✅ 消退完成，删除了 {len(decay_stats['chunks_to_forget'])} 条低激活记忆")
        except Exception as e:
            logger.warning(f"⚠️  消退失败: {e}")
    
    def demo_2_multi_turn_conversation(self):
        """演示2：多轮对话"""
        logger.info("\n" + "="*80)
        logger.info("演示2: 多轮对话")
        logger.info("="*80)
        
        questions = [
            "Who is Erik Hort?",
            "Where was Erik born?",
            "Tell me about Montebello."
        ]
        
        logger.info(f"\n💬 开始 {len(questions)} 轮对话\n")
        
        for i, question in enumerate(questions, 1):
            logger.info(f"轮次 {i}: {question}")
            
            # 检索
            try:
                results = self.rag.retrieve(question)
                logger.info(f"  ✅ 检索 {len(results)} 条结果")
            except:
                results = []
            
            # LLM 回答
            if self.use_llm and self.llm:
                try:
                    context = "\n".join([str(doc)[:200] for doc in results[:2]])
                    answer = self.llm.answer_question(question, context=context)
                    logger.info(f"  🤖 {answer[:100]}...")
                except:
                    pass
            
            # 最后一轮后应用消退
            if i == len(questions):
                logger.info(f"\n对话结束，应用消退...")
                try:
                    self.rag.apply_context_aware_memory_decay(
                        current_query=question,
                        retention_ratio=0.80,
                        auto_forget=True
                    )
                    logger.info(f"✅ 消退完成")
                except:
                    pass
    
    def demo_3_knowledge_update_with_conflict_resolution(self):
        """演示3：知识库更新 + 冲突处理"""
        logger.info("\n" + "="*80)
        logger.info("演示3: 知识库更新 + 冲突处理")
        logger.info("="*80)
        
        # 新文档
        new_docs = [
            "Updated: Erik Hort was born in Rockland County, not Montebello.",
            "New info: Rockland County is an important historical region."
        ]
        
        logger.info(f"\n📚 添加 {len(new_docs)} 个新文档...")
        try:
            self.rag.add(new_docs)
            logger.info("✅ 文档添加完成")
        except Exception as e:
            logger.warning(f"⚠️  添加失败: {e}")
        
        # 新事实
        new_facts = [
            ("Erik Hort", "birthplace", "Rockland County"),
            ("Montebello", "location", "Rockland County")
        ]
        
        logger.info(f"\n🔍 检测并解决冲突...")
        try:
            result = self.rag.detect_and_resolve_fact_conflicts(
                new_facts=new_facts,
                resolution_strategy='keep_new',
                auto_apply=True
            )
            logger.info(f"✅ 检测到 {result.get('conflicts_detected', 0)} 个冲突")
            logger.info(f"✅ 已用新值覆盖旧值")
        except Exception as e:
            logger.warning(f"⚠️  冲突处理失败: {e}")
    
    def demo_4_memory_cleanup(self):
        """演示4：内存清理（演练 + 执行）"""
        logger.info("\n" + "="*80)
        logger.info("演示4: 内存清理")
        logger.info("="*80)
        
        query = "What is Montebello?"
        
        # 演练模式
        logger.info(f"\n📋 步骤1: 预览要删除的项目 (dry_run=True)")
        try:
            preview = self.rag.manual_cleanup_low_activation_memories(
                current_query=query,
                activation_threshold=0.15,
                dry_run=True
            )
            logger.info(f"✅ 预览完成:")
            logger.info(f"   - 低激活 Chunks: {len(preview.get('chunks_to_delete', []))}")
        except Exception as e:
            logger.warning(f"⚠️  预览失败: {e}")
        
        # 执行删除
        logger.info(f"\n🔧 步骤2: 执行删除 (dry_run=False)")
        try:
            result = self.rag.manual_cleanup_low_activation_memories(
                current_query=query,
                activation_threshold=0.15,
                dry_run=False
            )
            if 'actually_deleted_count' in result:
                logger.info(f"✅ 已删除 {result['actually_deleted_count']} 个文档")
            else:
                logger.info(f"✅ 清理完成")
        except Exception as e:
            logger.warning(f"⚠️  删除失败: {e}")
    
    def demo_5_complete_workflow(self):
        """演示5：完整工作流"""
        logger.info("\n" + "="*80)
        logger.info("演示5: 完整工作流")
        logger.info("="*80 + "\n")
        
        # 调用所有演示
        self.demo_1_basic_qa()
        self.demo_2_multi_turn_conversation()
        self.demo_3_knowledge_update_with_conflict_resolution()
        self.demo_4_memory_cleanup()
        
        logger.info("\n" + "="*80)
        logger.info("✅ 完整工作流演示完成")
        logger.info("="*80)


# ============================================================================
# 使用示例
# ============================================================================

def main():
    """
    在你的 demo.py 或 main.py 中使用这个示例
    """
    
    # 假设你已经初始化了 HippoRAG
    # from src.hipporag import HippoRAG
    # rag = HippoRAG(your_config)
    
    # 创建增强版 RAG
    # enhanced_rag = EnhancedRAGDemo(rag, use_llm=True)
    
    # 运行演示
    # enhanced_rag.demo_1_basic_qa()
    # enhanced_rag.demo_2_multi_turn_conversation()
    # enhanced_rag.demo_3_knowledge_update_with_conflict_resolution()
    # enhanced_rag.demo_4_memory_cleanup()
    # enhanced_rag.demo_5_complete_workflow()
    
    print("""
    使用说明：
    
    1. 在你的 demo.py 中导入:
       from enhanced_rag_demo import EnhancedRAGDemo
    
    2. 在初始化 HippoRAG 后创建增强版 RAG:
       enhanced_rag = EnhancedRAGDemo(rag, use_llm=True)
    
    3. 调用演示方法:
       enhanced_rag.demo_1_basic_qa()
       enhanced_rag.demo_5_complete_workflow()
    """)


if __name__ == "__main__":
    main()
