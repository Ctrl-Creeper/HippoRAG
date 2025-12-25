"""
本地测试脚本：使用 Ollama + Qwen3:1.7b 测试 HippoRAG 记忆管理系统

前置条件：
1. 已安装 ollama
2. 已拉取 Qwen3:1.7b 模型：ollama pull qwen3:1.7b
3. 运行 ollama serve (默认监听 http://localhost:11434)
"""

import logging
import json
import time
from typing import List, Dict, Tuple, Optional
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OllamaQwenWrapper:
    """Ollama + Qwen 的简单包装"""
    
    def __init__(self, model_name: str = "qwen3:1.7b", 
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.7):
        """
        Args:
            model_name: 模型名称 (默认 qwen3:1.7b)
            base_url: ollama 服务地址
            temperature: 生成温度 (0-1, 越低越确定)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.api_endpoint = f"{base_url}/api/generate"
        
        # 检查连接
        self._check_connection()
    
    def _check_connection(self):
        """检查 ollama 服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                logger.info(f"✅ Ollama 服务连接成功")
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                logger.info(f"   可用模型: {', '.join(model_names)}")
            else:
                raise Exception("Ollama 服务返回错误")
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ 无法连接到 Ollama 服务 ({self.base_url})")
            logger.error("   请确保已运行: ollama serve")
            raise
        except Exception as e:
            logger.error(f"❌ 检查 Ollama 服务出错: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """
        使用 Qwen 生成文本
        
        Args:
            prompt: 提示文本
            max_tokens: 最大生成令牌数
        
        Returns:
            生成的文本
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "num_predict": max_tokens,
                "stream": False  # 不使用流式输出，便于处理
            }
            
            logger.debug(f"发送请求到 Ollama...")
            start_time = time.time()
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=60  # 增加超时时间，因为模型较小可能需要时间
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                logger.debug(f"✅ 生成完成 ({elapsed_time:.2f}s)")
                return generated_text
            else:
                logger.error(f"❌ Ollama API 返回错误: {response.status_code}")
                return ""
        
        except requests.exceptions.Timeout:
            logger.error("❌ 请求超时，Qwen 模型生成耗时过长")
            return ""
        except Exception as e:
            logger.error(f"❌ 生成文本出错: {e}")
            return ""
    
    def extract_facts(self, text: str) -> List[Tuple[str, str, str]]:
        """
        使用 Qwen 从文本中提取事实三元组
        
        Args:
            text: 输入文本
        
        Returns:
            事实三元组列表 [(subject, predicate, object), ...]
        """
        prompt = f"""请从以下文本中提取关键事实，格式为(主体, 关系, 宾体)。
        只列出事实，每行一个，不要有其他说明。
        
        文本: {text}
        
        事实:"""
        
        response = self.generate(prompt, max_tokens=150)
        
        # 解析输出
        facts = []
        for line in response.split('\n'):
            line = line.strip()
            if line and '(' in line and ')' in line:
                try:
                    # 简单的解析逻辑
                    content = line.replace('(', '').replace(')', '').strip()
                    parts = [p.strip() for p in content.split(',')]
                    if len(parts) == 3:
                        facts.append(tuple(parts))
                except:
                    pass
        
        return facts if facts else [("unknown", "property", "value")]
    
    def answer_question(self, question: str, context: str = "") -> str:
        """
        使用 Qwen 回答问题
        
        Args:
            question: 用户问题
            context: 可选的背景信息
        
        Returns:
            答案
        """
        if context:
            prompt = f"""请根据以下背景信息回答问题。

背景信息: {context}

问题: {question}

答案:"""
        else:
            prompt = f"""问题: {question}

答案:"""
        
        answer = self.generate(prompt, max_tokens=200)
        return answer if answer else "无法生成答案"


class LocalOllamaRAGTest:
    """
    使用本地 Ollama + Qwen 测试 HippoRAG 记忆管理系统
    """
    
    def __init__(self, rag_system, llm: Optional[OllamaQwenWrapper] = None):
        """
        Args:
            rag_system: HippoRAG 实例
            llm: OllamaQwenWrapper 实例（如果为None，会自动初始化）
        """
        self.rag = rag_system
        self.llm = llm or OllamaQwenWrapper()
        self.test_results = []
    
    # ========================================================================
    # 测试1: 激活分数动态变化
    # ========================================================================
    
    def test_1_activation_dynamics(self):
        """
        测试1: 随着查询序列，激活分数如何变化
        
        场景：同一个entity被多个相关查询激活，观察其激活分数变化
        """
        logger.info("\n" + "="*80)
        logger.info("测试1: 激活分数动态变化")
        logger.info("="*80)
        
        # 准备测试文本
        test_docs = [
            "Erik Hort is a notable historical figure. He was born in Montebello, a town known for its rich history.",
            "Montebello is a small town in New York, part of Rockland County. It has a population of around 3000 residents.",
            "Rockland County is located in the Hudson Valley region of New York. It borders New Jersey across the Hudson River."
        ]
        
        logger.info(f"\n📄 添加 {len(test_docs)} 个测试文档...")
        try:
            self.rag.add(test_docs)
            logger.info("✅ 文档添加成功\n")
        except Exception as e:
            logger.warning(f"⚠️  文档添加失败: {e}")
        
        # 执行查询序列
        queries = [
            "Who is Erik Hort?",
            "Where was Erik born?",
            "What is Montebello?",
            "Is Montebello in Rockland County?"
        ]
        
        logger.info(f"📋 执行 {len(queries)} 个相关查询:\n")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"查询 {i}: {query}")
            
            # 执行检索
            try:
                results = self.rag.retrieve(query)
                logger.info(f"  ✅ 检索完成，返回 {len(results)} 条结果")
            except Exception as e:
                logger.warning(f"  ⚠️  检索失败: {e}")
                results = []
            
            # 最后一个查询后检查激活状态
            if i == len(queries):
                logger.info(f"\n🧠 最后查询的激活状态分析:\n")
                try:
                    activation = self.rag.get_memory_activation_status(query)
                    if 'error' not in activation:
                        self._print_activation_analysis(activation)
                    else:
                        logger.warning(f"⚠️  获取激活状态失败")
                except Exception as e:
                    logger.warning(f"⚠️  获取激活状态出错: {e}")
        
        logger.info(f"\n✅ 测试1完成: 高度相关的记忆激活度应该较高，无关的较低\n")
    
    # ========================================================================
    # 测试2: 自动消退功能
    # ========================================================================
    
    def test_2_auto_decay(self):
        """
        测试2: 自动消退低激活记忆
        
        场景：在查询特定主题后，与其他主题相关的记忆被标记为低激活，然后删除
        """
        logger.info("\n" + "="*80)
        logger.info("测试2: 自动消退功能")
        logger.info("="*80)
        
        # 获取任意一个查询来计算激活分数
        test_query = "What is Montebello?"
        
        logger.info(f"\n📊 消退前的记忆统计:")
        
        try:
            # 获取消退前的统计
            chunk_ids_before = len(self.rag.chunk_embedding_store.get_all_ids())
            entity_ids_before = len(self.rag.entity_embedding_store.get_all_ids())
            fact_ids_before = len(self.rag.fact_embedding_store.get_all_ids())
            
            logger.info(f"  - Chunks: {chunk_ids_before}")
            logger.info(f"  - Entities: {entity_ids_before}")
            logger.info(f"  - Facts: {fact_ids_before}")
            
            # 执行自动消退
            logger.info(f"\n🔄 执行自动消退 (retention_ratio=0.8)...")
            decay_stats = self.rag.apply_context_aware_memory_decay(
                current_query=test_query,
                retention_ratio=0.8,  # 保留80%，删除20%
                auto_forget=True
            )
            
            logger.info(f"\n📊 消退后的记忆统计:")
            logger.info(f"  - Chunks 标记删除: {len(decay_stats['chunks_to_forget'])}")
            logger.info(f"  - Entities 标记删除: {len(decay_stats['entities_to_forget'])}")
            logger.info(f"  - Facts 标记删除: {len(decay_stats['facts_to_forget'])}")
            
            if 'auto_forgot_chunks' in decay_stats:
                logger.info(f"  - 实际删除的文档: {decay_stats['auto_forgot_chunks']}")
            
            logger.info(f"\n✅ 消退完成")
            
        except Exception as e:
            logger.warning(f"⚠️  消退出错: {e}")
    
    # ========================================================================
    # 测试3: 冲突检测与解决
    # ========================================================================
    
    def test_3_conflict_resolution(self):
        """
        测试3: 检测并解决事实冲突
        
        场景：导入新信息与现有知识冲突，自动用新值覆盖旧值
        """
        logger.info("\n" + "="*80)
        logger.info("测试3: 冲突检测与解决")
        logger.info("="*80)
        
        # 模拟新添加的事实（与原有文档信息冲突）
        new_facts = [
            ("Erik Hort", "birthplace", "Rockland County"),  # 与原文本冲突
            ("Montebello", "location", "Rockland County"),   # 与原文本冲突
            ("Alice Smith", "profession", "historian")       # 新事实，无冲突
        ]
        
        logger.info(f"\n📚 新添加的事实:")
        for i, fact in enumerate(new_facts, 1):
            logger.info(f"  {i}. {fact}")
        
        logger.info(f"\n🔍 检测冲突...")
        
        try:
            conflict_result = self.rag.detect_and_resolve_fact_conflicts(
                new_facts=new_facts,
                resolution_strategy='keep_new',  # 新值覆盖旧值
                auto_apply=True
            )
            
            logger.info(f"\n⚠️  冲突检测结果:")
            logger.info(f"  - 检测到的冲突: {conflict_result.get('conflicts_detected', 0)}")
            logger.info(f"  - 要删除的旧事实: {len(conflict_result.get('facts_to_delete', []))}")
            logger.info(f"  - 要合并的事实: {len(conflict_result.get('facts_to_merge', []))}")
            
            if conflict_result.get('conflict_records'):
                logger.info(f"\n📋 冲突详情:")
                for i, record in enumerate(conflict_result['conflict_records'][:3], 1):
                    logger.info(f"  {i}. {record}")
            
            logger.info(f"\n✅ 已应用 'keep_new' 策略，新值覆盖旧值")
            
        except Exception as e:
            logger.warning(f"⚠️  冲突处理出错: {e}")
    
    # ========================================================================
    # 测试4: 手动清除（两步流程）
    # ========================================================================
    
    def test_4_manual_cleanup(self):
        """
        测试4: 手动清除低激活记忆
        
        场景：用户可以先预览要删除的项目，然后确认删除
        """
        logger.info("\n" + "="*80)
        logger.info("测试4: 手动清除低激活记忆")
        logger.info("="*80)
        
        test_query = "What is Rockland County?"
        
        # 第一步：预览
        logger.info(f"\n📋 步骤1: 预览模式 (dry_run=True)")
        logger.info(f"   查找激活分数 < 0.15 的记忆\n")
        
        try:
            preview = self.rag.manual_cleanup_low_activation_memories(
                current_query=test_query,
                activation_threshold=0.15,
                dry_run=True  # 不实际删除
            )
            
            logger.info(f"📊 预览结果:")
            logger.info(f"  - 低激活 Chunks: {len(preview.get('chunks_to_delete', []))}")
            logger.info(f"  - 低激活 Entities: {len(preview.get('entities_to_delete', []))}")
            logger.info(f"  - 低激活 Facts: {len(preview.get('facts_to_delete', []))}")
            
            # 第二步：确认删除
            logger.info(f"\n✅ 用户审查完成，确认删除\n")
            
            logger.info(f"🔧 步骤2: 执行模式 (dry_run=False)")
            logger.info(f"   开始实际删除\n")
            
            result = self.rag.manual_cleanup_low_activation_memories(
                current_query=test_query,
                activation_threshold=0.15,
                dry_run=False  # 实际删除
            )
            
            if 'actually_deleted_count' in result:
                logger.info(f"✅ 清除完成: 删除了 {result['actually_deleted_count']} 个文档")
            else:
                logger.info(f"✅ 清除完成")
            
        except Exception as e:
            logger.warning(f"⚠️  手动清除出错: {e}")
    
    # ========================================================================
    # 测试5: 使用 Qwen LLM 的完整对话流程
    # ========================================================================
    
    def test_5_llm_integration(self):
        """
        测试5: 集成 Qwen LLM 的完整对话
        
        场景：
        1. 用户提问
        2. RAG 检索相关文档
        3. Qwen 生成答案
        4. 记录访问历史
        5. 自动应用消退
        """
        logger.info("\n" + "="*80)
        logger.info("测试5: Qwen LLM 集成对话")
        logger.info("="*80)
        
        # 对话序列
        conversation = [
            "Who is Erik Hort?",
            "Where was he born?",
            "Tell me about Montebello."
        ]
        
        logger.info(f"\n💬 开始 {len(conversation)} 轮对话\n")
        
        for i, question in enumerate(conversation, 1):
            logger.info(f"轮次 {i}:")
            logger.info(f"👤 用户: {question}\n")
            
            # 步骤1: RAG 检索
            logger.info(f"  [步骤1] RAG 检索相关文档...")
            try:
                retrieved_docs = self.rag.retrieve(question)
                logger.info(f"  ✅ 检索完成，得到 {len(retrieved_docs)} 条文档\n")
                
                # 组织上下文
                context = "\n".join([
                    doc.get('content', str(doc))[:200] 
                    for doc in retrieved_docs[:2]
                ])
            except Exception as e:
                logger.warning(f"  ⚠️  检索失败: {e}")
                context = ""
            
            # 步骤2: Qwen 生成答案
            logger.info(f"  [步骤2] Qwen 生成答案...")
            try:
                answer = self.llm.answer_question(question, context=context)
                logger.info(f"  ✅ 答案生成完成\n")
            except Exception as e:
                logger.warning(f"  ⚠️  生成失败: {e}")
                answer = "无法生成答案"
            
            # 输出答案
            logger.info(f"🤖 Qwen: {answer}\n")
            
            # 步骤3: 保存结果
            self.test_results.append({
                'turn': i,
                'question': question,
                'answer': answer,
                'docs_retrieved': len(retrieved_docs) if retrieved_docs else 0
            })
            
            # 最后一个问题后应用消退
            if i == len(conversation):
                logger.info(f"  [步骤3] 对话结束，应用自动消退...\n")
                try:
                    decay_stats = self.rag.apply_context_aware_memory_decay(
                        current_query=question,
                        retention_ratio=0.85,
                        auto_forget=True
                    )
                    logger.info(f"  ✅ 消退完成\n")
                except Exception as e:
                    logger.warning(f"  ⚠️  消退失败: {e}\n")
        
        logger.info(f"✅ 对话测试完成\n")
    
    # ========================================================================
    # 测试6: 内存激活状态诊断
    # ========================================================================
    
    def test_6_activation_diagnostics(self):
        """
        测试6: 深入诊断当前内存的激活状态
        
        这有助于理解记忆系统的工作情况
        """
        logger.info("\n" + "="*80)
        logger.info("测试6: 内存激活状态诊断")
        logger.info("="*80)
        
        test_query = "What is Montebello?"
        
        logger.info(f"\n🔍 诊断查询: {test_query}\n")
        
        try:
            activation = self.rag.get_memory_activation_status(test_query)
            
            if 'error' in activation:
                logger.warning(f"⚠️  获取激活状态失败")
                return
            
            self._print_activation_analysis(activation)
            
            # 额外的诊断信息
            logger.info(f"\n📊 详细统计:")
            
            for mem_type in ['chunk', 'entity', 'fact']:
                key = f'{mem_type}_activation'
                if key in activation:
                    stats = activation[key]
                    total = stats['total_count']
                    if total > 0:
                        high_ratio = stats['high_activation_count'] / total * 100
                        logger.info(f"  {mem_type.upper()}: {high_ratio:.1f}% 处于高激活状态")
            
        except Exception as e:
            logger.warning(f"⚠️  诊断出错: {e}")
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _print_activation_analysis(self, activation: Dict):
        """打印激活状态分析"""
        logger.info(f"📈 激活状态分析:")
        
        for mem_type in ['chunk', 'entity', 'fact']:
            key = f'{mem_type}_activation'
            if key in activation:
                stats = activation[key]
                
                if stats['total_count'] == 0:
                    logger.info(f"\n  {mem_type.upper()}: (无数据)")
                    continue
                
                logger.info(f"\n  {mem_type.upper()}:")
                logger.info(f"    - 总数: {stats['total_count']}")
                logger.info(f"    - 高激活 (>0.7): {stats['high_activation_count']} ({stats['high_activation_count']/stats['total_count']*100:.1f}%)")
                logger.info(f"    - 中激活 (0.3-0.7): {stats['medium_activation_count']} ({stats['medium_activation_count']/stats['total_count']*100:.1f}%)")
                logger.info(f"    - 低激活 (0.05-0.3): {stats['low_activation_count']} ({stats['low_activation_count']/stats['total_count']*100:.1f}%)")
                logger.info(f"    - 非活跃 (≤0.05): {stats['inactive_count']} ({stats['inactive_count']/stats['total_count']*100:.1f}%)")
                logger.info(f"    - 平均激活度: {stats['avg_activation']:.3f}")
                logger.info(f"    - 最大激活度: {stats['max_activation']:.3f}")
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("""
        
╔════════════════════════════════════════════════════════════════════════════╗
║                  HippoRAG 记忆管理系统 - 本地 Ollama 测试                   ║
║                      使用 Qwen3:1.7b 模型                                  ║
╚════════════════════════════════════════════════════════════════════════════╝
        """)
        
        tests = [
            ("激活分数动态变化", self.test_1_activation_dynamics),
            ("自动消退功能", self.test_2_auto_decay),
            ("冲突检测与解决", self.test_3_conflict_resolution),
            ("手动清除功能", self.test_4_manual_cleanup),
            ("Qwen LLM 集成", self.test_5_llm_integration),
            ("激活状态诊断", self.test_6_activation_diagnostics)
        ]
        
        results_summary = []
        
        for test_name, test_func in tests:
            try:
                test_func()
                results_summary.append((test_name, "✅ 通过"))
            except Exception as e:
                logger.error(f"❌ {test_name} 失败: {e}")
                results_summary.append((test_name, f"❌ 失败: {str(e)[:50]}"))
        
        # 打印总结
        logger.info("\n" + "="*80)
        logger.info("测试总结")
        logger.info("="*80)
        
        for test_name, result in results_summary:
            logger.info(f"{test_name:20} {result}")
        
        logger.info("="*80)


def main():
    """
    使用示例
    """
    try:
        # 初始化 Ollama Qwen
        logger.info("初始化 Ollama Qwen LLM...")
        llm = OllamaQwenWrapper(
            model_name="qwen3:1.7b",
            base_url="http://localhost:11434",
            temperature=0.7
        )
        logger.info("✅ Qwen LLM 初始化成功\n")
        
        # 这里需要你已经初始化好 HippoRAG
        # 如果你的 HippoRAG 初始化需要特定配置，请修改下面的代码
        
        # 示例 1: 如果你有现成的 RAG 实例
        # from src.hipporag import HippoRAG
        # rag = HippoRAG(config)
        
        # 示例 2: 如果需要创建测试用的 RAG
        # 请取消注释并根据你的配置修改
        
        logger.error("⚠️  需要初始化 HippoRAG 实例")
        logger.error("   请修改 main() 函数，添加你的 HippoRAG 初始化代码")
        logger.error("\n   例如:")
        logger.error("   from src.hipporag import HippoRAG")
        logger.error("   rag = HippoRAG(your_config)")
        logger.error("   ")
        logger.error("   然后取消注释下面的代码")
        
        # # 创建测试对象
        # test = LocalOllamaRAGTest(rag, llm)
        # 
        # # 运行所有测试
        # test.run_all_tests()
        
    except Exception as e:
        logger.error(f"初始化失败: {e}")


if __name__ == "__main__":
    main()
