"""
Ollama Qwen 本地测试 - 完整快速启动指南

五分钟快速上手
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_setup_guide():
    """打印完整的设置指南"""
    
    guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║           HippoRAG 本地测试指南 - Ollama Qwen3:1.7b                       ║
╚════════════════════════════════════════════════════════════════════════════╝


📋 第一步：环境准备 (1分钟)
═════════════════════════════════════════════════════════════════════════════

1.1 安装 Ollama
───────────────
访问 https://ollama.ai 下载并安装 Ollama

1.2 拉取 Qwen3:1.7b 模型
────────────────────────
打开终端，运行：
  $ ollama pull qwen3:1.7b

等待下载完成（3-5分钟），模型约3-5GB

1.3 验证安装
───────────
运行：
  $ ollama list

应该能看到 qwen3:1.7b 在列表中


📋 第二步：启动 Ollama 服务 (1分钟)
═════════════════════════════════════════════════════════════════════════════

在一个新的终端中运行：
  $ ollama serve

你会看到类似的输出：
  listening on 127.0.0.1:11434

保持这个终端打开。服务在后台运行。


📋 第三步：修改你的代码 (2分钟)
═════════════════════════════════════════════════════════════════════════════

在你的 demo.py 或 main.py 中添加：

──────────────────────────────────────────────────────────────────────────
from enhanced_rag_demo import EnhancedRAGDemo

# 假设你已有这个：
rag = HippoRAG(config)

# 创建增强版 RAG，支持本地 Qwen LLM
enhanced_rag = EnhancedRAGDemo(rag, use_llm=True)

# 运行完整演示
enhanced_rag.demo_5_complete_workflow()

# 或运行单个演示：
enhanced_rag.demo_1_basic_qa()
enhanced_rag.demo_2_multi_turn_conversation()
──────────────────────────────────────────────────────────────────────────


📋 第四步：运行测试 (立即开始)
═════════════════════════════════════════════════════════════════════════════

在另一个终端中运行你的程序：
  $ python your_demo.py

或如果你想快速测试内存管理功能：
  $ python test_with_local_ollama.py

(需要先在 test_with_local_ollama.py 中添加 HippoRAG 初始化代码)


═════════════════════════════════════════════════════════════════════════════
🚀 最小化起始代码（如果你没有现成的 HippoRAG）
═════════════════════════════════════════════════════════════════════════════

from test_with_local_ollama import OllamaQwenWrapper

# 初始化 LLM
llm = OllamaQwenWrapper()

# 测试文本生成
text = llm.generate("什么是人工智能？", max_tokens=200)
print(text)

# 测试事实提取
facts = llm.extract_facts("Erik Hort was born in Montebello, New York.")
print(facts)

# 测试回答问题
answer = llm.answer_question("Erik Hort 是谁？")
print(answer)


═════════════════════════════════════════════════════════════════════════════
📚 五个完整测试（包含在 EnhancedRAGDemo 中）
═════════════════════════════════════════════════════════════════════════════

测试1️⃣ : 基础问答 + 自动消退
  ├─ 检索相关文档
  ├─ Qwen 生成答案
  └─ 自动删除低激活记忆

测试2️⃣ : 多轮对话
  ├─ 三个相关的问题
  ├─ 每轮都获得 Qwen 的回答
  └─ 对话结束后应用消退

测试3️⃣ : 知识库更新 + 冲突处理
  ├─ 添加新文档
  ├─ 检测新旧知识的冲突
  └─ 用新值覆盖旧值

测试4️⃣ : 内存清理
  ├─ 预览要删除的项目 (dry_run=True)
  ├─ 用户审查确认
  └─ 执行实际删除 (dry_run=False)

测试5️⃣ : 完整工作流
  └─ 运行上面四个测试的组合


═════════════════════════════════════════════════════════════════════════════
🔧 配置选项
═════════════════════════════════════════════════════════════════════════════

OllamaQwenWrapper 的参数：

model_name: str = "qwen3:1.7b"
  ├─ 默认使用 Qwen3 1.7b
  └─ 可选: qwen3:4b, mistral, llama2 等

base_url: str = "http://localhost:11434"
  ├─ Ollama 服务地址
  └─ 如果使用不同端口，修改这里

temperature: float = 0.7
  ├─ 生成文本的随机性 (0-1)
  ├─ 0 = 确定性，1 = 随机性
  └─ 推荐值 0.5-0.8


HippoRAG 新 API 的参数：

apply_context_aware_memory_decay():
  ├─ retention_ratio: float = 0.9
  │  └─ 保留的比例 (0.8 = 删除20%)
  └─ auto_forget: bool = True
     └─ 自动执行删除

manual_cleanup_low_activation_memories():
  ├─ activation_threshold: float = 0.1
  │  └─ 删除激活度 < 0.1 的记忆
  └─ dry_run: bool = True
     └─ True 仅预览，False 执行删除

detect_and_resolve_fact_conflicts():
  ├─ resolution_strategy: str = 'keep_new'
  │  ├─ 'keep_new': 新值覆盖旧值
  │  ├─ 'keep_old': 保留旧值
  │  ├─ 'merge': 合并为不确定
  │  └─ 'keep_frequent': 基于访问频率
  └─ auto_apply: bool = False
     └─ 自动应用删除和合并


═════════════════════════════════════════════════════════════════════════════
⚡ 常见问题
═════════════════════════════════════════════════════════════════════════════

Q: 生成速度很慢？
A: 这是正常的。Qwen3 1.7b 在 CPU 上生成需要时间。
   - 如果有 GPU，确保 ollama 能用上它
   - 可以用更小的模型测试：ollama pull phi

Q: "无法连接到 Ollama"？
A: 检查 ollama serve 是否运行在另一个终端中

Q: "Qwen 模型未找到"？
A: 运行 ollama pull qwen3:1.7b 下载模型

Q: HippoRAG 初始化报错？
A: 检查 config 设置和依赖包是否完整

Q: 如何改用其他模型？
A: llm = OllamaQwenWrapper(model_name="mistral")


═════════════════════════════════════════════════════════════════════════════
📊 预期输出示例
═════════════════════════════════════════════════════════════════════════════

演示1: 基础问答

  问题: Who is Erik Hort?
  
  [步骤1] 检索相关文档...
  ✅ 检索完成，得到 3 条结果
  
  [步骤2] Qwen 生成答案...
  🤖 答案: Erik Hort was a notable historical figure born in Montebello...
  
  [步骤3] 应用自动消退...
  ✅ 消退完成，删除了 2 条低激活记忆


演示3: 冲突检测

  🔍 检测并解决冲突...
  ✅ 检测到 2 个冲突
  ✅ 已用新值覆盖旧值


═════════════════════════════════════════════════════════════════════════════
🎯 快速命令参考
═════════════════════════════════════════════════════════════════════════════

# 在终端1：启动 Ollama
$ ollama serve

# 在终端2：检查前置条件
$ python ollama_quickstart.py --check

# 在终端2：运行你的演示
$ python your_demo.py

# 快速测试 Qwen
$ ollama run qwen3:1.7b "你好，请问什么是机器学习？"


═════════════════════════════════════════════════════════════════════════════
📁 相关文件说明
═════════════════════════════════════════════════════════════════════════════

test_with_local_ollama.py
  └─ 完整的测试套件和诊断工具
    ├─ OllamaQwenWrapper: Ollama + Qwen 的包装器
    └─ LocalOllamaRAGTest: 6个测试

enhanced_rag_demo.py
  └─ EnhancedRAGDemo 类
    ├─ 5个完整的演示方法
    └─ 可直接集成到你的代码

ollama_quickstart.py
  └─ 快速启动和前置条件检查

quick_reference_api.py
  └─ 三个 API 的快速参考和代码片段


═════════════════════════════════════════════════════════════════════════════
✅ 验证清单
═════════════════════════════════════════════════════════════════════════════

□ 已安装 Ollama
□ 已下载 Qwen3:1.7b 模型 (ollama pull qwen3:1.7b)
□ Ollama 服务运行中 (ollama serve)
□ Python 依赖已安装 (pip install requests numpy pandas)
□ HippoRAG 初始化代码已添加
□ 运行测试成功

完成上述步骤后，你就可以开始测试了！
    """
    
    print(guide)


if __name__ == "__main__":
    print_setup_guide()
