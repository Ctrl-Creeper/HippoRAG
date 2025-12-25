#!/usr/bin/env python3
"""
环境检查脚本 - 验证是否可以运行 Ollama Qwen 测试

运行: python check_environment.py
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """检查 Python 版本"""
    logger.info("\n" + "="*80)
    logger.info("1️⃣  Python 版本检查")
    logger.info("="*80)
    
    version = sys.version_info
    print(f"当前 Python 版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        logger.info("✅ Python 版本符合要求 (需要 3.8+)")
        return True
    else:
        logger.error("❌ Python 版本过低，请升级到 3.8 或更高")
        return False


def check_required_packages():
    """检查必需的 Python 包"""
    logger.info("\n" + "="*80)
    logger.info("2️⃣  Python 包检查")
    logger.info("="*80)
    
    required_packages = {
        'requests': '网络请求库（连接 Ollama 需要）',
        'numpy': '数值计算库',
        'pandas': '数据处理库',
        'tqdm': '进度条库'
    }
    
    all_present = True
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✅ {package:15} - {description}")
        except ImportError:
            logger.error(f"❌ {package:15} - {description}")
            all_present = False
    
    if not all_present:
        logger.error("\n💡 安装缺失的包:")
        missing = [p for p in required_packages if __import__(p) is None]
        logger.error(f"   pip install {' '.join(missing)}")
    
    return all_present


def check_ollama_service():
    """检查 Ollama 服务"""
    logger.info("\n" + "="*80)
    logger.info("3️⃣  Ollama 服务检查")
    logger.info("="*80)
    
    try:
        import requests
    except ImportError:
        logger.error("⚠️  requests 库未安装，跳过 Ollama 检查")
        return False
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        
        if response.status_code == 200:
            logger.info("✅ Ollama 服务运行中 (http://localhost:11434)")
            return True
        else:
            logger.error(f"❌ Ollama 返回错误状态码: {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        logger.error("❌ 无法连接到 Ollama 服务 (http://localhost:11434)")
        logger.error("\n💡 解决方案:")
        logger.error("   1. 打开新终端")
        logger.error("   2. 运行: ollama serve")
        logger.error("   3. 保持该终端打开（在后台运行）")
        return False
    
    except Exception as e:
        logger.error(f"❌ 检查失败: {e}")
        return False


def check_qwen_model():
    """检查 Qwen 模型"""
    logger.info("\n" + "="*80)
    logger.info("4️⃣  Qwen 模型检查")
    logger.info("="*80)
    
    try:
        import requests
    except ImportError:
        logger.error("⚠️  requests 库未安装，跳过模型检查")
        return False
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        
        if response.status_code != 200:
            logger.error("❌ Ollama 服务返回错误")
            return False
        
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        logger.info("已安装的模型:")
        for name in model_names:
            logger.info(f"  - {name}")
        
        # 检查 Qwen
        qwen_found = any("qwen" in name.lower() for name in model_names)
        
        if qwen_found:
            logger.info("\n✅ Qwen 模型已安装")
            return True
        else:
            logger.error("\n❌ 未找到 Qwen 模型")
            logger.error("\n💡 解决方案:")
            logger.error("   运行: ollama pull qwen3:1.7b")
            logger.error("   等待下载完成（可能需要几分钟）")
            return False
    
    except Exception as e:
        logger.error(f"❌ 检查失败: {e}")
        return False


def check_hipporag():
    """检查 HippoRAG 库"""
    logger.info("\n" + "="*80)
    logger.info("5️⃣  HippoRAG 库检查")
    logger.info("="*80)
    
    try:
        from src.hipporag import HippoRAG
        logger.info("✅ HippoRAG 库可以导入")
        return True
    except ImportError as e:
        logger.error(f"❌ 无法导入 HippoRAG: {e}")
        logger.error("\n💡 解决方案:")
        logger.error("   1. 确保你在 HippoRAG 项目目录中")
        logger.error("   2. 安装必需的依赖: pip install -r requirements.txt")
        return False


def check_test_files():
    """检查测试文件"""
    logger.info("\n" + "="*80)
    logger.info("6️⃣  测试文件检查")
    logger.info("="*80)
    
    import os
    
    test_files = [
        'test_with_local_ollama.py',
        'enhanced_rag_demo.py',
        'ollama_quickstart.py',
        'quick_reference_api.py',
        'OLLAMA_QUICKSTART_ZH.md'
    ]
    
    all_present = True
    
    for filename in test_files:
        if os.path.exists(filename):
            logger.info(f"✅ {filename}")
        else:
            logger.error(f"❌ {filename} 不存在")
            all_present = False
    
    if not all_present:
        logger.error("\n⚠️  某些测试文件缺失")
    
    return all_present


def main():
    """运行所有检查"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║               HippoRAG Ollama Qwen 测试环境检查                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    results = {
        "Python 版本": check_python_version(),
        "Python 包": check_required_packages(),
        "Ollama 服务": check_ollama_service(),
        "Qwen 模型": check_qwen_model(),
        "HippoRAG 库": check_hipporag(),
        "测试文件": check_test_files()
    }
    
    # 打印总结
    logger.info("\n" + "="*80)
    logger.info("检查总结")
    logger.info("="*80)
    
    for check_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{check_name:20} {status}")
    
    # 整体判断
    all_passed = all(results.values())
    
    logger.info("="*80)
    
    if all_passed:
        logger.info("\n🎉 环境检查完全通过！")
        logger.info("\n✅ 你可以开始运行测试了：")
        logger.info("   python test_with_local_ollama.py")
        logger.info("   或")
        logger.info("   python your_demo.py")
        return 0
    else:
        logger.error("\n❌ 环境检查失败")
        logger.error("\n📋 需要修复的项目:")
        for check_name, result in results.items():
            if not result:
                logger.error(f"   - {check_name}")
        
        logger.error("\n💡 常见解决方案:")
        logger.error("   1. 安装 Python 包: pip install requests numpy pandas tqdm")
        logger.error("   2. 启动 Ollama: ollama serve (在另一个终端)")
        logger.error("   3. 下载模型: ollama pull qwen3:1.7b")
        logger.error("   4. 安装 HippoRAG 依赖: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
