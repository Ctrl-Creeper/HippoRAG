"""
三个核心API的快速参考 - 复制粘贴使用

在你的main.py或demo.py中直接使用这些代码片段
"""

# ============================================================================
# API #1: 自动消退 (自动删除低激活记忆)
# ============================================================================

def api_1_auto_decay():
    """
    使用场景：
    - 定期清理知识库中的不相关记忆
    - 系统性能下降时自动删除
    - 运行在后台自动维护
    """
    
    # 基础用法
    decay_stats = rag.apply_context_aware_memory_decay(
        current_query="Who is Erik Hort?",
        retention_ratio=0.9,      # 保留激活度top-90%的记忆
        auto_forget=True          # 自动执行删除
    )
    
    # 返回值详解
    print(f"总的chunks数: {decay_stats['total_chunks']}")
    print(f"要删除的chunks: {decay_stats['chunks_to_forget']}")
    print(f"实际删除的chunks: {decay_stats.get('auto_forgot_chunks', 0)}")
    
    # 更激进的清理（保留60%）
    decay_stats = rag.apply_context_aware_memory_decay(
        current_query="Query",
        retention_ratio=0.6,
        auto_forget=True
    )
    
    # 保守的清理（保留95%）
    decay_stats = rag.apply_context_aware_memory_decay(
        current_query="Query",
        retention_ratio=0.95,
        auto_forget=True
    )


# ============================================================================
# API #2: 手动清除 (查看后再删除，可选的演练模式)
# ============================================================================

def api_2_manual_cleanup():
    """
    使用场景：
    - 需要人工审核后再删除
    - 管理员监督的清理
    - 演练模式验证删除列表
    """
    
    # 第一步：演练模式，查看将要删除的项目（不实际删除）
    preview = rag.manual_cleanup_low_activation_memories(
        current_query="Who is Erik Hort?",
        activation_threshold=0.1,  # 删除激活分数<0.1的记忆
        dry_run=True               # 重要：dry_run=True时不执行删除！
    )
    
    print(f"要删除的chunks: {len(preview['chunks_to_delete'])}")
    print(f"要删除的entities: {len(preview['entities_to_delete'])}")
    print(f"要删除的facts: {len(preview['facts_to_delete'])}")
    
    # 用户审查后确认...
    user_confirmed = True  # 用户按确认按钮
    
    # 第二步：执行删除（dry_run=False）
    if user_confirmed:
        result = rag.manual_cleanup_low_activation_memories(
            current_query="Who is Erik Hort?",
            activation_threshold=0.1,
            dry_run=False  # 现在执行实际删除！
        )
        
        print(f"实际删除的文档数: {result.get('actually_deleted_count', 0)}")
    
    # 不同激活阈值的例子
    
    # 非常激进：删除<0.05的（几乎不活跃）
    preview = rag.manual_cleanup_low_activation_memories(
        current_query="Query",
        activation_threshold=0.05,
        dry_run=True
    )
    
    # 保守：删除<0.2的（低激活）
    preview = rag.manual_cleanup_low_activation_memories(
        current_query="Query",
        activation_threshold=0.2,
        dry_run=True
    )


# ============================================================================
# API #3: 冲突检测与解决 (处理新旧知识的矛盾)
# ============================================================================

def api_3_conflict_resolution():
    """
    使用场景：
    - 导入新的数据源
    - 更新过时的知识
    - 修正之前的错误信息
    
    四种解决策略：
    1. 'keep_new' - 新值覆盖旧值（默认，最常用）
    2. 'keep_old' - 保留旧值，拒绝新值
    3. 'merge' - 合并为"可能是X或Y"
    4. 'keep_frequent' - 保留访问频率更高的值
    """
    
    # 新事实（与现有知识库冲突）
    new_facts = [
        ("Erik Hort", "birthplace", "Rockland County"),  # 与旧值冲突
        ("Montebello", "location", "Rockland County"),   # 与旧值冲突
        ("Alice", "role", "CEO")  # 新事实，无冲突
    ]
    
    # 策略1：新值覆盖旧值（推荐）
    result = rag.detect_and_resolve_fact_conflicts(
        new_facts=new_facts,
        resolution_strategy='keep_new',
        auto_apply=True
    )
    
    print(f"检测到的冲突数: {result['conflicts_detected']}")
    print(f"要删除的旧事实: {len(result['facts_to_delete'])}")
    print(f"冲突记录: {result['conflict_records']}")
    
    # 策略2：保留旧值
    result = rag.detect_and_resolve_fact_conflicts(
        new_facts=new_facts,
        resolution_strategy='keep_old',
        auto_apply=True
    )
    
    # 策略3：合并不确定
    result = rag.detect_and_resolve_fact_conflicts(
        new_facts=new_facts,
        resolution_strategy='merge',
        auto_apply=True
    )
    
    # 策略4：基于访问频率选择
    result = rag.detect_and_resolve_fact_conflicts(
        new_facts=new_facts,
        resolution_strategy='keep_frequent',
        auto_apply=True
    )
    
    # 仅检测，不自动应用（进行手动审查）
    result = rag.detect_and_resolve_fact_conflicts(
        new_facts=new_facts,
        resolution_strategy='keep_new',
        auto_apply=False  # 只检测，不删除
    )


# ============================================================================
# 额外：查看记忆激活状态
# ============================================================================

def check_activation_status():
    """
    了解当前查询上下文中哪些记忆是活跃的
    """
    
    activation = rag.get_memory_activation_status("Who is Erik Hort?")
    
    print(f"当前查询: {activation['current_query']}")
    print(f"查询窗口大小: {activation['context_window_size']}")
    
    # Chunk激活度统计
    chunk_stats = activation['chunk_activation']
    print(f"\nChunk激活度：")
    print(f"  高激活 (>0.7): {chunk_stats['high_activation_count']}")
    print(f"  中激活 (0.3-0.7): {chunk_stats['medium_activation_count']}")
    print(f"  低激活 (0.05-0.3): {chunk_stats['low_activation_count']}")
    print(f"  非活跃 (<=0.05): {chunk_stats['inactive_count']}")
    print(f"  平均激活度: {chunk_stats['avg_activation']:.3f}")
    
    # 同样适用于entity和fact
    # activation['entity_activation']
    # activation['fact_activation']
    
    # 查看激活度最高的前5条
    top_5 = activation['top_5_chunks']
    print(f"\n激活度最高的前5条chunks:")
    for hash_id, score in top_5:
        print(f"  {hash_id}: {score:.3f}")


# ============================================================================
# 集成示例：完整工作流
# ============================================================================

def complete_workflow():
    """
    展示三个API的完整工作流
    """
    
    queries = [
        "Who is Erik Hort?",
        "Where was Erik born?",
        "What is Montebello?",
        "Is Montebello in Rockland County?"
    ]
    
    # 阶段1：处理多个查询
    print("\n=== 阶段1：处理多个查询 ===")
    for query in queries:
        results = rag.retrieve(query)  # 自动记录访问历史
        print(f"查询: {query}")
        print(f"  结果数: {len(results)}")
    
    # 阶段2：查看最后一个查询的激活状态
    print("\n=== 阶段2：检查激活状态 ===")
    last_query = queries[-1]
    activation = rag.get_memory_activation_status(last_query)
    print(f"总chunk数: {activation['chunk_activation']['total_count']}")
    
    # 阶段3：自动消退
    print("\n=== 阶段3：自动消退 ===")
    decay_result = rag.apply_context_aware_memory_decay(
        current_query=last_query,
        retention_ratio=0.85,
        auto_forget=True
    )
    print(f"删除的chunks: {len(decay_result['chunks_to_forget'])}")
    
    # 阶段4：冲突检测
    print("\n=== 阶段4：冲突检测与解决 ===")
    new_facts = [
        ("Erik Hort", "birthplace", "Updated Location"),
        ("New Entity", "property", "value")
    ]
    conflict_result = rag.detect_and_resolve_fact_conflicts(
        new_facts=new_facts,
        resolution_strategy='keep_new',
        auto_apply=True
    )
    print(f"冲突数: {conflict_result['conflicts_detected']}")
    
    # 阶段5：手动清除
    print("\n=== 阶段5：手动清除 ===")
    preview = rag.manual_cleanup_low_activation_memories(
        current_query=last_query,
        activation_threshold=0.15,
        dry_run=True
    )
    print(f"预览要删除的chunks: {len(preview['chunks_to_delete'])}")


# ============================================================================
# 参数调优建议
# ============================================================================

"""
参数调优指南：

1. retention_ratio (保留比例) - 用于apply_context_aware_memory_decay
   ────────────────────────────────────────────────────
   - 0.5 = 非常激进，只保留50%（大幅清理）
   - 0.7 = 激进，保留70%
   - 0.85 = 平衡，保留85%（推荐）
   - 0.95 = 保守，保留95%
   
   选择建议：
   - 知识库快速增长时：用0.7-0.8
   - 系统性能正常时：用0.85-0.9
   - 知识很重要，不愿删除时：用0.95+


2. activation_threshold (激活阈值) - 用于manual_cleanup_low_activation_memories
   ──────────────────────────────────────────────────────────────
   - 0.05 = 只删除几乎不使用的（<5%）
   - 0.10 = 删除不活跃的（<10%）（推荐）
   - 0.15 = 删除低激活的（<15%）
   - 0.20 = 删除激活度较低的（<20%）
   
   选择建议：
   - 第一次清理，保守：用0.05
   - 常规清理：用0.10
   - 激进清理：用0.15-0.20


3. resolution_strategy (冲突策略) - 用于detect_and_resolve_fact_conflicts
   ───────────────────────────────────────────────────────────
   - 'keep_new' = 新值覆盖旧值（推荐，95%情况下使用）
   - 'keep_old' = 保留旧值，拒绝新值（当旧数据更可信时）
   - 'merge' = 合并为不确定（不确定哪个对时）
   - 'keep_frequent' = 基于访问频率（让数据说话）
"""


if __name__ == "__main__":
    print("""
    快速参考卡片 - 三个核心API
    
    在这个文件中找到你需要的代码片段，复制到你的程序中即可。
    
    API #1: api_1_auto_decay()
            - 自动删除低激活记忆
    
    API #2: api_2_manual_cleanup()
            - 查看后手动删除
    
    API #3: api_3_conflict_resolution()
            - 处理新旧知识冲突
    
    辅助：
    - check_activation_status() - 查看激活状态
    - complete_workflow() - 完整工作流示例
    """)
