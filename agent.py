"""
Agent 模块 —— 你的第一个 AI Agent！
工作流: 观察 → 判断 → 决策 → 执行
每步都打印到终端，让你看清楚 agent 的"思考链"。
"""
from openai import OpenAI


# 不同类型对应的总结策略（系统提示词）
STRATEGIES = {
    "讲课": (
        "你是一个课程笔记助手。请用简体中文，把内容总结成结构化的知识点。"
        "格式要求：\n"
        "1. 先列出核心知识点（用 • 分点）\n"
        "2. 再列出重要细节\n"
        "3. 最后给出一个'一句话总结'"
    ),
    "新闻": (
        "你是一个新闻摘要助手。请用简体中文，按 5W1H 格式总结：\n"
        "- 发生了什么（What）\n"
        "- 涉及谁（Who）\n"
        "- 什么时候（When）\n"
        "- 在哪里（Where）\n"
        "- 为什么（Why）\n"
        "- 怎么发生的（How）"
    ),
    "闲聊": (
        "你是一个对话分析助手。请用简体中文，总结这段聊天的核心观点和有趣的观点。"
        "不需要逐字记录，提取关键想法即可。"
    ),
    "会议": (
        "你是一个会议纪要助手。请用简体中文，按以下格式总结：\n"
        "1. 会议主题\n"
        "2. 讨论要点\n"
        "3. 做出的决定\n"
        "4. 待办事项（如果有）"
    ),
    "通用": (
        "你是一个文字总结助手。请用简体中文，把内容总结成简洁的要点。"
        "保留关键信息，去掉废话。"
    ),
}


def classify_content(text, client, model):
    """
    Agent 步骤 1+2：观察 + 判断
    输入：一大段文字
    输出：内容类型（讲课 / 新闻 / 闲聊 / 会议 / 通用）
    """
    print("\n" + "=" * 50)
    print("🤖 Agent 启动！")
    print("=" * 50)

    # 步骤 1：观察
    print(f"\n🔍 [观察] Agent 正在阅读文字内容...")
    print(f"   文字长度: {len(text)} 字")

    # 步骤 2：判断（调 LLM 分类）
    print("\n🤔 [判断] Agent 正在分析内容类型...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个内容分类专家。阅读用户给的文字，判断它属于哪种类型。"
                    "只能回答以下五个词之一：讲课、新闻、闲聊、会议、通用。"
                    "不要输出任何其他内容，就一个词。\n\n"
                    "判断标准：\n"
                    "- 讲课：像老师在教东西，有知识讲解、概念解释\n"
                    "- 新闻：像新闻报道，有时效性信息、事件描述\n"
                    "- 闲聊：像日常聊天，话题松散，语气随意\n"
                    "- 会议：像开会讨论，有议题、有决策、有分工\n"
                    "- 通用：以上都不符合，或者无法确定"
                ),
            },
            {
                "role": "user",
                "content": f"请判断以下内容的类型（只回答一个词）：\n\n{text[:3000]}"
            },
        ],
        temperature=0.1,  # 极低温度，确保稳定输出
        max_tokens=10,
    )
    content_type = response.choices[0].message.content.strip()
    print(f"   ✅ Agent 判断结果: 【{content_type}】")

    return content_type


def get_strategy(content_type):
    """
    Agent 步骤 3：决策
    输入：内容类型
    输出：对应的总结策略（系统提示词）
    """
    print(f"\n📋 [决策] Agent 正在选择总结策略...")

    # 如果 LLM 返回的类型不在预定义列表里，用"通用"
    strategy = STRATEGIES.get(content_type, STRATEGIES["通用"])
    strategy_name = content_type if content_type in STRATEGIES else "通用"

    print(f"   ✅ Agent 选择策略: 【{strategy_name}】")
    return strategy, strategy_name


def execute_summary(text, strategy_prompt, strategy_name, client, model):
    """
    Agent 步骤 4：执行
    输入：文字 + 策略提示词
    输出：按策略总结好的文字
    """
    print(f"\n✍️  [执行] Agent 正在用【{strategy_name}】策略总结...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": strategy_prompt},
            {"role": "user", "content": f"请总结以下内容：\n\n{text}"},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    summary = response.choices[0].message.content
    print(f"   ✅ Agent 总结完成！")
    print("=" * 50)
    return summary
