"""
AI 总结模块
把长文字喂给大模型，让它吐出精炼的要点总结。
支持两种模式：
  1. summarize_text()        —— 直接总结（原来的方式）
  2. summarize_with_agent()  —— 先让 Agent 判断类型，再按类型总结（★新增）
"""
import os
from openai import OpenAI
from agent import classify_content, get_strategy, execute_summary


def _get_client():
    """创建 API 客户端（内部公用）"""
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "❌ 缺少 API_KEY 环境变量！\n"
            "请在终端设置: set API_KEY=你的key  然后再运行\n"
            "获取方式：去 deepseek.com 或 openai.com 注册 → API Keys 页面 → 创建"
        )
    base_url = os.getenv("API_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("API_MODEL", "deepseek-chat")
    return OpenAI(api_key=api_key, base_url=base_url), model


def summarize_text(text):
    """
    直接总结（原来的方式，保持不变）
    """
    client, model = _get_client()

    if len(text.strip()) < 50:
        return "（原文太短，无需总结）"

    print("⏳ 正在用 AI 总结文字...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一个文字总结助手。请用简体中文，把用户给的内容总结成简洁的要点。保留关键信息，去掉废话。"
            },
            {
                "role": "user",
                "content": f"请总结以下文字内容：\n\n{text}"
            }
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    print("✅ AI 总结完成")
    return response.choices[0].message.content


def summarize_with_agent(text):
    """
    ★ Agent 版本：先判断类型，再按类型总结
    Agent 工作流: 观察 → 判断 → 决策 → 执行
    """
    client, model = _get_client()

    if len(text.strip()) < 50:
        return "（原文太短，无需总结）"

    # 步骤 1+2：Agent 观察 + 判断内容类型
    content_type = classify_content(text, client, model)

    # 步骤 3：Agent 根据类型选策略
    strategy_prompt, strategy_name = get_strategy(content_type)

    # 步骤 4：Agent 用选中策略执行总结
    summary = execute_summary(text, strategy_prompt, strategy_name, client, model)

    return summary
