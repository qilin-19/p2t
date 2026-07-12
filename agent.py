"""
Agent 模块 —— 升级版！真正的 Agent 反馈闭环
=============================================

【教学】原来叫什么 vs 现在叫什么：
  原来：工作流: 观察 → 判断 → 决策 → 执行
  现在：工作流: 观察 → 判断 → 决策 → 执行 → 验证 → (不合格则带回意见重试)

  原来的流程是一条"直线"，数据流过去就不回头了。
  现在的流程是一个"环"，Agent 看到自己的输出后，会判断好
  不好、不好的话带着改进意见再来一次。最后还会告诉调用方
  "这份结果是否通过了质检"。

  Agent 的核心思想：不是"执行完就完了"，而是"执行完 →
  检查结果 → 发现不对 → 调整 → 再来"。这个"感知自己输
  出并据此调整行为"的循环，就是 Agent 区别于普通函数调
  用的本质。
"""
import time
from openai import OpenAI


# ============================================================
# 可配置常量
# ============================================================

# 【教学】原来：text[:3000] 直接写在代码里，是"魔法数字"
#        问题：别人看不懂为什么是 3000 而不是 2000 或 5000
#        改成：命名常量 + 注释解释原因
#        体现的思想：Agent 的每一步决策都应该有据可查

# 分类时截取文字的最大字符数。
# 为什么是 3000？
#   1. 前 3000 字通常已能判断内容类型（开头最暴露说话风格）
#   2. 截断节省 token 费用和响应时间
#   3. 中文字均 ~1.5 token，3000 字 ≈ 4500 token，兼容所有模型
MAX_CLASSIFY_CHARS = 3000

# Agent 最多重试总结的次数（不含首次执行）
# 设 2 的原因：1 次可能不够修正，3 次以上边际收益递减且费 token
MAX_SUMMARY_RETRIES = 2

# LLM 调用（网络问题）的最大重试次数
MAX_API_RETRIES = 2

# API 重试间隔（秒）
API_RETRY_DELAY = 2

# 合法内容类型集合
VALID_TYPES = {"讲课", "新闻", "闲聊", "会议", "通用"}


# ============================================================
# Agent 的"知识库"：每种内容类型对应的总结格式 + 质检标准
# ============================================================

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


# 【教学】新增：每个内容类型对应的"质检标准"。
#   有了这个，Agent 才知道"什么算好的总结"，
#   否则它无法判断自己的输出是否合格。

QUALITY_CHECKS = {
    "讲课": (
        "检查要点：\n"
        "- 是否包含明确的'核心知识点'部分？\n"
        "- 知识点是否准确反映了原文？\n"
        "- 是否有'一句话总结'？\n"
        "- 是否遗漏了原文中的重要概念？"
    ),
    "新闻": (
        "检查要点：\n"
        "- 是否覆盖了5W1H（What/Who/When/Where/Why/How）？\n"
        "- 关键事实是否准确？\n"
        "- 是否遗漏了重要的事件信息？"
    ),
    "闲聊": (
        "检查要点：\n"
        "- 是否提取了核心观点？\n"
        "- 是否过度记录了无关的闲聊细节？\n"
        "- 总结是否简洁？"
    ),
    "会议": (
        "检查要点：\n"
        "- 是否包含会议主题？\n"
        "- 是否列出了讨论要点？\n"
        "- 是否记录了决策和待办事项？\n"
        "- 是否遗漏了关键决策？"
    ),
    "通用": (
        "检查要点：\n"
        "- 是否简洁地保留了关键信息？\n"
        "- 是否有冗长或不相关的内容？\n"
        "- 要点是否清晰可读？"
    ),
}


# ============================================================
# 【教学】新增：LLM 调用统一入口，带网络重试
# ============================================================
# 原来：每个函数里直接调 client.chat.completions.create()
#      没有任何容错。网络一抖就崩溃。
# 现在：所有 LLM 调用都经过这个函数，统一做重试。
#
# Agent 思想：Agent 必须能处理"外部世界的不确定性"——
#      网络中断、API 限流、超时——这些都不可控。
#      Agent 不能假设每次调用都成功，必须有备用计划。

def _call_llm(client, model, messages, temperature, max_tokens, caller="LLM"):
    """
    所有 LLM 调用的统一入口，自动处理网络异常和重试。
    caller 参数只用于日志，让你看到是 Agent 的哪个环节在调用。
    """
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < MAX_API_RETRIES:
                print(f"   ⚠️ [{caller}] 网络异常，{API_RETRY_DELAY}秒后重试 "
                      f"({attempt + 1}/{MAX_API_RETRIES}): {type(e).__name__}")
                time.sleep(API_RETRY_DELAY)
            else:
                raise  # 所有重试都失败，向上抛出


# ============================================================
# Agent 步骤 1+2：观察 + 判断（升级：加分类容错重试）
# ============================================================
# 【教学】原来：LLM 返回什么就接受什么，不在合法集合就静默 fallback
#        问题1：静默 fallback 让你看不到"LLM 出错"这个事实
#        问题2：可能只是 LLM 格式输出异常（多带了标点），重问一次就好
# 现在：不合法 → 记录告警 → 重新询问 → 还不行才降级为"通用"
#
# Agent 思想：Agent 不盲目信任工具返回的结果，而是会
#      "验证工具输出是否可用"，不可用就重试或降级。

def classify_content(text, client, model):
    """
    观察 + 判断：读文字 → 调 LLM 判断内容类型。
    含容错：返回类型不合法时，重新询问一次。
    """
    print("\n" + "=" * 50)
    print("🤖 Agent 启动！")
    print("=" * 50)

    # 步骤 1：观察
    print(f"\n🔍 [观察] Agent 正在阅读文字内容...")
    print(f"   文字长度: {len(text)} 字")

    # 步骤 2：判断（调 LLM 分类，带容错重试）
    print("\n🤔 [判断] Agent 正在分析内容类型...")
    sample = text[:MAX_CLASSIFY_CHARS]

    content_type = None
    for attempt in range(2):  # 最多尝试 2 次
        content_type = _call_llm(
            client, model,
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
                    "content": (
                        f"请判断以下内容的类型（只回答一个词）：\n\n{sample}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=10,
            caller="分类器",
        )

        if content_type in VALID_TYPES:
            break

        if attempt == 0:
            print(f"   ⚠️ [容错] LLM 返回了非预期类型: '{content_type}'")
            print(f"   🔄 [容错] Agent 正在重新询问...")
    else:
        # 两次都失败 → 降级
        print(f"   ⚠️ [容错] 重试后仍无效，降级为: 【通用】")
        content_type = "通用"

    print(f"   ✅ Agent 判断结果: 【{content_type}】")
    return content_type


# ============================================================
# Agent 步骤 3：决策
# ============================================================

def get_strategy(content_type):
    """
    决策：根据内容类型，选择对应的总结策略提示词。
    """
    print(f"\n📋 [决策] Agent 正在选择总结策略...")
    strategy = STRATEGIES.get(content_type, STRATEGIES["通用"])
    strategy_name = content_type if content_type in STRATEGIES else "通用"
    print(f"   ✅ Agent 选择策略: 【{strategy_name}】")
    return strategy, strategy_name


# ============================================================
# 【教学】新增：自我验证模块 —— Agent 的"质检员"
# ============================================================
# 这是升级版的"灵魂"——Agent 不再只是一次性输出，而是
# 会回过头检查自己的输出。
#
# Agent 思想：感知 → 行动 → 观察结果 → 评估 → 再行动。
#      就像考试时做完题检查一遍——这是"元认知"的雏形。

def _verify_summary(original_text, summary, strategy_name, client, model):
    """
    验证：调 LLM 检查总结是否合格。
    返回: (是否合格: bool, 反馈意见: str)
    每次执行总结后都会调用，包括最后一次。
    """
    print(f"\n🔬 [验证] Agent 正在质检总结...")

    quality_check = QUALITY_CHECKS.get(strategy_name, QUALITY_CHECKS["通用"])

    verdict = _call_llm(
        client, model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个严格的质检员。请根据以下标准检查总结是否合格。\n\n"
                    f"{quality_check}\n\n"
                    "回复格式（严格遵守）：\n"
                    "- 如果合格，第一行写'合格'，后面不需要内容\n"
                    "- 如果不合格，第一行写'不合格'，"
                    "第二行起写具体问题，以及应该如何改进"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"【原文内容（片段）】\n{original_text[:2000]}\n\n"
                    f"【待检查的总结】\n{summary}\n\n"
                    f"请判断："
                ),
            },
        ],
        temperature=0.1,
        max_tokens=300,
        caller="质检员",
    )

    is_ok = verdict.startswith("合格")
    if is_ok:
        print(f"   ✅ 质检通过！总结符合【{strategy_name}】类的格式要求")
    else:
        print(f"   ❌ 质检不通过：")
        for line in verdict.split("\n")[1:6]:
            if line.strip():
                print(f"      {line.strip()}")

    return is_ok, verdict


# ============================================================
# 【教学】Agent 步骤 4+5 合体：执行 → 验证 → 重试
# ============================================================
# 原来：execute_summary 只生成总结，不检查质量
# 现在：生成 → 验证 → 不合格？→ 把质检意见注入提示词 → 重试
#      每次都会验证，包括最后一次。最后一次没通过也不丢弃，
#      而是带着"未通过"的标记返回。
#
# Agent 思想：这就是"反馈闭环"（Feedback Loop）。
#      Agent 观察自己的输出（验证），判断好坏（质检），
#      做出决策（重试 or 接受），然后带着新上下文再执行。
#      这与生物体的"刺激→反应→调节"是同一套逻辑。
#
# 返回值从单纯的字符串变成了字典，包含结果和质量标记，
# 这体现了 Agent 的"透明度"——不隐瞒自己的质量状况。

def execute_summary(text, strategy_prompt, strategy_name, client, model,
                    max_retries=MAX_SUMMARY_RETRIES):
    """
    执行 + 验证 + 重试 —— Agent 的反馈闭环。
    返回: {"summary": str, "passed_qa": bool, "retries_used": int}
    """
    print(f"\n✍️  [执行] Agent 正在用【{strategy_name}】策略总结...")

    current_prompt = strategy_prompt

    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"\n   🔄 [反馈环] 第 {attempt} 次重试，已将质检意见加入提示词...")

        # 4. 执行：生成总结
        summary = _call_llm(
            client, model,
            messages=[
                {"role": "system", "content": current_prompt},
                {"role": "user", "content": f"请总结以下内容：\n\n{text}"},
            ],
            temperature=0.3,
            max_tokens=2000,
            caller="总结器",
        )
        label = "初稿" if attempt == 0 else f"修订稿 v{attempt + 1}"
        print(f"   ✅ Agent 总结完成（{label}）！")

        # 5. 验证：每次都会做，包括最后一次
        is_ok, feedback = _verify_summary(
            text, summary, strategy_name, client, model
        )

        if is_ok:
            print(f"\n   🎉 Agent 在第 {attempt + 1} 次尝试后通过质检")
            print("=" * 50)
            return {
                "summary": summary,
                "passed_qa": True,
                "retries_used": attempt,
            }

        # 不合格，但还有重试机会
        if attempt < max_retries:
            # 把质检意见注入提示词，让下次尝试知道上次哪里不好
            current_prompt = (
                f"{strategy_prompt}\n\n"
                f"【⚠️ 上一次总结被质检驳回，问题如下，本次必须改正】\n{feedback}"
            )
        else:
            # 最后一次也没通过——不丢弃结果，但诚实标记
            print(f"\n   ⚠️ 已达最大重试次数 ({max_retries})，"
                  f"最终结果未通过质检，请谨慎使用")
            print("=" * 50)
            return {
                "summary": summary,
                "passed_qa": False,
                "retries_used": attempt,
            }


# ============================================================
# 【教学】新增：Agent 统一入口
# ============================================================
# 原来：summarize.py 需要分别调用 classify_content →
#      get_strategy → execute_summary，暴露了内部步骤
# 现在：一个 run_agent() 就是整个 Agent，返回结构化结果
#
# Agent 思想：Agent 对外是一个"黑盒"——你给它目标（文字）
#      和工具（client, model），它内部自己完成观察、判断、
#      决策、执行、验证、重试，最后给你结果 + 质量报告。

def run_agent(text, client, model):
    """
    Agent 统一入口。
    输入：文字 + 大模型客户端
    输出：{
        "summary": str,         # 总结文字
        "passed_qa": bool,      # 是否通过质检
        "retries_used": int,    # 用了多少次重试
        "content_type": str,    # 识别出的内容类型
        "strategy_name": str,   # 使用的总结策略名
    }
    """
    # 步骤 1+2：观察 + 判断（含分类容错重试）
    content_type = classify_content(text, client, model)

    # 步骤 3：决策
    strategy_prompt, strategy_name = get_strategy(content_type)

    # 步骤 4+5：执行 + 验证 + 重试（反馈闭环）
    result = execute_summary(
        text, strategy_prompt, strategy_name, client, model
    )

    # 把 Agent 的"思考过程"信息也带回去
    result["content_type"] = content_type
    result["strategy_name"] = strategy_name
    return result
