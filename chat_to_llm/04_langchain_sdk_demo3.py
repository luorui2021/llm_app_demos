"""
使用 langchain.agents.create_agent 构建带工具调用的chat agent。
相比于上一版本，这个版本的 agent 可以不仅可以自动判断何时需要调用工具，还能自动选择调用哪个
工具，并且可能会自动调用多次工具，直至达到目的或者用尽工具调用次数限制为止。
"""

import os
from datetime import datetime
import httpx
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from pydantic import SecretStr

# ===========================
# 配置部分
# ===========================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "api_key_not_set")
BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

# 代理设置(通过自定义httpx客户端实现，支持HTTP或SOCKS5)，并且忽略 SSL 验证
_http_client = httpx.Client(
    proxy="http://127.0.0.1:13128",
    verify=False,
)

# ===========================
# LangChain ChatOpenAI 客户端（含代理，忽略 SSL 验证）
# ===========================
llm = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model=MODEL,
    max_completion_tokens=1024,
    temperature=1.0,
    http_client=_http_client
)

# ===========================
# 工具定义
# ===========================
@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    print(f"\033[33m[工具执行: get_current_time()]\033[0m")
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气情况

    Args:
        city: 城市名称，例如：北京、上海
    """
    # Demo：固定返回天气数据
    print(f"\033[33m[工具执行: get_weather({city})]\033[0m")
    return f"{city}：晴，气温25°C，湿度60%，微风"


tools = [get_current_time, get_weather]
SYSTEM_PROMPT = "你是一个毒舌智能助手，喜欢嘲讽人，但也很有用。请使用中文回答。"
agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

# ===========================
# 聊天记录
# ===========================
conversation_history = []


# ===========================
# 获取模型回复（agent 自动处理工具调用循环，并流式输出）
# 参考文档：https://docs.langchain.com/oss/python/langchain/event-streaming
# ===========================
def get_response(user_input: str):
    global conversation_history

    conversation_history.append(HumanMessage(content=user_input))

    try:
        print("\033[91mAI:\033[0m ", end="", flush=True)
        input_messages = list(conversation_history)
        stream = agent.stream_events(
            {"messages": input_messages},
            version="v3",
        )

        for message in stream.messages:
            # `message.text` 是单条模型消息的流式文本视图。
            # 遍历它时拿到的是当前这次模型调用产生的文本增量；
            # 如果这一轮只是先生成工具调用而没有自然语言文本，这里可能为空。
            for delta in message.text:
                print(delta, end="", flush=True)

            print(end=f"\033[91m[消息分隔符]\033[0m", flush=True)

            # `message.output` 是单条模型消息完成后的完整对象。
            # 它通常是一个 AIMessage，除了最终文本外，还可能带有 tool_calls、
            # usage_metadata、content_blocks 等结构化字段，适合写入会话历史。
            completed_message = message.output
            if isinstance(completed_message, AIMessage):
                conversation_history.append(completed_message)

        # `stream.output` 是整轮 agent 执行完成后的完整状态。
        # 对 create_agent 来说，它通常是一个包含 `messages` 的状态字典，
        # 里面会汇总用户消息、带 tool_calls 的 AIMessage、ToolMessage 和最终 AIMessage。
        final_state = stream.output
        conversation_history = list(final_state["messages"])
        print()
        return conversation_history
    except Exception as e:
        print("请求出错:", e)
        return None


# ===========================
# 命令行交互
# ===========================
def main():
    global conversation_history

    print("=== 欢迎使用连续对话 CLI ===")
    print("输入 'exit' 退出程序\n")

    while True:
        user_input = prompt(HTML("<ansired>你:</ansired> ")).strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        get_response(user_input)
        print()


if __name__ == "__main__":
    main()
