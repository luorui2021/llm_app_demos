"""使用openai sdk代替requests实现命令行连续对话。"""

import os
import httpx
from openai import OpenAI
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

# ===========================
# 配置部分
# ===========================
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"

# 代理设置（支持HTTP或SOCKS5）
PROXY_URL = "http://127.0.0.1:13128"

# ===========================
# OpenAI 客户端（含代理，忽略 SSL 验证）
# ===========================
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    http_client=httpx.Client(
        proxy=PROXY_URL,
        verify=False,
    ),
)

# ===========================
# 聊天记录
# ===========================
conversation_history = []
conversation_history.append({
    "role": "system",
    "content": "你是一个数学专家，但是说话非常毒舌，喜欢用尖刻的语言回答问题。请使用中文回答。"
    })


# ===========================
# 获取模型回复（流式输出）
# ===========================
def get_response():
    try:
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=conversation_history,
            max_tokens=1024,
            temperature=1.0,
            stream=True,
        )

        print("\033[91mAI:\033[0m ", end="", flush=True)
        full_reply = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                print(delta, end="", flush=True)
                full_reply.append(delta)
        print()  # 换行
        return "".join(full_reply)
    except Exception as e:
        print("请求出错:", e)
        return ""


# ===========================
# 命令行交互
# ===========================
def main():
    print("=== 欢迎使用连续对话 CLI ===")
    print("输入 'exit' 退出程序\n")

    while True:
        user_input = prompt(HTML("<ansired>你:</ansired> ")).strip()
        # user_input 判空检查
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        conversation_history.append({"role": "user", "content": user_input})
        reply = get_response()
        conversation_history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()