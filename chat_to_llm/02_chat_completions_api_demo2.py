"""
演示使用Chat Completions API实现命令行连续对话。

相比于上一个版本，增加了流式输出和HTTP会话复用，提升了用户体验和性能。
"""

import os
import requests
import json
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

# ===========================
# 配置部分
# ===========================
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"

# 代理设置（支持HTTP或SOCKS5）
PROXIES = {
    "http": "http://127.0.0.1:13128",
    "https": "http://127.0.0.1:13128",
}
# 忽略 SSL 验证
VERIFY_SSL = False
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===========================
# HTTP 会话（复用连接）
# ===========================
session = requests.Session()
session.proxies = PROXIES
session.verify = VERIFY_SSL

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
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": conversation_history,
        "max_tokens": 1024,
        "temperature": 1.0,
        "stream": True,
    }

    try:
        with session.post(
            API_URL,
            headers=headers,
            data=json.dumps(data),
            stream=True,
        ) as response:
            response.raise_for_status()

            print("\033[91mAI:\033[0m ", end="", flush=True)
            full_reply = []
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[len("data: "):]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_reply.append(delta)
                except json.JSONDecodeError:
                    continue
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
    try:
        main()
    finally:
        session.close()