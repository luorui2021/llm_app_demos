"""
Demo Agent — 使用 raw requests（DeepSeek + MCP Server）实现带工具调用的命令行连续对话。

相比 02_chat_completions_api_demo5.py，本 demo 的工具不在本地定义，
而是通过 MCP Streamable HTTP 协议从远程 Server 动态发现并调用。
ToolManager 类对外屏蔽工具发现和缓存逻辑，只暴露 get_tools() 和 call() 接口。

运行前请先启动 MCP Server：
    uv run python enforce_agent_with_mcp/mcp_server.py

再运行本 Agent：
    uv run python enforce_agent_with_mcp/agent.py
"""

import json
import os
import urllib3

import requests
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===========================
# 配置部分
# ===========================
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"  # 该模型默认为思考模式，响应包含 reasoning_content 思维链

PROXIES = {
    "http": "http://127.0.0.1:13128",
    "https": "http://127.0.0.1:13128",
}
VERIFY_SSL = False

MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"

# ===========================
# HTTP 会话（复用连接，对接 DeepSeek）
# ===========================
http_session = requests.Session()
http_session.proxies = PROXIES
http_session.verify = VERIFY_SSL

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ===========================
# 聊天记录
# ===========================
conversation_history = [
    {
        "role": "system",
        "content": (
            "你是一个数学专家，但是说话非常毒舌，喜欢用尖刻的语言回答问题。"
            "请使用中文回答。"
            "当问题涉及计算、时间或天气时，优先调用可用工具。"
        ),
    }
]


# ===========================
# ToolManager：封装 MCP Streamable HTTP 会话，对外只暴露工具列表获取和调用
# ===========================
class ToolManager:
    """通过 MCP Streamable HTTP 协议与远程 Server 通信。
    对外只暴露两个接口：
      - get_tools()  → 获取 OpenAI 格式的工具列表
      - call()       → 调用指定工具并返回文本结果
    工具发现、Session 初始化、结果缓存等细节完全封装于内部。
    """

    def __init__(self, mcp_url: str) -> None:
        self._url = mcp_url
        self._mcp_session = requests.Session()  # MCP 专用 HTTP 会话（不走代理）
        self._session_id: str | None = None
        self._req_id = 0
        self._cache: list[dict] | None = None
        self._init_session()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_tools(self) -> list[dict]:
        """返回 OpenAI function-calling 格式的工具列表（结果自动缓存）。"""
        if self._cache is None:
            self._cache = self._fetch_tools()
        return self._cache

    def call(self, name: str, arguments_json: str) -> str:
        """调用 MCP Server 上的指定工具，返回文本结果。"""
        try:
            args = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            return f"参数解析失败: {exc}"

        try:
            result = self._rpc("tools/call", {"name": name, "arguments": args})
        except Exception as exc:
            return f"工具调用失败: {exc}"

        content = result.get("content", [])
        texts = [block["text"] for block in content if block.get("type") == "text"]
        return "\n".join(texts) if texts else "(无返回内容)"

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _post(self, payload: dict) -> dict | None:
        """发送 JSON-RPC 消息，返回解析后的响应体（通知类消息返回 None）。"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        resp = self._mcp_session.post(self._url, json=payload, headers=headers)
        resp.raise_for_status()

        # 更新 Session ID（首次初始化时服务端会下发）
        if "Mcp-Session-Id" in resp.headers:
            self._session_id = resp.headers["Mcp-Session-Id"]

        if not resp.content:
            return None  # 通知类消息无响应体（HTTP 202）

        ct = resp.headers.get("Content-Type", "")
        if "text/event-stream" in ct:
            # 强制 UTF-8 解码（text/event-stream 不带 charset 时 requests 默认 Latin-1，
            # 会将中文 UTF-8 多字节序列中的 0x85 解为 U+0085 NEL，导致 splitlines() 截断）
            text = resp.content.decode("utf-8")
            # 解析 SSE：提取第一条非空 data 行（按 \r\n 分割避免 splitlines() 副作用）
            for line in text.split("\r\n"):
                if line.startswith("data: ") and line[6:].strip():
                    return json.loads(line[6:])
            return None
        return resp.json()

    def _rpc(self, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求并返回 result 字段内容。"""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }
        response = self._post(payload)
        if response is None:
            raise RuntimeError(f"RPC {method} 返回空响应")
        if "error" in response:
            raise RuntimeError(f"RPC 错误: {response['error']}")
        return response.get("result", {})

    def _init_session(self) -> None:
        """完成 MCP 握手：initialize → notifications/initialized。"""
        self._rpc("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "sync-agent", "version": "1.0"},
        })
        # 发送 initialized 通知（无需等待响应）
        self._post({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })

    def _fetch_tools(self) -> list[dict]:
        """从 MCP Server 拉取工具列表并转换为 OpenAI tool schema 格式。"""
        result = self._rpc("tools/list", {})
        tools = []
        for t in result.get("tools", []):
            parameters = t.get("inputSchema") or {"type": "object", "properties": {}}
            tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": parameters,
                },
            })
        return tools


# ===========================
# 思维链打印
# ===========================
def print_reasoning(reasoning: str) -> None:
    """将思维链内容以灰色缩进格式打印。"""
    print("\033[90m[思维链]\033[0m")
    for line in reasoning.splitlines():
        print(f"\033[90m  {line}\033[0m")
    print()


# ===========================
# 工具调用循环（非流式，打印思维链）
# ===========================
def get_response_with_tools(tool_manager: ToolManager, user_input: str) -> str:
    conversation_history.append({"role": "user", "content": user_input})
    openai_tools = tool_manager.get_tools()

    try:
        while True:
            data = {
                "model": MODEL,
                "messages": conversation_history,
                "tools": openai_tools,
                "tool_choice": "auto",
                "max_tokens": 8192,
                "temperature": 1.0,
            }
            resp = http_session.post(API_URL, headers=HEADERS, data=json.dumps(data))
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
            tool_calls = message.get("tool_calls")

            # 打印本轮思维链（若存在）
            # 注意：reasoning_content字段并非 OpenAI API 标准输出，但已有多家模型提供商在响应中加入了该字段以输出思维链内容，使用时需根据实际情况调整
            reasoning = message.get("reasoning_content", "")
            if reasoning:
                print_reasoning(reasoning)

            if not tool_calls:
                reply = message.get("content", "")
                conversation_history.append({"role": "assistant", "content": reply})
                print(f"\033[91mAI:\033[0m {reply}")
                return reply

            conversation_history.append(message)
            for call in tool_calls:
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]
                print(f"\033[90m[tool] 调用 {name} 参数: {arguments}\033[0m")
                tool_result = tool_manager.call(name, arguments)
                print(f"\033[90m[tool] 结果: {tool_result}\033[0m")
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"result": tool_result}, ensure_ascii=False),
                })

    except Exception as exc:
        print(f"请求出错: {exc}")
        return ""


# ===========================
# 命令行交互
# ===========================
def main() -> None:
    print("正在连接 MCP Server →", MCP_SERVER_URL)
    tool_manager = ToolManager(MCP_SERVER_URL)
    tools = tool_manager.get_tools()

    print(f"已发现 {len(tools)} 个工具：", ", ".join(t["function"]["name"] for t in tools))
    print("\n=== 欢迎使用推理模型连续对话 CLI（含思维链输出）===")
    print("输入 'exit' 退出程序\n")

    while True:
        user_input = prompt(HTML("<ansired>你:</ansired> ")).strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        get_response_with_tools(tool_manager, user_input)
        print()


if __name__ == "__main__":
    main()
