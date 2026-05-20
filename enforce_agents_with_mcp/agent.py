"""
Demo Agent — 使用 raw requests（DeepSeek）+ mcp Python SDK（远程 MCP Server）
实现带工具调用的命令行连续对话。

相比 02_chat_completions_api_demo4.py，本 demo 的工具不在本地定义，
而是通过 MCP 协议从远程 Server 动态发现并调用。
ToolManager 类统一封装工具的发现、缓存与调用逻辑。

运行前请先启动 MCP Server：
    uv run python enforce_agents_with_mcp/mcp_server.py

再运行本 Agent：
    uv run python enforce_agents_with_mcp/agent.py
"""

import asyncio
import json
import os
import urllib3

import requests
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===========================
# 配置部分
# ===========================
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

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
# ToolManager：MCP 工具发现、缓存与调用
# ===========================
class ToolManager:
    """统一管理远程 MCP Server 上的工具：发现（含缓存）、Schema 转换、调用。"""

    def __init__(self, session: ClientSession) -> None:
        self._session = session
        self._cache: list[dict] | None = None  # OpenAI-format tools cache

    async def discover(self) -> list[dict]:
        """从 MCP Server 获取工具列表并转换为 OpenAI tool schema 格式。
        结果缓存于内存，后续调用直接返回缓存，不重复发网络请求。
        """
        if self._cache is not None:
            return self._cache
        result = await self._session.list_tools()
        self._cache = self._to_openai_schema(result.tools)
        return self._cache

    def _to_openai_schema(self, mcp_tools) -> list[dict]:
        """将 MCP ListToolsResult.tools 转换为 OpenAI function-calling tools 格式。"""
        tools = []
        for t in mcp_tools:
            # inputSchema 是 MCP 标准字段，包含 type/properties/required
            parameters = t.inputSchema if t.inputSchema else {"type": "object", "properties": {}}
            tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": parameters,
                },
            })
        return tools

    async def call(self, name: str, arguments_json: str) -> str:
        """调用 MCP Server 上的指定工具，返回文本结果。"""
        try:
            args = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            return f"参数解析失败: {exc}"

        try:
            result = await self._session.call_tool(name, args)
        except Exception as exc:
            return f"工具调用失败: {exc}"

        # result.content 是 list[TextContent | ImageContent | EmbeddedResource]
        texts = [
            block.text
            for block in result.content
            if hasattr(block, "text")
        ]
        return "\n".join(texts) if texts else "(无返回内容)"

    def invalidate(self) -> None:
        """清空缓存，下次 discover() 时重新从 Server 获取工具列表。"""
        self._cache = None


# ===========================
# 工具调用循环 + 流式输出最终回复
# ===========================
async def get_response_with_tools(tool_manager: ToolManager, user_input: str) -> str:
    conversation_history.append({"role": "user", "content": user_input})
    openai_tools = await tool_manager.discover()

    try:
        # --- 工具调用循环（非流式，便于解析 tool_calls）---
        while True:
            data = {
                "model": MODEL,
                "messages": conversation_history,
                "tools": openai_tools,
                "tool_choice": "auto",
                "max_tokens": 1024,
                "temperature": 1.0,
                "stream": False,
            }
            resp = http_session.post(API_URL, headers=HEADERS, data=json.dumps(data))
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                break

            conversation_history.append(message)

            for call in tool_calls:
                name = call["function"]["name"]
                arguments = call["function"]["arguments"]
                print(f"\033[90m[tool] 调用 {name} 参数: {arguments}\033[0m")
                tool_result = await tool_manager.call(name, arguments)
                print(f"\033[90m[tool] 结果: {tool_result}\033[0m")

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"result": tool_result}, ensure_ascii=False),
                })

        # --- 流式输出最终回复 ---
        data = {
            "model": MODEL,
            "messages": conversation_history,
            "tools": openai_tools,
            "tool_choice": "none",
            "max_tokens": 1024,
            "temperature": 1.0,
            "stream": True,
        }
        print("\033[91mAI:\033[0m ", end="", flush=True)
        full_reply: list[str] = []
        with http_session.post(API_URL, headers=HEADERS, data=json.dumps(data), stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
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
        print()

        reply = "".join(full_reply)
        conversation_history.append({"role": "assistant", "content": reply})
        return reply

    except Exception as exc:
        print(f"请求出错: {exc}")
        return ""


# ===========================
# 命令行交互
# ===========================
async def main() -> None:
    print("正在连接 MCP Server →", MCP_SERVER_URL)
    async with streamable_http_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tool_manager = ToolManager(session)
            tools = await tool_manager.discover()

            print(f"已发现 {len(tools)} 个工具：", ", ".join(t["function"]["name"] for t in tools))
            print("\n=== 欢迎使用支持 Remote MCP 工具调用的连续对话 CLI ===")
            print("输入 'exit' 退出程序\n")

            ps: PromptSession = PromptSession()
            while True:
                try:
                    user_input = await ps.prompt_async(HTML("<ansired>你:</ansired> "))
                except (EOFError, KeyboardInterrupt):
                    break
                user_input = user_input.strip()
                if not user_input:
                    continue
                if user_input.lower() == "exit":
                    break

                await get_response_with_tools(tool_manager, user_input)
                print()


if __name__ == "__main__":
    asyncio.run(main())
