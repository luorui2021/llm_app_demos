# LLM App Demos — Agent Instructions

## 项目概述

渐进式演示项目，展示从原始 HTTP 调用到 LangChain Agent 的 LLM 集成方式。

- `chat_to_llm/` — 单文件 CLI demo，按技术层级编号
- `enforce_agents_with_mcp/` — 独立子工程，演示 Agent 对接远程 MCP Server

## 构建与运行

```bash
uv sync                           # 安装依赖
uv run python chat_to_llm/04_langchain_sdk_demo3.py   # 运行指定 demo
python -m py_compile chat_to_llm/FILE.py              # 快速语法检查

# MCP demo（需两个终端）
uv run python enforce_agents_with_mcp/mcp_server.py   # 终端1：启动 MCP Server
uv run python enforce_agents_with_mcp/agent.py        # 终端2：启动 Agent
```

依赖管理使用 `uv`（非 pip/poetry），添加依赖用 `uv add <package>`。

## 文件结构与演进路线

### chat_to_llm/

| 文件前缀 | 底层技术 | 能力层级 |
|----------|----------|----------|
| `01_` | `requests` 原始 HTTP | 文本补全 (Completions API) |
| `02_` | `requests` / `httpx` | Chat Completions、流式、Function Calling、Pydantic Schema |
| `03_` | OpenAI SDK | 替代 requests，工具调用 |
| `04_` | LangChain + `ChatOpenAI` | 流式聊天、`@tool` 装饰器、自动 Agent |

新 demo 文件命名规则：`{序号}_{描述}_demo{变体}.py`，序号与技术层级对齐。

### enforce_agents_with_mcp/

| 文件 | 说明 |
|------|------|
| `mcp_server.py` | FastMCP Streamable HTTP Server，暴露4个示例工具（端口 8000） |
| `agent.py` | raw requests（DeepSeek LLM）+ mcp SDK（工具发现/调用），含 `ToolManager` 类 |

**`ToolManager` 类职责**：
- `discover()` — 从 MCP Server 获取工具列表并转换为 OpenAI function-calling schema，结果缓存于内存
- `call(name, arguments_json)` — 调用指定 MCP 工具，返回文本结果
- `invalidate()` — 清空缓存，触发下次重新发现

**关键技术决策**：
- MCP 传输协议选 Streamable HTTP（`/mcp` 路径），非 SSE 遗留协议
- Agent 整体为 `async`（MCP SDK 要求），入口用 `asyncio.run(main())`
- LLM 推理仍用同步 `requests`（在 async 函数内阻塞调用，demo 场景可接受）
- CLI 输入用 `PromptSession.prompt_async()`，避免阻塞事件循环

## 关键约定

### 配置区块
每个文件顶部必须有 `# 配置部分` 集中定义所有参数：
- API 密钥：大多数 demo 使用 `DEEPSEEK_API_KEY`（非 `OPENAI_API_KEY`）
- 代理：硬编码 `http://127.0.0.1:13128`，SSL 验证关闭（`verify=False`）
- 模型名、BASE_URL 均在此区块显式指定

### 代码风格
- `chat_to_llm/` 全部同步代码（无 `async/await`）
- `enforce_agents_with_mcp/agent.py` 为 async（MCP SDK 要求），LLM 调用仍同步
- CLI 交互使用 `prompt_toolkit`（非内置 `input()`）
- 消息历史用 `list[dict]`（raw HTTP/SDK）或 LangChain Message 对象
- 工具定义演进：手写 JSON Schema → `@pydantic_function_tool` → `@tool` → 远程 MCP

### LangChain 特定规则
- 流式优先使用 `agent.stream_events(..., version="v3")`
- 对话历史仅在 `get_response()` 函数内更新，`main()` 只负责读取输入和调用
- 当前版本：`langchain 1.3.1`、`langchain-openai 1.2.1`、`langgraph 1.2.0`

### FastMCP 注意事项
- `host` 和 `port` 是 `FastMCP.__init__()` 的参数，**不是** `run()` 的参数
- 正确写法：`FastMCP("名称", host="127.0.0.1", port=8000)` + `mcp.run(transport="streamable-http")`

## 环境要求

- Python ≥ 3.11，虚拟环境位于 `.venv/`
- 需设置环境变量 `DEEPSEEK_API_KEY`（或 `OPENAI_API_KEY`）
- 本地代理 `127.0.0.1:13128` 用于访问外部 API
- MCP demo 额外依赖：`mcp[cli]`（含 uvicorn、starlette）
