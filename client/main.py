import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import validate_call

# .envから環境変数をロードする
load_dotenv()


class MCPClient:
    """Weather API MCPサーバ にリクエストを送るクライアント"""

    def __init__(self) -> None:
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    @validate_call
    async def connect_to_server(self, server_script_path: str):
        """MCPサーバーに接続する

        Args:
            server_script_path: サーバースクリプトへのパス (.py または .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")

        if not (is_python or is_js):
            raise ValueError(
                "サーバースクリプトは .py または .js ファイルでなければなりません。"
            )

        command = "python" if is_python else "node"

        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # 使用可能なツールのリストを取得、表示
        response = await self.session.list_tools()
        tools = response.tools

        print(
            "\nツールでMCPサーバーにリクエストを送信できます:",
            [tool.name for tool in tools],
        )

    @validate_call
    async def process_query(self, query: str) -> str:
        """Claudeと利用可能なMツールを使用してクエリを処理する

        Args:
            query: ユーザが入力した文字列
        """
        messages = [
            {
                "role": "user",
                "content": query,
            }
        ]

        response = await self.session.list_tools()

        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        # Claudeの初期化
        # .. MCPサーバで使える機能 (tools)
        # .. ユーザが入力した文字列 を渡す
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
        )

        # レスポンス処理とtool callsの処理
        final_text = []
        assistant_message_content = []

        for content in response.content:
            if content.type == "text":
                final_text.append(content.text)
                assistant_message_content.append(content)

            elif content.type == "tool_use":
                tool_name = content.name
                tool_args = content.input

                # tool call の実行
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)

                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message_content,
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content,
                            }
                        ],
                    }
                )

                # Claudeから次の返答を得る
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """インタラクティブ・チャット・ループの実行"""
        print("\nMCP Client 開始!")
        print("クエリーを入力するか、「quit」で終了してください。")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nエラー: {str(e)}")

    async def cleanup(self):
        """クリーンアップ・リソース"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()

    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
