#!/usr/bin/env python3
import os
import logging
import pickle
from pathlib import Path
from typing import AsyncIterable
from collections.abc import Iterable
from dataclasses import dataclass
from dotenv import load_dotenv

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    RunContext,
    function_tool,
    RoomInputOptions,
    Agent,
    AgentSession,
    llm,
    ModelSettings,
)
from livekit.plugins import openai, silero, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from memobase import AsyncMemoBaseClient, MemoBaseClient, User, ChatBlob
from memobase.utils import string_to_uuid

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("memory-agent")
mb_client = AsyncMemoBaseClient(
    api_key=os.getenv("MEMOBASE_API_KEY"), project_url=os.getenv("MEMOBASE_URL")
)


class RAGEnrichedAgent(Agent):
    """
    An agent that can answer questions using RAG (Retrieval Augmented Generation).
    """

    def __init__(self) -> None:
        """Initialize the RAG-enabled agent."""
        super().__init__(
            instructions="You are a warm-hearted partner.You can remember past interactions and use them to inform your answers.",
        )
        self.user_name = os.getenv("MEMOBASE_USER_NAME", "test user")
        self.chat_log_index = 1

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        assert await mb_client.ping(), "Memobase is not reachable"
        user = await mb_client.get_or_create_user(string_to_uuid(self.user_name))
        # chat_ctx.items[0].content[0] += "\n" + "User name is Gus"
        if len(chat_ctx.items) > self.chat_log_index:
            need_to_update = chat_ctx.items[
                self.chat_log_index : len(chat_ctx.items) - 1
            ]
            if len(need_to_update):
                b = ChatBlob(
                    messages=[
                        {
                            "role": m.role,
                            "content": m.content[0],
                        }
                        for m in need_to_update
                        if m.role in ["user", "assistant"]
                    ]
                )
                await user.insert(b)
                await user.flush()
                self.chat_log_index = len(chat_ctx.items) - 1
        rag_context: str = await user.context(max_token_size=500)
        chat_ctx.add_message(content=rag_context, role="system")
        logger.info(f"Memobase context: {rag_context}")
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

    async def on_enter(self):
        """Called when the agent enters the session."""
        self.session.generate_reply(
            instructions="Briefly greet the user and offer your assistance"
        )


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(
            instructions="You are a helpful assistant with a pleasant voice.",
            voice="ash",
        ),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=RAGEnrichedAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
