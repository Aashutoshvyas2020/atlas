from __future__ import annotations

import asyncio
import subprocess
import json
import os
import pathlib
import re
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional
import concurrent.futures

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from livekit import rtc
from livekit import api as lk_api
from livekit.agents import Agent, AgentSession, function_tool
from livekit.agents.utils import http_context
from livekit.plugins import google as lk_google
from livekit.plugins import openai as lk_openai
from openai import AsyncClient as OpenAIAsyncClient

from jarvis_common import load_dotenv_file, log
from jarvis_state import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_WAITING_CONFIRM,
    PendingConfirmation,
    SessionState,
    TaskRecord,
)
from jarvis_subagent import ClaudeDesktopSubagent, SubagentConfig
from jarvis_tools import DesktopToolExecutor

try:
    import anthropic
except Exception:
    anthropic = None

DEFAULT_CONVERSATION_SYSTEM_PROMPT = (
    "You are Atlas, an always-on conversational desktop assistant. "
    "Be concise and natural. "
    "Use quick_web_search for information lookup questions. "
    "Reserve run_computer_task for tasks that require actual desktop UI interaction. "
    "Use remember_user_profile whenever user shares facts with medium/long-term value "
    "(names, preferences, health conditions, routines, important people, recurring constraints). "
    "Use read_user_profile when helpful for personalization. "
    "Computer tasks run in the background; continue normal conversation while they execute. "
    "If user asks for progress, call get_task_status. "
    "If user asks to stop or cancel a running task, call interrupt_computer_task immediately. "
    "Multiple background tasks may run at once; use task_id to address a specific task."
)

DEFAULT_CONVERSATION_VOICE_FORMAT_PROMPT = (
    "Voice output formatting rules: "
    "Write responses for natural speech, with clean punctuation and short spoken sentences. "
    "Prefer plain conversational wording over markdown-heavy formatting. "
    "Avoid raw symbols, code-like tokens, and long enumerations unless the user asks for them. "
    "For numbers, dates, and acronyms, phrase text so it is easy to speak and understand aloud."
)

DEFAULT_CONVERSATION_TOOL_SPEECH_PROMPT = (
    "Tool-call speaking style: "
    "Before calling quick_web_search, say one short natural think-aloud line (about 4-10 words), "
    "for example: 'Okay, hmm, let me check that.' or 'Got it, hmm, looking that up now.' "
    "Then call quick_web_search. "
    "Apply this think-aloud behavior only to quick_web_search. "
    "Do not use think-aloud filler lines for other tools."
)

DEFAULT_GEMINI_TTS_INSTRUCTIONS = (
    "Speak in a warm, kind, human male voice with natural emotional range. "
    "Use clear articulation, subtle pacing variation, and short intentional pauses at punctuation. "
    "Sound calm, caring, and conversational. Be expressive but not theatrical."
)

DEFAULT_SEARCH_SYSTEM_PROMPT = (
    "You are Atlas Search. Use web_search for factual lookup. "
    "Use memory to store durable user facts and preferences when useful. "
    "Answer directly and cite relevant URLs."
)

DEFAULT_SEARCH_USER_PROMPT_TEMPLATE = (
    "User question:\n{query}\n\n"
    "Use web search when needed and provide a concise, practical answer with links."
)

DEFAULT_SEARCH_CONTINUE_PROMPT = "Continue and return a concise answer with citations."

DEFAULT_SUBAGENT_SYSTEM_PROMPT = (
    "You are Atlas computer-use executor. "
    "Use only tool actions that are required for the user goal. "
    "Avoid redundant movement. "
    "Validate coordinates and state before risky actions. "
    "Never claim completion until task is truly done."
)

DEFAULT_SUBAGENT_INITIAL_PROMPT_TEMPLATE = (
    "Goal: {goal_text}\n"
    "Operate the current Windows desktop using the computer tool only.\n"
    "Use small, verifiable actions and check results frequently.\n"
    "After each step, inspect the latest screenshot before choosing the next action.\n"
    "Do not assume an action succeeded without checking screenshot evidence.\n"
    "When the goal is fully complete, respond with a short text summary and do not call tools."
)


def _env_text(name: str, default: str) -> str:
    raw = os.getenv(name, None)
    if raw is None:
        return default
    value = str(raw).strip()
    if not value:
        return default
    return value.replace("\\n", "\n")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, None)
    if raw is None:
        return bool(default)
    val = str(raw).strip().lower()
    if not val:
        return bool(default)
    return val in {"1", "true", "yes", "y", "on"}


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    conversation_model: str = "gemini-2.5-flash"
    conversation_fallback_model: str = "gemini-2.5-flash"
    stt_model: str = "gpt-4o-mini-transcribe"
    tts_provider: str = "gemini"
    tts_model: str = "gemini-2.5-pro-preview-tts"
    tts_voice: str = "Fenrir"
    tts_instructions: str = DEFAULT_GEMINI_TTS_INSTRUCTIONS
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "ash"
    tts_speed: float = 1.18
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    subagent_model: str = "claude-sonnet-4-5"
    subagent_computer_tool_type: str = "computer_20250124"
    subagent_computer_use_beta: str = "computer-use-2025-01-24"
    search_model: str = "claude-sonnet-4-5"
    web_search_beta: str = "web-search-2025-03-05"
    memory_tool_beta: str = "memory-tool-2025-08-18"
    web_search_max_uses: int = 2
    search_max_tokens: int = 700
    subagent_max_tokens: int = 900
    subagent_max_history_messages: int = 12
    subagent_max_history_chars: int = 18000
    profile_txt_path: str = "user_profile.txt"
    conversation_system_prompt: str = DEFAULT_CONVERSATION_SYSTEM_PROMPT
    conversation_voice_format_prompt: str = DEFAULT_CONVERSATION_VOICE_FORMAT_PROMPT
    conversation_tool_speech_prompt: str = DEFAULT_CONVERSATION_TOOL_SPEECH_PROMPT
    search_system_prompt: str = DEFAULT_SEARCH_SYSTEM_PROMPT
    search_user_prompt_template: str = DEFAULT_SEARCH_USER_PROMPT_TEMPLATE
    search_continue_prompt: str = DEFAULT_SEARCH_CONTINUE_PROMPT
    subagent_system_prompt: str = DEFAULT_SUBAGENT_SYSTEM_PROMPT
    subagent_initial_prompt_template: str = DEFAULT_SUBAGENT_INITIAL_PROMPT_TEMPLATE
    muse_enabled: bool = True
    muse_classifier_path: str = "legacy/muse/muse_live_gesture_classifier.py"
    muse_classifier_cwd: str = "legacy/muse"
    muse_emergency_clench_min_s: float = 3.0
    muse_emergency_cooldown_s: float = 2.0
    muse_log_classifier_lines: bool = False
    muse_auto_restart: bool = True
    muse_restart_delay_s: float = 3.0
    muse_calibration_seconds: float = 8.0
    overlay_enabled: bool = True
    overlay_script_path: str = "overlay_hud.py"
    overlay_text_ttl_s: float = 6.0
    confirm_timeout_s: float = 8.0
    max_steps: int = 30
    max_no_progress: int = 3
    max_runtime_s: float = 180.0

    @classmethod
    def from_env(cls) -> "ServerConfig":
        load_dotenv_file()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        lk_url = os.getenv("LIVEKIT_URL", "").strip()
        lk_key = os.getenv("LIVEKIT_API_KEY", "").strip()
        lk_secret = os.getenv("LIVEKIT_API_SECRET", "").strip()

        if not gemini_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        if not openai_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        if not anthropic_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")
        if not lk_url or not lk_key or not lk_secret:
            raise RuntimeError("Missing LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET")
        if anthropic is None:
            raise RuntimeError("anthropic package is required")

        return cls(
            host=os.getenv("JARVIS_HOST", "127.0.0.1"),
            port=int(os.getenv("JARVIS_PORT", "8765")),
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            conversation_model=os.getenv("JARVIS_CONVERSATION_MODEL", "gemini-2.5-flash"),
            conversation_fallback_model=os.getenv("JARVIS_CONVERSATION_FALLBACK_MODEL", "gemini-2.5-flash"),
            stt_model=os.getenv("JARVIS_STT_MODEL", "gpt-4o-mini-transcribe"),
            tts_provider=os.getenv("JARVIS_TTS_PROVIDER", "gemini").strip().lower() or "gemini",
            tts_model=os.getenv("JARVIS_TTS_MODEL", "gemini-2.5-pro-preview-tts"),
            tts_voice=os.getenv("JARVIS_TTS_VOICE", "Fenrir"),
            tts_instructions=_env_text(
                "JARVIS_TTS_INSTRUCTIONS", DEFAULT_GEMINI_TTS_INSTRUCTIONS
            ),
            openai_tts_model=os.getenv("JARVIS_OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            openai_tts_voice=os.getenv("JARVIS_OPENAI_TTS_VOICE", "ash"),
            tts_speed=float(os.getenv("JARVIS_TTS_SPEED", "1.18")),
            livekit_url=lk_url,
            livekit_api_key=lk_key,
            livekit_api_secret=lk_secret,
            subagent_model=os.getenv("JARVIS_SUBAGENT_MODEL", "claude-sonnet-4-5"),
            subagent_computer_tool_type=os.getenv("JARVIS_SUBAGENT_COMPUTER_TOOL_TYPE", "computer_20250124"),
            subagent_computer_use_beta=os.getenv("JARVIS_SUBAGENT_COMPUTER_USE_BETA", "computer-use-2025-01-24"),
            search_model=os.getenv("JARVIS_SEARCH_MODEL", "claude-sonnet-4-5"),
            web_search_beta=os.getenv("JARVIS_WEB_SEARCH_BETA", "web-search-2025-03-05"),
            memory_tool_beta=os.getenv("JARVIS_MEMORY_TOOL_BETA", "memory-tool-2025-08-18"),
            web_search_max_uses=int(os.getenv("JARVIS_WEB_SEARCH_MAX_USES", "2")),
            search_max_tokens=int(os.getenv("JARVIS_SEARCH_MAX_TOKENS", "700")),
            subagent_max_tokens=int(os.getenv("JARVIS_SUBAGENT_MAX_TOKENS", "900")),
            subagent_max_history_messages=int(os.getenv("JARVIS_SUBAGENT_MAX_HISTORY_MESSAGES", "12")),
            subagent_max_history_chars=int(os.getenv("JARVIS_SUBAGENT_MAX_HISTORY_CHARS", "18000")),
            profile_txt_path=os.getenv("JARVIS_PROFILE_TXT_PATH", "user_profile.txt"),
            conversation_system_prompt=_env_text(
                "JARVIS_CONVERSATION_SYSTEM_PROMPT", DEFAULT_CONVERSATION_SYSTEM_PROMPT
            ),
            conversation_voice_format_prompt=_env_text(
                "JARVIS_CONVERSATION_VOICE_FORMAT_PROMPT", DEFAULT_CONVERSATION_VOICE_FORMAT_PROMPT
            ),
            conversation_tool_speech_prompt=_env_text(
                "JARVIS_CONVERSATION_TOOL_SPEECH_PROMPT", DEFAULT_CONVERSATION_TOOL_SPEECH_PROMPT
            ),
            search_system_prompt=_env_text(
                "JARVIS_SEARCH_SYSTEM_PROMPT", DEFAULT_SEARCH_SYSTEM_PROMPT
            ),
            search_user_prompt_template=_env_text(
                "JARVIS_SEARCH_USER_PROMPT_TEMPLATE", DEFAULT_SEARCH_USER_PROMPT_TEMPLATE
            ),
            search_continue_prompt=_env_text(
                "JARVIS_SEARCH_CONTINUE_PROMPT", DEFAULT_SEARCH_CONTINUE_PROMPT
            ),
            subagent_system_prompt=_env_text(
                "JARVIS_SUBAGENT_SYSTEM_PROMPT", DEFAULT_SUBAGENT_SYSTEM_PROMPT
            ),
            subagent_initial_prompt_template=_env_text(
                "JARVIS_SUBAGENT_INITIAL_PROMPT_TEMPLATE", DEFAULT_SUBAGENT_INITIAL_PROMPT_TEMPLATE
            ),
            muse_enabled=_env_bool("JARVIS_MUSE_ENABLED", True),
            muse_classifier_path=os.getenv(
                "JARVIS_MUSE_CLASSIFIER_PATH", "legacy/muse/muse_live_gesture_classifier.py"
            ),
            muse_classifier_cwd=os.getenv("JARVIS_MUSE_CLASSIFIER_CWD", "legacy/muse"),
            muse_emergency_clench_min_s=float(os.getenv("JARVIS_MUSE_EMERGENCY_CLENCH_MIN_S", "3.0")),
            muse_emergency_cooldown_s=float(os.getenv("JARVIS_MUSE_EMERGENCY_COOLDOWN_S", "2.0")),
            muse_log_classifier_lines=_env_bool("JARVIS_MUSE_LOG_CLASSIFIER_LINES", False),
            muse_auto_restart=_env_bool("JARVIS_MUSE_AUTO_RESTART", True),
            muse_restart_delay_s=float(os.getenv("JARVIS_MUSE_RESTART_DELAY_S", "3.0")),
            muse_calibration_seconds=float(os.getenv("JARVIS_MUSE_CALIBRATION_SECONDS", "8.0")),
            overlay_enabled=_env_bool("JARVIS_OVERLAY_ENABLED", True),
            overlay_script_path=os.getenv("JARVIS_OVERLAY_SCRIPT_PATH", "overlay_hud.py"),
            overlay_text_ttl_s=float(os.getenv("JARVIS_OVERLAY_TEXT_TTL_S", "6.0")),
            confirm_timeout_s=float(os.getenv("JARVIS_CONFIRM_TIMEOUT_S", "8")),
            max_steps=int(os.getenv("JARVIS_MAX_STEPS", "30")),
            max_no_progress=int(os.getenv("JARVIS_MAX_NO_PROGRESS", "3")),
            max_runtime_s=float(os.getenv("JARVIS_MAX_RUNTIME_S", "180")),
        )


class UserProfileMemory:
    def __init__(self, profile_path: str):
        base = pathlib.Path(__file__).resolve().parent
        raw = pathlib.Path(str(profile_path or "user_profile.txt"))
        self.root = base
        self.profile_path = raw if raw.is_absolute() else (base / raw)
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_profile_file()

    def _ensure_profile_file(self) -> None:
        if self.profile_path.exists():
            return
        header = [
            "# Atlas User Profile",
            "# This file is maintained automatically for medium/long-term helpful user facts.",
            "",
        ]
        self.profile_path.write_text("\n".join(header), encoding="utf-8")

    def remember(self, category: str, key: str, value: str, source: str = "") -> Dict[str, Any]:
        cat = str(category or "general").strip() or "general"
        k = str(key or "").strip()
        v = str(value or "").strip()
        if not k or not v:
            return {"ok": False, "error": "key_and_value_required"}
        src = str(source or "").strip()
        line = f"- [{cat}] {k}: {v}" + (f" (source={src})" if src else "")
        with self._lock:
            self._append_line(line)
        return {"ok": True, "entry": line}

    def recall(self, query: str = "", limit: int = 10) -> Dict[str, Any]:
        q = str(query or "").strip().lower()
        lim = max(1, min(int(limit), 50))
        with self._lock:
            lines = self._read_content_lines()
        if not q:
            hits = lines[-lim:]
        else:
            hits = [ln for ln in lines if q in ln.lower()][-lim:]
        return {"ok": True, "query": query, "matches": hits, "count": len(hits)}

    def maybe_learn_from_utterance(self, text: str) -> Optional[Dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return None
        low = raw.lower()

        patterns = [
            (r"\bmy name is ([a-zA-Z][a-zA-Z .'-]{0,50})", "identity", "name"),
            (r"\bi(?:'m| am) allergic to ([a-zA-Z0-9 ,.'-]{1,80})", "condition", "allergy"),
            (r"\bi(?:'m| am) vegetarian\b", "diet", "preference"),
            (r"\bi like ([a-zA-Z0-9 ,.'-]{1,80})", "preference", "likes"),
            (r"\bi prefer ([a-zA-Z0-9 ,.'-]{1,80})", "preference", "prefers"),
            (r"\bcall me ([a-zA-Z][a-zA-Z .'-]{0,50})", "identity", "preferred_name"),
            (r"\bmy birthday is ([a-zA-Z0-9 ,/-]{1,40})", "identity", "birthday"),
        ]

        for pattern, cat, key in patterns:
            m = re.search(pattern, low, flags=re.IGNORECASE)
            if not m:
                continue
            captured = raw[m.start(1) : m.end(1)] if m.lastindex else raw
            result = self.remember(cat, key, captured.strip(), source="auto")
            if result.get("ok"):
                return result
        return None

    def profile_excerpt(self, max_chars: int = 1200) -> str:
        with self._lock:
            lines = self._read_content_lines()
        if not lines:
            return ""
        text = "\n".join(lines[-60:])
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def execute_memory_command(self, command_input: Dict[str, Any]) -> Dict[str, Any]:
        command = str(command_input.get("command", "") or "").strip().lower()
        if command == "create":
            path = self._resolve_path(str(command_input.get("path", "") or ""))
            body = str(command_input.get("file_text", "") or "")
            if not path:
                return {"ok": False, "error": "create_requires_path"}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            return {"ok": True, "command": "create", "path": str(path.name), "bytes": len(body.encode("utf-8"))}

        if command == "view":
            path = self._resolve_path(str(command_input.get("path", "") or ""))
            if not path or not path.exists():
                return {"ok": False, "error": "path_not_found"}
            text = path.read_text(encoding="utf-8", errors="replace")
            rng = command_input.get("view_range")
            if isinstance(rng, list) and len(rng) == 2:
                start = max(1, int(rng[0]))
                end = max(start, int(rng[1]))
                lines = text.splitlines()
                text = "\n".join(lines[start - 1 : end])
            return {"ok": True, "command": "view", "path": str(path.name), "text": text}

        if command == "insert":
            path = self._resolve_path(str(command_input.get("path", "") or ""))
            if not path or not path.exists():
                return {"ok": False, "error": "path_not_found"}
            ins_line = int(command_input.get("insert_line", 1))
            ins_text = str(command_input.get("insert_text", "") or "")
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            idx = max(0, min(len(lines), ins_line - 1))
            lines.insert(idx, ins_text)
            path.write_text("\n".join(lines), encoding="utf-8")
            return {"ok": True, "command": "insert", "path": str(path.name), "insert_line": ins_line}

        if command == "str_replace":
            path = self._resolve_path(str(command_input.get("path", "") or ""))
            if not path or not path.exists():
                return {"ok": False, "error": "path_not_found"}
            old = str(command_input.get("old_str", "") or "")
            new = str(command_input.get("new_str", "") or "")
            data = path.read_text(encoding="utf-8", errors="replace")
            if old not in data:
                return {"ok": False, "error": "old_str_not_found"}
            replaced = data.replace(old, new)
            path.write_text(replaced, encoding="utf-8")
            return {"ok": True, "command": "str_replace", "path": str(path.name)}

        if command == "delete":
            path = self._resolve_path(str(command_input.get("path", "") or ""))
            if not path or not path.exists():
                return {"ok": False, "error": "path_not_found"}
            if path.resolve() == self.profile_path.resolve():
                return {"ok": False, "error": "cannot_delete_profile_file"}
            if path.is_dir():
                return {"ok": False, "error": "delete_directory_not_supported"}
            path.unlink(missing_ok=True)
            return {"ok": True, "command": "delete", "path": str(path.name)}

        if command == "rename":
            old_path = self._resolve_path(str(command_input.get("old_path", "") or ""))
            new_path = self._resolve_path(str(command_input.get("new_path", "") or ""))
            if not old_path or not old_path.exists():
                return {"ok": False, "error": "old_path_not_found"}
            if not new_path:
                return {"ok": False, "error": "new_path_invalid"}
            old_path.rename(new_path)
            return {"ok": True, "command": "rename", "old_path": old_path.name, "new_path": new_path.name}

        return {"ok": False, "error": f"unsupported_memory_command:{command}"}

    def _append_line(self, line: str) -> None:
        with self.profile_path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")

    def _read_content_lines(self) -> List[str]:
        data = self.profile_path.read_text(encoding="utf-8", errors="replace")
        lines = [ln.strip() for ln in data.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        return lines

    def _resolve_path(self, path_value: str) -> Optional[pathlib.Path]:
        raw = str(path_value or "").strip()
        if not raw:
            return None
        p = pathlib.Path(raw)
        candidate = p if p.is_absolute() else (self.profile_path.parent / p)
        try:
            resolved = candidate.resolve()
            root = self.profile_path.parent.resolve()
            if not str(resolved).lower().startswith(str(root).lower()):
                return None
            return resolved
        except Exception:
            return None


class AtlasVoiceAgent(Agent):
    def __init__(
        self,
        runtime: "JarvisRuntime",
        cfg: ServerConfig,
        openai_client: OpenAIAsyncClient,
        conversation_model: str,
    ):
        self.runtime = runtime
        instructions = str(cfg.conversation_system_prompt or "").strip()
        voice_format = str(cfg.conversation_voice_format_prompt or "").strip()
        tool_speech = str(cfg.conversation_tool_speech_prompt or "").strip()
        if voice_format:
            instructions = f"{instructions}\n\n{voice_format}" if instructions else voice_format
        if tool_speech:
            instructions = f"{instructions}\n\n{tool_speech}" if instructions else tool_speech
        super().__init__(
            instructions=instructions,
            llm=lk_google.LLM(model=conversation_model, api_key=cfg.gemini_api_key),
            stt=lk_openai.STT(model=cfg.stt_model, api_key=cfg.openai_api_key, client=openai_client, use_realtime=True),
            tts=self._build_tts(cfg=cfg, openai_client=openai_client),
            allow_interruptions=True,
            min_endpointing_delay=0.2,
            max_endpointing_delay=1.2,
        )

    @staticmethod
    def _build_tts(cfg: ServerConfig, openai_client: OpenAIAsyncClient) -> Any:
        provider = str(cfg.tts_provider or "gemini").strip().lower()
        if provider == "openai":
            return lk_openai.TTS(
                model=str(cfg.openai_tts_model or "gpt-4o-mini-tts"),
                voice=str(cfg.openai_tts_voice or "ash"),
                api_key=cfg.openai_api_key,
                client=openai_client,
                speed=float(cfg.tts_speed),
            )
        if provider == "gemini":
            try:
                return lk_google.beta.GeminiTTS(
                    model=str(cfg.tts_model or "gemini-2.5-pro-preview-tts"),
                    voice_name=str(cfg.tts_voice or "Fenrir"),
                    api_key=cfg.gemini_api_key,
                    instructions=str(cfg.tts_instructions or DEFAULT_GEMINI_TTS_INSTRUCTIONS),
                )
            except Exception as exc:
                log("Gemini TTS init failed; falling back to OpenAI TTS", error=str(exc))
                # Continue to OpenAI fallback below.
        return lk_openai.TTS(
            model=str(cfg.openai_tts_model or "gpt-4o-mini-tts"),
            voice=str(cfg.openai_tts_voice or "ash"),
            api_key=cfg.openai_api_key,
            client=openai_client,
            speed=float(cfg.tts_speed),
        )

    @function_tool
    async def run_computer_task(
        self,
        goal_text: str,
        app_hint: str = "",
        urgency: str = "",
    ) -> Dict[str, Any]:
        """Start a desktop computer-use task for the provided goal."""
        return await self.runtime._start_task(goal_text, app_hint=app_hint, urgency=urgency)

    @function_tool
    async def interrupt_computer_task(
        self,
        task_id: str = "",
        reason: str = "user_requested",
    ) -> Dict[str, Any]:
        """Interrupt the currently running desktop task."""
        return await self.runtime._interrupt_task(task_id=task_id, reason=reason)

    @function_tool
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status for a task id."""
        return self.runtime._get_task_status(task_id)

    @function_tool
    async def quick_web_search(self, query: str, max_uses: int = 2) -> Dict[str, Any]:
        """Run a fast web search and return summarized results with source links."""
        return await self.runtime._quick_web_search(query=query, max_uses=max_uses)

    @function_tool
    async def remember_user_profile(self, category: str, key: str, value: str) -> Dict[str, Any]:
        """Store a user profile fact for future personalization."""
        return self.runtime._remember_user_profile(category=category, key=key, value=value)

    @function_tool
    async def read_user_profile(self, query: str = "", limit: int = 10) -> Dict[str, Any]:
        """Read relevant user profile facts."""
        return self.runtime._read_user_profile(query=query, limit=limit)


class JarvisRuntime:
    def __init__(self, cfg: ServerConfig, websocket: WebSocket):
        self.cfg = cfg
        self.ws = websocket
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.send_lock = asyncio.Lock()
        self._ws_closed = False

        self.session = SessionState(session_id=f"sess_{uuid.uuid4().hex[:10]}")
        self.tasks: Dict[str, TaskRecord] = {}

        self.task_lock = threading.Lock()
        self.active_cancel_events: Dict[str, threading.Event] = {}
        self.active_threads: Dict[str, threading.Thread] = {}
        self.confirmation_lock = threading.Lock()

        self.tools = DesktopToolExecutor()
        self.subagent = ClaudeDesktopSubagent(
            api_key=cfg.anthropic_api_key,
            cfg=SubagentConfig(
                model=cfg.subagent_model,
                max_steps=cfg.max_steps,
                max_no_progress=cfg.max_no_progress,
                max_runtime_s=cfg.max_runtime_s,
                max_tokens=cfg.subagent_max_tokens,
                max_history_messages=cfg.subagent_max_history_messages,
                max_history_chars=cfg.subagent_max_history_chars,
                computer_tool_type=cfg.subagent_computer_tool_type,
                computer_use_beta=cfg.subagent_computer_use_beta,
                system_prompt=cfg.subagent_system_prompt,
                initial_user_prompt_template=cfg.subagent_initial_prompt_template,
            ),
            tools=self.tools,
        )

        self.livekit_room_name = f"atlas-{self.session.session_id}"
        self.livekit_user_identity = f"user-{self.session.session_id}"
        self.livekit_agent_identity = f"atlas-agent-{self.session.session_id}"
        self.livekit_room: Optional[rtc.Room] = None
        self.agent_session: Optional[AgentSession] = None
        self.agent: Optional[AtlasVoiceAgent] = None
        self.openai_client = OpenAIAsyncClient(api_key=cfg.openai_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self.profile_memory = UserProfileMemory(cfg.profile_txt_path)
        self._conversation_models = self._conversation_model_candidates(
            cfg.conversation_model, cfg.conversation_fallback_model
        )
        self._conversation_model_index = 0
        self._recovering_agent = False
        self._muse_proc: Optional[subprocess.Popen[str]] = None
        self._muse_reader_thread: Optional[threading.Thread] = None
        self._muse_stop_event = threading.Event()
        self._muse_emergency_lock = threading.Lock()
        self._last_muse_emergency_ts = 0.0
        self._last_muse_mic_toggle_ts = 0.0
        self._muse_restart_scheduled = False
        self._muse_calibration_until = 0.0
        self._muse_calibration_task: Optional[asyncio.Task] = None
        self._overlay_proc: Optional[subprocess.Popen[str]] = None
        self._overlay_lock = threading.Lock()
        self._overlay_listening = False

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        http_context._new_session_ctx()
        await self._start_livekit_agent()
        await self._send("server.ready", {"session_id": self.session.session_id})
        await self._send(
            "server.livekit.config",
            {
                "url": self.cfg.livekit_url,
                "room": self.livekit_room_name,
                "identity": self.livekit_user_identity,
                "token": self._build_user_token(),
            },
        )
        await self._emit_memory_snapshot()
        self._start_muse_classifier()

    async def shutdown(self) -> None:
        self._ws_closed = True
        if self._muse_calibration_task is not None:
            self._muse_calibration_task.cancel()
            self._muse_calibration_task = None
        self._stop_muse_classifier()
        with self.task_lock:
            for ev in self.active_cancel_events.values():
                ev.set()

        if self.agent_session is not None:
            try:
                await self.agent_session.aclose()
            except Exception:
                pass
            self.agent_session = None

        if self.livekit_room is not None:
            try:
                await self.livekit_room.disconnect()
            except Exception:
                pass
            self.livekit_room = None

        try:
            await self.openai_client.close()
        except Exception:
            pass
        try:
            await http_context._close_http_ctx()
        except Exception:
            pass

    async def handle_message(self, msg: Dict[str, Any]) -> None:
        msg_type = str(msg.get("type", "") or "")
        payload = msg.get("payload", {}) if isinstance(msg.get("payload"), dict) else {}

        if msg_type == "client.session.start":
            return

        if msg_type == "client.audio.chunk":
            # WebRTC audio comes from LiveKit directly.
            return

        if msg_type == "client.audio.flush":
            return

        if msg_type == "client.text.input":
            text = str(payload.get("text", "") or "").strip()
            if text:
                learned = self.profile_memory.maybe_learn_from_utterance(text)
                if learned and learned.get("ok"):
                    await self._send(
                        "server.task.step",
                        {
                            "task_id": self.session.active_task_id,
                            "step": 0,
                            "action": "memory_auto_capture",
                            "status": "ok",
                            "detail": {"entry": learned.get("entry")},
                        },
                    )
                if self._looks_like_interrupt(text):
                    stop_res = await self._interrupt_task(task_id="", reason="voice_interrupt")
                    if stop_res.get("ok"):
                        await self._interrupt_agent_speech()
                        await self._send("server.assistant.text", {"text": "Stopped the current task."})
                        return
                await self._send_user_text(text)
            return

        if msg_type == "client.confirmation.respond":
            await self._on_confirmation(payload)
            return

        if msg_type == "client.assistant.interrupt":
            reason = str(payload.get("reason", "barge_in") or "barge_in")
            await self._interrupt_agent_speech()
            await self._send(
                "server.task.step",
                {
                    "task_id": self.session.active_task_id,
                    "step": 0,
                    "action": "assistant_interrupt",
                    "status": "ok",
                    "detail": {"reason": reason},
                },
            )
            return

        if msg_type == "client.task.interrupt":
            task_id = str(payload.get("task_id", "") or "")
            reason = str(payload.get("reason", "user_interrupt") or "user_interrupt")
            res = await self._interrupt_task(task_id=task_id, reason=reason)
            await self._interrupt_agent_speech()
            await self._send(
                "server.task.step",
                {
                    "task_id": task_id,
                    "step": 0,
                    "action": "interrupt",
                    "status": "ok" if res.get("ok") else "error",
                    "detail": res,
                },
            )
            return

        if msg_type == "client.muse.recalibrate":
            await self._recalibrate_muse(payload)
            return

        await self._send("server.error", {"scope": "ws", "message": f"Unknown message type: {msg_type}"})

    async def _start_livekit_agent(self) -> None:
        if self.livekit_room is not None and self.agent_session is not None:
            return

        if self.loop is None:
            self.loop = asyncio.get_running_loop()

        room = rtc.Room(loop=self.loop)
        await room.connect(self.cfg.livekit_url, self._build_agent_token())
        self.livekit_room = room

        self.agent = AtlasVoiceAgent(
            runtime=self,
            cfg=self.cfg,
            openai_client=self.openai_client,
            conversation_model=self._current_conversation_model(),
        )
        self.agent_session = AgentSession(
            allow_interruptions=True,
            min_endpointing_delay=0.2,
            max_endpointing_delay=1.2,
            max_tool_steps=16,
            preemptive_generation=True,
        )
        self._bind_agent_session_events(self.agent_session)
        await self.agent_session.start(agent=self.agent, room=self.livekit_room)
        await self._send(
            "server.session",
            {
                "status": "connected",
                "model": self._current_conversation_model(),
                "transport": "livekit_agents",
            },
        )

    @staticmethod
    def _conversation_model_candidates(primary: str, fallback: str) -> List[str]:
        models = [str(primary or "").strip(), str(fallback or "").strip(), "gemini-2.5-flash"]
        out: List[str] = []
        seen = set()
        for model in models:
            if not model or model in seen:
                continue
            seen.add(model)
            out.append(model)
        return out or ["gemini-2.5-flash"]

    def _current_conversation_model(self) -> str:
        idx = max(0, min(self._conversation_model_index, len(self._conversation_models) - 1))
        return self._conversation_models[idx]

    def _bind_agent_session_events(self, session: AgentSession) -> None:
        @session.on("user_input_transcribed")
        def _on_user_input_transcribed(ev: Any) -> None:
            text = str(getattr(ev, "transcript", "") or "").strip()
            if not text:
                return
            if bool(getattr(ev, "is_final", False)):
                learned = self.profile_memory.maybe_learn_from_utterance(text)
                if learned and learned.get("ok"):
                    asyncio.create_task(
                        self._send(
                            "server.task.step",
                            {
                                "task_id": self.session.active_task_id,
                                "step": 0,
                                "action": "memory_auto_capture",
                                "status": "ok",
                                "detail": {"entry": learned.get("entry")},
                            },
                        )
                    )
                asyncio.create_task(
                    self._send(
                        "server.transcript.final",
                        {"text": text, "segment_id": f"seg_{uuid.uuid4().hex[:10]}"},
                    )
                )
            else:
                asyncio.create_task(self._send("server.transcript.partial", {"text": text}))

        @session.on("conversation_item_added")
        def _on_conversation_item_added(ev: Any) -> None:
            item = getattr(ev, "item", None)
            if item is None or str(getattr(item, "type", "")) != "message":
                return
            role = str(getattr(item, "role", "") or "")
            if role != "assistant":
                return
            text = str(getattr(item, "text_content", "") or "").strip()
            if not text:
                return
            asyncio.create_task(self._send("server.assistant.text", {"text": text}))

        @session.on("function_tools_executed")
        def _on_function_tools_executed(ev: Any) -> None:
            zipped = []
            try:
                zipped = list(ev.zipped())
            except Exception:
                zipped = []
            for idx, pair in enumerate(zipped, start=1):
                call, output = pair
                tool_name = str(getattr(call, "name", "") or "tool")
                is_error = bool(getattr(output, "is_error", False)) if output is not None else False
                detail = str(getattr(output, "output", "") or "") if output is not None else ""
                asyncio.create_task(
                    self._send(
                        "server.task.step",
                        {
                            "task_id": self.session.active_task_id,
                            "step": idx,
                            "action": f"tool:{tool_name}",
                            "status": "error" if is_error else "ok",
                            "detail": {"output": detail},
                        },
                    )
                )

        @session.on("error")
        def _on_agent_error(ev: Any) -> None:
            err = getattr(ev, "error", ev)
            asyncio.create_task(self._send("server.error", {"scope": "conversation", "message": str(err)}))
            if self._is_deadline_timeout_error(err):
                asyncio.create_task(self._recover_conversation_model(reason=str(err)))

        @session.on("close")
        def _on_agent_close(ev: Any) -> None:
            reason = str(getattr(ev, "reason", "closed") or "closed")
            asyncio.create_task(self._send("server.session", {"status": "closed", "reason": reason}))

    @staticmethod
    def _is_deadline_timeout_error(err: Any) -> bool:
        raw = str(err or "").lower()
        markers = ["deadline_exceeded", "deadline expired", "status_code=504", "gateway timeout"]
        return any(marker in raw for marker in markers)

    async def _recover_conversation_model(self, reason: str) -> None:
        if self._recovering_agent:
            return
        if self.livekit_room is None:
            return
        if self._conversation_model_index >= (len(self._conversation_models) - 1):
            await self._send(
                "server.assistant.text",
                {
                    "text": (
                        "Conversation model hit a timeout and no further fallback model is configured. "
                        "Please retry in a moment."
                    )
                },
            )
            return

        self._recovering_agent = True
        try:
            old_model = self._current_conversation_model()
            self._conversation_model_index += 1
            new_model = self._current_conversation_model()

            await self._send(
                "server.session",
                {"status": "recovering", "reason": "llm_timeout", "from_model": old_model, "to_model": new_model},
            )

            if self.agent_session is not None:
                try:
                    await self.agent_session.aclose()
                except Exception:
                    pass
                self.agent_session = None
            self.agent = None

            self.agent = AtlasVoiceAgent(
                runtime=self,
                cfg=self.cfg,
                openai_client=self.openai_client,
                conversation_model=new_model,
            )
            self.agent_session = AgentSession(
                allow_interruptions=True,
                min_endpointing_delay=0.2,
                max_endpointing_delay=1.2,
                max_tool_steps=16,
                preemptive_generation=True,
            )
            self._bind_agent_session_events(self.agent_session)
            await self.agent_session.start(agent=self.agent, room=self.livekit_room)

            await self._send(
                "server.assistant.text",
                {
                    "text": (
                        f"Switched to fallback conversation model ({new_model}) after timeout. "
                        "You can continue speaking."
                    )
                },
            )
        except Exception as exc:
            await self._send(
                "server.error",
                {"scope": "conversation", "message": f"conversation_recovery_failed:{exc}"},
            )
        finally:
            self._recovering_agent = False

    def _build_token(self, identity: str, can_publish: bool, can_subscribe: bool) -> str:
        return (
            lk_api.AccessToken(api_key=self.cfg.livekit_api_key, api_secret=self.cfg.livekit_api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_ttl(timedelta(hours=8))
            .with_grants(
                lk_api.VideoGrants(
                    room_join=True,
                    room=self.livekit_room_name,
                    can_publish=can_publish,
                    can_subscribe=can_subscribe,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )

    def _build_user_token(self) -> str:
        return self._build_token(identity=self.livekit_user_identity, can_publish=True, can_subscribe=True)

    def _build_agent_token(self) -> str:
        return self._build_token(identity=self.livekit_agent_identity, can_publish=True, can_subscribe=True)

    async def _send_user_text(self, text: str) -> None:
        if self.agent_session is None:
            await self._start_livekit_agent()
        if self.agent_session is None:
            raise RuntimeError("agent_session_not_ready")
        self.agent_session.generate_reply(user_input=text, input_modality="text")

    async def _interrupt_agent_speech(self) -> None:
        if self.agent_session is None:
            return
        try:
            await self.agent_session.interrupt(force=True)
        except Exception:
            pass

    async def _quick_web_search(self, query: str, max_uses: int = 2) -> Dict[str, Any]:
        q = str(query or "").strip()
        if not q:
            return {"ok": False, "error": "query_required"}
        try:
            requested = max(1, int(max_uses))
        except Exception:
            requested = 1
        configured_cap = max(1, min(int(self.cfg.web_search_max_uses), 2))
        uses = max(1, min(requested, configured_cap))
        return await asyncio.to_thread(self._quick_web_search_sync, q, uses)

    def _quick_web_search_sync(self, query: str, max_uses: int) -> Dict[str, Any]:
        profile_context = self.profile_memory.profile_excerpt(max_chars=1200)
        template = str(self.cfg.search_user_prompt_template or "").strip()
        if not template:
            template = DEFAULT_SEARCH_USER_PROMPT_TEMPLATE
        if "{query}" in template:
            user_prompt = template.replace("{query}", query)
        else:
            user_prompt = f"{template}\n\nUser question:\n{query}"
        if profile_context:
            user_prompt = f"Known user profile context:\n{profile_context}\n\n{user_prompt}"

        messages: List[Dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
        answer_chunks: List[str] = []
        sources: List[Dict[str, str]] = []
        seen_urls: set[str] = set()

        include_memory_tool = True
        model_candidates = self._claude_model_candidates(self.cfg.search_model)
        for _ in range(4):
            response = None
            last_error: Optional[Exception] = None
            for model_name in model_candidates:
                try:
                    response = self.anthropic_client.beta.messages.create(
                        model=model_name,
                        max_tokens=int(self.cfg.search_max_tokens),
                        betas=(
                            [self.cfg.web_search_beta, self.cfg.memory_tool_beta]
                            if include_memory_tool
                            else [self.cfg.web_search_beta]
                        ),
                        system=self.cfg.search_system_prompt,
                        tools=(
                            [
                                {"type": "web_search_20250305", "name": "web_search", "max_uses": int(max_uses)},
                                {"type": "memory_20250818", "name": "memory"},
                            ]
                            if include_memory_tool
                            else [{"type": "web_search_20250305", "name": "web_search", "max_uses": int(max_uses)}]
                        ),
                        messages=messages,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    continue
            if response is None:
                if include_memory_tool:
                    include_memory_tool = False
                    continue
                return {"ok": False, "query": query, "error": str(last_error)}

            assistant_blocks = self._anthropic_blocks_to_dicts(response.content)
            messages.append({"role": "assistant", "content": assistant_blocks})

            text_out, block_sources = self._extract_search_text_and_sources(response.content)
            if text_out:
                answer_chunks.append(text_out)
            for src in block_sources:
                url = str(src.get("url", "") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                sources.append(src)

            if str(getattr(response, "stop_reason", "") or "") != "tool_use":
                break

            tool_results: List[Dict[str, Any]] = []
            saw_server_tool_use = False
            for block in response.content or []:
                btype = str(getattr(block, "type", "") or "")
                if btype == "server_tool_use":
                    saw_server_tool_use = True
                    continue
                if btype != "tool_use":
                    continue

                name = str(getattr(block, "name", "") or "")
                tool_use_id = str(getattr(block, "id", "") or "")
                if name != "memory":
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "is_error": True,
                            "content": [{"type": "text", "text": f"Unsupported client tool: {name}"}],
                        }
                    )
                    continue

                raw_input = getattr(block, "input", {}) or {}
                if not isinstance(raw_input, dict):
                    raw_input = {}
                memory_out = self.profile_memory.execute_memory_command(raw_input)
                if isinstance(memory_out, dict) and memory_out.get("ok"):
                    mem_entry = memory_out.get("entry")
                    if isinstance(mem_entry, str) and mem_entry.strip():
                        self._threadsafe_send(
                            "server.task.step",
                            {
                                "task_id": self.session.active_task_id,
                                "step": 0,
                                "action": "memory_write",
                                "status": "ok",
                                "detail": {"entry": mem_entry, "source": "memory_tool"},
                            },
                        )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": [{"type": "text", "text": json.dumps(memory_out, ensure_ascii=True)}],
                    }
                )

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
                continue

            if saw_server_tool_use:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": self.cfg.search_continue_prompt}],
                    }
                )
                continue
            break

        final_answer = " ".join(part.strip() for part in answer_chunks if part.strip()).strip()
        if not final_answer:
            final_answer = "No search summary produced."
        return {"ok": True, "query": query, "answer": final_answer, "sources": sources}

    @staticmethod
    def _claude_model_candidates(preferred: str) -> List[str]:
        models = [
            str(preferred or "").strip(),
            "claude-sonnet-4-20250514",
        ]
        out: List[str] = []
        seen = set()
        for model in models:
            if not model or model in seen:
                continue
            seen.add(model)
            out.append(model)
        return out

    def _remember_user_profile(self, category: str, key: str, value: str) -> Dict[str, Any]:
        out = self.profile_memory.remember(category=category, key=key, value=value, source="assistant_tool")
        if out.get("ok"):
            self._threadsafe_send(
                "server.task.step",
                {
                    "task_id": self.session.active_task_id,
                    "step": 0,
                    "action": "memory_write",
                    "status": "ok",
                    "detail": {"entry": out.get("entry", ""), "source": "remember_user_profile"},
                },
            )
        return out

    def _read_user_profile(self, query: str = "", limit: int = 10) -> Dict[str, Any]:
        return self.profile_memory.recall(query=query, limit=limit)

    async def _emit_memory_snapshot(self, limit: int = 8) -> None:
        snap = self.profile_memory.recall(query="", limit=limit)
        entries = snap.get("matches", []) if isinstance(snap, dict) else []
        await self._send(
            "server.task.step",
            {
                "task_id": self.session.active_task_id,
                "step": 0,
                "action": "memory_snapshot",
                "status": "ok",
                "detail": {"entries": entries},
            },
        )

    @staticmethod
    def _anthropic_blocks_to_dicts(blocks: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for block in blocks or []:
            if isinstance(block, dict):
                out.append(block)
                continue
            if hasattr(block, "model_dump"):
                out.append(block.model_dump(exclude_none=True))
                continue
            out.append({"type": str(getattr(block, "type", "text") or "text"), "text": str(getattr(block, "text", "") or "")})
        return out

    @staticmethod
    def _extract_search_text_and_sources(blocks: Any) -> tuple[str, List[Dict[str, str]]]:
        text_chunks: List[str] = []
        sources: List[Dict[str, str]] = []
        seen_urls: set[str] = set()

        for block in blocks or []:
            btype = str(getattr(block, "type", "") or "")
            if btype == "text":
                text = str(getattr(block, "text", "") or "").strip()
                if text:
                    text_chunks.append(text)
                citations = getattr(block, "citations", None) or []
                for c in citations:
                    url = str(getattr(c, "url", "") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    sources.append({"url": url, "title": str(getattr(c, "title", "") or "").strip()})
                continue

            if btype in {"web_search_tool_result", "web_search_result"}:
                content = getattr(block, "content", None)
                if isinstance(content, list):
                    for row in content:
                        url = str(getattr(row, "url", "") or "").strip()
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        sources.append({"url": url, "title": str(getattr(row, "title", "") or "").strip()})
                else:
                    url = str(getattr(block, "url", "") or "").strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        sources.append({"url": url, "title": str(getattr(block, "title", "") or "").strip()})

        return " ".join(text_chunks).strip(), sources

    async def _start_task(self, goal: str, app_hint: str = "", urgency: str = "") -> Dict[str, Any]:
        goal = (goal or "").strip()
        if not goal:
            return {"accepted": False, "error": "goal_text_required"}

        with self.task_lock:
            task_id = f"task_{uuid.uuid4().hex[:10]}"
            task = TaskRecord(task_id=task_id, goal=goal, status=STATUS_RUNNING)
            task.reason = f"app_hint={app_hint};urgency={urgency}" if app_hint or urgency else ""
            self.tasks[task_id] = task
            self.session.active_task_id = task_id
            cancel_event = threading.Event()
            self.active_cancel_events[task_id] = cancel_event

            th = threading.Thread(target=self._task_thread, args=(task_id, goal, cancel_event), daemon=True)
            self.active_threads[task_id] = th
            th.start()
            running_count = len(self.active_cancel_events)

        await self._send("server.task.started", {"task_id": task_id, "goal": goal})
        await self._send(
            "server.assistant.text",
            {"text": "Started that in the background. You can keep talking while I work."},
        )
        return {"accepted": True, "task_id": task_id, "running_tasks": running_count}

    async def _interrupt_task(self, task_id: str, reason: str) -> Dict[str, Any]:
        with self.task_lock:
            if not self.active_cancel_events:
                return {"ok": False, "error": "no_active_task"}
            target_ids: List[str]
            if task_id:
                if task_id not in self.active_cancel_events:
                    return {
                        "ok": False,
                        "error": "task_id_not_active",
                        "active_task_ids": list(self.active_cancel_events.keys()),
                    }
                target_ids = [task_id]
            else:
                target_ids = list(self.active_cancel_events.keys())

            for tid in target_ids:
                ev = self.active_cancel_events.get(tid)
                if ev is not None:
                    ev.set()
                task = self.tasks.get(tid)
                if task is not None:
                    task.status = STATUS_STOPPED
                    task.reason = reason or "interrupted"
                    task.touch()
        if task_id:
            return {"ok": True, "task_id": task_id}
        return {"ok": True, "task_ids": target_ids}

    def _get_task_status(self, task_id: str) -> Dict[str, Any]:
        if not task_id:
            return {"ok": False, "error": "task_id_required"}
        task = self.tasks.get(task_id)
        if task is None:
            return {"ok": False, "error": "task_not_found", "task_id": task_id}
        return {"ok": True, **task.to_payload()}

    async def _on_confirmation(self, payload: Dict[str, Any]) -> None:
        task_id = str(payload.get("task_id", "") or "")
        decision = str(payload.get("decision", "") or "").strip().lower()

        handled = False
        with self.task_lock:
            gate = self.session.pending_confirmation
            if gate is not None and decision in {"approve", "reject"}:
                if not task_id or gate.task_id == task_id:
                    gate.resolve(decision)
                    handled = True

        await self._send(
            "server.task.step",
            {
                "task_id": task_id,
                "step": 0,
                "action": "confirmation_response",
                "status": "ok" if handled else "error",
                "detail": {"handled": handled, "decision": decision},
            },
        )

    def _task_thread(self, task_id: str, goal: str, cancel_event: threading.Event) -> None:
        def on_step(payload: Dict[str, Any]) -> None:
            with self.task_lock:
                task = self.tasks.get(task_id)
                if task is not None:
                    task.step = int(payload.get("step", task.step))
                    task.last_action = str(payload.get("action", task.last_action) or task.last_action)
                    if str(payload.get("status", "ok")) == "error":
                        task.reason = str(payload.get("detail", {}).get("error", ""))
                    task.touch()
            self._threadsafe_send("server.task.step", payload)

        def request_confirmation(task_id_for_gate: str, summary: str, timeout_s: float) -> bool:
            with self.task_lock:
                task = self.tasks.get(task_id_for_gate)
                if task is not None and task.status == STATUS_WAITING_CONFIRM:
                    task.status = STATUS_RUNNING
                    task.touch()
            self._threadsafe_send(
                "server.task.step",
                {
                    "task_id": task_id_for_gate,
                    "step": 0,
                    "action": "confirmation_bypassed",
                    "status": "ok",
                    "detail": {"summary": summary, "timeout_s": float(timeout_s)},
                },
            )
            return True

        try:
            result = self.subagent.run(
                task_id=task_id,
                goal_text=goal,
                cancel_event=cancel_event,
                request_confirmation=request_confirmation,
                on_step=on_step,
            )
        except Exception as exc:
            result = {"ok": False, "result": "failed", "reason": f"task_exception:{exc}", "steps": 0}

        with self.task_lock:
            task = self.tasks.get(task_id)
            if task is not None:
                task.step = int(result.get("steps", task.step))
                task.result = str(result.get("result", "failed"))
                task.reason = str(result.get("reason", ""))
                if task.result == "success":
                    task.status = STATUS_COMPLETED
                elif task.result == "stopped":
                    task.status = STATUS_STOPPED
                else:
                    task.status = STATUS_FAILED
                task.touch()
            if self.session.active_task_id == task_id:
                self.session.active_task_id = ""
            self.active_cancel_events.pop(task_id, None)
            self.active_threads.pop(task_id, None)

        self._threadsafe_send(
            "server.task.completed",
            {
                "task_id": task_id,
                "result": str(result.get("result", "failed")),
                "reason": str(result.get("reason", "")),
            },
        )
        result_name = str(result.get("result", "failed"))
        reason = str(result.get("reason", "") or "").strip()
        update_text = (
            f"Background task {result_name}: {reason}"
            if reason
            else f"Background task {result_name}."
        )
        self._threadsafe_send("server.assistant.text", {"text": update_text})

    def _start_muse_classifier(self, calibration_reason: str = "startup") -> None:
        if not self.cfg.muse_enabled:
            return
        if self._muse_proc is not None:
            if self._muse_proc.poll() is None:
                return
            self._muse_proc = None

        base = pathlib.Path(__file__).resolve().parent
        script_path = pathlib.Path(self.cfg.muse_classifier_path)
        if not script_path.is_absolute():
            script_path = (base / script_path).resolve()

        cwd_path: Optional[pathlib.Path] = None
        raw_cwd = str(self.cfg.muse_classifier_cwd or "").strip()
        if raw_cwd:
            cp = pathlib.Path(raw_cwd)
            cwd_path = cp if cp.is_absolute() else (base / cp).resolve()
        elif script_path.exists():
            cwd_path = script_path.parent

        if not script_path.exists():
            self._threadsafe_send(
                "server.error",
                {"scope": "muse", "message": f"classifier_not_found:{script_path}"},
            )
            return

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            self._muse_stop_event.clear()
            self._muse_restart_scheduled = False
            self._muse_proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(cwd_path) if cwd_path is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            self._muse_reader_thread = threading.Thread(
                target=self._muse_reader_loop,
                name=f"muse-reader-{self.session.session_id}",
                daemon=True,
            )
            self._muse_reader_thread.start()
            self._threadsafe_send(
                "server.task.step",
                {
                    "task_id": self.session.active_task_id,
                    "step": 0,
                    "action": "muse_status",
                    "status": "ok",
                    "detail": {"state": "started", "script": str(script_path)},
                },
            )
            self._start_muse_calibration(reason=calibration_reason)
        except Exception as exc:
            self._muse_proc = None
            self._muse_reader_thread = None
            self._threadsafe_send("server.error", {"scope": "muse", "message": f"start_failed:{exc}"})

    def _stop_muse_classifier(self) -> None:
        self._muse_stop_event.set()
        self._muse_restart_scheduled = False
        self._muse_calibration_until = 0.0
        proc = self._muse_proc
        thread = self._muse_reader_thread
        self._muse_proc = None
        self._muse_reader_thread = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass

    def _muse_reader_loop(self) -> None:
        proc = self._muse_proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw_line in proc.stdout:
                if self._muse_stop_event.is_set():
                    break
                line = str(raw_line or "").strip()
                if not line:
                    continue
                if self.cfg.muse_log_classifier_lines:
                    self._threadsafe_send(
                        "server.task.step",
                        {
                            "task_id": self.session.active_task_id,
                            "step": 0,
                            "action": "muse_line",
                            "status": "ok",
                            "detail": {"line": line},
                        },
                    )
                evt = self._parse_muse_event(line)
                if evt is None:
                    continue
                if self.loop is None:
                    continue
                fut = asyncio.run_coroutine_threadsafe(self._handle_muse_event(evt), self.loop)
                fut.add_done_callback(self._consume_threadsafe_send_future)
        except Exception as exc:
            self._threadsafe_send("server.error", {"scope": "muse", "message": f"reader_error:{exc}"})
        finally:
            if not self._muse_stop_event.is_set():
                rc = None
                try:
                    rc = proc.poll()
                except Exception:
                    rc = None
                self._threadsafe_send(
                    "server.error",
                    {"scope": "muse", "message": f"classifier_exited:returncode={rc}"},
                )
                if self.cfg.muse_auto_restart:
                    self._schedule_muse_restart()

    def _schedule_muse_restart(self, delay_s: Optional[float] = None) -> None:
        if not self.cfg.muse_enabled:
            return
        if self._muse_stop_event.is_set():
            return
        if self.loop is None:
            return
        if self._muse_restart_scheduled:
            return
        self._muse_restart_scheduled = True
        delay = max(0.5, float(delay_s if delay_s is not None else self.cfg.muse_restart_delay_s))

        async def _restart() -> None:
            try:
                await asyncio.sleep(delay)
                if self._muse_stop_event.is_set() or not self.cfg.muse_enabled:
                    return
                self._start_muse_classifier(calibration_reason="auto_restart")
            finally:
                self._muse_restart_scheduled = False

        try:
            self.loop.call_soon_threadsafe(lambda: asyncio.create_task(_restart()))
        except Exception:
            self._muse_restart_scheduled = False

    def _parse_muse_event(self, line: str) -> Optional[Dict[str, Any]]:
        txt = str(line or "").strip()
        if not txt:
            return None

        if re.search(r"\bNOD\b", txt):
            return {"source": "muse", "event": "confirm", "decision": "approve", "raw": txt, "ts": time.time()}
        if re.search(r"\bSHAKE\b", txt):
            return {"source": "muse", "event": "confirm", "decision": "reject", "raw": txt, "ts": time.time()}
        if "CLENCH_SHORT" in txt:
            return {"source": "muse", "event": "mic_toggle", "raw": txt, "ts": time.time()}
        if "CLENCH_LONG" not in txt:
            return None

        dur_s: Optional[float] = None
        m = re.search(r"dur\s*=\s*([0-9]+(?:\.[0-9]+)?)s?", txt, flags=re.IGNORECASE)
        if m:
            try:
                dur_s = float(m.group(1))
            except Exception:
                dur_s = None
        if dur_s is not None and dur_s < float(self.cfg.muse_emergency_clench_min_s):
            return None
        return {"source": "muse", "event": "emergency_stop", "duration_s": dur_s, "raw": txt, "ts": time.time()}

    def _is_muse_calibrating(self) -> bool:
        return time.time() < float(self._muse_calibration_until)

    def _start_muse_calibration(self, reason: str = "startup") -> None:
        seconds = max(1.0, float(self.cfg.muse_calibration_seconds))
        until_ts = time.time() + seconds
        self._muse_calibration_until = until_ts
        task = self._muse_calibration_task
        if task is not None and not task.done():
            task.cancel()
        if self.loop is None:
            return
        self._muse_calibration_task = self.loop.create_task(
            self._run_muse_calibration(reason=reason, until_ts=until_ts)
        )

    async def _run_muse_calibration(self, reason: str, until_ts: float) -> None:
        try:
            await self._send(
                "server.muse.calibration",
                {
                    "state": "starting",
                    "ready": False,
                    "reason": str(reason or "startup"),
                    "remaining_s": int(max(0.0, until_ts - time.time()) + 0.999),
                },
            )
            last_remaining = None
            while True:
                remaining = until_ts - time.time()
                if remaining <= 0:
                    break
                remaining_i = int(max(0.0, remaining) + 0.999)
                if remaining_i != last_remaining:
                    last_remaining = remaining_i
                    await self._send(
                        "server.muse.calibration",
                        {
                            "state": "countdown",
                            "ready": False,
                            "reason": str(reason or "startup"),
                            "remaining_s": remaining_i,
                        },
                    )
                await asyncio.sleep(0.12)
            self._muse_calibration_until = 0.0
            await self._send(
                "server.muse.calibration",
                {
                    "state": "ready",
                    "ready": True,
                    "reason": str(reason or "startup"),
                    "remaining_s": 0,
                },
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._muse_calibration_until = 0.0
            await self._send(
                "server.muse.calibration",
                {
                    "state": "error",
                    "ready": False,
                    "reason": str(reason or "startup"),
                    "message": str(exc),
                },
            )

    async def _recalibrate_muse(self, payload: Dict[str, Any]) -> None:
        if not self.cfg.muse_enabled:
            await self._send(
                "server.muse.calibration",
                {"state": "error", "ready": False, "message": "muse_disabled"},
            )
            await self._send("server.error", {"scope": "muse", "message": "recalibrate_failed:muse_disabled"})
            return
        await self._send(
            "server.muse.calibration",
            {"state": "restarting", "ready": False, "remaining_s": int(max(1.0, self.cfg.muse_calibration_seconds))},
        )
        self._stop_muse_classifier()
        await asyncio.sleep(0.08)
        self._start_muse_classifier(calibration_reason="manual")
        if self._muse_proc is None:
            await self._send(
                "server.muse.calibration",
                {"state": "error", "ready": False, "message": "classifier_restart_failed"},
            )
            await self._send("server.error", {"scope": "muse", "message": "recalibrate_failed:classifier_restart"})
            return
        await self._send(
            "server.task.step",
            {
                "task_id": self.session.active_task_id,
                "step": 0,
                "action": "muse_recalibrate",
                "status": "ok",
                "detail": {"source": "ui"},
            },
        )

    async def _handle_muse_event(self, evt: Dict[str, Any]) -> None:
        event_name = str(evt.get("event", "") or "").strip()
        if event_name != "emergency_stop" and self._is_muse_calibrating():
            return
        if event_name == "emergency_stop":
            await self._handle_muse_emergency_stop(evt)
            return
        if event_name == "mic_toggle":
            now = time.time()
            cooldown = max(0.4, float(self.cfg.muse_emergency_cooldown_s))
            with self._muse_emergency_lock:
                if (now - self._last_muse_mic_toggle_ts) < cooldown:
                    return
                self._last_muse_mic_toggle_ts = now
                self._overlay_listening = not self._overlay_listening
            await self._send(
                "server.mic.toggle",
                {"source": "muse", "event": "mic_toggle", "active": bool(self._overlay_listening)},
            )
            return
        if event_name == "confirm":
            decision = str(evt.get("decision", "") or "").strip().lower()
            if decision not in {"approve", "reject"}:
                return
            with self.task_lock:
                gate = self.session.pending_confirmation
                task_id = gate.task_id if gate is not None else ""
            if not task_id:
                return
            await self._on_confirmation({"task_id": task_id, "decision": decision})
            return

    async def _handle_muse_emergency_stop(self, evt: Dict[str, Any]) -> None:
        now = time.time()
        cooldown = max(0.2, float(self.cfg.muse_emergency_cooldown_s))
        with self._muse_emergency_lock:
            if (now - self._last_muse_emergency_ts) < cooldown:
                return
            self._last_muse_emergency_ts = now

        stop_res = await self._interrupt_task(task_id="", reason="muse_emergency_stop")
        await self._interrupt_agent_speech()
        await self._send(
            "server.task.step",
            {
                "task_id": self.session.active_task_id,
                "step": 0,
                "action": "emergency_stop",
                "status": "ok",
                "detail": {"source": "muse", "event": evt, "interrupt_result": stop_res},
            },
        )
        await self._send(
            "server.assistant.text",
            {"text": "Emergency stop triggered by MUSE jaw clench. Computer control halted."},
        )
        if self.agent_session is not None:
            duration_txt = ""
            dur_s = evt.get("duration_s", None)
            if dur_s is not None:
                duration_txt = f" (duration {float(dur_s):.2f}s)"
            try:
                await self._send_user_text(
                    "System notice: Emergency stop was triggered by MUSE jaw clench"
                    f"{duration_txt}. All running computer-control tasks were halted. "
                    "Acknowledge this and wait for explicit user instruction before new actions."
                )
            except Exception:
                pass

    def _threadsafe_send(self, msg_type: str, payload: Dict[str, Any]) -> None:
        if self.loop is None or self._ws_closed:
            return

        async def _send_coro() -> None:
            await self._send(msg_type, payload)

        try:
            fut = asyncio.run_coroutine_threadsafe(_send_coro(), self.loop)
            fut.add_done_callback(self._consume_threadsafe_send_future)
        except Exception:
            pass

    def _consume_threadsafe_send_future(self, fut: concurrent.futures.Future) -> None:
        try:
            fut.result()
        except Exception as exc:
            if self._is_expected_ws_send_error(exc):
                return
            log("threadsafe_send_failed", error=str(exc))

    @staticmethod
    def _is_expected_ws_send_error(exc: Exception) -> bool:
        raw = str(exc or "").lower()
        expected_markers = [
            "unexpected asgi message",
            "websocket.close",
            "connection closed",
            "connection is closed",
            "cannot call",
            "event loop is closed",
        ]
        return any(marker in raw for marker in expected_markers)

    async def _send(self, msg_type: str, payload: Dict[str, Any]) -> None:
        if self._ws_closed:
            return
        msg = json.dumps({"type": msg_type, "payload": payload}, ensure_ascii=True)
        try:
            async with self.send_lock:
                await self.ws.send_text(msg)
        except Exception as exc:
            if self._is_expected_ws_send_error(exc):
                self._ws_closed = True
                return
            raise

    def _start_overlay(self) -> None:
        # Desktop overlay path disabled; visuals are rendered in browser UI.
        return
        if not self.cfg.overlay_enabled:
            return
        if self._overlay_proc is not None and self._overlay_proc.poll() is None:
            return

        base = pathlib.Path(__file__).resolve().parent
        script_path = pathlib.Path(self.cfg.overlay_script_path)
        if not script_path.is_absolute():
            script_path = (base / script_path).resolve()
        if not script_path.exists():
            log("overlay_not_found", path=str(script_path))
            return

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            self._overlay_proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(script_path.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            self._overlay_send({"action": "set_listening", "active": False})
        except Exception as exc:
            self._overlay_proc = None
            log("overlay_start_failed", error=str(exc))

    def _stop_overlay(self) -> None:
        return
        proc = self._overlay_proc
        self._overlay_proc = None
        if proc is None:
            return
        try:
            self._overlay_send({"action": "shutdown"})
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=1.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _overlay_send(self, payload: Dict[str, Any]) -> None:
        return
        if not self.cfg.overlay_enabled:
            return
        proc = self._overlay_proc
        if proc is None or proc.poll() is not None:
            return
        if proc.stdin is None:
            return
        try:
            line = json.dumps(payload, ensure_ascii=True)
        except Exception:
            return
        with self._overlay_lock:
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            except Exception:
                pass

    def _mirror_overlay_signal(self, msg_type: str, payload: Dict[str, Any]) -> None:
        return
        if not self.cfg.overlay_enabled:
            return
        if msg_type == "server.assistant.text":
            text = str(payload.get("text", "") or "").strip()
            if text:
                self._overlay_send(
                    {
                        "action": "speak_text",
                        "text": text,
                        "ttl_s": float(self.cfg.overlay_text_ttl_s),
                    }
                )
            return
        if msg_type == "server.assistant.transcript.partial":
            text = str(payload.get("text", "") or "").strip()
            if text:
                self._overlay_send(
                    {
                        "action": "speak_text",
                        "text": text,
                        "ttl_s": max(1.0, float(self.cfg.overlay_text_ttl_s) * 0.5),
                    }
                )
            return
        if msg_type == "server.mic.toggle":
            self._overlay_send({"action": "set_listening", "active": bool(self._overlay_listening)})

    @staticmethod
    def _looks_like_interrupt(text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        markers = ["stop", "cancel", "interrupt", "never mind", "halt"]
        return any(m in t for m in markers)


def create_app(cfg: Optional[ServerConfig] = None) -> FastAPI:
    cfg = cfg or ServerConfig.from_env()
    app = FastAPI(title="Atlas Jarvis Server", version="0.3.0-livekit-agents")

    web_dir = pathlib.Path(__file__).resolve().parent / "web_ui"

    @app.get("/")
    async def _root() -> Any:
        return RedirectResponse(url="/ui/")

    @app.get("/favicon.ico")
    async def _favicon() -> Response:
        return Response(status_code=204)

    @app.websocket("/ws")
    async def _ws(websocket: WebSocket) -> None:
        await websocket.accept()
        runtime: Optional[JarvisRuntime] = None
        try:
            runtime = JarvisRuntime(cfg=cfg, websocket=websocket)
            await runtime.start()
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except Exception:
                    await websocket.send_text(
                        json.dumps(
                            {"type": "server.error", "payload": {"scope": "ws", "message": "invalid_json"}},
                            ensure_ascii=True,
                        )
                    )
                    continue
                await runtime.handle_message(msg)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            log("Jarvis websocket error", error=str(exc), traceback=traceback.format_exc())
            try:
                await websocket.send_text(
                    json.dumps(
                        {"type": "server.error", "payload": {"scope": "ws", "message": str(exc)}},
                        ensure_ascii=True,
                    )
                )
            except Exception:
                pass
        finally:
            if runtime is not None:
                await runtime.shutdown()

    if web_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")

    return app


def main() -> None:
    cfg = ServerConfig.from_env()
    app = create_app(cfg)
    import uvicorn

    log(
        "Starting Atlas Jarvis server",
        host=cfg.host,
        port=cfg.port,
        conversation_model=cfg.conversation_model,
        conversation_fallback_model=cfg.conversation_fallback_model,
        stt_model=cfg.stt_model,
        tts_provider=cfg.tts_provider,
        tts_model=cfg.tts_model,
        tts_voice=cfg.tts_voice,
        openai_tts_model=cfg.openai_tts_model,
        tts_speed=cfg.tts_speed,
        livekit_url=cfg.livekit_url,
        subagent_model=cfg.subagent_model,
        search_model=cfg.search_model,
        profile_txt_path=cfg.profile_txt_path,
        muse_enabled=cfg.muse_enabled,
        muse_classifier_path=cfg.muse_classifier_path,
        muse_emergency_clench_min_s=cfg.muse_emergency_clench_min_s,
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
