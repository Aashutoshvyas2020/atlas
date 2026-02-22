from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
import time
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from jarvis_tools import DesktopToolExecutor

try:
    import anthropic
except Exception:
    anthropic = None

try:
    from PIL import Image
except Exception:
    Image = None


StepCallback = Callable[[Dict[str, Any]], None]
ConfirmCallback = Callable[[str, str, float], bool]

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


@dataclass
class SubagentConfig:
    model: str = "claude-sonnet-4-5"
    max_steps: int = 30
    max_no_progress: int = 3
    max_runtime_s: float = 180.0
    max_tokens: int = 900
    max_history_messages: int = 12
    max_history_chars: int = 18000
    computer_tool_type: str = "computer_20250124"
    computer_use_beta: str = "computer-use-2025-01-24"
    action_delay_s: float = 0.2
    system_prompt: str = DEFAULT_SUBAGENT_SYSTEM_PROMPT
    initial_user_prompt_template: str = DEFAULT_SUBAGENT_INITIAL_PROMPT_TEMPLATE


class ClaudeDesktopSubagent:
    ACTIONS = {
        "key",
        "type",
        "mouse_move",
        "left_click",
        "left_click_drag",
        "right_click",
        "middle_click",
        "double_click",
        "screenshot",
        "cursor_position",
        "scroll",
        "wait",
        "left_mouse_down",
        "left_mouse_up",
        "hold_key",
    }

    RISKY_TOKENS = {
        "buy",
        "purchase",
        "checkout",
        "submit",
        "confirm",
        "delete",
        "remove",
        "pay",
        "send",
        "transfer",
        "password",
        "login",
        "sign in",
        "sign-in",
        "account",
    }

    def __init__(
        self,
        api_key: str,
        cfg: Optional[SubagentConfig] = None,
        tools: Optional[DesktopToolExecutor] = None,
    ):
        if anthropic is None:
            raise RuntimeError("anthropic package is required")
        if not str(api_key or "").strip():
            raise RuntimeError("ANTHROPIC_API_KEY is required")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.cfg = cfg or SubagentConfig()
        self.tools = tools or DesktopToolExecutor()

    @classmethod
    def is_allowed_action(cls, name: str) -> bool:
        return str(name or "").strip() in cls.ACTIONS

    def run(
        self,
        task_id: str,
        goal_text: str,
        cancel_event: threading.Event,
        request_confirmation: ConfirmCallback,
        on_step: Optional[StepCallback] = None,
    ) -> Dict[str, Any]:
        started = time.time()
        no_progress = 0
        last_hash = ""

        real_w, real_h, display_w, display_h, scale = self._display_spec()
        messages: List[Dict[str, Any]] = []

        initial_png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
        last_hash = hashlib.sha256(initial_png).hexdigest()
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._initial_prompt(goal_text)},
                    self._image_block(initial_png),
                ],
            }
        )

        for step in range(1, self.cfg.max_steps + 1):
            if cancel_event.is_set():
                return {"ok": False, "result": "stopped", "reason": "interrupt_requested", "steps": step - 1}
            if float(self.cfg.max_runtime_s) > 0 and (time.time() - started) > float(self.cfg.max_runtime_s):
                return {"ok": False, "result": "stopped", "reason": "max_runtime_reached", "steps": step - 1}

            messages = self._trim_messages(messages)
            response = self._create_message(messages=messages, display_w=display_w, display_h=display_h)
            assistant_blocks = self._to_block_dicts(response.content)
            messages.append({"role": "assistant", "content": assistant_blocks})

            text_chunks = self._extract_text_blocks(response.content)
            if text_chunks:
                self._emit_step(
                    cb=on_step,
                    task_id=task_id,
                    step=step,
                    action="model_text",
                    status="ok",
                    detail={"text": " ".join(text_chunks)[:260]},
                )

            tool_uses = self._extract_computer_tool_uses(response.content)
            if not tool_uses:
                summary = " ".join(text_chunks).strip()
                if summary:
                    return {"ok": True, "result": "success", "reason": summary, "steps": step}
                # If model paused without tools or text, force a fresh screenshot turn.
                png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
                messages.append({"role": "user", "content": [self._image_block(png)]})
                continue

            tool_results: List[Dict[str, Any]] = []
            step_hash = last_hash

            for tool_use in tool_uses:
                tool_use_id = str(tool_use.get("id", "") or "")
                action_input = tool_use.get("input", {})
                if not isinstance(action_input, dict):
                    action_input = {}
                action = str(action_input.get("action", "") or "").strip()

                if not self.is_allowed_action(action):
                    tool_results.append(
                        self._tool_error(
                            tool_use_id=tool_use_id,
                            message=f"Unsupported computer action: {action or '[missing]'}",
                        )
                    )
                    self._emit_step(
                        cb=on_step,
                        task_id=task_id,
                        step=step,
                        action=action or "unknown_action",
                        status="error",
                        detail={"error": "unsupported_action"},
                    )
                    continue

                if self._requires_confirmation(action, action_input):
                    summary = self._confirmation_summary(action, action_input)
                    approved = request_confirmation(task_id, summary, 8.0)
                    if not approved:
                        return {
                            "ok": False,
                            "result": "stopped",
                            "reason": f"confirmation_rejected:{action}",
                            "steps": step,
                        }

                try:
                    outcome, screenshot_png = self._execute_computer_action(
                        action=action,
                        action_input=action_input,
                        coord_scale=scale,
                        display_w=display_w,
                        display_h=display_h,
                        real_w=real_w,
                        real_h=real_h,
                    )
                    if screenshot_png is not None:
                        step_hash = hashlib.sha256(screenshot_png).hexdigest()
                    tool_results.append(
                        self._tool_ok(tool_use_id=tool_use_id, action=action, outcome=outcome, screenshot_png=screenshot_png)
                    )
                    self._emit_step(cb=on_step, task_id=task_id, step=step, action=action, status="ok", detail=outcome)
                except Exception as exc:
                    err_png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
                    step_hash = hashlib.sha256(err_png).hexdigest()
                    tool_results.append(self._tool_error(tool_use_id=tool_use_id, message=str(exc), screenshot_png=err_png))
                    self._emit_step(
                        cb=on_step,
                        task_id=task_id,
                        step=step,
                        action=action,
                        status="error",
                        detail={"error": str(exc)},
                    )

            messages.append({"role": "user", "content": tool_results})

            if step_hash == last_hash:
                no_progress += 1
            else:
                no_progress = 0
                last_hash = step_hash

            if no_progress >= int(self.cfg.max_no_progress):
                return {"ok": False, "result": "stopped", "reason": "no_progress", "steps": step}

        return {"ok": False, "result": "stopped", "reason": "max_steps_reached", "steps": self.cfg.max_steps}

    def _create_message(self, messages: List[Dict[str, Any]], display_w: int, display_h: int) -> Any:
        last_error: Optional[Exception] = None
        for model_name in self._candidate_models(self.cfg.model):
            try:
                return self.client.beta.messages.create(
                    model=model_name,
                    max_tokens=int(self.cfg.max_tokens),
                    betas=[self.cfg.computer_use_beta],
                    system=self._system_prompt(),
                    tools=[
                        {
                            "type": self.cfg.computer_tool_type,
                            "name": "computer",
                            "display_width_px": int(display_w),
                            "display_height_px": int(display_h),
                        }
                    ],
                    messages=messages,
                )
            except Exception as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("No candidate model configured for Claude computer-use")

    def _trim_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(messages) <= 2:
            return messages

        keep_first = messages[:1]
        tail = list(messages[1:])
        max_history = max(4, int(self.cfg.max_history_messages))
        if len(tail) > max_history:
            tail = tail[-max_history:]

        max_chars = max(1000, int(self.cfg.max_history_chars))
        while len(tail) > 2 and self._estimate_messages_chars(keep_first + tail) > max_chars:
            tail.pop(0)
        return keep_first + tail

    @staticmethod
    def _estimate_messages_chars(messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                total += len(content)
                continue
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    total += len(str(block))
                    continue
                btype = str(block.get("type", "") or "")
                if btype == "text":
                    total += len(str(block.get("text", "") or ""))
                elif btype in {"image", "tool_result"}:
                    # Count image/tool blocks as a fixed budget unit; raw base64 is not useful for token estimate.
                    total += 220
                else:
                    total += len(json.dumps(block, ensure_ascii=True))
        return total

    @staticmethod
    def _candidate_models(preferred: str) -> List[str]:
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

    def _display_spec(self) -> Tuple[int, int, int, int, float]:
        screen_w, screen_h = self.tools.screen_size()
        scale = self._coordinate_scale(screen_w, screen_h)
        disp_w = max(1, int(round(screen_w * scale)))
        disp_h = max(1, int(round(screen_h * scale)))
        return int(screen_w), int(screen_h), disp_w, disp_h, scale

    @staticmethod
    def _coordinate_scale(width: int, height: int) -> float:
        long_edge = max(1, int(max(width, height)))
        total_pixels = max(1, int(width * height))
        long_edge_scale = 1568.0 / float(long_edge)
        total_pixels_scale = math.sqrt(1_150_000.0 / float(total_pixels))
        return float(min(1.0, long_edge_scale, total_pixels_scale))

    def _capture_model_screenshot(self, display_w: int, display_h: int) -> bytes:
        raw = self.tools.capture_screen_png()
        if Image is None:
            return raw
        try:
            with Image.open(BytesIO(raw)) as img:
                resized = img.resize((int(display_w), int(display_h)), Image.Resampling.LANCZOS)
                out = BytesIO()
                resized.save(out, format="PNG")
                return out.getvalue()
        except Exception:
            return raw

    def _initial_prompt(self, goal_text: str) -> str:
        goal = str(goal_text or "").strip()
        template = str(self.cfg.initial_user_prompt_template or "").strip()
        if not template:
            template = DEFAULT_SUBAGENT_INITIAL_PROMPT_TEMPLATE
        if "{goal_text}" in template:
            return template.replace("{goal_text}", goal)
        if "{{GOAL_TEXT}}" in template:
            return template.replace("{{GOAL_TEXT}}", goal)
        return f"{template.rstrip()}\nGoal: {goal}"

    def _system_prompt(self) -> str:
        prompt = str(self.cfg.system_prompt or "").strip()
        if prompt:
            return prompt
        return DEFAULT_SUBAGENT_SYSTEM_PROMPT

    @staticmethod
    def _to_block_dicts(blocks: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for block in blocks or []:
            if isinstance(block, dict):
                out.append(block)
                continue
            if hasattr(block, "model_dump"):
                out.append(block.model_dump(exclude_none=True))
                continue
            out.append(
                {
                    "type": str(getattr(block, "type", "text") or "text"),
                    "text": str(getattr(block, "text", "") or ""),
                }
            )
        return out

    @staticmethod
    def _extract_text_blocks(blocks: Any) -> List[str]:
        out: List[str] = []
        for block in blocks or []:
            btype = str(getattr(block, "type", "") or "")
            if btype == "text":
                text = str(getattr(block, "text", "") or "").strip()
                if text:
                    out.append(text)
        return out

    @staticmethod
    def _extract_computer_tool_uses(blocks: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for block in blocks or []:
            btype = str(getattr(block, "type", "") or "")
            if btype != "tool_use":
                continue
            name = str(getattr(block, "name", "") or "")
            if name != "computer":
                continue
            block_id = str(getattr(block, "id", "") or "")
            raw_input = getattr(block, "input", {}) or {}
            if not isinstance(raw_input, dict):
                raw_input = {}
            out.append({"id": block_id, "name": name, "input": raw_input})
        return out

    @staticmethod
    def _image_block(png_bytes: bytes) -> Dict[str, Any]:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png_bytes).decode("ascii"),
            },
        }

    def _tool_ok(
        self,
        tool_use_id: str,
        action: str,
        outcome: Dict[str, Any],
        screenshot_png: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps({"ok": True, "action": action, "result": outcome}, ensure_ascii=True),
            }
        ]
        if screenshot_png is not None:
            content.append(self._image_block(screenshot_png))
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}

    def _tool_error(
        self,
        tool_use_id: str,
        message: str,
        screenshot_png: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": str(message or "tool execution failed")}]
        if screenshot_png is not None:
            content.append(self._image_block(screenshot_png))
        return {"type": "tool_result", "tool_use_id": tool_use_id, "is_error": True, "content": content}

    @staticmethod
    def _parse_coordinate(
        action_input: Dict[str, Any],
        coord_scale: float,
        real_w: int,
        real_h: int,
    ) -> Tuple[int, int]:
        coord = action_input.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            cx, cy = int(coord[0]), int(coord[1])
        else:
            x = action_input.get("x")
            y = action_input.get("y")
            if x is None or y is None:
                raise RuntimeError("Action requires coordinate")
            cx, cy = int(x), int(y)
        scale = max(0.001, float(coord_scale))
        sx = int(round(float(cx) / scale))
        sy = int(round(float(cy) / scale))
        sx = max(0, min(int(real_w) - 1, sx))
        sy = max(0, min(int(real_h) - 1, sy))
        return sx, sy

    def _execute_computer_action(
        self,
        action: str,
        action_input: Dict[str, Any],
        coord_scale: float,
        display_w: int,
        display_h: int,
        real_w: int,
        real_h: int,
    ) -> Tuple[Dict[str, Any], Optional[bytes]]:
        if action == "screenshot":
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return {"ok": True, "bytes": len(png), "active_window": self.tools.get_active_window_title()}, png

        if action == "cursor_position":
            x, y = self.tools.cursor_position()
            scale = max(0.001, float(coord_scale))
            return {
                "ok": True,
                "x": x,
                "y": y,
                "model_x": int(round(float(x) * scale)),
                "model_y": int(round(float(y) * scale)),
                "active_window": self.tools.get_active_window_title(),
            }, None

        if action == "mouse_move":
            x, y = self._parse_coordinate(action_input, coord_scale=coord_scale, real_w=real_w, real_h=real_h)
            out = self.tools.move_mouse(x, y)
            time.sleep(max(0.0, float(self.cfg.action_delay_s)))
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "left_click":
            if "coordinate" in action_input or ("x" in action_input and "y" in action_input):
                x, y = self._parse_coordinate(action_input, coord_scale=coord_scale, real_w=real_w, real_h=real_h)
                out = self.tools.click_mouse(button="left", x=x, y=y, clicks=1)
            else:
                out = self.tools.click_mouse(button="left", clicks=1)
            time.sleep(max(0.0, float(self.cfg.action_delay_s)))
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "double_click":
            if "coordinate" in action_input or ("x" in action_input and "y" in action_input):
                x, y = self._parse_coordinate(action_input, coord_scale=coord_scale, real_w=real_w, real_h=real_h)
                out = self.tools.click_mouse(button="left", x=x, y=y, clicks=2)
            else:
                out = self.tools.click_mouse(button="left", clicks=2)
            time.sleep(max(0.0, float(self.cfg.action_delay_s)))
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "right_click":
            if "coordinate" in action_input or ("x" in action_input and "y" in action_input):
                x, y = self._parse_coordinate(action_input, coord_scale=coord_scale, real_w=real_w, real_h=real_h)
                out = self.tools.click_mouse(button="right", x=x, y=y, clicks=1)
            else:
                out = self.tools.click_mouse(button="right", clicks=1)
            time.sleep(max(0.0, float(self.cfg.action_delay_s)))
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "middle_click":
            if "coordinate" in action_input or ("x" in action_input and "y" in action_input):
                x, y = self._parse_coordinate(action_input, coord_scale=coord_scale, real_w=real_w, real_h=real_h)
                out = self.tools.click_mouse(button="middle", x=x, y=y, clicks=1)
            else:
                out = self.tools.click_mouse(button="middle", clicks=1)
            time.sleep(max(0.0, float(self.cfg.action_delay_s)))
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "left_click_drag":
            x, y = self._parse_coordinate(action_input, coord_scale=coord_scale, real_w=real_w, real_h=real_h)
            out = self.tools.drag_mouse_to(x, y, duration=0.2, button="left")
            time.sleep(max(0.0, float(self.cfg.action_delay_s)))
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "left_mouse_down":
            out = self.tools.mouse_down(button="left")
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "left_mouse_up":
            out = self.tools.mouse_up(button="left")
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "type":
            text = str(action_input.get("text", "") or "")
            out = self.tools.type_text(text)
            time.sleep(max(0.0, float(self.cfg.action_delay_s)))
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "key":
            key = str(action_input.get("text", "") or action_input.get("key", "") or "").strip()
            out = self.tools.key_combo(key)
            time.sleep(max(0.0, float(self.cfg.action_delay_s)))
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "hold_key":
            key = str(action_input.get("key", "") or action_input.get("text", "") or "").strip()
            seconds = float(action_input.get("seconds", 0.25))
            out = self.tools.hold_key(key=key, seconds=seconds)
            time.sleep(max(0.0, float(self.cfg.action_delay_s)))
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "scroll":
            direction = str(
                action_input.get("scroll_direction", "")
                or action_input.get("direction", "")
                or "down"
            ).strip().lower()
            amount = int(action_input.get("scroll_amount", action_input.get("amount", 600)))
            out = self.tools.scroll_view(direction=direction, amount=amount)
            time.sleep(max(0.0, float(self.cfg.action_delay_s)))
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return out, png

        if action == "wait":
            seconds = float(action_input.get("seconds", 1.0))
            seconds = max(0.0, min(seconds, 10.0))
            time.sleep(seconds)
            png = self._capture_model_screenshot(display_w=display_w, display_h=display_h)
            return {"ok": True, "slept_s": seconds}, png

        raise RuntimeError(f"Unsupported computer action: {action}")

    @classmethod
    def _requires_confirmation(cls, action: str, action_input: Dict[str, Any]) -> bool:
        if action not in {"type", "key", "left_click", "double_click", "right_click"}:
            return False
        joined = " ".join(
            [
                str(action_input.get("text", "") or ""),
                str(action_input.get("key", "") or ""),
                str(action_input.get("description", "") or ""),
                str(action_input.get("target", "") or ""),
            ]
        ).lower()
        return any(token in joined for token in cls.RISKY_TOKENS)

    @staticmethod
    def _confirmation_summary(action: str, action_input: Dict[str, Any]) -> str:
        if action == "type":
            text = str(action_input.get("text", "") or "")
            preview = (text[:80] + "...") if len(text) > 80 else text
            return f"Confirm typing text: {preview}"
        if action == "key":
            key = str(action_input.get("text", "") or action_input.get("key", "") or "")
            return f"Confirm key action: {key or '[unknown key]'}"
        if action in {"left_click", "double_click", "right_click"}:
            coord = action_input.get("coordinate")
            return f"Confirm {action} at {coord if coord is not None else 'current cursor'}"
        return f"Confirm action: {action}"

    @staticmethod
    def _emit_step(
        cb: Optional[StepCallback],
        task_id: str,
        step: int,
        action: str,
        status: str,
        detail: Dict[str, Any],
    ) -> None:
        if cb is None:
            return
        try:
            cb(
                {
                    "task_id": task_id,
                    "step": step,
                    "action": action,
                    "status": status,
                    "detail": detail,
                }
            )
        except Exception:
            pass
