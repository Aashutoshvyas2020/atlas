from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Any, Dict, Tuple

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    import mss
    import mss.tools
except Exception:
    mss = None

try:
    from pywinauto import Desktop
except Exception:
    Desktop = None


class DesktopToolExecutor:
    """UI-only desktop tool executor (no shell command execution or filesystem writes)."""

    def __init__(self):
        if pyautogui is None:
            raise RuntimeError("pyautogui is required")
        pyautogui.FAILSAFE = False

    def screen_size(self) -> Tuple[int, int]:
        return self._screen_size()

    def cursor_position(self) -> Tuple[int, int]:
        if pyautogui is None:
            return (0, 0)
        pos = pyautogui.position()
        return int(pos.x), int(pos.y)

    def move_mouse(self, x: Any, y: Any, duration: float = 0.12) -> Dict[str, Any]:
        xi, yi = self._clamp_xy(x, y)
        pyautogui.moveTo(xi, yi, duration=max(0.0, float(duration)))
        return {"ok": True, "x": xi, "y": yi}

    def click_mouse(
        self,
        button: str = "left",
        x: Any = None,
        y: Any = None,
        clicks: int = 1,
        interval: float = 0.05,
    ) -> Dict[str, Any]:
        btn = str(button or "left").strip().lower()
        if x is not None and y is not None:
            xi, yi = self._clamp_xy(x, y)
            pyautogui.click(xi, yi, clicks=max(1, int(clicks)), interval=max(0.0, float(interval)), button=btn)
            return {"ok": True, "x": xi, "y": yi, "button": btn, "clicks": max(1, int(clicks))}
        pyautogui.click(clicks=max(1, int(clicks)), interval=max(0.0, float(interval)), button=btn)
        cx, cy = self.cursor_position()
        return {"ok": True, "x": cx, "y": cy, "button": btn, "clicks": max(1, int(clicks))}

    def mouse_down(self, button: str = "left") -> Dict[str, Any]:
        btn = str(button or "left").strip().lower()
        pyautogui.mouseDown(button=btn)
        x, y = self.cursor_position()
        return {"ok": True, "x": x, "y": y, "button": btn}

    def mouse_up(self, button: str = "left") -> Dict[str, Any]:
        btn = str(button or "left").strip().lower()
        pyautogui.mouseUp(button=btn)
        x, y = self.cursor_position()
        return {"ok": True, "x": x, "y": y, "button": btn}

    def drag_mouse_to(self, x: Any, y: Any, duration: float = 0.2, button: str = "left") -> Dict[str, Any]:
        xi, yi = self._clamp_xy(x, y)
        pyautogui.dragTo(xi, yi, duration=max(0.0, float(duration)), button=str(button or "left").strip().lower())
        return {"ok": True, "x": xi, "y": yi, "button": str(button or "left").strip().lower()}

    def type_text(self, text: str, interval: float = 0.008) -> Dict[str, Any]:
        body = str(text or "")
        pyautogui.write(body, interval=max(0.0, float(interval)))
        return {"ok": True, "typed_chars": len(body)}

    def key_combo(self, keys: Any) -> Dict[str, Any]:
        if isinstance(keys, str):
            parts = [p.strip() for p in re.split(r"[+]", keys) if p.strip()]
            if not parts:
                parts = [keys.strip()]
            norm = [self._key_alias(p) for p in parts if p.strip()]
        elif isinstance(keys, list):
            norm = [self._key_alias(str(k)) for k in keys if str(k).strip()]
        else:
            raise RuntimeError("key_combo requires string or non-empty key list")

        if not norm:
            raise RuntimeError("key_combo resolved to empty key sequence")
        if len(norm) == 1:
            pyautogui.press(norm[0])
        else:
            pyautogui.hotkey(*norm)
        return {"ok": True, "keys": norm}

    def hold_key(self, key: str, seconds: float = 0.25) -> Dict[str, Any]:
        k = self._key_alias(str(key or "").strip())
        if not k:
            raise RuntimeError("hold_key requires key")
        pyautogui.keyDown(k)
        try:
            time.sleep(max(0.0, min(float(seconds), 5.0)))
        finally:
            pyautogui.keyUp(k)
        return {"ok": True, "key": k, "seconds": max(0.0, min(float(seconds), 5.0))}

    def scroll_view(self, direction: str, amount: int = 600) -> Dict[str, Any]:
        d = str(direction or "down").strip().lower()
        amt = max(1, int(abs(amount)))
        if d == "up":
            pyautogui.scroll(amt)
        elif d == "down":
            pyautogui.scroll(-amt)
        elif d == "left":
            if hasattr(pyautogui, "hscroll"):
                pyautogui.hscroll(-amt)
            else:
                pyautogui.keyDown("shift")
                pyautogui.scroll(amt)
                pyautogui.keyUp("shift")
        elif d == "right":
            if hasattr(pyautogui, "hscroll"):
                pyautogui.hscroll(amt)
            else:
                pyautogui.keyDown("shift")
                pyautogui.scroll(-amt)
                pyautogui.keyUp("shift")
        else:
            raise RuntimeError(f"Unsupported scroll direction: {d}")
        return {"ok": True, "direction": d, "amount": amt}

    @staticmethod
    def _screen_size() -> Tuple[int, int]:
        if pyautogui is None:
            return (1920, 1080)
        size = pyautogui.size()
        return int(size.width), int(size.height)

    @staticmethod
    def _clamp_xy(x: Any, y: Any) -> Tuple[int, int]:
        w, h = DesktopToolExecutor._screen_size()
        xi = int(float(x))
        yi = int(float(y))
        xi = max(0, min(w - 1, xi))
        yi = max(0, min(h - 1, yi))
        return xi, yi

    @staticmethod
    def _key_alias(key: str) -> str:
        mapping = {
            "control": "ctrl",
            "ctrl": "ctrl",
            "return": "enter",
            "enter": "enter",
            "spacebar": "space",
            "escape": "esc",
            "windows": "win",
            "command": "win",
            "option": "alt",
            "cmd": "win",
            "page_down": "pagedown",
            "pageup": "pageup",
            "page_up": "pageup",
            "arrowup": "up",
            "arrowdown": "down",
            "arrowleft": "left",
            "arrowright": "right",
        }
        k = str(key or "").strip().lower()
        return mapping.get(k, k)

    def get_active_window_title(self) -> str:
        if Desktop is None:
            return ""
        try:
            win = Desktop(backend="uia").get_active()
            return str(win.window_text() or "").strip()
        except Exception:
            return ""

    def capture_screen_png(self) -> bytes:
        if mss is None:
            raise RuntimeError("mss is required")
        with mss.mss() as sct:
            mon = sct.monitors[1]
            shot = sct.grab(mon)
            return mss.tools.to_png(shot.rgb, shot.size)

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        action = str(name or "").strip()
        if action == "open_application":
            return self._open_application(str(args.get("app_name", "")).strip())
        if action == "focus_window":
            return self._focus_window(str(args.get("title_contains", "")).strip())
        if action == "move_mouse":
            return self.move_mouse(args.get("x", 0), args.get("y", 0), duration=0.12)
        if action == "left_click":
            return self.click_mouse(button="left", x=args.get("x", 0), y=args.get("y", 0), clicks=1)
        if action == "double_click":
            return self.click_mouse(button="left", x=args.get("x", 0), y=args.get("y", 0), clicks=2)
        if action == "right_click":
            return self.click_mouse(button="right", x=args.get("x", 0), y=args.get("y", 0), clicks=1)
        if action == "type_text":
            return self.type_text(str(args.get("text", "")), interval=0.008)
        if action == "press_keys":
            keys = args.get("keys", [])
            if not isinstance(keys, list) or not keys:
                raise RuntimeError("press_keys requires non-empty keys[]")
            return self.key_combo(keys)
        if action == "scroll":
            direction = str(args.get("direction", "down")).strip().lower()
            amount = int(args.get("amount", 600))
            return self.scroll_view(direction=direction, amount=amount)
        if action == "wait":
            seconds = float(args.get("seconds", 1.0))
            seconds = max(0.0, min(10.0, seconds))
            time.sleep(seconds)
            return {"ok": True, "slept_s": seconds}
        if action == "capture_screen":
            png = self.capture_screen_png()
            return {"ok": True, "bytes": len(png), "active_window": self.get_active_window_title()}
        if action == "done":
            return {"ok": True, "done": True, "summary": str(args.get("summary", ""))}
        raise RuntimeError(f"Unsupported action: {action}")

    def _open_application(self, app_name: str) -> Dict[str, Any]:
        if not app_name:
            raise RuntimeError("open_application requires app_name")

        common = {
            "chrome": "chrome.exe",
            "google chrome": "chrome.exe",
            "edge": "msedge.exe",
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "paint": "mspaint.exe",
            "explorer": "explorer.exe",
        }
        target = common.get(app_name.lower(), app_name)

        # os.startfile launches registered apps/docs without exposing shell command execution to the model.
        try:
            if os.path.exists(target):
                os.startfile(target)  # type: ignore[attr-defined]
            else:
                subprocess.Popen([target], creationflags=0x08000000)
        except Exception as exc:
            raise RuntimeError(f"Failed to open app '{app_name}': {exc}")
        return {"ok": True, "app_name": app_name}

    def _focus_window(self, title_contains: str) -> Dict[str, Any]:
        if not title_contains:
            raise RuntimeError("focus_window requires title_contains")
        if Desktop is None:
            raise RuntimeError("pywinauto is required for focus_window")

        needle = title_contains.strip().lower()
        try:
            wins = Desktop(backend="uia").windows()
            for win in wins:
                title = str(win.window_text() or "").strip()
                if needle in title.lower():
                    try:
                        win.set_focus()
                    except Exception:
                        pass
                    return {"ok": True, "title": title}
        except Exception as exc:
            raise RuntimeError(f"focus_window failed: {exc}")

        raise RuntimeError(f"No window found matching: {title_contains}")
