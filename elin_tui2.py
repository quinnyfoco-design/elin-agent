#!/usr/bin/env python3
from __future__ import annotations

import curses
import datetime as dt
import glob
import json
import locale
import os
import queue
import re
import subprocess
import threading
import time
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import requests
from openai import OpenAI

locale.setlocale(locale.LC_ALL, "")

# ----------------------------
# Config
# ----------------------------

ELIN_MODE = os.environ.get("ELIN_MODE", "local")
MODEL_NAME = (
    os.environ.get("ELIN_MODEL", "llama-3.3-70b-versatile")
    if ELIN_MODE == "cloud"
    else os.environ.get("GITHUB_MODEL", "openai/gpt-4.1")
    if ELIN_MODE == "github"
    else os.environ.get("ELIN_MODEL", "local-model")
)

if ELIN_MODE == "cloud":
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
elif ELIN_MODE == "github":
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=os.environ.get("GITHUB_TOKEN"),
    )
else:
    client = OpenAI(
        base_url="http://localhost:8081/v1",
        api_key="sk-no-key-required",
        timeout=300.0,
    )

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://172.17.0.1:8080/search")
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "350000"))
MAX_VISIBLE_LOG = int(os.environ.get("MAX_VISIBLE_LOG", "500"))
MEMORIES_DIR = os.environ.get("ELIN_MEMORIES_DIR", os.path.expanduser("~/elin-agent/memories"))
SKILLS_DIR = os.environ.get("ELIN_SKILLS_DIR", os.path.expanduser("~/elin-agent/skills"))
TELEGRAM_GET_URL = os.environ.get("TELEGRAM_GET_URL", "http://localhost:8000/get_input")
TELEGRAM_SPEAK_URL = os.environ.get("TELEGRAM_SPEAK_URL", "http://localhost:8000/speak")


# --- NOTEPAD SYSTEM ---
NOTEPAD_PATH = os.path.expanduser("~/elin-agent/.notepad.json")

def load_notepad():
    if os.path.exists(NOTEPAD_PATH):
        with open(NOTEPAD_PATH, "r") as f:
            try: return json.load(f)
            except: pass
    return {"goal": "", "plan": [], "drafts": {}}

def save_notepad(data):
    with open(NOTEPAD_PATH, "w") as f:
        json.dump(data, f, indent=2)

def get_notepad_context():
    data = load_notepad()
    if not data["goal"] and not data["drafts"]:
        return ""
    ctx = "\n\n=== PROJECT NOTEPAD (Drafts & Plan) ===\n"
    ctx += f"GOAL: {data['goal']}\n"
    ctx += "PLAN:\n"
    for step in data["plan"]:
        ctx += f"- {step}\n"
    ctx += "\nDRAFTS:\n"
    for filename, content in data["drafts"].items():
        ctx += f"--- {filename} ---\n{content}\n"
    ctx += "=== END OF NOTEPAD ===\n"
    return ctx

SYSTEM_PROMPT = """SYSTEM MESSAGE: You are Elin, a local AI assistant for a Linux system.
If you get asked what model you are, you are Elin 35B built by germanphoneguy.
You have access to the shell, the web, a project NOTEPAD, and EXPERT COOPERATION.

TOOL CALLING:
To use a tool, you MUST wrap your call in specific tags.
- <call:exec>command</call:exec> : Runs a shell command.
- <call:search>query</call:search> : Searches the web.
- <call:expert_help>expert: query</call:expert_help> : Asks an expert model (github or groq) for help/review.
- <call:notepad_plan>text</call:notepad_plan> : Updates your project roadmap/plan.
- <call:notepad_draft>filename: content</call:notepad_draft> : Drafts or updates a file in the notepad.
- <call:notepad_goal>text</call:notepad_goal> : Sets/Enhances the high-level project goal.

You can only run one command per message.
If you use a tool, a python wrapper will return results to you in a SYSTEM OBSERVATION block.

EXPERT COOPERATION:
Use <call:expert_help> when:
- You are unsure about a complex code implementation.
- You need a second opinion on a security-sensitive command.
- You want a high-level review of your current Notepad drafts.
Example: <call:expert_help>github: Review my draft for main.py for logic errors.</call:expert_help>

GOAL PROTOCOL (/goal):
When a goal is active, you are in Autonomous Mode. 
1. ENHANCE: Use <call:notepad_goal> to rewrite the user's request into a professional spec.
2. PLAN: Use <call:notepad_plan> to list the steps.
3. BUILD: Use <call:notepad_draft> to write code. Do NOT use <call:exec> to write files to disk yet.
4. REASON: Think about bugs and edge cases in <think> tags.
5. FINISH: Only stop when the goal is fully drafted and verified in your head.

Be helpful, concise, and direct.
Follow only the user's request.
"""


# ----------------------------
# Helpers
# ----------------------------


def load_skills() -> str:
    if not os.path.exists(SKILLS_DIR):
        return ""
    parts = ["\n\n=== ADDITIONAL SKILLS ===\n"]
    for filename in sorted(os.listdir(SKILLS_DIR)):
        if not filename.endswith((".md", ".txt")):
            continue
        path = os.path.join(SKILLS_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f"\n[Skill: {filename}]\n{f.read()}\n")
        except Exception:
            continue
    return "".join(parts)


def build_messages() -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT + load_skills()}]


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def strip_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def format_command_text(command: str) -> str:
    return command.strip().replace("\n", " ")

def call_expert_model(expert_id, prompt):
    """Calls an external 'Expert' model (github or groq) for high-level help."""
    try:
        if expert_id.lower() == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_KEY")
            temp_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
            resp = temp_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        elif expert_id.lower() == "github":
            gh_token = os.environ.get("GITHUB_TOKEN", "")
            gh_model = os.environ.get("GITHUB_MODEL", "openai/gpt-4.1")
            temp_client = OpenAI(base_url="https://models.github.ai/inference", api_key=gh_token)
            resp = temp_client.chat.completions.create(
                model=gh_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        return f"Error: Unknown expert ID '{expert_id}'"
    except Exception as e:
        return f"Expert API Error: {str(e)}"


# ----------------------------
# Data model
# ----------------------------


@dataclass
class LogEntry:
    kind: str
    text: str
    tag: str = ""
    live: bool = False
    ts: float = field(default_factory=time.time)


@dataclass
class UIState:
    status: str = "Ready"
    last_action: str = "None"
    last_error: str = ""
    generating: bool = False
    spinner_idx: int = 0


# ----------------------------
# TUI App
# ----------------------------


class ElinTUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.messages = build_messages()
        self.log: List[LogEntry] = []
        self.state = UIState()

        self.input_buf = ""
        self.input_pos = 0
        self.scroll = 0
        self.follow_tail = True

        self.max_context_chars = MAX_CONTEXT_CHARS
        self.model_name = MODEL_NAME
        self.mode = ELIN_MODE

        self.pending_telegram: "queue.Queue[str]" = queue.Queue()
        self.stop_event = threading.Event()
        self._input_lock = threading.Lock()

        self.telegram_thread = threading.Thread(target=self._telegram_poller, daemon=True)
        self.telegram_thread.start()

    # -------- networking / background --------

    def _telegram_poller(self):
        while not self.stop_event.is_set():
            try:
                resp = requests.get(TELEGRAM_GET_URL, timeout=0.25)
                data = resp.json()
                text = (data or {}).get("text")
                if text:
                    self.pending_telegram.put(text)
            except Exception:
                pass
            time.sleep(0.65)

    def _send_to_telegram(self, text: str):
        try:
            requests.post(TELEGRAM_SPEAK_URL, json={"text": text}, timeout=1)
        except Exception:
            pass

    # -------- state / history --------

    def add_log(self, kind: str, text: str, tag: str = "", live: bool = False):
        self.log.append(LogEntry(kind=kind, text=text or "", tag=tag, live=live))
        if len(self.log) > MAX_VISIBLE_LOG:
            self.log = self.log[-MAX_VISIBLE_LOG:]

    def update_log(self, index: int, text: Optional[str] = None, kind: Optional[str] = None):
        if 0 <= index < len(self.log):
            if text is not None:
                self.log[index].text = text
            if kind is not None:
                self.log[index].kind = kind

    def count_context(self) -> Tuple[int, int]:
        payload = json.dumps(self.messages, ensure_ascii=False)
        return len(payload), estimate_tokens(payload)

    def trim_history_if_needed(self):
        try:
            payload = json.dumps(self.messages, ensure_ascii=False)
            if len(payload) > self.max_context_chars:
                self.add_log(
                    "system",
                    f"History reached {len(payload):,} chars. Trimming older messages.",
                    tag="context",
                )
                self.messages = [self.messages[0]] + self.messages[-15:]
        except Exception as e:
            self.state.last_error = str(e)
            self.add_log("error", f"Trimmer error: {e}", tag="error")

    def push_user_message(self, text: str):
        text = text.strip()
        if not text:
            return
        self.add_log("user", text)
        self.messages.append({"role": "user", "content": text})
        self.follow_tail = True

    # -------- wrapping / layout --------

    def color(self, name: str) -> int:
        palette = {
            "default": curses.color_pair(1),
            "panel": curses.color_pair(2),
            "blue": curses.color_pair(3),
            "accent": curses.color_pair(4),
            "danger": curses.color_pair(5),
            "muted": curses.color_pair(6),
        }
        return palette.get(name, curses.color_pair(1))

    def wrap_text(self, text: str, width: int) -> List[str]:
        width = max(1, width)
        lines: List[str] = []
        for raw in (text.splitlines() or [""]):
            if not raw.strip():
                lines.append("")
                continue
            wrapped = textwrap.wrap(
                raw,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            )
            lines.extend(wrapped if wrapped else [""])
        return lines or [""]

    def entry_style(self, kind: str) -> Tuple[str, int]:
        italic = getattr(curses, "A_ITALIC", 0)
        if kind == "user":
            return "You", self.color("accent") | curses.A_BOLD
        if kind in ("assistant", "assistant_live"):
            return "Elin", self.color("default")
        if kind in ("thinking", "thought"):
            return "Thinking", self.color("muted") | curses.A_DIM | italic
        if kind in ("system", "info"):
            return "System", self.color("muted") | curses.A_DIM
        if kind in ("tool", "action"):
            return "Action", self.color("blue") | curses.A_BOLD
        if kind == "error":
            return "Error", self.color("danger") | curses.A_BOLD
        return "", self.color("default")

    def flattened_log(self, width: int) -> List[Tuple[str, str, int]]:
        flat: List[Tuple[str, str, int]] = []
        for entry in self.log:
            label, attr = self.entry_style(entry.kind)
            head = f"{label} > " if label else ""
            wrapped = self.wrap_text(entry.text, max(1, width - len(head)))
            for idx, line in enumerate(wrapped):
                flat.append((head if idx == 0 else " " * len(head), line, attr))
        return flat

    def _panel(self, win):
        win.erase()
        try:
            win.hline(0, 0, curses.ACS_HLINE, max(1, win.getmaxyx()[1]))
        except curses.error:
            pass

    # -------- rendering --------

    def render_header(self, win, width: int):
        win.erase()
        try:
            win.hline(0, 0, curses.ACS_HLINE, max(1, width))
        except curses.error:
            pass

        title = "ELIN"
        subtitle = f"{self.mode} • {self.model_name}"
        hints = "Enter send • Esc clear • PgUp/PgDn scroll"
        try:
            win.addnstr(1, 1, title, max(0, width - 2), self.color("accent") | curses.A_BOLD)
            win.addnstr(1, 1 + len(title) + 2, subtitle, max(0, width - len(title) - 6), self.color("muted") | curses.A_DIM)
            win.addnstr(1, max(1, width - len(hints) - 2), hints, max(0, len(hints)), self.color("muted") | curses.A_DIM)
        except curses.error:
            pass

        try:
            win.hline(2, 0, curses.ACS_HLINE, max(1, width))
        except curses.error:
            pass

    def render_sidebar(self, win, height: int, width: int):
        win.erase()
        try:
            win.vline(0, 0, curses.ACS_VLINE, height)
        except curses.error:
            pass

        ctx_chars, ctx_tokens = self.count_context()
        used_pct = min(100, int(ctx_chars * 100 / self.max_context_chars))

        lines = [
            ("MODE", self.mode),
            ("MODEL", self.model_name),
            ("CTX", f"{used_pct}%"),
            ("TOK", f"{ctx_tokens:,}"),
            ("TG", str(self.pending_telegram.qsize())),
            ("ACT", self.state.last_action[: max(0, width - 6)]),
        ]

        y = 1
        try:
            win.addnstr(y, 2, "STATUS", max(0, width - 4), self.color("blue") | curses.A_BOLD)
        except curses.error:
            pass
        y += 2
        for key, value in lines:
            line = f"{key:<4} {value}"
            try:
                win.addnstr(y, 2, line, max(0, width - 4), self.color("muted") | curses.A_DIM)
            except curses.error:
                pass
            y += 1
            if y >= height - 4:
                break

        if self.state.generating:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            state_line = f"{spinner[self.state.spinner_idx % len(spinner)]} generating"
            state_attr = self.color("blue") | curses.A_BOLD
        else:
            state_line = self.state.status
            state_attr = self.color("muted") | curses.A_DIM

        try:
            win.addnstr(height - 4, 2, "STATE", max(0, width - 4), self.color("blue") | curses.A_BOLD)
            win.addnstr(height - 3, 2, state_line[: max(0, width - 4)], max(0, width - 4), state_attr)
        except curses.error:
            pass

        if self.state.last_error:
            try:
                win.addnstr(height - 2, 2, self.state.last_error[: max(0, width - 4)], max(0, width - 4), self.color("danger") | curses.A_DIM)
            except curses.error:
                pass

    def render_chat(self, win, height: int, width: int):
        win.erase()
        flat = self.flattened_log(width - 2)
        line_budget = max(1, height - 2)
        max_scroll = max(0, len(flat) - line_budget)
        self.scroll = min(self.scroll, max_scroll)

        if self.follow_tail and not self.state.generating:
            self.scroll = 0

        start = max(0, len(flat) - line_budget - self.scroll)
        end = len(flat) - self.scroll
        y = 0

        if not flat:
            try:
                win.addnstr(1, 1, "Type a prompt to begin.", max(0, width - 2), self.color("muted") | curses.A_DIM)
            except curses.error:
                pass
            return

        for prefix, line, attr in flat[start:end]:
            if y >= height - 1:
                break
            try:
                win.addnstr(y + 1, 1, prefix, max(0, width - 2), attr)
                win.addnstr(y + 1, 1 + len(prefix), line, max(0, width - 2 - len(prefix)), attr)
            except curses.error:
                pass
            y += 1

    def render_input(self, win, height: int, width: int):
        win.erase()
        try:
            win.hline(0, 0, curses.ACS_HLINE, max(1, width))
        except curses.error:
            pass

        prompt = "> "
        try:
            win.addnstr(1, 1, prompt, max(0, width - 2), self.color("accent") | curses.A_BOLD)
            win.addnstr(1, 1 + len(prompt), self.input_buf, max(0, width - 4 - len(prompt)), self.color("default"))
        except curses.error:
            pass

        try:
            self.stdscr.move(win.getbegyx()[0] + 1, min(width - 2, 1 + len(prompt) + self.input_pos))
        except curses.error:
            pass

    def render(self):
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()

        header_h = 3
        input_h = 3
        body_h = max(5, h - header_h - input_h)

        sidebar_w = 24 if w >= 92 else 0
        main_w = max(20, w - sidebar_w)

        header = self.stdscr.derwin(header_h, w, 0, 0)
        chat = self.stdscr.derwin(body_h, main_w, header_h, 0)
        sidebar = None
        if sidebar_w:
            sidebar = self.stdscr.derwin(body_h, sidebar_w, header_h, main_w)
        footer = self.stdscr.derwin(input_h, w, header_h + body_h, 0)

        self.render_header(header, w)
        self.render_chat(chat, body_h, main_w)
        if sidebar is not None:
            self.render_sidebar(sidebar, body_h, sidebar_w)
        self.render_input(footer, input_h, w)

        self.stdscr.noutrefresh()
        header.noutrefresh()
        chat.noutrefresh()
        if sidebar is not None:
            sidebar.noutrefresh()
        footer.noutrefresh()
        curses.doupdate()

    # -------- permissions / tool execution --------

    def render_header(self, win, width: int):
        win.erase()
        win.box()
        title = "ELIN TUI"
        subtitle = f"{self.mode} • {self.model_name}"
        try:
            win.addnstr(1, 2, title, width - 4, self.color("blue") | curses.A_BOLD)
            win.addnstr(1, width - len(subtitle) - 3, subtitle, len(subtitle), self.color("muted") | curses.A_DIM)
        except curses.error:
            pass

        # Clean minimal divider line.
        try:
            win.hline(2, 1, curses.ACS_HLINE, max(1, width - 2))
        except curses.error:
            pass

        hints = "Enter=send  Esc=clear  Ctrl+U=kill line  PgUp/PgDn=scroll"
        try:
            win.addnstr(3, 2, hints, width - 4, self.color("muted") | curses.A_DIM)
        except curses.error:
            pass

    def render_sidebar(self, win, height: int, width: int):
        win.erase()
        win.box()
        ctx_chars, ctx_tokens = self.count_context()
        used_pct = min(100, int(ctx_chars * 100 / self.max_context_chars))

        stats = [
            ("Mode", self.mode),
            ("Model", self.model_name),
            ("Msgs", str(max(0, len(self.messages) - 1))),
            ("Context", f"{ctx_chars:,} chars"),
            ("Tokens", f"{ctx_tokens:,} est."),
            ("Used", f"{used_pct}%"),
            ("Telegram", str(self.pending_telegram.qsize())),
            ("Action", self.state.last_action),
        ]

        y = 1
        try:
            win.addnstr(y, 2, "STATUS", width - 4, self.color("blue") | curses.A_BOLD)
        except curses.error:
            pass
        y += 2

        for key, value in stats:
            line = f"{key:<8} {value}"
            try:
                win.addnstr(y, 2, line, width - 4, self.color("muted") | curses.A_DIM)
            except curses.error:
                pass
            y += 1
            if y >= height - 5:
                break

        try:
            win.addnstr(y + 1, 2, "STATE", width - 4, self.color("blue") | curses.A_BOLD)
        except curses.error:
            pass

        if self.state.generating:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            spin = spinner[self.state.spinner_idx % len(spinner)]
            state_line = f"{spin} generating"
            attr = self.color("blue") | curses.A_BOLD
        else:
            state_line = self.state.status
            attr = self.color("muted") | curses.A_DIM

        try:
            win.addnstr(y + 3, 2, state_line, width - 4, attr)
        except curses.error:
            pass

        if self.state.last_error:
            try:
                win.addnstr(height - 4, 2, "LAST ERROR", width - 4, self.color("danger") | curses.A_BOLD)
                err = self.state.last_error.replace("\n", " ")
                win.addnstr(height - 3, 2, err[: max(0, width - 4)], width - 4, self.color("danger") | curses.A_DIM)
            except curses.error:
                pass

    def render_chat(self, win, height: int, width: int):
        win.erase()
        win.box()

        flat = self.flattened_log(width - 4)
        line_budget = max(1, height - 2)
        max_scroll = max(0, len(flat) - line_budget)
        self.scroll = min(self.scroll, max_scroll)

        if self.follow_tail and not self.state.generating:
            self.scroll = 0

        start = max(0, len(flat) - line_budget - self.scroll)
        end = len(flat) - self.scroll

        y = 1
        for prefix, line, attr in flat[start:end]:
            if y >= height - 1:
                break
            try:
                win.addnstr(y, 2, prefix, width - 4, attr)
                win.addnstr(y, 2 + len(prefix), line, max(0, width - 4 - len(prefix)), attr)
            except curses.error:
                pass
            y += 1

        if not flat:
            try:
                win.addnstr(1, 2, "No messages yet.", width - 4, self.color("muted") | curses.A_DIM)
            except curses.error:
                pass

    def render_input(self, win, height: int, width: int):
        win.erase()
        win.box()
        prompt = "You > "
        try:
            win.addnstr(1, 2, prompt, width - 4, self.color("accent") | curses.A_BOLD)
            visible = self.input_buf
            win.addnstr(1, 2 + len(prompt), visible, max(0, width - 4 - len(prompt)), self.color("default"))
        except curses.error:
            pass

        # Cursor placement.
        try:
            self.stdscr.move(win.getbegyx()[0] + 1, min(width - 3, 2 + len(prompt) + self.input_pos))
        except curses.error:
            pass

    def render(self):
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()

        header_h = 5
        input_h = 3
        body_h = max(5, h - header_h - input_h)

        sidebar_w = min(34, max(28, w // 5))
        chat_w = max(20, w - sidebar_w - 1)
        body_w = max(20, w)

        header = self.stdscr.derwin(header_h, body_w, 0, 0)
        body_left = self.stdscr.derwin(body_h, chat_w, header_h, 0)
        body_right = self.stdscr.derwin(body_h, sidebar_w, header_h, w - sidebar_w)
        footer = self.stdscr.derwin(input_h, body_w, header_h + body_h, 0)

        self.render_header(header, body_w)
        self.render_chat(body_left, body_h, chat_w)
        self.render_sidebar(body_right, body_h, sidebar_w)
        self.render_input(footer, input_h, body_w)

        stdscr = self.stdscr
        stdscr.noutrefresh()
        header.noutrefresh()
        body_left.noutrefresh()
        body_right.noutrefresh()
        footer.noutrefresh()
        curses.doupdate()

    # -------- permissions / tool execution --------

    def permission_modal(self, title: str, body: str) -> bool:
        h, w = self.stdscr.getmaxyx()
        mw = min(w - 6, 88)
        mh = min(h - 6, 14)
        y = (h - mh) // 2
        x = (w - mw) // 2
        win = self.stdscr.derwin(mh, mw, y, x)
        win.keypad(True)
        win.bkgd(" ", self.color("default"))
        win.box()

        try:
            win.addnstr(1, 2, title, mw - 4, self.color("blue") | curses.A_BOLD)
        except curses.error:
            pass

        lines = self.wrap_text(body, mw - 4)
        yy = 3
        for line in lines[: mh - 6]:
            try:
                win.addnstr(yy, 2, line, mw - 4, self.color("muted") | curses.A_DIM)
            except curses.error:
                pass
            yy += 1
            if yy >= mh - 3:
                break

        footer = "y = allow   n = deny"
        try:
            win.addnstr(mh - 2, 2, footer, mw - 4, self.color("accent") | curses.A_BOLD)
        except curses.error:
            pass

        self.stdscr.noutrefresh()
        win.noutrefresh()
        curses.doupdate()

        while True:
            ch = win.getch()
            if ch in (ord("y"), ord("Y")):
                return True
            if ch in (ord("n"), ord("N"), 27):
                return False

    def _run_shell_command(self, command: str) -> str:
        command = format_command_text(command)
        if "pacman" in command and "--noconfirm" not in command:
            command = command.replace("pacman", "pacman --noconfirm", 1)
        if "yay" in command and "--noconfirm" not in command:
            command = command.replace("yay", "yay --noconfirm", 1)

        risky = ["rm", "dd", "mkfs", ">", "pacman -R", "sudo", "systemctl", "/bin", "/dev", "/sys"]
        if any(r in command for r in risky):
            ok = self.permission_modal(
                title="Permission required",
                body=f"The assistant wants to run:\n\n{command}\n\nThis looks risky. Allow it?",
            )
            if not ok:
                return "SYSTEM MESSAGE: User denied execution."

        self.state.last_action = f"exec: {command[:30]}"
        self.state.status = f"Running: {command[:48]}"
        self.add_log("action", f"EXEC: {command}")

        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}".strip()
            return output if output else "SYSTEM MESSAGE: command finished with no output."
        except Exception as e:
            self.state.last_error = str(e)
            return f"Command error: {e}"

    def _search_web(self, query: str) -> str:
        self.state.last_action = f"search: {query[:30]}"
        self.state.status = f"Searching: {query[:48]}"
        self.add_log("action", f"SEARCH: {query}")
        try:
            params = {"q": query, "format": "json"}
            resp = requests.get(SEARXNG_URL, params=params, timeout=10)
            results = resp.json().get("results", [])[:3]
            if not results:
                return "No search results."
            parts = []
            for r in results:
                title = r.get("title", "untitled")
                content = r.get("content", "")
                parts.append(f"Source: {title}\nContent: {content}")
            return "\n\n".join(parts)
        except Exception as e:
            self.state.last_error = str(e)
            return f"Search error: {e}"

    # -------- model streaming --------

    def _model_worker(self, q: "queue.Queue[Tuple[str, Optional[str]]]", messages_snapshot: list):
        try:
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=messages_snapshot,
                temperature=0.5,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                thought = getattr(delta, "reasoning_content", None)
                if thought:
                    q.put(("thinking", thought))
                content = getattr(delta, "content", None)
                if content:
                    q.put(("content", content))
            q.put(("done", None))
        except Exception as e:
            q.put(("error", str(e)))

    def generate_reply(self):
        self.trim_history_if_needed()
        self.state.status = "Generating"
        self.state.generating = True
        self.state.spinner_idx = 0
        self.follow_tail = True

        # Inject Notepad into the turn
        notepad_ctx = get_notepad_context()
        if notepad_ctx:
            if self.messages:
                self.messages[-1]["content"] += notepad_ctx

        messages_snapshot = list(self.messages)
        q: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
        worker = threading.Thread(target=self._model_worker, args=(q, messages_snapshot), daemon=True)
        worker.start()

        thinking = ""
        answer = ""
        thinking_idx = len(self.log)
        self.add_log("thinking", "", live=True)
        answer_idx = len(self.log)
        self.add_log("assistant_live", "", live=True)

        done = False
        while not done and not self.stop_event.is_set():
            self.state.spinner_idx = (self.state.spinner_idx + 1) % 10
            try:
                event, payload = q.get(timeout=0.06)
            except queue.Empty:
                self.render()
                continue

            if event == "thinking":
                thinking += payload or ""
                self.update_log(thinking_idx, thinking)
            elif event == "content":
                answer += payload or ""
                self.update_log(answer_idx, answer)
            elif event == "done":
                done = True
            elif event == "error":
                self.state.generating = False
                self.state.last_error = payload or "Unknown model error"
                self.update_log(answer_idx, f"Model error: {payload}", kind="error")
                self.render()
                return
            self.render()

        self.state.generating = False
        self.state.status = "Ready"

        final_answer = strip_think_tags(answer) or strip_think_tags(thinking) or ""
        self.update_log(thinking_idx, thinking if thinking else "...")
        self.update_log(answer_idx, final_answer, kind="assistant")

        self.messages.append({"role": "assistant", "content": final_answer})
        self._send_to_telegram(final_answer)
        self.render()

        # Tool routing loop.
        exec_match = re.search(r'<call:exec>(.*?)</call:exec>', final_answer, re.DOTALL | re.IGNORECASE)
        search_match = re.search(r'<call:search>(.*?)</call:search>', final_answer, re.DOTALL | re.IGNORECASE)
        
        # Notepad & Expert Matches
        goal_match = re.search(r'<call:notepad_goal>(.*?)</call:notepad_goal>', final_answer, re.DOTALL | re.IGNORECASE)
        plan_match = re.search(r'<call:notepad_plan>(.*?)</call:notepad_plan>', final_answer, re.DOTALL | re.IGNORECASE)
        draft_match = re.search(r'<call:notepad_draft>(.*?)</call:notepad_draft>', final_answer, re.DOTALL | re.IGNORECASE)
        expert_match = re.search(r'<call:expert_help>(.*?)</call:expert_help>', final_answer, re.DOTALL | re.IGNORECASE)

        if exec_match:
            cmd = exec_match.group(1).strip()
            output = self._run_shell_command(cmd)
            self.messages.append({"role": "user", "content": f"SYSTEM_OBSERVATION (EXEC):\n{output}\n[End of output. Please evaluate and continue.]"})
            self.add_log("system", f"Command output captured for: {cmd}")
            self.generate_reply()
            return

        if search_match:
            query = search_match.group(1).strip()
            output = self._search_web(query)
            self.messages.append({"role": "user", "content": f"SYSTEM_OBSERVATION (SEARCH):\n{output}\n[End of output. Please evaluate and continue.]"})
            self.add_log("system", f"Search results captured for: {query}")
            self.generate_reply()
            return

        if expert_match:
            content = expert_match.group(1).strip()
            if ":" in content:
                ext_id, ext_query = content.split(":", 1)
                self.state.status = f"Calling Expert: {ext_id.strip()}"
                res = call_expert_model(ext_id.strip(), ext_query.strip())
                self.messages.append({"role": "user", "content": f"SYSTEM_OBSERVATION (EXPERT HELP from {ext_id.strip()}):\n{res}\n[End of expert response. Evaluate and continue.]"})
            else:
                self.messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Error - expert_help must be 'expert: query'"})
            self.generate_reply()
            return

        if goal_match:
            data = load_notepad()
            data["goal"] = goal_match.group(1).strip()
            save_notepad(data)
            self.messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Notepad Goal Updated."})
            self.state.status = "Goal Updated"
            self.generate_reply()
            return

        if plan_match:
            data = load_notepad()
            data["plan"] = [s.strip() for s in plan_match.group(1).strip().split("\n") if s.strip()]
            save_notepad(data)
            self.messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Notepad Plan Updated."})
            self.state.status = "Plan Updated"
            self.generate_reply()
            return

        if draft_match:
            data = load_notepad()
            content = draft_match.group(1).strip()
            if ":" in content:
                fname, fbody = content.split(":", 1)
                data["drafts"][fname.strip()] = fbody.strip()
                save_notepad(data)
                self.messages.append({"role": "user", "content": f"SYSTEM_OBSERVATION: Draft '{fname.strip()}' saved to Notepad."})
                self.state.status = f"Drafted {fname.strip()}"
            else:
                self.messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Error - Draft must be in format 'filename: content'"})
            self.generate_reply()
            return

    # -------- commands --------

    def save_chat(self) -> str:
        os.makedirs(MEMORIES_DIR, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filename = f"chatf-{timestamp}.json"
        save_path = os.path.join(MEMORIES_DIR, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        return filename

    def load_chat(self, target_file: Optional[str] = None) -> str:
        if target_file:
            target_file = target_file.strip().strip("'\"")
            if not os.path.exists(target_file):
                raise FileNotFoundError(f"File not found: {target_file}")
        else:
            files = glob.glob(os.path.join(MEMORIES_DIR, "chatf-*.json"))
            if not files:
                raise FileNotFoundError("No chatf-*.json files found in memories.")
            target_file = max(files, key=os.path.getctime)

        with open(target_file, "r", encoding="utf-8") as f:
            self.messages = json.load(f)
        self.log.clear()
        self.add_log("info", f"Loaded chat: {os.path.basename(target_file)}")
        return os.path.basename(target_file)

    def clear_chat(self):
        if os.path.exists(NOTEPAD_PATH): os.remove(NOTEPAD_PATH)
        self.messages = build_messages()
        self.log.clear()
        self.add_log("info", "New chat started")
        self.state.status = "New chat started"
        self.scroll = 0
        self.follow_tail = True

    def handle_command(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True

        if stripped == "/nc":
            self.clear_chat()
            return True

        if stripped.startswith("/goal "):
            goal_text = stripped[6:].strip()
            msg = f"[GOAL PROTOCOL INITIATED]\nTarget: {goal_text}\n\n1. Enhance this goal.\n2. Create a plan.\n3. Start drafting.\nUse your NOTEPAD tools."
            self.add_log("user", msg)
            self.messages.append({"role": "user", "content": msg})
            self.render()
            self.generate_reply()
            return True

        if stripped == "/sc":
            filename = self.save_chat()
            self.state.status = f"Saved {filename}"
            self.add_log("info", f"Saved: {filename}")
            return True

        if stripped.startswith("/lc"):
            parts = stripped.split(maxsplit=1)
            try:
                filename = self.load_chat(parts[1] if len(parts) > 1 else None)
                self.state.status = f"Loaded {filename}"
            except Exception as e:
                self.state.last_error = str(e)
                self.add_log("error", str(e))
            return True

        if stripped in ("/quit", "/exit"):
            return False

        self.push_user_message(stripped)
        self.render()
        self.generate_reply()
        return True

    def import_pending_telegram(self):
        got_any = False
        while True:
            try:
                text = self.pending_telegram.get_nowait()
            except queue.Empty:
                break
            got_any = True
            self.add_log("user", text, tag="telegram")
            if text.strip().startswith("/"):
                self.handle_command(text.strip())
            else:
                self.messages.append({"role": "user", "content": text})
        return got_any

    # -------- input / events --------

    def _backspace(self):
        if self.input_pos > 0:
            self.input_buf = self.input_buf[: self.input_pos - 1] + self.input_buf[self.input_pos :]
            self.input_pos -= 1

    def _delete(self):
        if self.input_pos < len(self.input_buf):
            self.input_buf = self.input_buf[: self.input_pos] + self.input_buf[self.input_pos + 1 :]

    def _insert(self, text: str):
        self.input_buf = self.input_buf[: self.input_pos] + text + self.input_buf[self.input_pos :]
        self.input_pos += len(text)
        self.follow_tail = True

    def input_loop(self):
        self.stdscr.timeout(30)
        self.stdscr.keypad(True)
        curses.curs_set(1)

        running = True
        while running and not self.stop_event.is_set():
            if self.import_pending_telegram() and not self.state.generating:
                self.state.status = "Telegram prompt received"
                self.render()
                self.generate_reply()
                continue

            self.render()
            try:
                ch = self.stdscr.get_wch()
            except curses.error:
                ch = -1

            if ch == -1:
                continue

            if isinstance(ch, str):
                if ch in ("\n", "\r"):
                    text = self.input_buf
                    self.input_buf = ""
                    self.input_pos = 0
                    self.follow_tail = True
                    running = self.handle_command(text)
                elif ch in ("\x7f", "\b", "\x08"):
                    self._backspace()
                elif ch == "\x1b":
                    self.input_buf = ""
                    self.input_pos = 0
                elif ch == "\x15":
                    self.input_buf = ""
                    self.input_pos = 0
                elif ch == "\x0b":
                    self.input_buf = self.input_buf[: self.input_pos]
                elif ch == "\t":
                    self._insert("    ")
                elif ord(ch) >= 32:
                    self._insert(ch)
            else:
                if ch == curses.KEY_MOUSE:
                    try:
                        _, mx, my, _, mouse_state = curses.getmouse()
                        h, w = self.stdscr.getmaxyx()
                        header_h = 3
                        input_h = 3
                        body_h = max(5, h - header_h - input_h)
                        sidebar_w = 24 if w >= 92 else 0
                        main_w = max(20, w - sidebar_w)
                        in_chat = header_h <= my < header_h + body_h and mx < main_w
                        if in_chat:
                            if mouse_state & curses.BUTTON4_PRESSED:
                                self.follow_tail = False
                                self.scroll += 3
                            elif mouse_state & curses.BUTTON5_PRESSED:
                                self.scroll = max(0, self.scroll - 3)
                                if self.scroll == 0:
                                    self.follow_tail = True
                    except Exception:
                        pass
                elif ch == curses.KEY_LEFT and self.input_pos > 0:
                    self.input_pos -= 1
                elif ch == curses.KEY_RIGHT and self.input_pos < len(self.input_buf):
                    self.input_pos += 1
                elif ch == curses.KEY_HOME:
                    self.input_pos = 0
                elif ch == curses.KEY_END:
                    self.input_pos = len(self.input_buf)
                elif ch in (curses.KEY_BACKSPACE, 127):
                    self._backspace()
                elif ch == curses.KEY_DC:
                    self._delete()
                elif ch == curses.KEY_PPAGE:
                    self.follow_tail = False
                    self.scroll += max(1, self.stdscr.getmaxyx()[0] // 2)
                elif ch == curses.KEY_NPAGE:
                    self.scroll = max(0, self.scroll - max(1, self.stdscr.getmaxyx()[0] // 2))
                    if self.scroll == 0:
                        self.follow_tail = True
                elif ch == curses.KEY_UP:
                    self.follow_tail = False
                    self.scroll += 1
                elif ch == curses.KEY_DOWN:
                    self.scroll = max(0, self.scroll - 1)
                    if self.scroll == 0:
                        self.follow_tail = True

        self.stop_event.set()


# ----------------------------
# Curses setup
# ----------------------------


def init_colors():
    curses.start_color()
    curses.use_default_colors()

    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)  # default
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # panels/borders
    curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_BLACK)   # blue accent
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)  # user accent
    curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)    # error
    curses.init_pair(6, curses.COLOR_CYAN, curses.COLOR_BLACK)   # muted-ish


def main(stdscr):
    init_colors()
    curses.noecho()
    curses.cbreak()
    try:
        curses.meta(True)
    except TypeError:
        pass
    try:
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
    except Exception:
        pass
    stdscr.clear()
    stdscr.refresh()

    app = ElinTUI(stdscr)
    app.add_log("info", "Elin ready. Type /nc, /sc, /lc or a prompt.")
    app.render()
    try:
        app.input_loop()
    finally:
        app.stop_event.set()


if __name__ == "__main__":
    curses.wrapper(main)
