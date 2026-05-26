#!/usr/bin/env python3
import curses
import datetime as dt
import glob
import json
import locale
import os
import queue
import re
import select
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from openai import OpenAI

locale.setlocale(locale.LC_ALL, "")

ELIN_MODE = os.environ.get("ELIN_MODE", "local")
if ELIN_MODE == "cloud":
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    MODEL_NAME = "llama-3.3-70b-versatile"
elif ELIN_MODE == "github":
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=os.environ.get("GITHUB_TOKEN"),
    )
    MODEL_NAME = os.environ.get("GITHUB_MODEL", "openai/gpt-4.1")
else:
    client = OpenAI(
        base_url="http://localhost:8081/v1",
        api_key="sk-no-key-required",
        timeout=300.0,
    )
    MODEL_NAME = "local-model"

SEARXNG_URL = "http://172.17.0.1:8080/search"
MAX_CONTEXT_CHARS = 350000
MAX_VISIBLE_LOG = 500


def load_skills() -> str:
    skills_dir = os.path.expanduser("~/elin-agent/skills")
    if not os.path.exists(skills_dir):
        return ""
    parts = ["\n\n=== ADDITIONAL SKILLS ===\n"]
    for filename in sorted(os.listdir(skills_dir)):
        if filename.endswith((".md", ".txt")):
            path = os.path.join(skills_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    parts.append(f"\n[Skill: {filename}]\n{f.read()}\n")
            except Exception:
                pass
    return "".join(parts)


def estimate_tokens(text: str) -> int:
    """Best-effort token estimate; uses tiktoken if available."""
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

Be "human", nice, and helpful. Use punctuation freely (!, ?, <, >, etc.) but not more.
Follow ONLY the users request.
THINKING PROCESS:
- You MUST wrap your internal thoughts and plan inside <think> tags.
- Example: <think>I need to see the file first.</think> <call:exec>cat /path/to/file</call:exec>
- Anything outside <think> must be either a direct message to the user or a tool call (<call:exec>/<call:search>).
REASONING RULES:
- If the task is "simple" (chat), skip thinking.
- If the task involves "Code", "Fix", "Debug", or "Research", you MUST use <think>.
- You have 20 steps. Use them to verify your code works (e.g., <call:exec>python3 script.py</call:exec>).
Formatting: **bold**, __italic__, `code`, ```block```, ||spoiler||."""


def build_messages() -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT + load_skills()}]


@dataclass
class LogEntry:
    kind: str
    text: str
    tag: str = ""
    live: bool = False
    ts: float = field(default_factory=time.time)


class ElinTUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.messages = build_messages()
        self.log: List[LogEntry] = []
        self.input_buf = ""
        self.input_pos = 0
        self.scroll = 0
        self.spinner_idx = 0
        self.generating = False
        self.status = "Ready"
        self.last_action = "None"
        self.last_error = ""
        self.pending_telegram = queue.Queue()
        self.stop_event = threading.Event()
        self.telegram_thread = threading.Thread(target=self._telegram_poller, daemon=True)
        self.telegram_thread.start()
        self.max_context_chars = MAX_CONTEXT_CHARS
        self.model_name = MODEL_NAME
        self.mode = ELIN_MODE
        self.pending_permission = None

    def _telegram_poller(self):
        while not self.stop_event.is_set():
            try:
                resp = requests.get("http://localhost:8000/get_input", timeout=0.25)
                data = resp.json()
                text = (data or {}).get("text")
                if text:
                    self.pending_telegram.put(text)
            except Exception:
                pass
            time.sleep(0.65)

    def _send_to_telegram(self, text):
        try:
            requests.post("http://localhost:8000/speak", json={"text": text}, timeout=1)
        except Exception:
            pass

    def add_log(self, kind: str, text: str, tag: str = "", live: bool = False):
        if text is None:
            text = ""
        self.log.append(LogEntry(kind=kind, text=text, tag=tag, live=live))
        if len(self.log) > MAX_VISIBLE_LOG:
            self.log = self.log[-MAX_VISIBLE_LOG:]

    def update_log(self, index: int, text: Optional[str] = None, kind: Optional[str] = None):
        if 0 <= index < len(self.log):
            if text is not None:
                self.log[index].text = text
            if kind is not None:
                self.log[index].kind = kind

    def count_context(self):
        payload = json.dumps(self.messages, ensure_ascii=False)
        return len(payload), estimate_tokens(payload)

    def trim_history_if_needed(self):
        try:
            payload = json.dumps(self.messages, ensure_ascii=False)
            if len(payload) > self.max_context_chars:
                self.add_log(
                    "system",
                    f"History is {len(payload)} chars. Trimming to keep the prompt usable.",
                    tag="context",
                )
                self.messages = [self.messages[0]] + self.messages[-15:]
        except Exception as e:
            self.last_error = str(e)
            self.add_log("error", f"Trimmer error: {e}", tag="error")

    def draw_box(self, win):
        win.box()

    def color(self, name: str) -> int:
        return {
            "default": curses.color_pair(1),
            "grey": curses.color_pair(2),
            "blue": curses.color_pair(3),
            "accent": curses.color_pair(4),
            "danger": curses.color_pair(5),
            "muted": curses.color_pair(6),
        }.get(name, curses.color_pair(1))

    def wrap_text(self, text: str, width: int) -> List[str]:
        lines: List[str] = []
        for raw in text.splitlines() or [""]:
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
        return lines if lines else [""]

    def render_entry(self, win, y, x, width, entry: LogEntry):
        label = ""
        attr = self.color("default")
        if entry.kind == "user":
            label = "You"
            attr = self.color("accent") | curses.A_BOLD
        elif entry.kind in ("assistant", "assistant_live"):
            label = "Elin"
            attr = self.color("default")
        elif entry.kind in ("thinking", "thought"):
            label = "Thinking"
            attr = self.color("grey") | curses.A_DIM | getattr(curses, "A_ITALIC", 0)
        elif entry.kind in ("system", "info"):
            label = "System"
            attr = self.color("grey") | curses.A_DIM
        elif entry.kind in ("tool", "action"):
            label = "Action"
            attr = self.color("grey") | curses.A_BOLD
        elif entry.kind == "error":
            label = "Error"
            attr = self.color("danger") | curses.A_BOLD

        head = f"{label} > " if label else ""
        head_len = len(head)
        wrapped = self.wrap_text(entry.text, max(1, width - head_len))

        first = True
        for line in wrapped:
            if y >= win.getmaxyx()[0] - 1:
                break
            prefix = head if first else " " * head_len
            try:
                win.addnstr(y, x, prefix, width, attr)
                win.addnstr(y, x + head_len, line, max(0, width - head_len), attr)
            except curses.error:
                pass
            y += 1
            first = False
        return y

    def render(self):
        stdscr = self.stdscr
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        header_h = 8
        footer_h = 4
        sidebar_w = min(34, max(28, w // 5))
        chat_w = max(20, w - sidebar_w - 3)
        chat_h = max(5, h - header_h - footer_h - 1)

        # Header.
        header = stdscr.derwin(header_h, w, 0, 0)
        header.erase()
        header.attrset(self.color("grey") | curses.A_DIM)
        header.box()
        header.attrset(self.color("default"))
        logo_lines = [
            "███████╗██╗     ██╗███╗   ██╗",
            "██╔════╝██║     ██║████╗  ██║",
            "█████╗  ██║     ██║██╔██╗ ██║",
            "██╔══╝  ██║     ██║██║╚██╗██║",
            "███████╗███████╗██║██║ ╚████║",
            "╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝",
        ]
        logo_x = max(2, (w - len(logo_lines[0])) // 2)
        for i, line in enumerate(logo_lines[:6]):
            if 1 + i >= header_h - 1:
                break
            try:
                header.addstr(1 + i, logo_x, line[: max(0, w - logo_x - 2)], self.color("blue") | curses.A_BOLD)
            except curses.error:
                pass
        subtitle = f"Elin TUI  <>  {self.mode} mode"
        try:
            header.addnstr(header_h - 2, 2, subtitle, w - 4, self.color("grey") | curses.A_DIM)
        except curses.error:
            pass

        # Sidebar.
        sb_x = w - sidebar_w
        sidebar = stdscr.derwin(h - header_h - footer_h, sidebar_w, header_h, sb_x)
        sidebar.erase()
        sidebar.attrset(self.color("grey") | curses.A_DIM)
        sidebar.box()
        sidebar.attrset(self.color("default"))

        ctx_chars, ctx_tokens = self.count_context()
        used_pct = min(100, int(ctx_chars * 100 / self.max_context_chars))
        pending = self.pending_telegram.qsize()
        stats = [
            ("Mode", self.mode),
            ("Model", self.model_name),
            ("Msgs", str(max(0, len(self.messages) - 1))),
            ("Context", f"{ctx_chars:,} chars"),
            ("Tokens", f"{ctx_tokens:,} est."),
            ("Used", f"{used_pct}%"),
            ("Telegram", str(pending)),
            ("Action", self.last_action),
        ]
        y = 1
        try:
            sidebar.addstr(y, 2, "STATUS", self.color("blue") | curses.A_BOLD)
        except curses.error:
            pass
        y += 2
        for k, v in stats:
            line = f"{k:<8} {v}"
            try:
                sidebar.addnstr(y, 2, line, sidebar_w - 4, self.color("grey") | curses.A_DIM)
            except curses.error:
                pass
            y += 1

        if self.generating:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            spin = spinner[self.spinner_idx % len(spinner)]
            status_line = f"{spin} generating..."
            attr = self.color("blue") | curses.A_BOLD
        else:
            status_line = self.status
            attr = self.color("grey") | curses.A_DIM

        try:
            sidebar.addnstr(y + 1, 2, "STATE", sidebar_w - 4, self.color("blue") | curses.A_BOLD)
            sidebar.addnstr(y + 3, 2, status_line[: sidebar_w - 4], sidebar_w - 4, attr)
        except curses.error:
            pass

        if self.last_error:
            try:
                sidebar.addnstr(h - header_h - footer_h - 3, 2, "LAST ERROR", sidebar_w - 4, self.color("danger") | curses.A_BOLD)
                sidebar.addnstr(h - header_h - footer_h - 2, 2, self.last_error[: sidebar_w - 4], sidebar_w - 4, self.color("danger") | curses.A_DIM)
            except curses.error:
                pass

        # Chat pane.
        chat = stdscr.derwin(h - header_h - footer_h, chat_w, header_h, 0)
        chat.erase()
        chat.attrset(self.color("grey") | curses.A_DIM)
        chat.box()
        chat.attrset(self.color("default"))
        usable_w = max(10, chat_w - 4)

        visible_lines = []
        for entry in self.log:
            visible_lines.append(entry)

        # Render from the bottom with scroll support.
        line_budget = max(1, h - header_h - footer_h - 2)
        rendered_blocks: List[List[str]] = []
        for entry in visible_lines:
            label = ""
            if entry.kind in ("user",):
                label = "You > "
            elif entry.kind in ("assistant", "assistant_live"):
                label = "Elin > "
            elif entry.kind in ("thinking", "thought"):
                label = "Thinking > "
            elif entry.kind in ("tool", "action"):
                label = "Action > "
            elif entry.kind in ("system", "info"):
                label = "System > "
            elif entry.kind == "error":
                label = "Error > "
            wrapped = self.wrap_text(entry.text, max(1, usable_w - len(label)))
            rendered_blocks.append(wrapped)

        flattened = []
        for idx, entry in enumerate(visible_lines):
            label = ""
            attr = self.color("default")
            if entry.kind == "user":
                label = "You > "
                attr = self.color("accent") | curses.A_BOLD
            elif entry.kind in ("assistant", "assistant_live"):
                label = "Elin > "
                attr = self.color("default")
            elif entry.kind in ("thinking", "thought"):
                label = "Thinking > "
                attr = self.color("grey") | curses.A_DIM | getattr(curses, "A_ITALIC", 0)
            elif entry.kind in ("system", "info"):
                label = "System > "
                attr = self.color("grey") | curses.A_DIM
            elif entry.kind in ("tool", "action"):
                label = "Action > "
                attr = self.color("grey") | curses.A_BOLD
            elif entry.kind == "error":
                label = "Error > "
                attr = self.color("danger") | curses.A_BOLD
            wrapped = self.wrap_text(entry.text, max(1, usable_w - len(label)))
            for i, line in enumerate(wrapped):
                flattened.append((label if i == 0 else " " * len(label), line, attr))

        if self.scroll > max(0, len(flattened) - line_budget):
            self.scroll = max(0, len(flattened) - line_budget)
        start = max(0, len(flattened) - line_budget - self.scroll)
        end = len(flattened) - self.scroll

        y = 1
        for prefix, line, attr in flattened[start:end]:
            try:
                chat.addnstr(y, 2, prefix, usable_w, attr)
                chat.addnstr(y, 2 + len(prefix), line, max(0, usable_w - len(prefix)), attr)
            except curses.error:
                pass
            y += 1
            if y >= chat.getmaxyx()[0] - 1:
                break

        # Input pane.
        footer = stdscr.derwin(footer_h, w, h - footer_h, 0)
        footer.erase()
        footer.attrset(self.color("grey") | curses.A_DIM)
        footer.box()
        footer.attrset(self.color("default"))
        prompt = "You > "
        try:
            footer.addnstr(1, 2, prompt, w - 4, self.color("accent") | curses.A_BOLD)
            footer.addnstr(1, 2 + len(prompt), self.input_buf, max(0, w - len(prompt) - 6), self.color("default"))
        except curses.error:
            pass

        # Cursor.
        cur_x = min(w - 3, 2 + len(prompt) + self.input_pos)
        try:
            stdscr.move(h - 3, cur_x)
        except curses.error:
            pass

        stdscr.noutrefresh()
        header.noutrefresh()
        sidebar.noutrefresh()
        chat.noutrefresh()
        footer.noutrefresh()
        curses.doupdate()

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

        self.last_action = f"exec: {command[:22]}"
        self.status = f"Running command: {command}"
        self.add_log("action", f"EXEC: {command}")

        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}".strip()
            return output if output else "SYSTEM MESSAGE: command finished with no output."
        except Exception as e:
            self.last_error = str(e)
            return f"Command error: {e}"

    def _search_web(self, query: str) -> str:
        self.last_action = f"search: {query[:22]}"
        self.status = f"Searching web: {query}"
        self.add_log("action", f"SEARCH: {query}")
        try:
            params = {"q": query, "format": "json"}
            resp = requests.get(SEARXNG_URL, params=params, timeout=10)
            results = resp.json().get("results", [])[:3]
            if not results:
                return "No search results."
            parts = []
            for r in results:
                parts.append(f"Source: {r.get('title', 'untitled')}\nContent: {r.get('content', '')}")
            return "\n\n".join(parts)
        except Exception as e:
            self.last_error = str(e)
            return f"Search error: {e}"

    def permission_modal(self, title: str, body: str) -> bool:
        h, w = self.stdscr.getmaxyx()
        mw = min(w - 6, 88)
        mh = min(h - 6, 14)
        y = (h - mh) // 2
        x = (w - mw) // 2
        win = self.stdscr.derwin(mh, mw, y, x)
        win.keypad(True)
        win.bkgd(" ", self.color("default"))
        win.attrset(self.color("grey") | curses.A_DIM)
        win.box()
        win.attrset(self.color("default"))
        try:
            win.addnstr(1, 2, title, mw - 4, self.color("blue") | curses.A_BOLD)
        except curses.error:
            pass

        lines = self.wrap_text(body, mw - 4)
        yy = 3
        for line in lines[: mh - 6]:
            try:
                win.addnstr(yy, 2, line, mw - 4, self.color("grey") | curses.A_DIM)
            except curses.error:
                pass
            yy += 1
            if yy >= mh - 3:
                break

        footer = "y = allow   n = deny"
        try:
            win.addnstr(mh - 2, 2, footer, mw - 4, self.color("blue") | curses.A_BOLD)
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

    def push_user_message(self, text: str):
        text = text.strip()
        if not text:
            return
        self.add_log("user", text)
        self.messages.append({"role": "user", "content": text})

    def save_chat(self) -> str:
        mem_dir = os.path.expanduser("~/elin-agent/memories")
        os.makedirs(mem_dir, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filename = f"chatf-{timestamp}.json"
        save_path = os.path.join(mem_dir, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)
        return filename

    def load_chat(self, target_file: Optional[str] = None) -> str:
        mem_dir = os.path.expanduser("~/elin-agent/memories")
        if target_file:
            target_file = target_file.strip().strip("'\"")
            if not os.path.exists(target_file):
                raise FileNotFoundError(f"File not found: {target_file}")
        else:
            files = glob.glob(os.path.join(mem_dir, "chatf-*.json"))
            if not files:
                raise FileNotFoundError("No chatf-*.json files found in memories.")
            target_file = max(files, key=os.path.getctime)

        with open(target_file, "r", encoding="utf-8") as f:
            self.messages = json.load(f)
        return os.path.basename(target_file)

    def _model_worker(self, q: queue.Queue, messages_snapshot: list):
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
        self.status = "Generating..."
        self.generating = True
        self.spinner_idx = 0

        # Inject Notepad into the turn
        notepad_ctx = get_notepad_context()
        if notepad_ctx:
            if self.messages:
                self.messages[-1]["content"] += notepad_ctx

        messages_snapshot = list(self.messages)
        q: queue.Queue = queue.Queue()
        worker = threading.Thread(target=self._model_worker, args=(q, messages_snapshot), daemon=True)
        worker.start()

        thinking = ""
        answer = ""
        thinking_idx = None
        answer_idx = None

        # live entries
        thinking_idx = len(self.log)
        self.add_log("thinking", "", live=True)
        answer_idx = len(self.log)
        self.add_log("assistant_live", "", live=True)

        done = False
        while not done:
            self.spinner_idx = (self.spinner_idx + 1) % 10
            try:
                event, payload = q.get(timeout=0.06)
            except queue.Empty:
                self.render()
                continue

            if event == "thinking":
                thinking += payload
                self.update_log(thinking_idx, thinking)
            elif event == "content":
                answer += payload
                self.update_log(answer_idx, answer)
            elif event == "done":
                done = True
            elif event == "error":
                self.generating = False
                self.update_log(answer_idx, f"Model error: {payload}", kind="error")
                self.last_error = payload
                self.render()
                return
            self.render()

        self.generating = False
        self.status = "Ready"

        final_answer = strip_think_tags(answer)
        if not final_answer and thinking:
            final_answer = strip_think_tags(thinking)

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
            self.add_log("system", f"Command output captured:\n{output}")
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
            self.status = "Goal Updated"
            self.generate_reply()
            return

        if plan_match:
            data = load_notepad()
            data["plan"] = [s.strip() for s in plan_match.group(1).strip().split("\n") if s.strip()]
            save_notepad(data)
            self.messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Notepad Plan Updated."})
            self.status = "Plan Updated"
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
                self.status = f"Drafted {fname.strip()}"
            else:
                self.messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Error - Draft must be in format 'filename: content'"})
            self.generate_reply()
            return

    def handle_command(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True

        if stripped == "/nc":
            if os.path.exists(NOTEPAD_PATH): os.remove(NOTEPAD_PATH)
            self.messages = build_messages()
            self.log.clear()
            self.status = "New chat started"
            self.add_log("info", "new chat started")
            return True

        if stripped.startswith("/goal "):
            goal_text = stripped[6:].strip()
            msg = f"[GOAL PROTOCOL INITIATED]\nTarget: {goal_text}\n\n1. Enhance this goal.\n2. Create a plan.\n3. Start drafting.\nUse your NOTEPAD tools."
            self.push_user_message(msg)
            self.render()
            self.generate_reply()
            return True

        if stripped == "/sc":
            filename = self.save_chat()
            self.status = f"Saved {filename}"
            self.add_log("info", f"Saved: {filename}")
            return True

        if stripped.startswith("/lc"):
            parts = stripped.split(maxsplit=1)
            try:
                filename = self.load_chat(parts[1] if len(parts) > 1 else None)
                self.status = f"Loaded {filename}"
                self.add_log("info", f"Loaded: {filename}")
            except Exception as e:
                self.last_error = str(e)
                self.add_log("error", str(e))
            return True

        if stripped in ("/quit", "/exit"):
            return False

        self.push_user_message(stripped)
        self.render()
        self.generate_reply()
        return True

    def import_pending_telegram(self):
        should_generate = False
        while True:
            try:
                text = self.pending_telegram.get_nowait()
            except queue.Empty:
                break
            self.add_log("user", text, tag="telegram")
            if text.strip().startswith("/"):
                self.handle_command(text.strip())
            else:
                self.messages.append({"role": "user", "content": text})
                should_generate = True
        return should_generate

    def input_loop(self):
        self.stdscr.timeout(30)
        self.stdscr.keypad(True)
        curses.curs_set(1)
        
        running = True
        while running and not self.stop_event.is_set():
            if self.import_pending_telegram() and not self.generating:
                self.status = "Telegram prompt received"
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
                    self.render()
                    running = self.handle_command(text)
                elif ch in ("\x7f", "\b", "\x08"):
                    if self.input_pos > 0:
                        self.input_buf = self.input_buf[: self.input_pos - 1] + self.input_buf[self.input_pos :]
                        self.input_pos -= 1
                elif ch == "\x1b":
                    self.input_buf = ""
                    self.input_pos = 0
                elif ch == "\x15":  # Ctrl+U
                    self.input_buf = ""
                    self.input_pos = 0
                elif ch == "\x0b":  # Ctrl+K
                    self.input_buf = self.input_buf[: self.input_pos]
                elif ch == "\t":
                    self.input_buf = self.input_buf[: self.input_pos] + "    " + self.input_buf[self.input_pos :]
                    self.input_pos += 4
                elif ord(ch) >= 32:
                    self.input_buf = self.input_buf[: self.input_pos] + ch + self.input_buf[self.input_pos :]
                    self.input_pos += len(ch)
            else:
                if ch == curses.KEY_LEFT and self.input_pos > 0:
                    self.input_pos -= 1
                elif ch == curses.KEY_RIGHT and self.input_pos < len(self.input_buf):
                    self.input_pos += 1
                elif ch == curses.KEY_HOME:
                    self.input_pos = 0
                elif ch == curses.KEY_END:
                    self.input_pos = len(self.input_buf)
                elif ch == curses.KEY_BACKSPACE:
                    if self.input_pos > 0:
                        self.input_buf = self.input_buf[: self.input_pos - 1] + self.input_buf[self.input_pos :]
                        self.input_pos -= 1
                elif ch == curses.KEY_DC:
                    if self.input_pos < len(self.input_buf):
                        self.input_buf = self.input_buf[: self.input_pos] + self.input_buf[self.input_pos + 1 :]
                elif ch == curses.KEY_PPAGE:
                    self.scroll += max(1, (self.stdscr.getmaxyx()[0] // 2))
                elif ch == curses.KEY_NPAGE:
                    self.scroll = max(0, self.scroll - max(1, (self.stdscr.getmaxyx()[0] // 2)))
                elif ch == curses.KEY_UP:
                    self.scroll += 1
                elif ch == curses.KEY_DOWN:
                    self.scroll = max(0, self.scroll - 1)

        self.stop_event.set()


def init_colors():
    curses.start_color()
    curses.use_default_colors()

    # pairs: fg, bg
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)   # default
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)   # grey-ish / borders
    curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_BLACK)     # blue accents
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)    # accent for user
    curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)      # errors
    curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_BLACK)    # muted fallback


def main(stdscr):
    init_colors()
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
