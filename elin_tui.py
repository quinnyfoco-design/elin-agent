#!/usr/bin/env python3
import curses
import datetime as dt
import glob
import json
import locale
import os
import queue
import re
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

# --- Configuration ---
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
MAX_CONTEXT_CHARS = 600000
MAX_VISIBLE_LOG = 500

STATUSBAR_H = 1
INPUT_H = 2
BG = -1

# --- Helpers ---
def load_skills() -> str:
    skills_dir = os.path.expanduser("~/elin-agent/skills")
    if not os.path.exists(skills_dir):
        return ""
    parts = ["\n\n=== ADDITIONAL SKILLS ===\n"]
    for filename in sorted(os.listdir(skills_dir)):
        if filename.endswith((".md", ".txt")):
            path = os.path.join(skills_dir, filename)
            try:
                with open(path, "r") as f:
                    parts.append(f"\n[Skill: {filename}]\n{f.read()}\n")
            except Exception:
                pass
    return "".join(parts)

def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
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

# --- Prompts ---
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
- You have 20 steps. Use them to verify your code works (e.g., <call:exec>python3 script.py</call:exec>)."""

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
        self.last_action = ""
        self.last_error = ""
        self.pending_telegram = queue.Queue()
        self.stop_event = threading.Event()
        self.telegram_thread = threading.Thread(target=self._telegram_poller, daemon=True)
        self.telegram_thread.start()
        self.max_context_chars = MAX_CONTEXT_CHARS
        self.model_name = MODEL_NAME
        self.mode = ELIN_MODE
        self.pending_permission = None
        # Cache for streaming display
        self._reply_thinking = ""
        self._reply_answer = ""

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
        if text is None: text = ""
        self.log.append(LogEntry(kind=kind, text=text, tag=tag, live=live))
        if len(self.log) > MAX_VISIBLE_LOG:
            self.log = self.log[-MAX_VISIBLE_LOG:]

    def update_log(self, index: int, text: Optional[str] = None, kind: Optional[str] = None):
        if 0 <= index < len(self.log):
            if text is not None: self.log[index].text = text
            if kind is not None: self.log[index].kind = kind

    def count_context(self):
        payload = json.dumps(self.messages, ensure_ascii=False)
        return len(payload), estimate_tokens(payload)

    def trim_history_if_needed(self):
        try:
            payload = json.dumps(self.messages, ensure_ascii=False)
            if len(payload) > self.max_context_chars:
                self.add_log("system", f"History trimmed ({len(payload)} chars).", tag="context")
                self.messages = [self.messages[0]] + self.messages[-15:]
        except Exception as e:
            self.last_error = str(e)
            self.add_log("error", f"Trimmer error: {e}", tag="error")

    def color(self, name: str) -> int:
        return {
            "default": curses.color_pair(1), # White
            "muted": curses.color_pair(2),
            "accent": curses.color_pair(3),
            "user": curses.color_pair(4),    # Dark Blue
            "danger": curses.color_pair(5),
            "meta": curses.color_pair(6),    # White (Borders/Actions)
            "think": curses.color_pair(7),   # Light Grey
        }.get(name, curses.color_pair(1))

    def wrap_text(self, text: str, width: int) -> List[str]:
        lines: List[str] = []
        for raw in text.splitlines() or [""]:
            if not raw.strip():
                lines.append("")
                continue
            wrapped = textwrap.wrap(
                raw, width=width, replace_whitespace=False, drop_whitespace=False,
                break_long_words=True, break_on_hyphens=False
            )
            lines.extend(wrapped if wrapped else [""])
        return lines if lines else [""]

    def render(self):
        stdscr = self.stdscr
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        chat_bottom = h - STATUSBAR_H - INPUT_H
        chat_h = max(1, chat_bottom)

        ctx_chars, ctx_tokens = self.count_context()
        used_pct = min(100, int(ctx_chars * 100 / self.max_context_chars))
        pending = self.pending_telegram.qsize()

        ITALIC = getattr(curses, 'A_ITALIC', curses.A_DIM)

        # ── Chat pane ──
        chat = stdscr.derwin(chat_h, w, 0, 0)
        chat.erase()
        usable_w = max(10, w - 4)

        flattened = []
        prev_kind = None
        for entry in self.log:
            prefix_char = " "
            # Default AI/System attribute
            attr = self.color("default")
            is_assistant = False
            
            if entry.kind == "user":
                prefix_char = "\u2503"
                attr = self.color("user") | curses.A_BOLD # User line: Dark Blue & Fat
            elif entry.kind in ("assistant", "assistant_live"):
                prefix_char = "\u2503"
                attr = self.color("default") | curses.A_BOLD # AI Default: White & Fat
                is_assistant = True
            elif entry.kind in ("system", "info"):
                attr = self.color("default")
            elif entry.kind in ("tool", "action"):
                prefix_char = "\u2503"
                attr = self.color("default")
            elif entry.kind == "error":
                prefix_char = "\u2503"
                attr = self.color("danger")

            if prev_kind == "user" and is_assistant:
                flattened.append((" ", "", self.color("default")))
            prev_kind = entry.kind

            md_lines = entry.text.split("\n")
            think_mode = False
            first_line = True
            
            for md_line in md_lines:
                if not md_line.strip() and not first_line:
                    flattened.append((" ", "", attr))
                    continue
                
                wrapped = self.wrap_text(md_line, usable_w - 2) if md_line.strip() else [""]
                for wl in wrapped:
                    # Logic to switch between thinking and answer
                    if is_assistant and wl.startswith("\u2517") and not think_mode:
                        think_mode = True
                        wl = wl[1:]
                        if first_line:
                            flattened.append((f"{prefix_char}   \u22ef ", "", self.color("default")))
                    
                    # If we find a closing marker
                    if think_mode and "\u2517" in wl:
                        parts = wl.split("\u2517", 1)
                        if parts[0]:
                            flattened.append((f"{prefix_char}  ", parts[0], self.color("think") | ITALIC))
                        think_mode = False
                        wl = parts[1] if len(parts) > 1 else ""
                        if not wl: continue

                    line_attr = self.color("think") | ITALIC if (is_assistant and think_mode) else attr
                    indent = "  " if is_assistant else " "
                    
                    if first_line:
                        flattened.append((f"{prefix_char}{indent}", wl, line_attr))
                        first_line = False
                    else:
                        flattened.append((f"{prefix_char}{indent}", wl, line_attr))

        # Scroll logic
        line_budget = max(1, chat_h)
        if self.scroll > max(0, len(flattened) - line_budget):
            self.scroll = max(0, len(flattened) - line_budget)
        start_idx = max(0, len(flattened) - line_budget - self.scroll)
        end_idx = len(flattened) - self.scroll

        y = 0
        for prefix, line, line_attr in flattened[start_idx:end_idx]:
            if y >= chat_h: break
            try:
                chat.addnstr(y, 0, prefix, 2, self.color("default"))
                chat.addnstr(y, 2, line, usable_w, line_attr)
            except curses.error: pass
            y += 1

        # ── Input area ──
        footer = stdscr.derwin(INPUT_H, w, h - INPUT_H - STATUSBAR_H, 0)
        footer.erase()
        input_width = max(10, w - 6)
        display_text = self.input_buf if self.input_buf else "type a message..."
        display_attr = self.color("default") if self.input_buf else self.color("muted")
        
        try:
            # Border stays white
            footer.addnstr(0, 2, "\u2501" * (w - 4), w - 4, self.color("default"))
            # Prompt: Dark Blue & Fat
            footer.addnstr(1, 2, "> ", w, self.color("user") | curses.A_BOLD)
            footer.addnstr(1, 4, display_text, input_width, display_attr)
            # Hints stay white
            hints = "/nc /sc /lc /quit"
            footer.addnstr(1, w - len(hints) - 2, hints, w, self.color("default"))
        except curses.error: pass

        cur_x = min(w - 3, 4 + self.input_pos)
        try: stdscr.move(h - STATUSBAR_H - INPUT_H + 1, cur_x)
        except curses.error: pass

        # ── Status bar ──
        statusbar = stdscr.derwin(STATUSBAR_H, w, h - STATUSBAR_H, 0)
        statusbar.erase()
        left_text = f"  {self.status}"
        if self.generating:
            spinner = "\u2812\u2832\u28b2\u28b6\u28be\u28de\u28ee\u28e6\u28e2\u28c2"
            spin = spinner[self.spinner_idx % len(spinner)]
            left_text = f"  {spin} generating..."
        
        try:
            statusbar.addnstr(0, 0, left_text, w, self.color("default"))
            mid = f" {used_pct}%  {ctx_tokens}k"
            statusbar.addnstr(0, w // 2 - len(mid) // 2, mid, w, self.color("default"))
            right = f"\u25a3 {self.model_name[:18]} "
            if pending: right = f"tg:{pending}  {right}"
            statusbar.addnstr(0, w - len(right) - 1, right, w, self.color("default"))
        except curses.error: pass

        stdscr.noutrefresh()
        chat.noutrefresh()
        footer.noutrefresh()
        statusbar.noutrefresh()
        curses.doupdate()

    def _run_shell_command(self, command: str) -> str:
        command = format_command_text(command)
        risky = ["rm", "dd", "mkfs", ">", "sudo", "systemctl"]
        if any(r in command for r in risky):
            if not self.permission_modal("Permission required", f"Run risky command:\n\n{command}"):
                return "SYSTEM MESSAGE: User denied execution."
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return (result.stdout + result.stderr).strip() or "(no output)"
        except Exception as e: return f"Error: {e}"

    def _search_web(self, query: str) -> str:
        self.add_log("action", f"SEARCH: {query}")
        try:
            resp = requests.get(SEARXNG_URL, params={"q": query, "format": "json"}, timeout=10)
            results = resp.json().get("results", [])[:3]
            return "\n\n".join([f"Source: {r.get('title')}\n{r.get('content')}" for r in results]) or "No results."
        except Exception as e: return f"Search error: {e}"

    def permission_modal(self, title: str, body: str) -> bool:
        h, w = self.stdscr.getmaxyx()
        mw, mh = min(w - 8, 64), min(h - 8, 10)
        win = self.stdscr.derwin(mh, mw, (h - mh) // 2, (w - mw) // 2)
        win.keypad(True)
        win.erase()
        win.box()
        win.addstr(1, 2, title, curses.A_BOLD)
        lines = self.wrap_text(body, mw - 4)
        for i, line in enumerate(lines[:mh-4]):
            win.addstr(3 + i, 2, line)
        win.addstr(mh - 2, 2, "y = allow    n = deny", self.color("muted"))
        win.refresh()
        while True:
            ch = win.getch()
            if ch in (ord("y"), ord("Y")): return True
            if ch in (ord("n"), ord("N"), 27): return False

    def push_user_message(self, text: str):
        if not text.strip(): return
        self.add_log("user", text)
        self.messages.append({"role": "user", "content": text})

    def save_chat(self) -> str:
        path = os.path.expanduser(f"~/elin-agent/memories/chatf-{dt.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f: json.dump(self.messages, f, indent=2)
        return os.path.basename(path)

    def load_chat(self, target: Optional[str] = None) -> str:
        mem_dir = os.path.expanduser("~/elin-agent/memories")
        if not target:
            files = glob.glob(os.path.join(mem_dir, "chatf-*.json"))
            if not files: raise FileNotFoundError("No saves found")
            target = max(files, key=os.path.getctime)
        with open(target, "r") as f: self.messages = json.load(f)
        return os.path.basename(target)

    def _model_worker(self, q: queue.Queue, messages_snapshot: list):
        try:
            stream = client.chat.completions.create(model=self.model_name, messages=messages_snapshot, temperature=0.5, stream=True)
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    q.put(("thinking", delta.reasoning_content))
                if delta.content:
                    q.put(("content", delta.content))
            q.put(("done", None))
        except Exception as e: q.put(("error", str(e)))

    def generate_reply(self):
        self.trim_history_if_needed()
        self.status = "Generating..."
        self.generating = True
        q = queue.Queue()
        
        # Inject Notepad into the turn
        notepad_ctx = get_notepad_context()
        if notepad_ctx:
            # We append it to the LAST message in the history snapshot
            # This ensures the model always sees the current state of the notepad
            if self.messages:
                self.messages[-1]["content"] += notepad_ctx

        threading.Thread(target=self._model_worker, args=(q, list(self.messages)), daemon=True).start()

        thinking, answer = "", ""
        answer_idx = len(self.log)
        self.add_log("assistant_live", "")

        while True:
            self.spinner_idx = (self.spinner_idx + 1) % 10
            try: event, payload = q.get(timeout=0.06)
            except queue.Empty: self.render(); continue

            if event == "thinking":
                thinking += payload
                display = f"\u2517{thinking}\u2517\n{answer}" if answer else f"\u2517{thinking}\u2517"
                self.update_log(answer_idx, display)
            elif event == "content":
                answer += payload
                display = f"\u2517{thinking}\u2517\n{answer}" if thinking else answer
                self.update_log(answer_idx, display)
            elif event == "done": break
            elif event == "error":
                self.update_log(answer_idx, f"Error: {payload}", kind="error")
                break
            self.render()

        self.generating, self.status = False, "Ready"
        final_answer = strip_think_tags(answer) or strip_think_tags(thinking)
        self.messages.append({"role": "assistant", "content": final_answer})
        self._send_to_telegram(final_answer)
        
        exec_match = re.search(r'<call:exec>(.*?)</call:exec>', final_answer, re.DOTALL | re.IGNORECASE)
        search_match = re.search(r'<call:search>(.*?)</call:search>', final_answer, re.DOTALL | re.IGNORECASE)
        
        # Notepad Matches
        goal_match = re.search(r'<call:notepad_goal>(.*?)</call:notepad_goal>', final_answer, re.DOTALL | re.IGNORECASE)
        plan_match = re.search(r'<call:notepad_plan>(.*?)</call:notepad_plan>', final_answer, re.DOTALL | re.IGNORECASE)
        draft_match = re.search(r'<call:notepad_draft>(.*?)</call:notepad_draft>', final_answer, re.DOTALL | re.IGNORECASE)
        expert_match = re.search(r'<call:expert_help>(.*?)</call:expert_help>', final_answer, re.DOTALL | re.IGNORECASE)

        if exec_match:
            cmd = exec_match.group(1).strip()
            out = self._run_shell_command(cmd)
            self.messages.append({"role": "user", "content": f"SYSTEM_OBSERVATION (EXEC):\n{out}\n[End of output. Please evaluate and continue.]"})
            self.generate_reply()
        elif search_match:
            query = search_match.group(1).strip()
            out = self._search_web(query)
            self.messages.append({"role": "user", "content": f"SYSTEM_OBSERVATION (SEARCH):\n{out}\n[End of output. Please evaluate and continue.]"})
            self.generate_reply()
        elif expert_match:
            content = expert_match.group(1).strip()
            if ":" in content:
                ext_id, ext_query = content.split(":", 1)
                res = call_expert_model(ext_id.strip(), ext_query.strip())
                self.messages.append({"role": "user", "content": f"SYSTEM_OBSERVATION (EXPERT HELP from {ext_id.strip()}):\n{res}\n[End of expert response. Evaluate and continue.]"})
            else:
                self.messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Error - expert_help must be 'expert: query'"})
            self.generate_reply()
        elif goal_match:
            data = load_notepad()
            data["goal"] = goal_match.group(1).strip()
            save_notepad(data)
            self.messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Notepad Goal Updated."})
            self.status = "Goal Updated"
            self.generate_reply()
        elif plan_match:
            data = load_notepad()
            data["plan"] = [s.strip() for s in plan_match.group(1).strip().split("\n") if s.strip()]
            save_notepad(data)
            self.messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Notepad Plan Updated."})
            self.status = "Plan Updated"
            self.generate_reply()
        elif draft_match:
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
        self.render()


    def handle_command(self, text: str) -> bool:
        cmd = text.strip()
        if not cmd: return True
        if cmd == "/nc":
            if os.path.exists(NOTEPAD_PATH): os.remove(NOTEPAD_PATH)
            self.messages, self.log = build_messages(), []
            return True
        if cmd == "/sc": self.status = f"Saved: {self.save_chat()}"; return True
        if cmd.startswith("/lc"):
            try: self.status = f"Loaded: {self.load_chat(cmd.split()[1] if len(cmd.split())>1 else None)}"
            except Exception as e: self.last_error = str(e)
            return True
        if cmd.startswith("/goal "):
            goal_text = cmd[6:].strip()
            msg = f"[GOAL PROTOCOL INITIATED]\nTarget: {goal_text}\n\n1. Enhance this goal.\n2. Create a plan.\n3. Start drafting.\nUse your NOTEPAD tools."
            self.push_user_message(msg)
            self.generate_reply()
            return True
        if cmd in ("/quit", "/exit"): return False
        self.push_user_message(cmd)
        self.generate_reply()
        return True

    def input_loop(self):
        self.stdscr.timeout(30)
        while not self.stop_event.is_set():
            # Handle telegram input
            while not self.pending_telegram.empty():
                self.handle_command(self.pending_telegram.get_nowait())
            
            self.render()
            try: ch = self.stdscr.get_wch()
            except: ch = -1
            if ch == -1: continue

            if isinstance(ch, str):
                if ch in ("\n", "\r"):
                    buf = self.input_buf; self.input_buf, self.input_pos = "", 0
                    if not self.handle_command(buf): break
                elif ch in ("\x7f", "\b"):
                    if self.input_pos > 0:
                        self.input_buf = self.input_buf[:self.input_pos-1] + self.input_buf[self.input_pos:]
                        self.input_pos -= 1
                elif ord(ch) >= 32:
                    self.input_buf = self.input_buf[:self.input_pos] + ch + self.input_buf[self.input_pos:]
                    self.input_pos += 1
            else:
                if ch == curses.KEY_LEFT: self.input_pos = max(0, self.input_pos - 1)
                elif ch == curses.KEY_RIGHT: self.input_pos = min(len(self.input_buf), self.input_pos + 1)
                elif ch == curses.KEY_UP: self.scroll += 1
                elif ch == curses.KEY_DOWN: self.scroll = max(0, self.scroll - 1)

def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, 231, BG) # Default White
    curses.init_pair(2, 244, BG) # Muted
    curses.init_pair(3, 183, BG) # Accent
    curses.init_pair(4, 27,  BG) # User/Prompt: Dark Blue
    curses.init_pair(5, 204, BG) # Danger
    curses.init_pair(6, 231, BG) # Borders/Actions: White
    curses.init_pair(7, 248, BG) # AI Thinking: Light Grey

def main(stdscr):
    init_colors()
    curses.curs_set(1)
    app = ElinTUI(stdscr)
    app.render()
    app.input_loop()

if __name__ == "__main__":
    curses.wrapper(main)
