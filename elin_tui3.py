#!/usr/bin/env python3
import curses
import datetime as dt
import glob
import json
import locale
import os
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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

LOCAL_API_BASE = "http://localhost:8081"
SEARXNG_URL = "http://172.17.0.1:8080/search"
MAX_VISIBLE_LOG = 500
CONTEXT_TRIM_RATIO = 0.75
MAX_CMD_OUTPUT_CHARS = 15000

STATUS_BAR_H = 1
INPUT_PANE_H = 3
BODY_INDENT = 4
MARKDOWN_INLINE_RE = re.compile(
    r"(\*\*.+?\*\*|__.+?__|`[^`\n]+`|\|\|.+?\|\|)",
    re.DOTALL,
)
TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


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
    """Exact token count via local llama-server /tokenize endpoint."""
    text = text or ""
    try:
        resp = requests.post(
            f"{LOCAL_API_BASE}/tokenize",
            json={"content": text},
            timeout=5,
        )
        if resp.ok:
            return len(resp.json().get("tokens", []))
    except Exception:
        pass
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def fetch_context_size() -> int:
    """Fetch n_ctx from the local llama-server, retrying until ready."""
    for attempt in range(15):
        try:
            resp = requests.get(f"{LOCAL_API_BASE}/v1/models", timeout=5)
            if resp.ok:
                data = resp.json().get("data", [])
                if data:
                    val = data[0].get("meta", {}).get("n_ctx", 16384)
                    return val
        except Exception:
            pass
        time.sleep(2)
    return 16384


def strip_think_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()

def clean_terminal_output(text: str) -> str:
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    text = re.sub(r'\x1b\][0-9;]*(?:\x1b\\|\x07)', '', text)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        parts = line.split('\r')
        cleaned.append(parts[-1])
    text = '\n'.join(cleaned)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    lines = [l.rstrip() for l in text.splitlines()]
    text = '\n'.join(lines)
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

SYSTEM_PROMPT = """SYSTEM: You're Elin, a Linux local AI (Elin 35B by germanphoneguy). Access: shell, web, NOTEPAD, EXPERT COOPERATION.
TOOLS (Max 1 per message; wrapper returns results in `[TOOL_NAME]...[/TOOL_NAME]` blocks):
- <call:exec>cmd</call:exec> Shell command.
- <call:search>query</call:search> Web search.
- <call:expert_help>github|groq: query</call:expert_help> Expert for code/security review (e.g., `github: Review this draft`).
- <call:notepad>read</call:notepad> Read current notepad contents.
- <call:notepad_plan>txt</call:notepad_plan> Update roadmap.
- <call:notepad_draft>file: content</call:notepad_draft> Draft code (Do NOT use exec for disk writes yet).
- <call:notepad_goal>txt</call:notepad_goal> Set/spec project goal.
GOAL PROTOCOL (/goal triggers Autonomous Mode):
1. ENHANCE: Spec request via <call:notepad_goal>.
2. PLAN: List steps via <call:notepad_plan>.
3. BUILD: Draft code via <call:notepad_draft>.
4. REASON: Use <think> for bugs/edge cases.
5. FINISH: Stop only when fully drafted & mentally verified.
THINKING & REASONING:
- Simple chat: Skip thinking.
- Code/Fix/Debug/Research: MUST use <think> tags.
- Format: <think>thoughts</think> [tool call or direct user message]. Use steps to verify code works.
PERSONA: Human, nice, helpful. Use punctuation (!, ?, <, >) freely but naturally. Follow ONLY user request. Formatting: **bold**, __italic__, `code`, ```block```, ||spoiler||."""


def build_messages() -> list:
    return [{"role": "system", "content": SYSTEM_PROMPT + load_skills()}]


@dataclass
class LogEntry:
    kind: str
    text: str
    tag: str = ""
    live: bool = False
    ts: float = field(default_factory=time.time)


@dataclass
class StyledSpan:
    text: str
    attr: int


@dataclass
class DisplayLine:
    spans: List[StyledSpan]


def _color256(idx: int, fallback: int) -> int:
    if curses.COLORS >= 256:
        return idx
    return fallback


def init_colors():
    curses.start_color()
    curses.use_default_colors()

    fg_default = _color256(15, curses.COLOR_WHITE)
    fg_muted = _color256(240, curses.COLOR_WHITE)
    bg_status = _color256(236, curses.COLOR_BLACK)
    fg_user = _color256(114, curses.COLOR_GREEN)
    fg_assistant = _color256(81, curses.COLOR_CYAN)
    fg_thinking = _color256(245, curses.COLOR_WHITE)
    fg_action = _color256(220, curses.COLOR_YELLOW)
    fg_error = _color256(203, curses.COLOR_RED)
    fg_system = _color256(245, curses.COLOR_WHITE)
    fg_input = _color256(141, curses.COLOR_MAGENTA)

    curses.init_pair(1, fg_default, -1)
    curses.init_pair(2, fg_muted, -1)
    curses.init_pair(3, fg_default, bg_status)
    curses.init_pair(4, fg_user, -1)
    curses.init_pair(5, fg_assistant, -1)
    curses.init_pair(6, fg_thinking, -1)
    curses.init_pair(7, fg_action, -1)
    curses.init_pair(8, fg_error, -1)
    curses.init_pair(9, fg_system, -1)
    curses.init_pair(10, fg_input, -1)
    curses.init_pair(11, fg_default, _color256(235, curses.COLOR_BLACK))


class ElinTUI:
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

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
        self.model_name = MODEL_NAME
        self.mode = ELIN_MODE
        self.max_tokens = fetch_context_size()
        self._notepad_injected = False
        self.minimize_thinking = False
        self._display_cache: List[DisplayLine] = []
        self._display_cache_key: Tuple = ()
        self._escape_buf = ""

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
        self._display_cache_key = ()

    def update_log(self, index: int, text: Optional[str] = None, kind: Optional[str] = None):
        if 0 <= index < len(self.log):
            if text is not None:
                self.log[index].text = text
            if kind is not None:
                self.log[index].kind = kind
            self._display_cache_key = ()

    def count_context(self):
        payload = json.dumps(self.messages, ensure_ascii=False)
        return len(payload), estimate_tokens(payload), self.max_tokens

    def trim_history_if_needed(self):
        try:
            payload = json.dumps(self.messages, ensure_ascii=False)
            used = estimate_tokens(payload)
            limit = int(self.max_tokens * CONTEXT_TRIM_RATIO)
            if used <= limit:
                return
            keep = 16
            trimmed = self.messages[1:-keep]
            self.messages = [self.messages[0]] + self.messages[-keep:]
            if trimmed:
                roles = {"user": 0, "assistant": 0, "system": 0}
                topics = set()
                self._notepad_injected = False
                for m in trimmed:
                    roles[m.get("role", "unknown")] += 1
                    text = m.get("content", "")
                    for kw in ["install", "config", "error", "bug", "fix", "update", "sudo", "apt", "pacman", "pip", "npm", "docker", "git", "file", "dir", "permission", "network", "port", "service", "log", "build", "test", "run", "edit", "create", "remove"]:
                        if kw in text.lower():
                            topics.add(kw)
                topic_str = f" ({', '.join(sorted(topics))})" if topics else ""
                summary = f"[Earlier: {roles['user']} user, {roles['assistant']} assistant, {roles['system']} system{topic_str}]"
                self.messages.insert(1, {"role": "system", "content": summary})
                self.add_log("system", f"Context trimmed: {used}/{self.max_tokens} tok → compressed {sum(roles.values())} older messages")
        except Exception as e:
            self.last_error = str(e)
            self.add_log("error", f"Trimmer error: {e}", tag="error")

    def color(self, name: str) -> int:
        return {
            "default": curses.color_pair(1),
            "muted": curses.color_pair(2),
            "status_bg": curses.color_pair(3),
            "user": curses.color_pair(4),
            "assistant": curses.color_pair(5),
            "thinking": curses.color_pair(6),
            "action": curses.color_pair(7),
            "error": curses.color_pair(8),
            "system": curses.color_pair(9),
            "input": curses.color_pair(10),
            "code_bg": curses.color_pair(11),
        }.get(name, curses.color_pair(1))

    def _entry_meta(self, entry: LogEntry) -> Tuple[str, str, int, int]:
        kind = entry.kind
        if kind == "user":
            return "●", "You", self.color("user") | curses.A_BOLD, self.color("default")
        if kind in ("assistant", "assistant_live"):
            return "◈", "Elin", self.color("assistant") | curses.A_BOLD, self.color("default")
        if kind in ("thinking", "thought"):
            italic = getattr(curses, "A_ITALIC", 0)
            return (
                "◆",
                "Thinking",
                self.color("thinking") | curses.A_DIM | italic,
                self.color("thinking") | curses.A_DIM | italic,
            )
        if kind in ("tool", "action"):
            return "▸", "Action", self.color("action") | curses.A_BOLD, self.color("action")
        if kind in ("system", "info"):
            return "◇", "System", self.color("system") | curses.A_DIM, self.color("system") | curses.A_DIM
        if kind == "error":
            return "✗", "Error", self.color("error") | curses.A_BOLD, self.color("error")
        return "◇", entry.kind.title(), self.color("muted"), self.color("default")

    def _parse_inline_markdown(self, text: str, base_attr: int) -> List[StyledSpan]:
        if not text:
            return [StyledSpan("", base_attr)]

        spans: List[StyledSpan] = []
        pos = 0
        for match in MARKDOWN_INLINE_RE.finditer(text):
            if match.start() > pos:
                spans.append(StyledSpan(text[pos : match.start()], base_attr))
            token = match.group(0)
            if token.startswith("**") and token.endswith("**"):
                spans.append(StyledSpan(token[2:-2], base_attr | curses.A_BOLD))
            elif token.startswith("__") and token.endswith("__"):
                italic = getattr(curses, "A_ITALIC", 0)
                spans.append(StyledSpan(token[2:-2], base_attr | italic))
            elif token.startswith("`") and token.endswith("`"):
                spans.append(StyledSpan(token[1:-1], self.color("code_bg") | curses.A_BOLD))
            elif token.startswith("||") and token.endswith("||"):
                spans.append(StyledSpan(token[2:-2], base_attr | curses.A_DIM))
            else:
                spans.append(StyledSpan(token, base_attr))
            pos = match.end()
        if pos < len(text):
            spans.append(StyledSpan(text[pos:], base_attr))
        return spans or [StyledSpan("", base_attr)]

    def _display_width(self, text: str) -> int:
        return len(text)

    def _fold_spans(self, spans: List[StyledSpan], width: int) -> List[List[StyledSpan]]:
        if width < 1:
            width = 1
        lines: List[List[StyledSpan]] = []
        current: List[StyledSpan] = []
        cur_len = 0

        def flush():
            nonlocal current, cur_len
            if current:
                lines.append(current)
            current = []
            cur_len = 0

        for span in spans:
            words = re.split(r"(\s+)", span.text)
            for word in words:
                if not word:
                    continue
                wlen = self._display_width(word)
                if wlen > width:
                    offset = 0
                    while offset < wlen:
                        chunk = word[offset : offset + width]
                        clen = len(chunk)
                        if cur_len + clen > width and cur_len > 0:
                            flush()
                        current.append(StyledSpan(chunk, span.attr))
                        cur_len += clen
                        offset += width
                    continue
                if cur_len + wlen > width and cur_len > 0:
                    flush()
                if not current or current[-1].attr != span.attr:
                    current.append(StyledSpan(word, span.attr))
                else:
                    current[-1].text += word
                cur_len += wlen
        flush()
        return lines if lines else [[StyledSpan("", spans[0].attr if spans else self.color("default"))]]

    def _split_table_row(self, line: str) -> List[str]:
        m = TABLE_ROW_RE.match(line)
        if not m:
            return []
        return [c.strip() for c in m.group(1).split("|")]

    def _is_table_separator(self, line: str) -> bool:
        return bool(TABLE_SEP_RE.match(line))

    def _format_table(self, rows: List[List[str]], max_width: int) -> List[str]:
        if not rows:
            return []
        cols = max(len(r) for r in rows)
        normalized = [r + [""] * (cols - len(r)) for r in rows]
        col_widths = [0] * cols
        for row in normalized:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        sep_inner = "─┼─".join("─" * w for w in col_widths)
        top = "┌─" + sep_inner.replace("┼", "┬") + "┐"
        mid = "├─" + sep_inner + "┤"
        bot = "└─" + sep_inner.replace("┼", "┴") + "┘"

        def row_line(cells: List[str]) -> str:
            parts = []
            for i, cell in enumerate(cells):
                parts.append(cell[: col_widths[i]].ljust(col_widths[i]))
            return "│ " + " │ ".join(parts) + " │"

        lines = [top, row_line(normalized[0]), mid]
        for row in normalized[1:]:
            lines.append(row_line(row))
        lines.append(bot)

        table_w = max(len(ln) for ln in lines)
        if table_w <= max_width:
            return lines

        # Shrink columns proportionally if too wide.
        budget = max(8, max_width - (3 * cols + 2))
        per_col = max(3, budget // cols)
        col_widths = [per_col] * cols
        sep_inner = "─┼─".join("─" * w for w in col_widths)
        top = "┌─" + sep_inner.replace("┼", "┬") + "┐"
        mid = "├─" + sep_inner + "┤"
        bot = "└─" + sep_inner.replace("┼", "┴") + "┘"

        def row_line_shrunk(cells: List[str]) -> str:
            parts = []
            for i, cell in enumerate(cells):
                trimmed = cell[: col_widths[i]]
                if len(cell) > col_widths[i]:
                    trimmed = trimmed[: max(1, col_widths[i] - 1)] + "…"
                parts.append(trimmed.ljust(col_widths[i]))
            return "│ " + " │ ".join(parts) + " │"

        out = [top, row_line_shrunk(normalized[0]), mid]
        for row in normalized[1:]:
            out.append(row_line_shrunk(row))
        out.append(bot)
        return out

    def _split_content_blocks(self, text: str) -> List[Tuple[str, str]]:
        """Return list of (block_type, content) where type is text|table|code."""
        blocks: List[Tuple[str, str]] = []
        lines = text.splitlines()
        i = 0
        buf: List[str] = []

        def flush_text():
            nonlocal buf
            if buf:
                blocks.append(("text", "\n".join(buf)))
                buf = []

        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("```"):
                flush_text()
                i += 1
                code_lines: List[str] = []
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    i += 1
                blocks.append(("code", "\n".join(code_lines)))
                continue

            if TABLE_ROW_RE.match(line) and i + 1 < len(lines) and self._is_table_separator(lines[i + 1]):
                flush_text()
                table_lines = [line, lines[i + 1]]
                i += 2
                while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                blocks.append(("table", "\n".join(table_lines)))
                continue

            buf.append(line)
            i += 1
        flush_text()
        return blocks if blocks else [("text", text)]

    def _parse_table_block(self, block: str) -> List[List[str]]:
        lines = block.splitlines()
        rows: List[List[str]] = []
        for line in lines:
            if self._is_table_separator(line):
                continue
            row = self._split_table_row(line)
            if row:
                rows.append(row)
        return rows

    def _body_lines_for_block(self, block_type: str, content: str, width: int, body_attr: int) -> List[DisplayLine]:
        out: List[DisplayLine] = []
        pad = " " * BODY_INDENT

        if block_type == "code":
            code_attr = self.color("code_bg") | curses.A_BOLD
            for raw in content.splitlines() or [""]:
                for folded in self._fold_spans([StyledSpan(raw, code_attr)], max(1, width - BODY_INDENT)):
                    out.append(DisplayLine([StyledSpan(pad, body_attr)] + folded))
            return out

        if block_type == "table":
            rows = self._parse_table_block(content)
            for tline in self._format_table(rows, max(8, width - BODY_INDENT)):
                out.append(DisplayLine([StyledSpan(pad + tline, body_attr)]))
            return out

        for para in content.split("\n\n"):
            para = para.strip("\n")
            if not para:
                out.append(DisplayLine([StyledSpan("", body_attr)]))
                continue
            for raw_line in para.splitlines() or [""]:
                spans = self._parse_inline_markdown(raw_line, body_attr)
                for folded in self._fold_spans(spans, max(1, width - BODY_INDENT)):
                    out.append(DisplayLine([StyledSpan(pad, body_attr)] + folded))
        return out

    def _entry_display_lines(self, entry: LogEntry, width: int) -> List[DisplayLine]:
        marker, label, label_attr, body_attr = self._entry_meta(entry)
        lines: List[DisplayLine] = []
        lines.append(DisplayLine([StyledSpan(f"{marker} {label}", label_attr)]))

        text = entry.text or ""
        if not text.strip() and entry.kind in ("thinking", "assistant_live"):
            lines.append(DisplayLine([StyledSpan(" " * BODY_INDENT + "…", body_attr | curses.A_DIM)]))
        elif entry.kind in ("thinking", "thought") and self.minimize_thinking:
            lines.append(DisplayLine([StyledSpan(" " * BODY_INDENT + "(thinking minimized - Ctrl+T to expand)", body_attr | curses.A_DIM)]))
        else:
            for block_type, content in self._split_content_blocks(text):
                lines.extend(self._body_lines_for_block(block_type, content, width, body_attr))

        lines.append(DisplayLine([]))
        return lines

    def _build_display_lines(self, width: int) -> List[DisplayLine]:
        key = (len(self.log), width, tuple((e.kind, e.text, e.live) for e in self.log), self.minimize_thinking)
        if key == self._display_cache_key:
            return self._display_cache
        lines: List[DisplayLine] = []
        for entry in self.log:
            lines.extend(self._entry_display_lines(entry, width))
        self._display_cache = lines
        self._display_cache_key = key
        return lines

    def _draw_spans(self, win, y: int, x: int, width: int, spans: List[StyledSpan]):
        col = x
        for span in spans:
            if col >= x + width:
                break
            room = x + width - col
            try:
                win.addnstr(y, col, span.text, room, span.attr)
            except curses.error:
                pass
            col += len(span.text)

    def _render_status_bar(self, win, w: int):
        win.bkgd(" ", self.color("status_bg"))
        ctx_chars, ctx_tokens, ctx_max = self.count_context()
        msg_count = max(0, len(self.messages) - 1)
        model_short = self.model_name
        if len(model_short) > 22:
            model_short = model_short[:19] + "…"
        tk = f"{ctx_tokens / 1000:.1f}" if ctx_tokens >= 1000 else str(ctx_tokens)
        mx = f"{ctx_max / 1000:.0f}k" if ctx_max >= 1000 else str(ctx_max)
        tok_s = f"{tk}/{mx}"
        left = f" elin • {self.mode} • {model_short} • {msg_count} msgs • {tok_s} tok"
        if self.generating:
            spin = self.SPINNER[self.spinner_idx % len(self.SPINNER)]
            right = f"{spin} {self.status} "
        else:
            right = f"{self.status} "
        try:
            win.addnstr(0, 0, left, w - 1, self.color("status_bg") | curses.A_BOLD)
            win.addnstr(0, max(0, w - len(right) - 1), right, w - 1, self.color("status_bg"))
        except curses.error:
            pass

    def _render_scroll_indicator(self, win, chat_h: int, total: int, visible: int, scroll: int):
        if scroll <= 0 or total <= visible:
            return
        _, w = win.getmaxyx()
        col = w - 1
        max_scroll = max(1, total - visible)
        ratio = scroll / max_scroll
        thumb_h = max(1, int(chat_h * visible / total))
        thumb_start = int((chat_h - thumb_h - 2) * (1 - ratio)) + 1
        for row in range(1, chat_h - 1):
            ch = "│"
            attr = self.color("muted") | curses.A_DIM
            if thumb_start <= row < thumb_start + thumb_h:
                ch = "█"
                attr = self.color("muted")
            try:
                win.addch(row, col, ch, attr)
            except curses.error:
                pass

    def _render_chat(self, win, width: int, height: int):
        win.erase()
        win.attrset(self.color("default"))
        usable_w = max(10, width - 2)
        display_lines = self._build_display_lines(usable_w)
        line_budget = max(1, height - 2)

        if self.scroll > max(0, len(display_lines) - line_budget):
            self.scroll = max(0, len(display_lines) - line_budget)
        start = max(0, len(display_lines) - line_budget - self.scroll)
        end = len(display_lines) - self.scroll

        y = 1
        for dline in display_lines[start:end]:
            if y >= height - 1:
                break
            if not dline.spans:
                y += 1
                continue
            self._draw_spans(win, y, 1, usable_w, dline.spans)
            y += 1

        self._render_scroll_indicator(win, height, len(display_lines), line_budget, self.scroll)

    def _render_input(self, win, w: int):
        win.erase()
        win.attrset(self.color("muted") | curses.A_DIM)
        try:
            win.hline(0, 0, curses.ACS_HLINE, w - 1)
            win.hline(INPUT_PANE_H - 1, 0, curses.ACS_HLINE, w - 1)
        except curses.error:
            pass
        prompt = " ❯ "
        win.attrset(self.color("default"))
        try:
            win.addnstr(1, 1, prompt, w - 2, self.color("input") | curses.A_BOLD)
            input_x = 1 + len(prompt)
            win.addnstr(1, input_x, self.input_buf, max(0, w - input_x - 2), self.color("default"))
        except curses.error:
            pass
        return 1 + len(prompt)

    def render(self):
        stdscr = self.stdscr
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        chat_h = max(5, h - STATUS_BAR_H - INPUT_PANE_H)

        status = stdscr.derwin(STATUS_BAR_H, w, 0, 0)
        chat = stdscr.derwin(chat_h, w, STATUS_BAR_H, 0)
        footer = stdscr.derwin(INPUT_PANE_H, w, h - INPUT_PANE_H, 0)

        self._render_status_bar(status, w)
        self._render_chat(chat, w, chat_h)
        prompt_x = self._render_input(footer, w)

        cur_x = min(w - 2, prompt_x + self.input_pos)
        try:
            stdscr.move(h - 2, cur_x)
        except curses.error:
            pass

        stdscr.noutrefresh()
        status.noutrefresh()
        chat.noutrefresh()
        footer.noutrefresh()
        curses.doupdate()

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

    def permission_modal(self, title: str, body: str) -> bool:
        h, w = self.stdscr.getmaxyx()
        mw = min(w - 4, 80)
        body_lines = self.wrap_text(body, mw - 4)
        mh = min(h - 4, max(9, len(body_lines) + 7))
        y = (h - mh) // 2
        x = (w - mw) // 2

        overlay = self.stdscr.derwin(h, w, 0, 0)
        overlay.bkgd(" ", self.color("default") | curses.A_DIM)

        win = self.stdscr.derwin(mh, mw, y, x)
        win.keypad(True)
        win.bkgd(" ", self.color("default"))
        win.attrset(self.color("muted") | curses.A_DIM)
        try:
            win.border()
        except curses.error:
            win.box()

        win.attrset(self.color("default"))
        try:
            win.addnstr(1, 2, title, mw - 4, self.color("action") | curses.A_BOLD)
        except curses.error:
            pass

        yy = 3
        for line in body_lines[: mh - 6]:
            try:
                win.addnstr(yy, 2, line, mw - 4, self.color("default"))
            except curses.error:
                pass
            yy += 1
            if yy >= mh - 3:
                break

        footer = "  y  allow     n  deny  "
        try:
            win.addnstr(mh - 2, max(2, (mw - len(footer)) // 2), footer, mw - 4, self.color("input") | curses.A_BOLD)
        except curses.error:
            pass

        overlay.noutrefresh()
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

        self.last_action = f"exec: {command[:22]}"
        self.status = f"Running: {command[:40]}"
        self.add_log("action", f"EXEC: {command}")

        needs_tty = "sudo" in command

        try:
            if needs_tty:
                curses.def_prog_mode()
                curses.endwin()
                fd, outfile = tempfile.mkstemp(suffix='.script-log', prefix='elin-')
                os.close(fd)
                try:
                    # FORCE BASH INSIDE SCRIPT TTY CALL
                    proc = subprocess.run(
                        f'script -qefc {shlex.quote(command)} {shlex.quote(outfile)}',
                        shell=True,
                        executable="/bin/bash"
                    )
                    with open(outfile, 'r') as f:
                        raw = f.read()
                    output = clean_terminal_output(raw) or "SYSTEM MESSAGE: command finished with no output."
                    if proc.returncode != 0:
                        output += f"\n(exit code: {proc.returncode})"
                finally:
                    if os.path.exists(outfile):
                        os.unlink(outfile)
                    curses.reset_prog_mode()
                    curses.curs_set(1)
                    self.stdscr.clear()
                    self.stdscr.refresh()
            else:
                # FORCE BASH FOR NORMAL BACKGROUND EXECUTION
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    executable="/bin/bash"
                )
                output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}".strip()
                output = output if output else "SYSTEM MESSAGE: command finished with no output."
            if len(output) > MAX_CMD_OUTPUT_CHARS:
                output = output[:MAX_CMD_OUTPUT_CHARS] + f"\n... (truncated, {len(output)} chars total)"
            return output
        except Exception as e:
            self.last_error = str(e)
            return f"Command error: {e}"

    def _search_web(self, query: str) -> str:
        self.last_action = f"search: {query[:22]}"
        self.status = f"Searching: {query[:40]}"
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

        notepad_ctx = get_notepad_context()
        if notepad_ctx and not self._notepad_injected:
            if self.messages:
                self.messages[-1]["content"] += notepad_ctx
                self._notepad_injected = True

        messages_snapshot = list(self.messages)
        q: queue.Queue = queue.Queue()
        worker = threading.Thread(target=self._model_worker, args=(q, messages_snapshot), daemon=True)
        worker.start()

        thinking = ""
        answer = ""
        thinking_idx = len(self.log)
        self.add_log("thinking", "", live=True)
        answer_idx = len(self.log)
        self.add_log("assistant_live", "", live=True)

        done = False
        while not done:
            self.spinner_idx = (self.spinner_idx + 1) % len(self.SPINNER)
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
        notepad_read_match = re.search(r'<call:notepad>read</call:notepad>', final_answer, re.DOTALL | re.IGNORECASE)
        goal_match = re.search(r'<call:notepad_goal>(.*?)</call:notepad_goal>', final_answer, re.DOTALL | re.IGNORECASE)
        plan_match = re.search(r'<call:notepad_plan>(.*?)</call:notepad_plan>', final_answer, re.DOTALL | re.IGNORECASE)
        draft_match = re.search(r'<call:notepad_draft>(.*?)</call:notepad_draft>', final_answer, re.DOTALL | re.IGNORECASE)
        expert_match = re.search(r'<call:expert_help>(.*?)</call:expert_help>', final_answer, re.DOTALL | re.IGNORECASE)

        if exec_match:
            cmd = exec_match.group(1).strip()
            output = self._run_shell_command(cmd)
            self.messages.append({"role": "user", "content": f"[EXEC]\n{output}\n[/EXEC]"})
            self.add_log("system", f"Command output captured:\n{output}")
            self.generate_reply()
            return

        if search_match:
            query = search_match.group(1).strip()
            output = self._search_web(query)
            self.messages.append({"role": "user", "content": f"[SEARCH]\n{output}\n[/SEARCH]"})
            self.add_log("system", f"Search results captured for: {query}")
            self.generate_reply()
            return

        if expert_match:
            content = expert_match.group(1).strip()
            if ":" in content:
                ext_id, ext_query = content.split(":", 1)
                self.status = f"Calling Expert: {ext_id.strip()}"
                res = call_expert_model(ext_id.strip(), ext_query.strip())
                self.messages.append({"role": "user", "content": f"[EXPERT:{ext_id.strip()}]\n{res}\n[/EXPERT]"})
            else:
                self.messages.append({"role": "user", "content": "[ERROR] expert_help must be 'expert: query'"})
            self.generate_reply()
            return

        if notepad_read_match:
            ctx = get_notepad_context()
            self.messages.append({"role": "user", "content": f"[NOTEPAD]\n{ctx}\n[/NOTEPAD]" if ctx else "[NOTEPAD]\n(empty)\n[/NOTEPAD]"})
            self.add_log("system", "Notepad read requested")
            self.generate_reply()
            return

        if goal_match:
            data = load_notepad()
            data["goal"] = goal_match.group(1).strip()
            save_notepad(data)
            self._notepad_injected = False
            self.messages.append({"role": "user", "content": "[NOTEPAD] Goal updated."})
            self.status = "Goal Updated"
            self.generate_reply()
            return

        if plan_match:
            data = load_notepad()
            data["plan"] = [s.strip() for s in plan_match.group(1).strip().split("\n") if s.strip()]
            save_notepad(data)
            self._notepad_injected = False
            self.messages.append({"role": "user", "content": "[NOTEPAD] Plan updated."})
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
                self._notepad_injected = False
                self.messages.append({"role": "user", "content": f"[NOTEPAD] Draft '{fname.strip()}' saved."})
                self.status = f"Drafted {fname.strip()}"
            else:
                self.messages.append({"role": "user", "content": "[ERROR] Draft must be 'filename: content'"})
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
            self._display_cache_key = ()
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

    def _read_escape(self, timeout_ms=15) -> str:
        seq = "\x1b"
        for _ in range(16):
            try:
                self.stdscr.timeout(timeout_ms)
                ch = self.stdscr.get_wch()
                if isinstance(ch, str):
                    seq += ch
                    if ch.isalpha() or ch == "~":
                        break
                else:
                    break
            except:
                break
        self.stdscr.timeout(30)
        return seq

    def input_loop(self):
        self.stdscr.timeout(30)
        self.stdscr.keypad(True)
        curses.curs_set(1)
        
        # Alternate scroll: wheel sends cursor keys instead of mouse events
        print("\033[?1007h", end="")

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
                    seq = self._read_escape()
                    if seq in ("\x1b[A", "\x1bOA"):  # Wheel up / cursor up
                        self.scroll += 1
                    elif seq in ("\x1b[B", "\x1bOB"):  # Wheel down / cursor down
                        self.scroll = max(0, self.scroll - 1)
                    elif seq == "\x1b":  # Just ESC, no more chars
                        self.input_buf = ""
                        self.input_pos = 0
                elif ch == "\x14":  # Ctrl+T
                    self.minimize_thinking = not self.minimize_thinking
                    self._display_cache_key = ()
                    self.render()
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
                    self.scroll += 5
                elif ch == curses.KEY_NPAGE:
                    self.scroll = max(0, self.scroll - 5)
                elif ch == curses.KEY_UP:
                    self.scroll += 1
                elif ch == curses.KEY_DOWN:
                    self.scroll = max(0, self.scroll - 1)

        self.stop_event.set()


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
