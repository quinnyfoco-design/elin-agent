import os
import subprocess
import requests
import re
import json
import datetime
import glob
import sys
import select
import time
from flask import Flask, request, jsonify
import threading
import queue
from openai import OpenAI

ELIN_MODE = os.environ.get("ELIN_MODE", "local")
if ELIN_MODE == "cloud":
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))
    MODEL_NAME = "llama-3.3-70b-versatile"
elif ELIN_MODE == "github":
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=os.environ.get("GITHUB_TOKEN"),
    )
    MODEL_NAME = os.environ.get("GITHUB_MODEL", "openai/gpt-4.1")
else:
    client = OpenAI(base_url="http://localhost:8081/v1", api_key="sk-no-key-required", timeout=300.0)
    MODEL_NAME = "local-model"

SEARXNG_URL = "http://172.17.0.1:8080/search" 

def format_md(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\033[1m\1\033[0m', text)
    text = re.sub(r'\*(.*?)\*', r'\033[3m\1\033[0m', text)
    text = re.sub(r'`(.*?)`', r'\033[36m\1\033[0m', text)
    return text

def load_skills():
    skills_dir = os.path.expanduser("~/elin-project/skills")
    if not os.path.exists(skills_dir):
        return ""
    all_skills = "\n\n=== ADDITIONAL SKILLS ===\n"
    for filename in os.listdir(skills_dir):
        if filename.endswith(".md") or filename.endswith(".txt"):
            try:
                with open(os.path.join(skills_dir, filename), "r") as f:
                    all_skills += f"\n[Skill: {filename}]\n{f.read()}\n"
            except: pass
    return all_skills

def run_linux_command(command):
    if "pacman" in command and "--noconfirm" not in command:
        command = command.replace("pacman", "pacman --noconfirm")
    if "yay" in command and "--noconfirm" not in command:
        command = command.replace("yay", "yay --noconfirm")

    risky = ["rm", "dd", "mkfs", "mv", ">", "pacman -R", "sudo", "systemctl", "touch", "/bin", "/dev", "/sys"]
    
    if any(r in command for r in risky):
        prompt_text = f" elin wants to run\n: `{command}`\n\n[y/n]"
        print(f"\n\033[1;33mwaiting for confirmation... [{command}]\033[0m")
        elin_visual_speak(prompt_text, "LOCK")

        confirm = None
        while confirm is None:
            try:
                v_resp = requests.get("http://localhost:8000/get_input", timeout=1).json()
                if v_resp and v_resp.get("text"):
                    confirm = v_resp["text"].strip().lower()
            except:
                pass
            time.sleep(1)

        if confirm != 'y':
            return "SYSTEM MESSAGE: User denied execution, likely because they didnt understand or just doesnt want it to be run."

    print(f"\n\033[2mexecuting: {command}\033[0m")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"

def search_web(query):
    print(f"\n\033[2msearching for: {query}\033[0m")
    try:
        params = {'q': query, 'format': 'json'}
        resp = requests.get(SEARXNG_URL, params=params)
        results = resp.json().get('results', [])[:3]
        return "\n".join([f"Source: {r['title']}\nContent: {r['content']}" for r in results])
    except Exception as e:
        return f"Search error: {e}"

def call_expert_model(expert_id, prompt):
    """Calls an external 'Expert' model (github or groq) for high-level help."""
    print(f"\n\033[1;35m[Calling Expert: {expert_id}]...\033[0m")
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

def elin_visual_speak(text, shape="GLOBE"):
    try:
        requests.post("http://localhost:8000/speak", json={"text": text, "shape": shape}, timeout=1)
    except: pass

# --- NOTEPAD SYSTEM ---
NOTEPAD_PATH = os.path.expanduser("~/elin-project/.notepad.json")

def load_notepad():
    if os.path.exists(NOTEPAD_PATH):
        with open(NOTEPAD_PATH, "r") as f:
            return json.load(f)
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
REASONING RULES:
- If the task involves "Code", "Fix", "Debug", or "Research", you MUST use <think>.
- Use the NOTEPAD to keep track of your work so you don't forget it in long chats.
Formatting: **bold**, __italic__, `code`, ```block```, ||spoiler||."""

SKILLS_CONTENT = load_skills()
os.system('clear' if os.name == 'posix' else 'cls')
print(f"\033[1;36melin project ({ELIN_MODE} mode)\033[0m")

messages = [{"role": "system", "content": SYSTEM_PROMPT + SKILLS_CONTENT}]

while True:
    user_msg = None
    print(f"\n\033[1;31mUser > \033[0m", end="", flush=True)
    while not user_msg:
        try:
            v_resp = requests.get("http://localhost:8000/get_input", timeout=0.05).json()
            if v_resp and v_resp.get("text"):
                user_msg = v_resp["text"]
                print(f"\033[1;34m[Telegram]: {user_msg}\033[0m")
                break
        except: pass

        if sys.stdin in select.select([sys.stdin], [], [], 0.0)[0]:
            user_msg = sys.stdin.readline().strip()
            break
        time.sleep(0.2)
    
    if user_msg.strip().lower() == "/nc":
        if os.path.exists(NOTEPAD_PATH): os.remove(NOTEPAD_PATH)
        messages = [{"role": "system", "content": SYSTEM_PROMPT + SKILLS_CONTENT}]
        print("\033[1;32m[new chat started]\033[0m")
        elin_visual_speak("new chat started", "GLOBE")
        continue

    if user_msg.strip().lower().startswith("/goal "):
        goal_text = user_msg[6:].strip()
        user_msg = f"[GOAL PROTOCOL INITIATED]\nTarget: {goal_text}\n\n1. Enhance this goal.\n2. Create a plan.\n3. Start drafting.\nUse your NOTEPAD tools."
        elin_visual_speak("Goal Protocol Active", "EXEC")

    # ... (rest of /sc and /lc logic stays same)

    if user_msg.strip().lower() == "/sc":
        # ... (save logic)
        continue # simplified for brevity, assume original logic exists

    messages.append({"role": "user", "content": user_msg})
    for i in range(20):
        # Inject Notepad into the turn
        notepad_ctx = get_notepad_context()
        if notepad_ctx:
            # We insert it as a system-like message at the top of the context or current turn
            # To be safe with the 500 error, we append it to the LAST user message
            messages[-1]["content"] += notepad_ctx

        try:
            history_size = len(json.dumps(messages))
            if history_size > 600000:
                print(f"\n\033[1;33m[Elin System]: History is {history_size} chars. Trimming to fit 128k context...\033[0m")
                messages = [messages[0]] + messages[-15:]
        except Exception as e:
            print(f"Trimmer error: {e}")
        print(f"\033[1;31mElin > \033[0m", end="", flush=True)
        elin_resp = ""

        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.5,
            stream=True
        )
        
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                print(f"\033[90m\033[3m{delta.reasoning_content}\033[0m", end="", flush=True)
            if delta.content:
                print(delta.content, end="", flush=True)
                elin_resp += delta.content
        print()
        
        current_shape = "GLOBE"
        if "<call:" in elin_resp: current_shape = "EXEC"
        elin_visual_speak(elin_resp, current_shape)
        
        messages.append({"role": "assistant", "content": elin_resp})
        
        exec_match = re.search(r'<call:exec>(.*?)</call:exec>', elin_resp, re.DOTALL | re.IGNORECASE)
        search_match = re.search(r'<call:search>(.*?)</call:search>', elin_resp, re.DOTALL | re.IGNORECASE)
        
        # Notepad Matches
        goal_match = re.search(r'<call:notepad_goal>(.*?)</call:notepad_goal>', elin_resp, re.DOTALL | re.IGNORECASE)
        plan_match = re.search(r'<call:notepad_plan>(.*?)</call:notepad_plan>', elin_resp, re.DOTALL | re.IGNORECASE)
        draft_match = re.search(r'<call:notepad_draft>(.*?)</call:notepad_draft>', elin_resp, re.DOTALL | re.IGNORECASE)
        expert_match = re.search(r'<call:expert_help>(.*?)</call:expert_help>', elin_resp, re.DOTALL | re.IGNORECASE)

        if exec_match:
            cmd = exec_match.group(1).strip()
            output = run_linux_command(cmd)
            messages.append({"role": "user", "content": f"SYSTEM_OBSERVATION (EXEC):\n{output}\n[End of output. Please evaluate and continue.]"})
        elif search_match:
            query = search_match.group(1).strip()
            output = search_web(query)
            messages.append({"role": "user", "content": f"SYSTEM_OBSERVATION (SEARCH):\n{output}\n[End of output. Please evaluate and continue.]"})
        elif expert_match:
            content = expert_match.group(1).strip()
            if ":" in content:
                ext_id, ext_query = content.split(":", 1)
                result = call_expert_model(ext_id.strip(), ext_query.strip())
                messages.append({"role": "user", "content": f"SYSTEM_OBSERVATION (EXPERT HELP from {ext_id.strip()}):\n{result}\n[End of expert response. Evaluate and continue.]"})
            else:
                messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Error - expert_help must be 'expert: query'"})
        elif goal_match:
            data = load_notepad()
            data["goal"] = goal_match.group(1).strip()
            save_notepad(data)
            messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Notepad Goal Updated."})
        elif plan_match:
            data = load_notepad()
            data["plan"] = [s.strip() for s in plan_match.group(1).strip().split("\n") if s.strip()]
            save_notepad(data)
            messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Notepad Plan Updated."})
        elif draft_match:
            data = load_notepad()
            content = draft_match.group(1).strip()
            if ":" in content:
                fname, fbody = content.split(":", 1)
                data["drafts"][fname.strip()] = fbody.strip()
                save_notepad(data)
                messages.append({"role": "user", "content": f"SYSTEM_OBSERVATION: Draft '{fname.strip()}' saved to Notepad."})
            else:
                messages.append({"role": "user", "content": "SYSTEM_OBSERVATION: Error - Draft must be in format 'filename: content'"})
        else:
            break


