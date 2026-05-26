import os
import re
import threading
import time
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIG ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "replace")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

pending_messages = []
current_chat_id = None

def md_to_html(text):
    """Translates Elin's Markdown into Telegram-safe HTML"""
    # 1. Escape basic characters that actually break HTML
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # 2. Convert Bold (**fat**)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # 3. Convert Italics (__italic__)
    text = re.sub(r'__(.*?)__', r'<i>\1</i>', text)
    # 4. Convert Spoilers (||secret||)
    text = re.sub(r'\|\|(.*?)\|\|', r'<tg-spoiler>\1</tg-spoiler>', text)
    # 5. Convert Code Blocks (```code```)
    text = re.sub(r'```(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    # 6. Convert Inline Code (`cmd`)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    return text

@app.route('/get_input', methods=['GET'])
def get_input():
    if pending_messages:
        msg = pending_messages.pop(0)
        return jsonify({"text": msg})
    return jsonify({})

@app.route('/speak', methods=['POST'])
def speak():
    global current_chat_id
    data = request.json
    text = data.get("text", "")
    
    if current_chat_id and text:
        # Convert Markdown to HTML so Telegram doesn't have a tantrum
        html_text = md_to_html(text)
        
        payload = {
            "chat_id": current_chat_id,
            "text": html_text,
            "parse_mode": "HTML"
        }
        r = requests.post(f"{TELEGRAM_API_URL}/sendMessage", json=payload)
        if r.status_code != 200:
            print(f"!!! TG ERROR: {r.text}")
            
    return jsonify({"status": "success"})

def poll_telegram():
    global current_chat_id
    last_update_id = 0
    while True:
        try:
            resp = requests.get(f"{TELEGRAM_API_URL}/getUpdates?offset={last_update_id}&timeout=10")
            updates = resp.json().get("result", [])
            for update in updates:
                last_update_id = update["update_id"] + 1
                if "message" in update and "text" in update["message"]:
                    text = update["message"]["text"]
                    current_chat_id = update["message"]["chat"]["id"]
                    pending_messages.append(text)
                    print(f"\n[Telegram Bridge] Received: {text}")
        except Exception as e:
            print(f"Telegram polling error: {e}")
        time.sleep(1)

if __name__ == '__main__':
    threading.Thread(target=poll_telegram, daemon=True).start()
    print("Telegram Bridge running on port 8000...")
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
