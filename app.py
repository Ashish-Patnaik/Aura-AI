# app.py
import os
import json
import re
import requests
import edge_tts
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from flask import Flask, render_template
from flask_cors import CORS
from flask_sock import Sock

nest_asyncio.apply()

load_dotenv()

# --- Configuration ---
GEMINI_API = os.getenv('GEMINI_API_KEY')
if not GEMINI_API:
    raise ValueError("GEMINI_API_KEY environment variable not set!")

Model = "gemini-1.5-pro"

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{Model}:streamGenerateContent?key={GEMINI_API}&alt=sse"
FIXED_VOICE = "en-US-JennyNeural"
SYSTEM_PROMPT = """
You are "Aura," a friendly and helpful AI voice assistant. Keep your responses concise, natural, and to the point, as if you were speaking in a real conversation. Do not use markdown or formatting.
"""

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)

# --- Helper to stream TTS audio to the websocket ---
async def tts_streamer(text_chunk: str, websocket):
    try:
        communicate = edge_tts.Communicate(text=text_chunk, voice=FIXED_VOICE)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                if websocket.connected:
                    # Send audio as raw binary data
                    websocket.send(chunk["data"])
                else:
                    break
    except Exception as e:
        print(f"Error during TTS generation: {e}")


# --- Main WebSocket Route (Now Long-Lived) ---
@sock.route('/stream')
def stream(ws):
    """
    Handles a long-lived WebSocket connection for a continuous conversation.
    """
    print("WebSocket connection established. Initializing conversation history.")
    loop = asyncio.get_event_loop()

    # --- Conversation History Management ---
    # This history will be maintained for the duration of the WebSocket connection.
    history = [
        {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
        {"role": "model", "parts": [{"text": "Okay, I'm ready to help."}]}
    ]

    # Loop to process messages for the entire connection duration
    while ws.connected:
        try:
            user_prompt = ws.receive(timeout=3600) # Keep connection open for 1 hour of inactivity
            if user_prompt is None:
                continue

            print(f"Received prompt: {user_prompt}")
            
            # 1. Add user prompt to history
            history.append({"role": "user", "parts": [{"text": user_prompt}]})

            # 2. Prepare and send request to Gemini with full history
            headers = {'Content-Type': 'application/json'}
            payload = {"contents": history}
            response = requests.post(GEMINI_API_URL, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            # 3. Stream response and process
            text_buffer = ""
            full_model_response = ""
            sentence_end_re = re.compile(r'(.*?[.!?]["”’]?)', re.DOTALL)
            
            # Signal the client that a new AI response is starting
            ws.send(json.dumps({"type": "start_of_response"}))

            for line in response.iter_lines():
                if not ws.connected: break
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        try:
                            chunk = json.loads(decoded_line[6:])
                            text_chunk = chunk["candidates"][0]["content"]["parts"][0]["text"]
                            text_buffer += text_chunk
                            full_model_response += text_chunk

                            match = sentence_end_re.search(text_buffer)
                            while match:
                                sentence = match.group(1).strip()
                                text_buffer = text_buffer[match.end():].lstrip()
                                
                                if sentence:
                                    # Send the text chunk to the client for display
                                    ws.send(json.dumps({"type": "text_chunk", "data": sentence}))
                                    # Stream the audio for this sentence
                                    loop.run_until_complete(tts_streamer(sentence, ws))
                                
                                if not ws.connected: break
                                match = sentence_end_re.search(text_buffer)
                            if not ws.connected: break
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
            
            # Process any remaining text in the buffer
            if ws.connected and text_buffer.strip():
                remaining_text = text_buffer.strip()
                ws.send(json.dumps({"type": "text_chunk", "data": remaining_text}))
                loop.run_until_complete(tts_streamer(remaining_text, ws))

            # 4. Add the complete model response to history for the next turn
            if full_model_response.strip():
                history.append({"role": "model", "parts": [{"text": full_model_response.strip()}]})

            # Signal the end of the AI's turn
            if ws.connected:
                ws.send(json.dumps({"type": "end_of_response"}))

        except Exception as e:
            print(f"An error occurred in the WebSocket loop: {e}")
            break # Exit the loop on error
    
    print("WebSocket processing finished. Closing connection.")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
