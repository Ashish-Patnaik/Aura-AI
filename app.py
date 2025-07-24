# app.py
import os
import json
import re
import asyncio
import nest_asyncio
import aiohttp 
from gevent.timeout import Timeout

from flask import Flask, render_template
from flask_cors import CORS
from flask_sock import Sock

# Apply nest_asyncio to allow asyncio to run within gevent/flask
nest_asyncio.apply()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set! Please set it in your Render environment.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent"
FIXED_VOICE = "en-US-JennyNeural"
SYSTEM_PROMPT = """
You are "Aura," a friendly and helpful AI voice assistant. Keep your responses concise, natural, and to the point, as if you were speaking in a real conversation. Do not use markdown or formatting.
"""

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)

# --- Text-to-Speech Helper ---
async def tts_streamer(text_chunk: str, websocket):
    """
    Uses edge-tts to generate audio for a text chunk and streams it to the websocket.
    """
    # === THE KEY FIX: Import edge_tts inside the async function ===
    # This ensures the module is available in the async context where it's executed.
    import edge_tts 
    
    try:
        communicate = edge_tts.Communicate(text=text_chunk, voice=FIXED_VOICE)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                if websocket.connected:
                    websocket.send(chunk["data"])
                else: 
                    # Stop if the client has disconnected
                    break
    except Exception as e:
        print(f"Error during TTS generation: {e}")

# --- Main WebSocket Route ---
@sock.route('/stream')
def stream(ws):
    """Handles a long-lived WebSocket connection for a continuous conversation."""
    
    print("WebSocket connection established. Initializing conversation history.")
    history = [
        {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
        {"role": "model", "parts": [{"text": "Okay, I'm ready to help."}]}
    ]

    async def main_conversation_loop():
        """The main async loop to be run for the duration of the connection."""
        while ws.connected:
            try:
                # Wait for a message from the client
                user_prompt = ws.receive(timeout=3600) # 1 hour timeout
                if user_prompt is None:
                    continue # If timeout occurs, continue waiting

                print(f"Received prompt: {user_prompt}")
                history.append({"role": "user", "parts": [{"text": user_prompt}]})
                
                headers = {'Content-Type': 'application/json'}
                payload = {"contents": history}
                full_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}&alt=sse"

                # Use aiohttp for the async request to Gemini API
                async with aiohttp.ClientSession() as session:
                    async with session.post(full_url, headers=headers, json=payload) as resp:
                        if not resp.ok:
                            error_text = await resp.text()
                            print(f"Gemini API Error: {resp.status} - {error_text}")
                            if ws.connected:
                                ws.send(json.dumps({"type": "error", "message": "The AI is having trouble, please try again."}))
                            continue # Wait for next prompt

                        # Process the streaming response
                        text_buffer = ""
                        full_model_response = ""
                        sentence_end_re = re.compile(r'(.*?[.!?]["”’]?)', re.DOTALL)
                        
                        if ws.connected:
                            ws.send(json.dumps({"type": "start_of_response"}))
                        
                        async for line in resp.content:
                            if not ws.connected: break
                            decoded_line = line.decode('utf-8')
                            
                            if decoded_line.startswith('data: '):
                                try:
                                    chunk_str = decoded_line[6:]
                                    chunk = json.loads(chunk_str)
                                    text_chunk = chunk["candidates"][0]["content"]["parts"][0]["text"]
                                    
                                    text_buffer += text_chunk
                                    full_model_response += text_chunk

                                    match = sentence_end_re.search(text_buffer)
                                    while match:
                                        sentence = match.group(1).strip()
                                        text_buffer = text_buffer[match.end():].lstrip()
                                        if sentence:
                                            # Send text chunk for display and generate audio
                                            ws.send(json.dumps({"type": "text_chunk", "data": sentence}))
                                            await tts_streamer(sentence, ws)
                                        if not ws.connected: break
                                        match = sentence_end_re.search(text_buffer)
                                    if not ws.connected: break
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    pass # Ignore incomplete JSON chunks
                        
                        # Handle any text remaining in the buffer
                        if ws.connected and text_buffer.strip():
                            remaining_text = text_buffer.strip()
                            ws.send(json.dumps({"type": "text_chunk", "data": remaining_text}))
                            await tts_streamer(remaining_text, ws)
                        
                        # Update history with the full response for conversational context
                        if full_model_response.strip():
                            history.append({"role": "model", "parts": [{"text": full_model_response.strip()}]})

                        if ws.connected:
                            ws.send(json.dumps({"type": "end_of_response"}))

            except Timeout:
                print("WebSocket timed out while waiting for a message.")
                break # Exit the loop if client is inactive for too long
            except Exception as e:
                print(f"An unexpected error occurred in the WebSocket loop: {e}")
                break

    # Get the patched asyncio loop and run our main async function in it
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main_conversation_loop())
    finally:
        print("Main conversation loop has ended. Closing WebSocket connection.")

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

if __name__ == '__main__':
    # This block is for local development only.
    # On Render, you should be using a Gunicorn command.
    print("Starting Flask server for local development...")
    app.run(host='0.0.0.0', port=5000, debug=True)
