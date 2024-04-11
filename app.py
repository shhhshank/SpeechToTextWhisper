from flask import Flask,render_template,request
from flask_socketio import SocketIO, emit
import subprocess

import argparse
import numpy as np
import whisper
import torch
import base64
import traceback

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

model = None

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)
    print("Model loaded: ", model)
    return audio_model

def process_wav_bytes(webm_bytes: bytes, sample_rate: int = 16000):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
        temp_file.write(webm_bytes)
        temp_file.flush()
        waveform = whisper.load_audio(temp_file.name, sr=sample_rate)
        return waveform

def recorder_callback(ws):
    while not ws.closed:
        message = ws.receive()
        if message:
            print('message received', len(message), type(message))
            try:
                if isinstance(message, str):
                    message = base64.b64decode(message)
                audio = process_wav_bytes(bytes(message)).reshape(1, -1)
                audio = whisper.pad_or_trim(audio)
                transcription = whisper.transcribe(
                    model,
                    audio
                )
            except Exception as e:
                traceback.print_exc()

if __name__ == "__main__":
    model = setup()
    app = Flask(__name__)
    socketio = SocketIO(app, debug=True, cors_allowed_origins='*', async_mode='eventlet')


    @app.route('/home')
    def main():
        return render_template('index.html')

    @socketio.on("ping")
    def checkping():
        for x in range(5):
            cmd = 'ping -c 1 8.8.8.8|head -2|tail -1'
            listing1 = subprocess.run(cmd,stdout=subprocess.PIPE,text=True,shell=True)
            sid = request.sid
            emit('server', {"data1":x, "data":listing1.stdout}, room=sid)
            socketio.sleep(1)

    @socketio.on("audio_event")
    def recorder_callback(ws):
        while not ws.closed:
            message = ws.receive()
            if message:
                print('message received', len(message), type(message))
                try:
                    if isinstance(message, str):
                        message = base64.b64decode(message)
                    audio = process_wav_bytes(bytes(message)).reshape(1, -1)
                    audio = whisper.pad_or_trim(audio)
                    transcription = whisper.transcribe(
                        model,
                        audio
                    )
                except Exception as e:
                    traceback.print_exc()
    
    socketio.run(app, debug=True, port=5003)