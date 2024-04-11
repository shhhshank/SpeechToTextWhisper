from flask import Flask, render_template
from flask_socketio import SocketIO
from sys import platform
from queue import Queue
from datetime import datetime, timedelta
import numpy as np

import argparse
import whisper
import base64
import traceback
import tempfile
import torch

app = Flask(__name__)
app.config['SECRET_KEY'] = "secret_key"
socket = SocketIO(app, debug=True, cors_allowed_origins='*', async_mode='eventlet')

args = None
phrase_time = None
data_queue = None
audio_model = None
record_timeout = None
phrase_timeout = None
transcription = None


def process_wav_bytes(webm_bytes: bytes, sample_rate: int = 16000):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
        temp_file.write(webm_bytes)
        temp_file.flush()
        waveform = whisper.load_audio(temp_file.name, sr=sample_rate)
        return waveform

@app.route('/')
def main():
    return render_template('app.html')


@socket.on("message")
def handle_message(msg):
        global args, audio_model, record_timeout, phrase_timeout, data_queue, phrase_time, transcription

        message = msg
        if message:
            print('message received', len(message), type(message))
            try:
                if isinstance(message, str):
                    message = base64.b64decode(message)
                data_queue.put(bytes(message))

                now = datetime.utcnow()
                if not data_queue.empty():
                    phrase_complete = False
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        phrase_complete = True
                    phrase_time = now
                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()
                    audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    result = audio_model.transcribe(audio, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text
                    print(transcription, result)
                    socket.send('Output: ' + ''.join(transcription))
                    

            except Exception as e:
                traceback.print_exc()

def setup():
    global args, audio_model, record_timeout, phrase_timeout, data_queue, phrase_time, transcription
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
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()

    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']
    print("Model loaded: ", model)
    return audio_model

if __name__ == '__main__':
    setup()
    socket.run(app, port=5005, debug=True)