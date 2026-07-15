import os
import subprocess
import wave
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    fs = 44100
    t = np.linspace(0, 3.0, int(fs * 3.0)) # 3 seconds
    clean_signal = np.sin(2 * np.pi * 440 * t) 
    noise = 0.5 * np.sin(2 * np.pi * 5000 * t) + 0.3 * np.random.randn(len(t))
    signal = clean_signal + noise
    
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    
    filepath = os.path.join(UPLOAD_FOLDER, 'input.wav')
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(fs)
        wav_file.writeframes(signal.tobytes())
        
    return jsonify({"status": "success", "file": "input.wav"})

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, 'input.wav')
    file.save(filepath)
    return jsonify({"status": "success", "file": "input.wav"})

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    method = data.get('method', 'baseline')
    
    input_wav = os.path.join(UPLOAD_FOLDER, 'input.wav')
    if not os.path.exists(input_wav):
        return jsonify({"error": "No audio file found"}), 400
        
    with wave.open(input_wav, 'r') as wav_file:
        fs = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        audio_data /= 32768.0 

    in_bin = os.path.join(UPLOAD_FOLDER, 'input.bin')
    out_bin = os.path.join(UPLOAD_FOLDER, 'output.bin')
    
    with open(in_bin, 'wb') as f:
        f.write(audio_data.tobytes())

    # ./dsp_engine method chunk_size num_matrices input output
    # Kita menggunakan matriks berukuran 512x512 sebanyak 8 buah
    # Agar komputasinya cukup berat (ratusan juta operasi) sehingga 
    # kecepatan OpenMP (Paralel) bisa melampaui "overhead" pembuatan thread.
    cmd = ['./dsp_engine', method, '512', '8', in_bin, out_bin]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        engine_time = float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Engine error: {e.stderr}"}), 500
        
    with open(out_bin, 'rb') as f:
        out_data = np.frombuffer(f.read(), dtype=np.float64)
    
    out_data = np.clip(out_data, -1.0, 1.0)
    out_data = np.int16(out_data * 32767)
    
    output_wav_name = f'output_{method}.wav'
    output_wav = os.path.join(UPLOAD_FOLDER, output_wav_name)
    with wave.open(output_wav, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(fs)
        wav_file.writeframes(out_data.tobytes())
        
    return jsonify({
        "status": "success", 
        "file": output_wav_name,
        "time": engine_time
    })

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
