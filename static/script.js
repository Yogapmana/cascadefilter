document.getElementById('btn-generate').addEventListener('click', async () => {
    const res = await fetch('/generate_audio', { method: 'POST' });
    const data = await res.json();
    setAudioInput(data.file);
});

document.getElementById('file-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    const res = await fetch('/upload_audio', {
        method: 'POST',
        body: formData
    });
    const data = await res.json();
    setAudioInput(data.file);
});

function setAudioInput(filename) {
    const container = document.getElementById('input-audio-container');
    const audio = document.getElementById('audio-input');
    audio.src = `/audio/${filename}?t=${new Date().getTime()}`;
    container.classList.remove('hidden');
}

async function processAudio(method) {
    const loader = document.getElementById('loader');
    loader.classList.remove('hidden');
    
    const res = await fetch('/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ method: method })
    });
    const data = await res.json();
    loader.classList.add('hidden');
    
    if (data.error) {
        alert(data.error);
        return;
    }
    
    const container = document.getElementById('output-audio-container');
    const audio = document.getElementById('audio-output');
    audio.src = `/audio/${data.file}?t=${new Date().getTime()}`;
    container.classList.remove('hidden');
    
    const statsList = document.getElementById('stats-list');
    const li = document.createElement('li');
    li.textContent = `[${method.toUpperCase()}] Waktu Eksekusi di Server: ${data.time.toFixed(5)} detik`;
    statsList.appendChild(li);
}
