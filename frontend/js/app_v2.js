const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const resultsContainer = document.getElementById('results-container');
const chatWindow = document.getElementById('chat-window');
const chatInput = document.getElementById('chat-input');
const sendChatBtn = document.getElementById('send-chat-btn');

let currentFile = null;
let chatMessages = [];
let currentPredictionContext = null;

// Navigation Logic
const btnStartDetect = document.getElementById('btn-start-detect');
if (btnStartDetect) {
    btnStartDetect.addEventListener('click', () => {
        document.getElementById('upload-section').scrollIntoView({ behavior: 'smooth' });
    });
}
const navDashboard = document.getElementById('nav-dashboard');
if (navDashboard) {
    navDashboard.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

// File Upload Logic
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--accent)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'rgba(126, 145, 167, 0.4)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(126, 145, 167, 0.4)';
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) return;
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Analyze Logic
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    loadingOverlay.style.display = 'flex';
    resultsContainer.style.display = 'none';

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (res.ok) {
            displayResults(data);
        } else {
            alert('Error: ' + (data.detail || 'Failed to analyze'));
        }
    } catch (err) {
        alert('Network error: ' + err.message);
    } finally {
        loadingOverlay.style.display = 'none';
    }
});

function displayResults(data) {
    const ai = data.ai_response || {};
    const disease = data.disease_info || {};

    document.getElementById('res-headline').innerText = ai.headline || disease.status || data.label;
    document.getElementById('res-summary').innerText = ai.summary || disease.info || '';
    
    const confidence = (data.confidence || 0) * 100;
    console.log("Confidence value:", confidence);
    
    const confidenceText = document.getElementById('res-confidence');
    if (confidenceText) {
        confidenceText.textContent = confidence.toFixed(1) + '%';
    }
    
    // Reset and animate the circular chart
    const circle = document.getElementById('confidence-circle');
    if (circle) {
        circle.setAttribute('stroke-dasharray', '0, 100');
        setTimeout(() => {
            circle.setAttribute('stroke-dasharray', `${confidence}, 100`);
        }, 100);
    }

    document.getElementById('res-severity').innerText = disease.severity || 'Menengah';
    document.getElementById('res-label').innerText = data.label;

    const actionsList = document.getElementById('res-actions');
    actionsList.innerHTML = '';
    const actions = (ai.actions && ai.actions.length > 0) ? ai.actions : [disease.treatment];
    actions.forEach(action => {
        const li = document.createElement('li');
        li.innerText = action;
        actionsList.appendChild(li);
    });

    resultsContainer.style.display = 'flex';

    // Reset Chat
    currentPredictionContext = data;
    chatMessages = [];
    chatWindow.innerHTML = '';
    addChatMessage('assistant', `Halo! Hasil analisa menunjukkan **${data.label}**. Ada yang ingin ditanyakan lebih lanjut mengenai hasil ini?`);
}

// Chat Logic
function addChatMessage(role, text) {
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    // simple markdown bold replacement
    div.innerHTML = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendChat() {
    const text = chatInput.value.trim();
    if (!text || !currentPredictionContext) return;

    addChatMessage('user', text);
    chatMessages.push({ role: 'user', content: text });
    chatInput.value = '';

    // Create api payload
    const contextMsg = {
        role: "system",
        content: `Kamu adalah asisten pertanian ahli penyakit pisang. User sedang melihat hasil deteksi gambar daun pisangnya dengan hasil: ${currentPredictionContext.label} (Tingkat Keyakinan: ${(currentPredictionContext.confidence * 100).toFixed(2)}%). Berikan jawaban yang membantu, singkat, dan berhubungan dengan penyakit tersebut.`
    };

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: [contextMsg, ...chatMessages] })
        });
        const data = await res.json();
        if (res.ok && data.response) {
            addChatMessage('assistant', data.response);
            chatMessages.push({ role: 'assistant', content: data.response });
        } else {
            addChatMessage('assistant', '⚠️ Gagal mendapatkan respon.');
        }
    } catch (err) {
        addChatMessage('assistant', '⚠️ Kesalahan jaringan saat menghubungi AI.');
    }
}

sendChatBtn.addEventListener('click', sendChat);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChat();
});
