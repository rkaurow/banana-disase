// Dynamic API URL path resolver to support trailing slashes, subpaths, and tunnels
const getApiUrl = (endpoint) => {
    const path = window.location.pathname;
    if (path.endsWith('/') || !path.includes('.')) {
        const base = path.endsWith('/') ? path : path + '/';
        return base + endpoint;
    }
    const base = path.substring(0, path.lastIndexOf('/') + 1);
    return base + endpoint;
};

// ==================== 1. HOMEPAGE & DASHBOARD STAGE ROUTER ====================
const viewHomepage = document.getElementById('view-homepage');
const viewAppDashboard = document.getElementById('view-app-dashboard');

const homeBtnEnter = document.getElementById('home-btn-enter');
const heroBtnStart = document.getElementById('hero-btn-start');
const ctaBtnEnter = document.getElementById('cta-btn-enter');
const sidebarBrandLogo = document.getElementById('sidebar-brand-logo');

function enterDashboard() {
    viewHomepage.classList.add('hidden');
    viewAppDashboard.classList.remove('hidden');
    
    // Ensure scroll resets to top of dashboard when entering
    window.scrollTo({ top: 0 });
    
    // Default to the Leaf Diagnosis Center tab
    switchTab('tab-btn-detect');
}

function returnToHomepage() {
    viewAppDashboard.classList.add('hidden');
    viewHomepage.classList.remove('hidden');
    
    // Scroll to top of homepage
    window.scrollTo({ top: 0 });
}

// Bind CTA clicks
if (homeBtnEnter) homeBtnEnter.addEventListener('click', enterDashboard);
if (heroBtnStart) heroBtnStart.addEventListener('click', enterDashboard);
if (ctaBtnEnter) ctaBtnEnter.addEventListener('click', enterDashboard);
if (sidebarBrandLogo) sidebarBrandLogo.addEventListener('click', returnToHomepage);


// ==================== 2. THEME TOGGLE LOGIC ====================
const themeToggleBtn = document.getElementById('theme-toggle');
const themeToggleDarkIcon = document.getElementById('theme-toggle-dark-icon');
const themeToggleLightIcon = document.getElementById('theme-toggle-light-icon');

// Initialize theme from localStorage or system prefers-color-scheme
if (localStorage.getItem('color-theme') === 'dark' || (!('color-theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark');
    if (themeToggleLightIcon) themeToggleLightIcon.classList.remove('hidden');
} else {
    document.documentElement.classList.remove('dark');
    if (themeToggleDarkIcon) themeToggleDarkIcon.classList.remove('hidden');
}

if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', function() {
        if (themeToggleDarkIcon) themeToggleDarkIcon.classList.toggle('hidden');
        if (themeToggleLightIcon) themeToggleLightIcon.classList.toggle('hidden');

        if (localStorage.getItem('color-theme')) {
            if (localStorage.getItem('color-theme') === 'light') {
                document.documentElement.classList.add('dark');
                localStorage.setItem('color-theme', 'dark');
            } else {
                document.documentElement.classList.remove('dark');
                localStorage.setItem('color-theme', 'light');
            }
        } else {
            if (document.documentElement.classList.contains('dark')) {
                document.documentElement.classList.remove('dark');
                localStorage.setItem('color-theme', 'light');
            } else {
                document.documentElement.classList.add('dark');
                localStorage.setItem('color-theme', 'dark');
            }
        }
    });
}


// ==================== 3. SAAS DYNAMIC TAB SWITCHING ====================
const tabBtnDetect = document.getElementById('tab-btn-detect');
const tabBtnChat = document.getElementById('tab-btn-chat');
const tabBtnDatabase = document.getElementById('tab-btn-database');

const viewDetect = document.getElementById('view-detect');
const viewChat = document.getElementById('view-chat');
const viewDatabase = document.getElementById('view-database');

const tabs = [
    { button: tabBtnDetect, view: viewDetect },
    { button: tabBtnChat, view: viewChat },
    { button: tabBtnDatabase, view: viewDatabase }
];

function switchTab(activeId) {
    tabs.forEach(tab => {
        if (!tab.button || !tab.view) return;
        if (tab.button.id === activeId) {
            // Show Active View
            tab.view.classList.remove('hidden');
            
            // Set Selected Button Styles
            tab.button.className = "w-full px-4 py-3 text-sm font-semibold rounded-xl text-yellow-600 dark:text-yellow-450 bg-yellow-500/10 dark:bg-yellow-500/10 border-l-4 border-yellow-500 flex items-center gap-3 transition-all duration-200 shadow-sm";
        } else {
            // Hide Inactive View
            tab.view.classList.add('hidden');
            
            // Set Unselected Button Styles
            tab.button.className = "w-full px-4 py-3 text-sm font-medium rounded-xl text-slate-500 dark:text-slate-400 hover:text-slate-850 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-900/50 flex items-center gap-3 transition-all duration-200 border-l-4 border-transparent";
        }
    });
}

if (tabBtnDetect) tabBtnDetect.addEventListener('click', () => switchTab('tab-btn-detect'));
if (tabBtnChat) tabBtnChat.addEventListener('click', () => switchTab('tab-btn-chat'));
if (tabBtnDatabase) tabBtnDatabase.addEventListener('click', () => switchTab('tab-btn-database'));


// ==================== 4. ENCYCLOPEDIA FILTER SEARCH ====================
const diseaseSearchInput = document.getElementById('disease-search');
const diseaseCardsGrid = document.getElementById('disease-cards-grid');
const diseaseCards = document.querySelectorAll('.disease-card');
const diseaseSearchEmpty = document.getElementById('disease-search-empty');

if (diseaseSearchInput) {
    diseaseSearchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        let matchCount = 0;
        
        diseaseCards.forEach(card => {
            const dataName = card.getAttribute('data-name').toLowerCase();
            if (dataName.includes(query)) {
                card.classList.remove('hidden');
                matchCount++;
            } else {
                card.classList.add('hidden');
            }
        });
        
        if (matchCount === 0) {
            if (diseaseCardsGrid) diseaseCardsGrid.classList.add('hidden');
            if (diseaseSearchEmpty) diseaseSearchEmpty.classList.remove('hidden');
        } else {
            if (diseaseCardsGrid) diseaseCardsGrid.classList.remove('hidden');
            if (diseaseSearchEmpty) diseaseSearchEmpty.classList.add('hidden');
        }
    });
}


// ==================== 5. LEAF DIAGNOSTIC CENTER CORE LOGIC ====================
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingOverlay = document.getElementById('loading-overlay');

const detectPlaceholder = document.getElementById('detect-placeholder');
const resultsContainer = document.getElementById('results-container');
const chatWindow = document.getElementById('chat-window');
const chatInput = document.getElementById('chat-input');
const sendChatBtn = document.getElementById('send-chat-btn');

let currentFile = null;
let chatMessages = [];
let currentPredictionContext = null;

// File Upload Trigger Event
if (dropZone) {
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-yellow-400');
        dropZone.classList.remove('border-slate-200', 'dark:border-slate-800');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-yellow-400');
        dropZone.classList.add('border-slate-200', 'dark:border-slate-800');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-yellow-400');
        dropZone.classList.add('border-slate-200', 'dark:border-slate-800');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}

if (fileInput) {
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFile(fileInput.files[0]);
        }
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) return;
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        if (previewImage) {
            previewImage.src = e.target.result;
            previewImage.classList.remove('hidden');
        }
        if (analyzeBtn) analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Model Analysis Action
if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        if (loadingOverlay) loadingOverlay.classList.remove('hidden');
        
        // Smoothly close active result displays while processing
        if (resultsContainer) resultsContainer.classList.add('hidden');
        if (detectPlaceholder) detectPlaceholder.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const res = await fetch(getApiUrl('api/predict'), {
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
            if (loadingOverlay) loadingOverlay.classList.add('hidden');
        }
    });
}

function displayResults(data) {
    const ai = data.ai_response || {};
    const disease = data.disease_info || {};

    // Map Headline & Description text
    const resHeadline = document.getElementById('res-headline');
    const resSummary = document.getElementById('res-summary');
    if (resHeadline) resHeadline.innerText = ai.headline || disease.status || data.label;
    if (resSummary) resSummary.innerText = ai.summary || disease.info || '';
    
    const confidence = (data.confidence || 0) * 100;
    
    const confidenceText = document.getElementById('res-confidence');
    if (confidenceText) {
        confidenceText.textContent = confidence.toFixed(1) + '%';
    }
    
    // Reset and animate the circular SVG gauge progress circle
    const circle = document.getElementById('confidence-circle');
    if (circle) {
        circle.setAttribute('stroke-dasharray', '0, 100');
        setTimeout(() => {
            circle.setAttribute('stroke-dasharray', `${confidence}, 100`);
        }, 100);
    }

    const resSeverity = document.getElementById('res-severity');
    const resLabel = document.getElementById('res-label');
    if (resSeverity) resSeverity.innerText = disease.severity || 'Menengah';
    if (resLabel) resLabel.innerText = data.label;

    // Render Checklist Actions
    const actionsList = document.getElementById('res-actions');
    if (actionsList) {
        actionsList.innerHTML = '';
        const actions = (ai.actions && ai.actions.length > 0) ? ai.actions : [disease.treatment];
        
        actions.forEach(action => {
            const li = document.createElement('li');
            li.className = "flex items-start gap-2.5 bg-emerald-500/[0.04] dark:bg-emerald-500/[0.02] border border-emerald-500/10 dark:border-emerald-500/5 p-3.5 rounded-2xl text-slate-700 dark:text-slate-355 text-xs md:text-sm leading-relaxed transition-all duration-300 hover:translate-x-1 hover:border-emerald-500/30";
            li.innerHTML = `
                <span class="w-5.5 h-5.5 rounded-full bg-emerald-500/10 dark:bg-emerald-500/5 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shrink-0 text-[10px] font-bold mt-0.5 shadow-sm">
                    ✓
                </span>
                <span>${action}</span>
            `;
            actionsList.appendChild(li);
        });
    }

    // Toggle Dynamic Displays
    if (detectPlaceholder) detectPlaceholder.classList.add('hidden');
    if (resultsContainer) resultsContainer.classList.remove('hidden');
    
    // Scroll smoothly to results card
    if (resultsContainer) resultsContainer.scrollIntoView({ behavior: 'smooth' });

    // Initialize Conversation Context
    currentPredictionContext = data;
    chatMessages = [];
    if (chatWindow) {
        chatWindow.innerHTML = '';
        // Insert welcome greeting bubble
        addChatMessage('assistant', `Halo! Hasil analisa menunjukkan daun pisang mengidap **${data.label}**. Ada yang ingin ditanyakan lebih lanjut mengenai hasil ini?`);
    }
}


// ==================== 6. Conversational CHATBOT LOGIC ====================
function addChatMessage(role, text) {
    if (!chatWindow) return;
    const div = document.createElement('div');
    div.className = "p-3.5 rounded-2xl max-w-[85%] text-xs md:text-sm leading-relaxed transition-all duration-300 flex flex-col gap-1 shadow-sm animate-[fadeIn_0.3s_ease-out]";
    
    if (role === 'user') {
        div.className += ' bg-gradient-to-r from-yellow-400 to-amber-500 text-slate-950 font-medium self-end rounded-br-sm';
    } else {
        div.className += ' bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/60 text-slate-700 dark:text-slate-300 self-start rounded-bl-sm';
    }
    
    // Replace markdown double asterisk bold triggers with bold text tags
    div.innerHTML = text.replace(/\*\*(.*?)\*\*/g, '<strong class="font-extrabold text-slate-950 dark:text-white">$1</strong>');
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendChat() {
    if (!chatInput) return;
    const text = chatInput.value.trim();
    if (!text) return;
    
    // If user has not performed an image analysis yet, establish standard context
    const contextLabel = currentPredictionContext ? currentPredictionContext.label : "Healthy / Belum Ada";
    const contextConfidence = currentPredictionContext ? (currentPredictionContext.confidence * 100).toFixed(2) + "%" : "100%";

    addChatMessage('user', text);
    chatMessages.push({ role: 'user', content: text });
    chatInput.value = '';

    const contextMsg = {
        role: "system",
        content: `Kamu adalah asisten pertanian ahli penyakit pisang. User sedang melihat hasil deteksi gambar daun pisangnya dengan hasil: ${contextLabel} (Tingkat Keyakinan: ${contextConfidence}). Berikan jawaban yang membantu, singkat, dan berhubungan dengan penyakit tersebut.`
    };

    try {
        const res = await fetch(getApiUrl('api/chat'), {
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

if (sendChatBtn) sendChatBtn.addEventListener('click', sendChat);
if (chatInput) {
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChat();
    });
}


// ==================== 7. CHATBOT SUGGESTION CHIPS LOGIC ====================
const suggestionChips = document.querySelectorAll('.chat-chip');
suggestionChips.forEach(chip => {
    chip.addEventListener('click', () => {
        if (!chatInput) return;
        const queryText = chip.textContent.trim();
        chatInput.value = queryText;
        sendChat();
    });
});
