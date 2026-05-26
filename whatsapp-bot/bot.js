require('dotenv').config();
const {
    default: makeWASocket,
    useMultiFileAuthState,
    DisconnectReason,
    downloadContentFromMessage,
    fetchLatestBaileysVersion,
    Browsers
} = require('@whiskeysockets/baileys');
const qrcode = require('qrcode-terminal');
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
const pino = require('pino');

// Load environment variables
const BACKEND_API_URL = process.env.BACKEND_API_URL || 'http://localhost:8000';

// Initialize session state in memory
const userSessions = new Map(); // JID -> { lastPrediction: data, chatHistory: [] }

// Helper function to safely extract the actual message content (bypasses ephemeral/viewOnce wrappers)
function getMessage(m) {
    if (!m) return null;
    let message = m.message;
    if (!message) return null;
    
    if (message.ephemeralMessage) {
        message = message.ephemeralMessage.message;
    }
    if (message.viewOnceMessage) {
        message = message.viewOnceMessage.message;
    }
    if (message.viewOnceMessageV2) {
        message = message.viewOnceMessageV2.message;
    }
    return message;
}

// Helper function to extract text content from a message
function getMessageText(message) {
    if (!message) return '';
    if (message.conversation) return message.conversation;
    if (message.extendedTextMessage?.text) return message.extendedTextMessage.text;
    if (message.imageMessage?.caption) return message.imageMessage.caption;
    return '';
}

async function connectToWhatsApp() {
    console.log('🤖 Menginisialisasi koneksi WhatsApp...');
    
    // Fetch latest WhatsApp version dynamically with fallback
    let version = [2, 3000, 1015901307]; // Safe fallback version
    try {
        const latest = await fetchLatestBaileysVersion();
        version = latest.version;
        console.log(`🔄 Menggunakan versi WhatsApp Web terbaru: v${version.join('.')}`);
    } catch (e) {
        console.log(`⚠️ Gagal mengambil versi WA terbaru, menggunakan fallback: v${version.join('.')}`);
    }
    
    // Auth state directory
    const authFolder = path.join(__dirname, 'auth_info');
    const { state, saveCreds } = await useMultiFileAuthState(authFolder);
    
    // Create socket connection
    const sock = makeWASocket({
        version,
        auth: state,
        browser: Browsers.macOS('Desktop'),
        logger: pino({ level: 'silent' }), // Suppress verbose log noise
        printQRInTerminal: false // We will handle rendering QR manually using qrcode-terminal
    });
    
    // Handle credentials update to persist sessions
    sock.ev.on('creds.update', saveCreds);
    
    // Connection updates handler
    sock.ev.on('connection.update', async (update) => {
        const { connection, lastDisconnect, qr } = update;
        
        if (qr) {
            console.log('\n======================================================');
            console.log('⚡ PAIRED QR CODE DIBAWAH INI DENGAN WHATSAPP ANDA:');
            console.log('======================================================\n');
            qrcode.generate(qr, { small: true });
            console.log('\n💡 Tips: Buka WhatsApp -> Perangkat Tertaut -> Tautkan Perangkat.');
        }
        
        if (connection === 'close') {
            const statusCode = lastDisconnect?.error?.output?.statusCode || lastDisconnect?.error?.statusCode;
            const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
            console.log(`🔌 Koneksi terputus (status code: ${statusCode}). Reconnect: ${shouldReconnect}`);
            console.error('Error detail:', lastDisconnect?.error);
            
            if (shouldReconnect) {
                connectToWhatsApp();
            } else {
                console.log('❌ Anda telah logout dari perangkat. Hapus folder auth_info untuk scan ulang.');
            }
        } else if (connection === 'open') {
            console.log('\n======================================================');
            console.log('✅ BANANA DOCTOR AI BOT SUDAH AKTIF DAN TERKONEKSI! 🍌');
            console.log('======================================================\n');
        }
    });
    
    // Message handler
    sock.ev.on('messages.upsert', async (m) => {
        // We only care about new messages
        if (m.type !== 'notify') return;
        
        for (const msg of m.messages) {
            // Ignore messages from status, group broadcasts, or sent by ourselves
            if (msg.key.fromMe) continue;
            
            const senderJid = msg.key.remoteJid;
            // Ignore group messages for simple 1-on-1 private chat support
            if (senderJid.endsWith('@g.us')) continue;
            
            const message = getMessage(msg);
            if (!message) continue;
            
            const isImage = !!message.imageMessage;
            const text = getMessageText(message).trim();
            
            // 1. Process Image upload (Banana Leaf Analysis)
            if (isImage) {
                console.log(`📸 Menerima foto dari ${senderJid}, memulai analisa...`);
                
                try {
                    // Send processing state to user
                    await sock.sendMessage(senderJid, { 
                        text: '📥 *Gambar diterima!* Sedang menganalisa daun pisang menggunakan AI...' 
                    });
                    
                    // Download image content stream
                    const imageMessage = message.imageMessage;
                    const stream = await downloadContentFromMessage(imageMessage, 'image');
                    let buffer = Buffer.from([]);
                    for await (const chunk of stream) {
                        buffer = Buffer.concat([buffer, chunk]);
                    }
                    
                    // Prepare multipart form data
                    const form = new FormData();
                    form.append('file', buffer, {
                        filename: 'leaf.jpg',
                        contentType: 'image/jpeg'
                    });
                    
                    console.log(`📡 Mengirim gambar ke backend: ${BACKEND_API_URL}/api/predict...`);
                    const response = await axios.post(`${BACKEND_API_URL}/api/predict`, form, {
                        headers: {
                            ...form.getHeaders(),
                            'Accept': 'application/json'
                        },
                        timeout: 30000 // 30 seconds timeout for ML + LLM generation
                    });
                    
                    const data = response.data;
                    console.log(`✅ Diagnosa selesai: ${data.label} (confidence: ${(data.confidence * 100).toFixed(2)}%)`);
                    
                    // Format response for WhatsApp markdown
                    const confidencePct = (data.confidence * 100).toFixed(2);
                    const severity = data.disease_info?.severity || 'Perlu ditinjau';
                    
                    let responseText = '';
                    if (data.label === 'Healthy') {
                        responseText = `🍌 *HASIL DIAGNOSA BANANA DOCTOR AI* 🍌\n\n` +
                            `*Hasil:* Daun Sehat ✨\n` +
                            `*Confidence:* ${confidencePct}%\n` +
                            `*Tingkat Keparahan:* -\n\n` +
                            `*Penjelasan:*\n` +
                            `Daun pisang Anda tampak sehat dan tidak menunjukkan gejala penyakit aktif. Tetap lakukan pemeliharaan rutin!\n\n` +
                            `🛡️ *Langkah Pemeliharaan:*\n` +
                            `- Pastikan drainase tanah bekerja dengan baik agar air tidak menggenang.\n` +
                            `- Bersihkan gulma di sekitar pangkal pohon pisang secara teratur.\n` +
                            `- Lakukan pemupukan seimbang (NPK) sesuai dengan fase pertumbuhan tanaman.\n` +
                            `- Pantau kesehatan daun baru secara berkala untuk deteksi dini gejala lain.\n\n` +
                            `💬 _Jika ada pertanyaan seputar pemeliharaan pisang, silakan ketik pesan Anda langsung di sini!_`;
                    } else {
                        const ai = data.ai_response;
                        const headline = ai?.headline || data.disease_info?.status || data.label;
                        const summary = ai?.summary || data.disease_info?.info || 'Informasi detail belum tersedia.';
                        const actions = ai?.actions || [data.disease_info?.treatment || 'Lakukan pengecekan lapangan.'];
                        const prevention = ai?.prevention || ['Lakukan monitoring berkala.', 'Jaga kebersihan area kebun.'];
                        const warning = ai?.warning || 'Segera konsultasikan dengan ahli agronomi jika gejala meluas.';
                        
                        responseText = `🍌 *HASIL DIAGNOSA BANANA DOCTOR AI* 🍌\n\n` +
                            `*Hasil:* ${headline}\n` +
                            `*Penyakit Terdeteksi:* ${data.label}\n` +
                            `*Confidence:* ${confidencePct}%\n` +
                            `*Tingkat Keparahan:* ${severity}\n\n` +
                            `*Penjelasan:*\n` +
                            `${summary}\n\n` +
                            `📋 *Saran Tindakan:*\n` +
                            actions.map((act, index) => `${index + 1}. ${act}`).join('\n') + `\n\n` +
                            `🛡️ *Langkah Pencegahan:*\n` +
                            prevention.map((prev) => `- ${prev}`).join('\n') + `\n\n` +
                            `⚠️ *Peringatan:*\n` +
                            `_${warning}_\n\n` +
                            `💬 _Anda dapat membalas pesan ini langsung untuk bertanya lebih lanjut mengenai penyakit ini! Asisten AI kami siap menjawab._`;
                    }
                    
                    // Save context in session
                    userSessions.set(senderJid, {
                        lastPrediction: data,
                        chatHistory: [
                            {
                                role: 'assistant',
                                content: `Halo! Hasil analisa menunjukkan daun pisang Anda terdeteksi: *${data.label}* dengan tingkat keyakinan *${confidencePct}%*. Apakah ada yang ingin ditanyakan mengenai penanganan penyakit ini?`
                            }
                        ]
                    });
                    
                    // Reply to the user
                    await sock.sendMessage(senderJid, { text: responseText });
                    
                } catch (error) {
                    console.error('❌ Error processing image:', error.message);
                    let errMsg = '❌ *Gagal menganalisa gambar.*\n\nPastikan server backend Anda sedang aktif dan silakan coba lagi beberapa saat lagi.';
                    if (error.code === 'ECONNREFUSED') {
                        errMsg = '❌ *Koneksi ke backend gagal.*\n\nServer FastAPI di `localhost:8000` tidak merespon. Pastikan server FastAPI sudah dijalankan menggunakan `uvicorn api:app --reload`.';
                    }
                    await sock.sendMessage(senderJid, { text: errMsg });
                }
                
            } 
            // 2. Process Text message (Chat assistant or Greeting commands)
            else if (text) {
                const lowerText = text.toLowerCase();
                
                // Reset session command
                if (lowerText === 'reset' || lowerText === 'restart') {
                    userSessions.delete(senderJid);
                    await sock.sendMessage(senderJid, { 
                        text: '🔄 *Sesi obrolan Anda telah direset.* Silakan kirimkan foto daun pisang baru untuk memulai analisa.' 
                    });
                    continue;
                }
                
                const session = userSessions.get(senderJid);
                
                // If there is an active prediction session, route to LLM Chatbot
                if (session && session.lastPrediction) {
                    console.log(`💬 Menerima pertanyaan chatbot dari ${senderJid}: "${text}"`);
                    
                    try {
                        // Send typing indicator
                        await sock.sendPresenceUpdate('composing', senderJid);
                        
                        // Push user message to history
                        session.chatHistory.push({
                            role: 'user',
                            content: text
                        });
                        
                        // Formulate messages with specialized agriculture system context
                        const systemMsg = {
                            role: 'system',
                            content: `Kamu adalah asisten agrikultur ahli penyakit pisang. User sedang melihat hasil deteksi gambar daun pisangnya dengan hasil: ${session.lastPrediction.label} (Tingkat Keyakinan: ${(session.lastPrediction.confidence*100).toFixed(2)}%). Berikan jawaban yang membantu, singkat, dan berhubungan dengan penyakit tersebut dalam Bahasa Indonesia. Rujuk nama fungisida komersial atau agens hayati jika ditanya.`
                        };
                        
                        // Keep history bounded to avoid token issues
                        const messageWindow = session.chatHistory.slice(-10); // Keep last 10 exchanges
                        const payloadMessages = [systemMsg, ...messageWindow];
                        
                        console.log(`📡 Menghubungi API Chatbot...`);
                        const chatResponse = await axios.post(`${BACKEND_API_URL}/api/chat`, {
                            messages: payloadMessages
                        }, {
                            timeout: 20000
                        });
                        
                        const botReplyText = chatResponse.data.response;
                        
                        // Push assistant reply to history
                        session.chatHistory.push({
                            role: 'assistant',
                            content: botReplyText
                        });
                        
                        // Reply to WhatsApp user
                        await sock.sendMessage(senderJid, { text: botReplyText });
                        
                    } catch (error) {
                        console.error('❌ Error in chat chatbot:', error.message);
                        await sock.sendMessage(senderJid, { 
                            text: '⚠️ *Koneksi ke asisten chatbot terganggu.* Silakan coba tanyakan kembali.' 
                        });
                    }
                } 
                // Welcome / Info message if no active session
                else {
                    const welcomeText = `👋 *Halo! Selamat datang di Banana Doctor AI Bot.*` +
                        `\n\nSaya adalah asisten pintar yang dapat membantu mendeteksi penyakit daun pisang Anda secara instan menggunakan Computer Vision & LLM AI.` +
                        `\n\n📸 *Cara Penggunaan:*` +
                        `\n1. *Kirimkan foto* daun pisang Anda yang mengalami gejala penyakit.` +
                        `\n2. Tunggu beberapa detik selagi AI melakukan analisa mendalam.` +
                        `\n3. Anda akan menerima hasil diagnosa lengkap beserta *saran penanganan spesifik* (termasuk rekomendasi merk produk pertanian).` +
                        `\n4. Setelah menerima hasil, Anda dapat *membalas langsung* dengan pertanyaan seputar penyakit atau tips perawatannya.` +
                        `\n\n🔄 Ketik *reset* kapan saja untuk menghapus memori obrolan dan memulai dari awal.` +
                        `\n\n_Silakan kirimkan foto daun pisang Anda sekarang untuk mulai mendeteksi!_ 🍌`;
                    
                    await sock.sendMessage(senderJid, { text: welcomeText });
                }
            }
        }
    });
}

// Start connection
connectToWhatsApp().catch(err => {
    console.error('❌ Terjadi error kritis saat startup:', err);
});
