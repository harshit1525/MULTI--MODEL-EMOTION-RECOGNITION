/**
 * MULTIMODAL EMOTION AI — UNIFIED CLIENT APPLICATION
 * Pairs Video (ResNet-50), Audio (Wav2Vec2), and Text (Linguistic) modalities
 * into ONE unified final emotion prediction via Learned Cross-Modal Attention Fusion.
 */

const API_BASE = window.location.origin;

// Application State
let state = {
    audioFile: null,
    videoFile: null,
    mmAudioFile: null,
    mmVideoFile: null,
    history: []
};

// Emotion color tokens
const EMOTION_COLORS = {
    anger: '#ef4444',
    disgust: '#84cc16',
    fear: '#a855f7',
    joy: '#10b981',
    neutral: '#64748b',
    sadness: '#38bdf8',
    surprise: '#f59e0b'
};

const EMOTION_EMOJIS = {
    anger: '😡',
    disgust: '🤢',
    fear: '😨',
    joy: '😄',
    neutral: '😐',
    sadness: '😢',
    surprise: '😲'
};

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initDropzones();
    initTextCounter();
    checkBackendHealth();
    loadWebcamStatus();
    loadSessionHistory();
});

function initNavigation() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.getAttribute('data-tab');
            switchTab(targetId);
        });
    });
}

function switchTab(tabId) {
    document.querySelectorAll('.nav-tab').forEach(t => {
        if (t.getAttribute('data-tab') === tabId) {
            t.classList.add('active');
        } else {
            t.classList.remove('active');
        }
    });

    document.querySelectorAll('.tab-pane').forEach(p => {
        if (p.id === tabId) {
            p.classList.add('active');
        } else {
            p.classList.remove('active');
        }
    });

    if (tabId === 'tab-webcam') {
        loadWebcamStatus();
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function checkBackendHealth() {
    const statusText = document.getElementById('status-text');
    const statusDot = document.querySelector('.status-dot');
    const deviceText = document.getElementById('device-text');

    try {
        const resp = await fetch(`${API_BASE}/health`);
        if (resp.ok) {
            const data = await resp.json();
            statusText.innerText = 'System Online (v2.0 Multimodal)';
            statusDot.style.backgroundColor = '#10b981';
            deviceText.innerText = `DEVICE: ${data.device.toUpperCase()}`;
        } else {
            throw new Error(`HTTP ${resp.status}`);
        }
    } catch (err) {
        statusText.innerText = 'Backend Offline';
        statusDot.style.backgroundColor = '#ef4444';
        statusDot.classList.remove('pulsing');
        deviceText.innerText = 'DEVICE: DISCONNECTED';
        showToast('Cannot connect to backend server. Make sure backend is running.', 'error');
    }
}

// ============================================================================
// DROPZONE SETUP
// ============================================================================
function initDropzones() {
    setupDropzone(
        document.getElementById('audio-dropzone'),
        document.getElementById('audio-file-input'),
        (file) => handleAudioFileSelection(file)
    );

    setupDropzone(
        document.getElementById('text-dropzone'),
        document.getElementById('text-file-input'),
        (file) => handleTextFileSelection(file)
    );

    setupDropzone(
        document.getElementById('video-dropzone'),
        document.getElementById('video-file-input'),
        (file) => handleVideoFileSelection(file)
    );

    setupDropzone(
        document.getElementById('mm-audio-dropzone'),
        document.getElementById('mm-audio-input'),
        (file) => handleMMAudioSelection(file)
    );

    setupDropzone(
        document.getElementById('mm-video-dropzone'),
        document.getElementById('mm-video-input'),
        (file) => handleMMVideoSelection(file)
    );
}

function setupDropzone(dropzoneEl, inputEl, onFileSelected) {
    if (!dropzoneEl || !inputEl) return;

    dropzoneEl.addEventListener('click', () => inputEl.click());

    inputEl.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            onFileSelected(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzoneEl.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzoneEl.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzoneEl.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzoneEl.classList.remove('dragover');
        });
    });

    dropzoneEl.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            onFileSelected(e.dataTransfer.files[0]);
        }
    });
}

function initTextCounter() {
    const textInput = document.getElementById('text-input');
    const mmTextInput = document.getElementById('mm-text-input');

    if (textInput) {
        textInput.addEventListener('input', () => {
            const val = textInput.value;
            document.getElementById('char-count').innerText = `${val.length} characters`;
            document.getElementById('word-count').innerText = `${val.trim() ? val.trim().split(/\s+/).length : 0} words`;
        });
    }

    if (mmTextInput) {
        mmTextInput.addEventListener('input', () => {
            updateMMActiveCount();
        });
    }
}

// ============================================================================
// MULTIMODAL UNIFIED WORKSPACE
// ============================================================================
function handleMMAudioSelection(file) {
    state.mmAudioFile = file;
    document.getElementById('mm-audio-dropzone').style.display = 'none';
    const info = document.getElementById('mm-audio-info');
    info.style.display = 'flex';
    document.getElementById('mm-audio-name').innerText = file.name;
    updateMMActiveCount();
    showToast(`Audio attached: ${file.name}`, 'info');
}

function clearMMAudio() {
    state.mmAudioFile = null;
    const input = document.getElementById('mm-audio-input');
    if (input) input.value = '';
    document.getElementById('mm-audio-dropzone').style.display = 'block';
    document.getElementById('mm-audio-info').style.display = 'none';
    updateMMActiveCount();
}

function handleMMVideoSelection(file) {
    state.mmVideoFile = file;
    document.getElementById('mm-video-dropzone').style.display = 'none';
    const info = document.getElementById('mm-video-info');
    info.style.display = 'flex';
    document.getElementById('mm-video-name').innerText = file.name;

    // Optional preview player
    const playerWrapper = document.getElementById('mm-video-player-wrapper');
    const player = document.getElementById('mm-video-player');
    if (playerWrapper && player) {
        player.src = URL.createObjectURL(file);
        playerWrapper.style.display = 'block';
    }

    updateMMActiveCount();
    showToast(`Video attached: ${file.name} (Audio & transcript will sync automatically)`, 'info');
}

function clearMMVideo() {
    state.mmVideoFile = null;
    const input = document.getElementById('mm-video-input');
    if (input) input.value = '';
    document.getElementById('mm-video-dropzone').style.display = 'block';
    document.getElementById('mm-video-info').style.display = 'none';

    const playerWrapper = document.getElementById('mm-video-player-wrapper');
    if (playerWrapper) playerWrapper.style.display = 'none';

    updateMMActiveCount();
}

function updateMMActiveCount() {
    let count = 0;
    if (state.mmAudioFile) count++;
    if (document.getElementById('mm-text-input')?.value.trim()) count++;
    if (state.mmVideoFile) count++;

    const countEl = document.getElementById('mm-active-count');
    if (countEl) countEl.innerText = count;

    const btn = document.getElementById('btn-run-fusion');
    if (btn) btn.disabled = (count === 0);
}

function resetMultimodalStudio() {
    clearMMAudio();
    clearMMVideo();
    const txt = document.getElementById('mm-text-input');
    if (txt) txt.value = '';
    updateMMActiveCount();
    const res = document.getElementById('unified-result-container');
    if (res) res.style.display = 'none';
    showToast('Multimodal workspace cleared', 'info');
}

// Preset Loader
async function loadPresetTrimodal(presetId) {
    showToast('Loading multimodal preset...', 'info');

    if (presetId === 'dia0_utt0') {
        document.getElementById('mm-text-input').value = "Oh my God, he’s lost it. He’s totally lost it.";
        try {
            const respA = await fetch('/static/samples/dia0_utt0.wav');
            if (respA.ok) {
                const blobA = await respA.blob();
                handleMMAudioSelection(new File([blobA], "dia0_utt0.wav", { type: 'audio/wav' }));
            }
            const respV = await fetch('/static/samples/dia0_utt0.mp4');
            if (respV.ok) {
                const blobV = await respV.blob();
                handleMMVideoSelection(new File([blobV], "dia0_utt0.mp4", { type: 'video/mp4' }));
            }
            showToast('MELD dia0_utt0 Loaded (Trimodal Set: Phoebe - Sadness)', 'success');
        } catch (e) {
            showToast(`Preset error: ${e.message}`, 'error');
        }
    } else if (presetId === 'dia0_utt1') {
        document.getElementById('mm-text-input').value = "What?";
        try {
            const respA = await fetch('/static/samples/dia0_utt1.wav');
            if (respA.ok) {
                const blobA = await respA.blob();
                handleMMAudioSelection(new File([blobA], "dia0_utt1.wav", { type: 'audio/wav' }));
            }
            const respV = await fetch('/static/samples/dia0_utt1.mp4');
            if (respV.ok) {
                const blobV = await respV.blob();
                handleMMVideoSelection(new File([blobV], "dia0_utt1.mp4", { type: 'video/mp4' }));
            }
            showToast('MELD dia0_utt1 Loaded (Trimodal Set: Monica - Surprise)', 'success');
        } catch (e) {
            showToast(`Preset error: ${e.message}`, 'error');
        }
    } else if (presetId === 'conflict_scenario') {
        // Conflicting: Sadness video + Neutral audio + Surprise text
        document.getElementById('mm-text-input').value = "Oh my God, wow, I really didn't see that coming at all!";
        try {
            const respV = await fetch('/static/samples/dia0_utt0.mp4');
            if (respV.ok) {
                const blobV = await respV.blob();
                handleMMVideoSelection(new File([blobV], "dia0_utt0.mp4", { type: 'video/mp4' }));
            }
            const respA = await fetch('/static/samples/dia0_utt1.wav');
            if (respA.ok) {
                const blobA = await respA.blob();
                handleMMAudioSelection(new File([blobA], "dia0_utt1.wav", { type: 'audio/wav' }));
            }
            showToast('Conflicting Modalities Scenario Loaded (Sadness Video vs Neutral Audio vs Surprise Text)', 'info');
        } catch (e) {
            showToast(`Preset error: ${e.message}`, 'error');
        }
    } else if (presetId === 'joy_sample') {
        clearMMAudio();
        clearMMVideo();
        document.getElementById('mm-text-input').value = "I am so incredibly happy and proud of what we accomplished today!";
        updateMMActiveCount();
        showToast('Joyful dialogue statement loaded', 'info');
    } else if (presetId === 'anger_sample') {
        clearMMAudio();
        clearMMVideo();
        document.getElementById('mm-text-input').value = "Stop doing that right now, I am completely furious with you!";
        updateMMActiveCount();
        showToast('Confrontational anger statement loaded', 'info');
    }
}

// Unified Prediction Execution
async function predictMultimodal() {
    const textVal = document.getElementById('mm-text-input').value.trim();

    if (!state.mmAudioFile && !textVal && !state.mmVideoFile) {
        showToast('Please provide at least one input stream (Video, Audio, or Text).', 'error');
        return;
    }

    const btn = document.getElementById('btn-run-fusion');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '⏳ Executing Multimodal Attention Fusion...';

    const formData = new FormData();
    if (state.mmAudioFile) formData.append('audio_file', state.mmAudioFile);
    if (textVal) formData.append('text_content', textVal);
    if (state.mmVideoFile) formData.append('video_file', state.mmVideoFile);

    try {
        const t0 = performance.now();
        const resp = await fetch(`${API_BASE}/predict/multimodal`, {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Multimodal prediction failed');
        }

        const data = await resp.json();
        const latency = Math.round(performance.now() - t0);

        renderUnifiedMultimodalResults(data);
        recordHistory('Multimodal', `Active: [${data.active_modalities.join(', ')}]`, data.fused_predicted_class, data.fused_confidence, data.modality_agreement, latency);
        showToast(`FINAL PREDICTION: ${data.fused_predicted_class.toUpperCase()} (${(data.fused_confidence * 100).toFixed(1)}%)`, 'success');
    } catch (err) {
        showToast(`Multimodal error: ${err.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

function renderUnifiedMultimodalResults(data) {
    const container = document.getElementById('unified-result-container');
    container.style.display = 'block';

    const emotion = data.fused_predicted_class.toLowerCase();
    const color = EMOTION_COLORS[emotion] || '#4f46e5';
    const emoji = EMOTION_EMOJIS[emotion] || ' Affect';

    // 1. ONE Prominent Final Emotion Badge
    const emoBadge = document.getElementById('unified-emotion-badge');
    emoBadge.innerHTML = `<span style="font-size: 2.4rem;">${emoji}</span> ${emotion.toUpperCase()}`;
    emoBadge.style.color = color;
    emoBadge.style.borderColor = color;
    emoBadge.style.boxShadow = `0 8px 32px ${color}33`;

    // 2. Calibrated Confidence
    const confVal = (data.fused_confidence * 100).toFixed(1);
    document.getElementById('unified-confidence-num').innerText = `${confVal}%`;
    const calBadge = document.getElementById('unified-calibration-badge');
    if (calBadge) {
        const temp = data.temperature_scaling?.temperature || 0.60;
        calBadge.innerText = `✓ Temperature Scaled (T=${temp.toFixed(2)})`;
    }

    // 3. Modality Coverage Indicators
    const cov = data.modality_coverage || {};
    setCoveragePill('pill-cov-video', cov.video, '🎥 Video (Visual Expressions)');
    setCoveragePill('pill-cov-audio', cov.audio, '🎙️ Audio (Vocal Prosody)');
    setCoveragePill('pill-cov-text', cov.text, '✍️ Text (Dialogue Transcript)');

    // 4. Transcript Analyzed
    const transCard = document.getElementById('unified-transcript-card');
    const transText = document.getElementById('unified-transcript-text');
    if (data.transcript_used && data.transcript_used.trim()) {
        transText.innerText = `"${data.transcript_used.trim()}"`;
        transCard.style.display = 'block';
    } else {
        transCard.style.display = 'none';
    }

    // 5. Consensus & Conflict Reasoning
    const agreeTag = document.getElementById('unified-agreement-tag');
    agreeTag.innerText = data.modality_agreement.toUpperCase();
    agreeTag.className = 'agreement-tag';
    if (data.modality_agreement === 'Full Consensus') {
        agreeTag.classList.add('agreement-full');
    } else if (data.modality_agreement === 'Partial Consensus') {
        agreeTag.classList.add('agreement-partial');
    } else {
        agreeTag.classList.add('agreement-divergent');
    }

    document.getElementById('unified-reasoning-text').innerText = data.unified_explanation;
    document.getElementById('unified-fusion-method-name').innerText = data.fusion_method;

    // 6. Dynamic Modality Attention Meters
    const w = data.modality_weights || {};
    setAttentionMeter('att-video', (w.video || 0) * 100, '#38bdf8');
    setAttentionMeter('att-text', (w.text || 0) * 100, '#10b981');
    setAttentionMeter('att-audio', (w.audio || 0) * 100, '#f59e0b');

    // 7. Full 7-Class Unified Distribution Bars
    renderProbabilityBars(data.fused_probabilities, 'unified-prob-bars', emotion);

    // 8. Secondary Diagnostics Panel (Transparent Supporting Views)
    renderCompCard('video', data.individual_predictions?.video);
    renderCompCard('audio', data.individual_predictions?.audio);
    renderCompCard('text', data.individual_predictions?.text);

    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setCoveragePill(elId, isActive, label) {
    const el = document.getElementById(elId);
    if (!el) return;
    if (isActive) {
        el.className = 'coverage-pill active';
        el.innerHTML = `✓ ${label}`;
    } else {
        el.className = 'coverage-pill missing';
        el.innerHTML = `✕ ${label} (Missing)`;
    }
}

function setAttentionMeter(prefix, pctVal, color) {
    const fill = document.getElementById(`${prefix}-fill`);
    const val = document.getElementById(`${prefix}-val`);
    if (fill) {
        fill.style.width = `${pctVal.toFixed(1)}%`;
        fill.style.background = color;
    }
    if (val) {
        val.innerText = `${pctVal.toFixed(1)}%`;
    }
}

function toggleDiagnostics() {
    const panel = document.getElementById('secondary-diagnostics-content');
    const icon = document.getElementById('diag-toggle-icon');
    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        if (icon) icon.innerText = '▲ Hide Diagnostics';
    } else {
        panel.style.display = 'none';
        if (icon) icon.innerText = '▼ Show Diagnostics';
    }
}

function renderCompCard(modality, res) {
    const tagEl = document.getElementById(`comp-${modality}-tag`);
    const confEl = document.getElementById(`comp-${modality}-conf`);
    const barsEl = document.getElementById(`comp-${modality}-bars`);

    if (!tagEl || !confEl || !barsEl) return;

    if (!res || !res.predicted_class) {
        tagEl.innerText = 'OFFLINE / UNUSED';
        tagEl.style.color = 'var(--text-faint)';
        confEl.innerText = 'Modality was not provided in this run.';
        barsEl.innerHTML = '';
        return;
    }

    tagEl.innerText = res.predicted_class.toUpperCase();
    tagEl.style.color = EMOTION_COLORS[res.predicted_class] || '#fff';
    confEl.innerText = `Raw Model Confidence: ${(res.confidence * 100).toFixed(1)}%`;

    barsEl.innerHTML = '';
    const sorted = Object.entries(res.probabilities || {}).sort((a, b) => b[1] - a[1]).slice(0, 3);
    for (const [emo, p] of sorted) {
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.style.fontSize = '0.72rem';
        row.innerHTML = `
            <span class="prob-label" style="width: 55px;">${emo}</span>
            <div class="prob-track" style="height: 6px;">
                <div class="prob-fill" style="width: ${(p * 100).toFixed(1)}%; background: ${EMOTION_COLORS[emo]};"></div>
            </div>
            <span class="prob-val" style="width: 35px;">${(p * 100).toFixed(0)}%</span>
        `;
        barsEl.appendChild(row);
    }
}

// ============================================================================
// STANDALONE AUDIO / TEXT / VIDEO INFERENCE (PRESERVED)
// ============================================================================
function handleAudioFileSelection(file) {
    state.audioFile = file;
    document.getElementById('audio-dropzone').style.display = 'none';
    const info = document.getElementById('audio-file-info');
    info.style.display = 'flex';
    document.getElementById('audio-filename').innerText = file.name;
    document.getElementById('audio-filesize').innerText = formatFileSize(file.size);
    document.getElementById('btn-predict-audio').disabled = false;
    showToast(`Loaded audio: ${file.name}`, 'info');
}

function clearAudioInput() {
    state.audioFile = null;
    document.getElementById('audio-file-input').value = '';
    document.getElementById('audio-dropzone').style.display = 'block';
    document.getElementById('audio-file-info').style.display = 'none';
    document.getElementById('btn-predict-audio').disabled = true;
    document.getElementById('audio-result-content').style.display = 'none';
    document.getElementById('audio-result-placeholder').style.display = 'block';
}

async function loadSampleAudio(filename) {
    showToast(`Loading sample ${filename}...`, 'info');
    try {
        const resp = await fetch(`/static/samples/${filename}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const blob = await resp.blob();
        const file = new File([blob], filename, { type: 'audio/wav' });
        handleAudioFileSelection(file);
    } catch (err) {
        showToast(`Could not load sample: ${err.message}`, 'error');
    }
}

async function predictAudio() {
    if (!state.audioFile) return;
    const btn = document.getElementById('btn-predict-audio');
    btn.disabled = true;
    btn.innerText = 'Running Audio Inference...';

    const formData = new FormData();
    formData.append('file', state.audioFile);

    try {
        const t0 = performance.now();
        const resp = await fetch(`${API_BASE}/predict/audio`, {
            method: 'POST',
            body: formData
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Audio prediction failed');
        const data = await resp.json();
        const lat = Math.round(performance.now() - t0);

        renderAudioResults(data);
        recordHistory('Audio', state.audioFile.name, data.predicted_class, data.confidence, 'Unimodal', lat);
        showToast(`Audio Prediction: ${data.predicted_class.toUpperCase()} (${(data.confidence * 100).toFixed(1)}%)`, 'success');
    } catch (err) {
        showToast(`Audio error: ${err.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerText = '⚡ Run Audio Emotion Inference';
    }
}

function renderAudioResults(data) {
    document.getElementById('audio-result-placeholder').style.display = 'none';
    const content = document.getElementById('audio-result-content');
    content.style.display = 'block';

    const emoTag = document.getElementById('audio-predicted-emotion');
    emoTag.innerText = data.predicted_class.toUpperCase();
    emoTag.style.color = EMOTION_COLORS[data.predicted_class] || '#fff';

    document.getElementById('audio-confidence-val').innerText = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('audio-latency').innerText = `${data.inference_time_ms} ms`;
    document.getElementById('audio-model-name').innerText = data.model_used;

    renderProbabilityBars(data.probabilities, 'audio-prob-bars', data.predicted_class);
}

// Standalone Text
function handleTextFileSelection(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('text-input').value = e.target.result;
        document.getElementById('text-input').dispatchEvent(new Event('input'));
        showToast(`Loaded text file: ${file.name}`, 'info');
    };
    reader.readAsText(file);
}

function loadSampleText(sentence) {
    const input = document.getElementById('text-input');
    input.value = sentence;
    input.dispatchEvent(new Event('input'));
    showToast('Sample dialogue loaded', 'info');
}

async function predictText() {
    const text = document.getElementById('text-input').value.trim();
    if (!text) {
        showToast('Please type or paste some text first.', 'error');
        return;
    }

    const btn = document.getElementById('btn-predict-text');
    btn.disabled = true;
    btn.innerText = 'Running Text Inference...';

    try {
        const t0 = performance.now();
        const resp = await fetch(`${API_BASE}/predict/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Text prediction failed');
        const data = await resp.json();
        const lat = Math.round(performance.now() - t0);

        renderTextResults(data);
        recordHistory('Text', text.slice(0, 30) + '...', data.predicted_class, data.confidence, 'Unimodal', lat);
        showToast(`Text Prediction: ${data.predicted_class.toUpperCase()} (${(data.confidence * 100).toFixed(1)}%)`, 'success');
    } catch (err) {
        showToast(`Text error: ${err.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerText = '⚡ Run Text Emotion Inference';
    }
}

function renderTextResults(data) {
    document.getElementById('text-result-placeholder').style.display = 'none';
    const content = document.getElementById('text-result-content');
    content.style.display = 'block';

    const emoTag = document.getElementById('text-predicted-emotion');
    emoTag.innerText = data.predicted_class.toUpperCase();
    emoTag.style.color = EMOTION_COLORS[data.predicted_class] || '#fff';

    document.getElementById('text-confidence-val').innerText = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('text-latency').innerText = `${data.inference_time_ms} ms`;
    document.getElementById('text-model-name').innerText = data.model_used;

    renderProbabilityBars(data.probabilities, 'text-prob-bars', data.predicted_class);
}

// Standalone Video
function handleVideoFileSelection(file) {
    state.videoFile = file;
    document.getElementById('video-dropzone').style.display = 'none';
    const info = document.getElementById('video-file-info');
    info.style.display = 'flex';
    document.getElementById('video-filename').innerText = file.name;
    document.getElementById('video-filesize').innerText = formatFileSize(file.size);

    const playerWrapper = document.getElementById('video-player-wrapper');
    const player = document.getElementById('video-player');
    if (playerWrapper && player) {
        player.src = URL.createObjectURL(file);
        playerWrapper.style.display = 'block';
    }

    document.getElementById('btn-predict-video').disabled = false;
    showToast(`Loaded video: ${file.name}`, 'info');
}

function clearVideoInput() {
    state.videoFile = null;
    document.getElementById('video-file-input').value = '';
    document.getElementById('video-dropzone').style.display = 'block';
    document.getElementById('video-file-info').style.display = 'none';
    const playerWrapper = document.getElementById('video-player-wrapper');
    if (playerWrapper) playerWrapper.style.display = 'none';
    document.getElementById('btn-predict-video').disabled = true;
    document.getElementById('video-result-content').style.display = 'none';
    document.getElementById('video-result-placeholder').style.display = 'block';
}

async function loadSampleVideo(filename) {
    showToast(`Loading video ${filename}...`, 'info');
    try {
        const resp = await fetch(`/static/samples/${filename}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const blob = await resp.blob();
        const file = new File([blob], filename, { type: 'video/mp4' });
        handleVideoFileSelection(file);
    } catch (err) {
        showToast(`Could not load video sample: ${err.message}`, 'error');
    }
}

async function predictVideo() {
    if (!state.videoFile) return;
    const btn = document.getElementById('btn-predict-video');
    btn.disabled = true;
    btn.innerText = 'Extracting Frames & Predicting...';

    const formData = new FormData();
    formData.append('file', state.videoFile);

    try {
        const t0 = performance.now();
        const resp = await fetch(`${API_BASE}/predict/video`, {
            method: 'POST',
            body: formData
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Video prediction failed');
        const data = await resp.json();
        const lat = Math.round(performance.now() - t0);

        renderVideoResults(data);
        recordHistory('Video', state.videoFile.name, data.predicted_class, data.confidence, 'Unimodal', lat);
        showToast(`Video Prediction: ${data.predicted_class.toUpperCase()} (${(data.confidence * 100).toFixed(1)}%)`, 'success');
    } catch (err) {
        showToast(`Video error: ${err.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerText = '⚡ Run Video Emotion Inference';
    }
}

function renderVideoResults(data) {
    document.getElementById('video-result-placeholder').style.display = 'none';
    const content = document.getElementById('video-result-content');
    content.style.display = 'block';

    const emoTag = document.getElementById('video-predicted-emotion');
    emoTag.innerText = data.predicted_class.toUpperCase();
    emoTag.style.color = EMOTION_COLORS[data.predicted_class] || '#fff';

    document.getElementById('video-confidence-val').innerText = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('video-latency').innerText = `${data.inference_time_ms} ms`;
    document.getElementById('video-model-name').innerText = data.model_used;

    const thumbContainer = document.getElementById('video-thumbnails-container');
    const thumbRow = document.getElementById('video-thumbnails-row');
    if (data.details && data.details.preview_thumbnails && data.details.preview_thumbnails.length > 0) {
        thumbRow.innerHTML = '';
        data.details.preview_thumbnails.forEach((src) => {
            const img = document.createElement('img');
            img.src = src;
            img.className = 'thumb-img';
            thumbRow.appendChild(img);
        });
        thumbContainer.style.display = 'block';
    } else {
        thumbContainer.style.display = 'none';
    }

    renderProbabilityBars(data.probabilities, 'video-prob-bars', data.predicted_class);
}

// ============================================================================
// PROBABILITY BARS RENDERING
// ============================================================================
function renderProbabilityBars(probs, containerId, topClass) {
    const container = document.getElementById(containerId);
    if (!container || !probs) return;

    container.innerHTML = '';
    const emotionOrder = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'];

    emotionOrder.forEach(emo => {
        const p = probs[emo] || 0.0;
        const pct = (p * 100).toFixed(1);
        const color = EMOTION_COLORS[emo] || '#4f46e5';
        const isTop = (emo.toLowerCase() === topClass.toLowerCase());

        const row = document.createElement('div');
        row.className = 'prob-row';

        row.innerHTML = `
            <span class="prob-label" style="${isTop ? 'color: ' + color + '; font-weight: 700;' : ''}">${EMOTION_EMOJIS[emo]} ${emo}</span>
            <div class="prob-track">
                <div class="prob-fill" style="width: 0%; background: ${color};" data-target-width="${pct}%"></div>
            </div>
            <span class="prob-val" style="${isTop ? 'color: white; font-weight: 700;' : ''}">${pct}%</span>
        `;
        container.appendChild(row);
    });

    requestAnimationFrame(() => {
        setTimeout(() => {
            container.querySelectorAll('.prob-fill').forEach(fill => {
                fill.style.width = fill.getAttribute('data-target-width');
            });
        }, 50);
    });
}

// ============================================================================
// HISTORY TRACKING
// ============================================================================
function recordHistory(modality, inputId, predictedClass, confidence, agreement, latency) {
    const item = {
        time: new Date().toLocaleTimeString(),
        modality: modality,
        input: inputId,
        predicted: predictedClass,
        confidence: `${(confidence * 100).toFixed(1)}%`,
        agreement: agreement || 'N/A',
        latency: `${latency}ms`
    };

    state.history.unshift(item);
    if (state.history.length > 50) state.history.pop();
    renderHistoryTable();
}

function renderHistoryTable() {
    const tbody = document.getElementById('history-tbody');
    const countBadge = document.getElementById('history-count');
    if (countBadge) countBadge.innerText = state.history.length;
    if (!tbody) return;

    if (state.history.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 30px;">
                    No predictions recorded yet. Run a unified multimodal analysis!
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = '';
    state.history.forEach(item => {
        const color = EMOTION_COLORS[item.predicted.toLowerCase()] || '#fff';
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><span class="history-time">${item.time}</span></td>
            <td><span class="badge ${item.modality.toLowerCase()}">${item.modality}</span></td>
            <td style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${item.input}</td>
            <td><strong style="color: ${color};">${item.predicted.toUpperCase()}</strong></td>
            <td>${item.confidence}</td>
            <td><span class="badge ${item.agreement === 'Full Consensus' ? 'badge-success' : 'badge-warning'}">${item.agreement}</span></td>
            <td>${item.latency}</td>
        `;
        tbody.appendChild(tr);
    });
}

function clearHistory() {
    state.history = [];
    renderHistoryTable();
    showToast('History cleared', 'info');
}

function loadSessionHistory() {
    renderHistoryTable();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function showToast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerText = msg;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ============================================================================
// REAL-TIME WEBCAM EMOTION DETECTION & MULTIMODAL STREAM ENGINE
// ============================================================================
let webcamState = {
    stream: null,
    audioStream: null,
    mediaRecorder: null,
    audioChunks: [],
    isRunning: false,
    isProcessing: false,
    intervalId: null,
    audioTimerId: null,
    micEnabled: false,
    history: [],
    lastFrameTimestamp: 0,
    fpsHistory: [],
    selectedCameraId: '',
    ws: null,
    wsConnected: false,
    lastWsSendTime: 0,
    isCalibrating: false,
    hasCalibratedOnce: false,
    calibrateCount: 0,
    clientId: 'webcam_client_' + Math.floor(Math.random() * 1000000)
};

function connectWebcamWebSocket() {
    try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/predict/webcam/ws`;
        const ws = new WebSocket(wsUrl);
        ws.onopen = () => {
            webcamState.ws = ws;
            webcamState.wsConnected = true;
            console.log('Real-time webcam WebSocket connected (Low Latency Mode)');
        };
        ws.onmessage = (evt) => {
            try {
                const data = JSON.parse(evt.data);
                const roundtripMs = performance.now() - webcamState.lastWsSendTime;
                renderWebcamInference(data, roundtripMs);
            } catch (e) {
                console.warn('WS message parse error:', e);
            }
        };
        ws.onclose = () => {
            webcamState.ws = null;
            webcamState.wsConnected = false;
        };
        ws.onerror = (err) => {
            console.warn('WebSocket error, falling back to HTTP transport:', err);
            webcamState.ws = null;
            webcamState.wsConnected = false;
        };
    } catch (err) {
        console.warn('WebSocket init exception:', err);
        webcamState.ws = null;
        webcamState.wsConnected = false;
    }
}

function startWebcamCalibration() {
    if (!webcamState.isRunning) {
        showToast('Please start the camera first.', 'warning');
        return;
    }
    webcamState.isCalibrating = true;
    webcamState.calibrateCount = 0;
    const btn = document.getElementById('btn-calibrate-webcam');
    if (btn) {
        btn.classList.add('calibrating');
        btn.innerHTML = '<span>⏳</span> HOLD NEUTRAL...';
    }
    showToast('Calibration started: Relax your face and look at the screen for 2-3 seconds.', 'info');
}

async function resetWebcamCalibration() {
    webcamState.hasCalibratedOnce = false;
    webcamState.isCalibrating = false;
    webcamState.calibrateCount = 0;
    const btn = document.getElementById('btn-calibrate-webcam');
    if (btn) {
        btn.classList.remove('calibrating');
        btn.innerHTML = '<span>🎯</span> CALIBRATE FACE';
    }
    try {
        await fetch('/predict/webcam/reset-calibration', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ client_id: webcamState.clientId })
        });
    } catch (e) {
        console.warn('Reset calibration request failed:', e);
    }
    showToast('Neutral face calibration reset.', 'info');
}

async function populateWebcamDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        const select = document.getElementById('webcam-camera-select');
        if (!select) return;

        select.innerHTML = '';
        if (videoDevices.length === 0) {
            select.innerHTML = '<option value="">Default Camera</option>';
            return;
        }

        videoDevices.forEach((device, idx) => {
            const opt = document.createElement('option');
            opt.value = device.deviceId;
            opt.textContent = device.label || `Camera ${idx + 1}`;
            select.appendChild(opt);
        });

        if (webcamState.selectedCameraId) {
            select.value = webcamState.selectedCameraId;
        }
    } catch (e) {
        console.warn('Could not enumerate camera devices:', e);
    }
}

async function switchWebcamCamera() {
    const select = document.getElementById('webcam-camera-select');
    if (select) {
        webcamState.selectedCameraId = select.value;
    }
    if (webcamState.isRunning) {
        await restartWebcam();
    }
}

function updateWebcamModelDisplay(data) {
    if (!data) return;
    const select = document.getElementById('webcam-backbone-select');
    if (select) {
        if (Array.isArray(data.available_backbones)) {
            data.available_backbones.forEach(item => {
                const opt = Array.from(select.options).find(o => o.value === item.id);
                if (opt) {
                    opt.textContent = item.available ? item.label : `${item.label} [Missing]`;
                    opt.disabled = !item.available;
                }
            });
        }
        if (data.backbone) {
            select.value = data.backbone;
        }
    }

    const modelPill = document.getElementById('webcam-model-pill');
    if (modelPill && data.backbone) {
        const shortName = data.backbone === 'resnet18_raf'
            ? `ResNet-18 (${data.img_size || 128}px)`
            : `EfficientNet-B0 (${data.img_size || 96}px)`;
        modelPill.innerHTML = `Model: <strong>${shortName}</strong>`;
    }

    const confLabel = document.getElementById('webcam-conf-label');
    if (confLabel && typeof data.calibrated_temperature === 'number') {
        confLabel.textContent = `Calibrated Confidence (T*=${data.calibrated_temperature.toFixed(2)})`;
    }
}

async function loadWebcamStatus() {
    try {
        const resp = await fetch('/predict/webcam/status');
        if (!resp.ok) return;
        const data = await resp.json();
        updateWebcamModelDisplay(data);
    } catch (e) {
        console.warn('Could not load webcam status:', e);
    }
}

async function switchWebcamBackbone() {
    const select = document.getElementById('webcam-backbone-select');
    if (!select) return;
    const backbone = select.value;
    select.disabled = true;
    try {
        const resp = await fetch('/predict/webcam/backbone', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backbone })
        });
        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.detail || `HTTP ${resp.status}`);
        }
        updateWebcamModelDisplay(data);
        webcamState.hasCalibratedOnce = false;
        webcamState.isCalibrating = false;
        webcamState.calibrateCount = 0;
        const calibBtn = document.getElementById('btn-calibrate-webcam');
        if (calibBtn) {
            calibBtn.classList.remove('calibrating');
            calibBtn.innerHTML = '<span>🎯</span> CALIBRATE FACE';
        }
        showToast(`Active webcam model switched to ${select.options[select.selectedIndex].text}`, 'success');
    } catch (e) {
        console.warn('Failed to switch webcam backbone:', e);
        showToast(`Failed to switch model: ${e.message}`, 'error');
        await loadWebcamStatus();
    } finally {
        select.disabled = false;
    }
}

async function restartWebcam() {
    stopWebcam();
    await new Promise(r => setTimeout(r, 250));
    await startWebcam();
}

async function startWebcam() {
    const alertBox = document.getElementById('webcam-alert-box');
    if (alertBox) alertBox.style.display = 'none';

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showWebcamAlert('Webcam access is not supported by your browser. Please use Chrome, Edge, or Firefox.', 'danger');
        return;
    }

    try {
        const select = document.getElementById('webcam-camera-select');
        const selectedId = select ? select.value : '';

        const videoConstraints = {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            aspectRatio: { ideal: 1.777777778 },
            facingMode: 'user'
        };
        if (selectedId) {
            videoConstraints.deviceId = { exact: selectedId };
        }

        const stream = await navigator.mediaDevices.getUserMedia({
            video: videoConstraints,
            audio: false
        });
        webcamState.stream = stream;

        // Refresh device names now that camera permission is granted
        populateWebcamDevices();

        const video = document.getElementById('webcam-video');
        const canvas = document.getElementById('webcam-canvas');
        const placeholder = document.getElementById('webcam-placeholder');
        const feedWrapper = document.getElementById('webcam-feed-wrapper');

        video.srcObject = stream;
        video.style.display = 'block';
        canvas.style.display = 'block';
        placeholder.style.display = 'none';
        feedWrapper.classList.add('active-stream');

        await video.play();

        // Dynamically lock overlay canvas resolution to the stream's native dimensions
        const syncCanvas = () => {
            if (video.videoWidth && video.videoHeight) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            }
        };
        syncCanvas();
        video.onloadedmetadata = syncCanvas;

        webcamState.isRunning = true;
        document.getElementById('btn-start-webcam').disabled = true;
        document.getElementById('btn-stop-webcam').disabled = false;
        const restartBtn = document.getElementById('btn-restart-webcam');
        if (restartBtn) restartBtn.disabled = false;
        const calibBtn = document.getElementById('btn-calibrate-webcam');
        if (calibBtn) calibBtn.disabled = false;
        const resetCalibBtn = document.getElementById('btn-reset-calib-webcam');
        if (resetCalibBtn) resetCalibBtn.disabled = false;

        connectWebcamWebSocket();
        
        const camPill = document.getElementById('webcam-cam-pill');
        if (camPill) {
            camPill.innerHTML = 'Camera: <strong>Active (HD 16:9)</strong>';
            camPill.className = 'stat-pill online';
        }
        document.getElementById('webcam-status-text').textContent = 'LIVE STREAMING';
        document.getElementById('webcam-status-text').style.color = '#34d399';

        // Reset server-side EMA smoothing session
        fetch('/predict/webcam/reset-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ client_id: webcamState.clientId })
        }).catch(e => console.warn('Reset session error:', e));

        // Auto-run neutral resting face calibration on first camera start
        if (!webcamState.hasCalibratedOnce) {
            setTimeout(() => {
                if (webcamState.isRunning && !webcamState.isCalibrating && !webcamState.hasCalibratedOnce) {
                    startWebcamCalibration();
                }
            }, 1200);
        }

        // Start frame capture loop (~20 FPS: every 50ms, with isProcessing lock dropping stale frames)
        webcamState.intervalId = setInterval(captureAndProcessWebcamFrame, 50);

        // Check if microphone is already toggled on
        if (webcamState.micEnabled) {
            startMicrophoneStream();
        }

        showToast('Webcam started (HD 16:9 Preview Active)', 'success');

    } catch (err) {
        console.error('Camera startup error:', err);
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            showWebcamAlert('⚠️ Camera permission was denied. Please allow camera permissions in your browser address bar and try again.', 'danger');
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            showWebcamAlert('⚠️ No webcam device was found on this computer. Please connect a webcam.', 'danger');
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            showWebcamAlert('⚠️ Camera is currently in use by another application. Please close other camera apps.', 'danger');
        } else {
            showWebcamAlert(`⚠️ Could not access webcam: ${err.message}`, 'danger');
        }
        stopWebcam();
    }
}

function stopWebcam() {
    webcamState.isRunning = false;

    if (webcamState.intervalId) {
        clearInterval(webcamState.intervalId);
        webcamState.intervalId = null;
    }
    if (webcamState.audioTimerId) {
        clearInterval(webcamState.audioTimerId);
        webcamState.audioTimerId = null;
    }

    // Stop video tracks
    if (webcamState.stream) {
        webcamState.stream.getTracks().forEach(track => track.stop());
        webcamState.stream = null;
    }

    // Stop audio tracks
    if (webcamState.audioStream) {
        webcamState.audioStream.getTracks().forEach(track => track.stop());
        webcamState.audioStream = null;
    }

    if (webcamState.mediaRecorder && webcamState.mediaRecorder.state !== 'inactive') {
        webcamState.mediaRecorder.stop();
        webcamState.mediaRecorder = null;
    }

    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const placeholder = document.getElementById('webcam-placeholder');
    const feedWrapper = document.getElementById('webcam-feed-wrapper');

    if (video) {
        video.srcObject = null;
        video.style.display = 'none';
    }
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvas.style.display = 'none';
    }
    if (placeholder) placeholder.style.display = 'flex';
    if (feedWrapper) feedWrapper.classList.remove('active-stream');

    document.getElementById('btn-start-webcam').disabled = false;
    document.getElementById('btn-stop-webcam').disabled = true;
    const restartBtn = document.getElementById('btn-restart-webcam');
    if (restartBtn) restartBtn.disabled = true;
    const calibBtn = document.getElementById('btn-calibrate-webcam');
    if (calibBtn) {
        calibBtn.disabled = true;
        calibBtn.classList.remove('calibrating');
        calibBtn.innerHTML = '<span>🎯</span> CALIBRATE FACE';
    }
    const resetCalibBtn = document.getElementById('btn-reset-calib-webcam');
    if (resetCalibBtn) resetCalibBtn.disabled = true;
    webcamState.isCalibrating = false;

    if (webcamState.ws) {
        try { webcamState.ws.close(); } catch(e){}
        webcamState.ws = null;
        webcamState.wsConnected = false;
    }

    const camPill = document.getElementById('webcam-cam-pill');
    if (camPill) {
        camPill.innerHTML = 'Camera: <strong>Inactive</strong>';
        camPill.className = 'stat-pill';
    }
    const facePill = document.getElementById('webcam-face-pill');
    if (facePill) {
        facePill.innerHTML = 'Face: <strong>No Face</strong>';
        facePill.className = 'stat-pill';
    }
    const fpsPill = document.getElementById('webcam-fps-pill');
    if (fpsPill) fpsPill.innerHTML = 'FPS: <strong>--</strong>';
    const latPill = document.getElementById('webcam-latency-pill');
    if (latPill) latPill.innerHTML = 'Latency: <strong>-- ms</strong>';

    document.getElementById('webcam-status-text').textContent = 'STANDBY';
    document.getElementById('webcam-status-text').style.color = 'var(--text-muted)';

    // Reset session
    fetch('/predict/webcam/reset-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: webcamState.clientId })
    }).catch(e => {});

    showToast('Webcam stream stopped and hardware released', 'info');
}

async function captureAndProcessWebcamFrame() {
    if (!webcamState.isRunning || webcamState.isProcessing) return;

    const video = document.getElementById('webcam-video');
    if (!video || video.paused || video.ended || video.readyState < 2) return;

    webcamState.isProcessing = true;
    const t0 = performance.now();

    try {
        // Fast offscreen canvas snapshot (16:9 native 320x180 aspect ratio preserving facial proportions with low latency)
        if (!webcamState.snapCanvas) {
            webcamState.snapCanvas = document.createElement('canvas');
            webcamState.snapCanvas.width = 320;
            webcamState.snapCanvas.height = 180;
        }
        const snap = webcamState.snapCanvas;
        const sctx = snap.getContext('2d');
        sctx.drawImage(video, 0, 0, 320, 180);
        const dataUrl = snap.toDataURL('image/jpeg', 0.85);

        const isCalib = webcamState.isCalibrating;
        if (isCalib) {
            webcamState.calibrateCount++;
            const btn = document.getElementById('btn-calibrate-webcam');
            if (btn) {
                btn.innerHTML = `<span>🎯</span> CALIBRATING (${Math.min(25, webcamState.calibrateCount)}/25)...`;
            }
        }

        const payload = {
            frame: dataUrl,
            client_id: webcamState.clientId,
            apply_smoothing: true,
            calibrate: isCalib
        };

        // Prefer WebSocket full-duplex stream for sub-10ms latency
        if (webcamState.ws && webcamState.ws.readyState === WebSocket.OPEN) {
            webcamState.lastWsSendTime = performance.now();
            webcamState.ws.send(JSON.stringify(payload));
            webcamState.isProcessing = false;
            return;
        }

        // HTTP fallback transport
        const resp = await fetch('/predict/webcam/frame-json', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!resp.ok) {
            throw new Error(`HTTP error ${resp.status}`);
        }

        const data = await resp.json();
        const roundtripMs = performance.now() - t0;

        renderWebcamInference(data, roundtripMs);

    } catch (err) {
        console.warn('Frame inference cycle skipped:', err);
    } finally {
        webcamState.isProcessing = false;
    }
}

function renderWebcamInference(data, roundtripMs) {
    const canvas = document.getElementById('webcam-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Check calibration completion status
    if (data.calibration_status === 'complete' && webcamState.isCalibrating) {
        webcamState.isCalibrating = false;
        webcamState.hasCalibratedOnce = true;
        const btn = document.getElementById('btn-calibrate-webcam');
        if (btn) {
            btn.classList.remove('calibrating');
            btn.innerHTML = '<span>✅</span> CALIBRATED';
        }
        showToast('🎯 Neutral face calibrated! Personal resting-face bias removed.', 'success');
    }

    // Update FPS and Latency Pills
    const fpsPill = document.getElementById('webcam-fps-pill');
    if (fpsPill && data.fps) {
        fpsPill.innerHTML = `FPS: <strong>${data.fps.toFixed(1)}</strong>`;
    }
    const latPill = document.getElementById('webcam-latency-pill');
    if (latPill) {
        const transportTag = (webcamState.ws && webcamState.ws.readyState === WebSocket.OPEN) ? ' (WS)' : '';
        latPill.innerHTML = `Latency: <strong>${Math.round(roundtripMs)} ms${transportTag}</strong>`;
    }

    const facePill = document.getElementById('webcam-face-pill');
    const alertBox = document.getElementById('webcam-alert-box');

    if (!data.face_detected || !data.bbox) {
        if (facePill) {
            facePill.innerHTML = 'Face: <strong>No Face</strong>';
            facePill.className = 'stat-pill';
        }
        document.getElementById('webcam-emotion-name').textContent = 'NO FACE DETECTED';
        document.getElementById('webcam-emotion-name').style.color = '#94a3b8';
        document.getElementById('webcam-emotion-emoji').textContent = '👤';
        document.getElementById('webcam-confidence-val').textContent = '0.0%';
        document.getElementById('webcam-confidence-val').style.color = 'var(--text-muted)';
        
        const tierBadge = document.getElementById('webcam-tier-badge');
        if (tierBadge) {
            tierBadge.textContent = 'AWAITING FACE';
            tierBadge.className = 'tier-badge tier-subtle';
        }
        renderWebcamProbBars({ anger: 0, disgust: 0, fear: 0, joy: 0, neutral: 0, sadness: 0, surprise: 0 });
        return;
    }

    // Face is detected
    const expr = (data.apparent_expression || 'NEUTRAL').toUpperCase();
    const conf = (data.calibrated_confidence || 0.0);
    const color = EMOTION_COLORS[expr.toLowerCase()] || '#10b981';
    const emoji = EMOTION_EMOJIS[expr.toLowerCase()] || '😐';

    if (facePill) {
        const score = data.face_score ? Math.round(data.face_score * 100) : 95;
        const calibTag = data.is_calibrated ? ' • <span style="color:#10b981;">CALIBRATED FOR USER</span>' : '';
        facePill.innerHTML = `Face: <strong>Locked (${score}%)${calibTag}</strong>`;
        facePill.className = 'stat-pill locked';
    }

    // Bounding Box coordinates on canvas (mapped to mirrored video display coordinates)
    const bx = canvas.width - (data.bbox.x + data.bbox.width) * canvas.width;
    const by = data.bbox.y * canvas.height;
    const bw = data.bbox.width * canvas.width;
    const bh = data.bbox.height * canvas.height;

    ctx.save();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;

    // Rounded rectangle bounding box
    const r = 8;
    ctx.beginPath();
    ctx.moveTo(bx + r, by);
    ctx.lineTo(bx + bw - r, by);
    ctx.quadraticCurveTo(bx + bw, by, bx + bw, by + r);
    ctx.lineTo(bx + bw, by + bh - r);
    ctx.quadraticCurveTo(bx + bw, by + bh, bx + bw - r, by + bh);
    ctx.lineTo(bx + r, by + bh);
    ctx.quadraticCurveTo(bx, by + bh, bx, by + bh - r);
    ctx.lineTo(bx, by + r);
    ctx.quadraticCurveTo(bx, by, bx + r, by);
    ctx.closePath();
    ctx.stroke();

    // High-tech corner accents
    const cLen = Math.min(16, Math.min(bw, bh) * 0.22);
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(bx, by + cLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cLen, by);
    ctx.moveTo(bx + bw - cLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cLen);
    ctx.moveTo(bx + bw, by + bh - cLen); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw - cLen, by + bh);
    ctx.moveTo(bx + cLen, by + bh); ctx.lineTo(bx, by + bh); ctx.lineTo(bx, by + bh - cLen);
    ctx.stroke();

    // Render 5 Facial Landmarks (Eyes, Nose Tip, Mouth Corners) mapped to mirrored video display coordinates
    if (data.landmarks && Array.isArray(data.landmarks) && data.landmarks.length > 0) {
        data.landmarks.forEach(([lx, ly]) => {
            const px = canvas.width - lx * canvas.width;
            const py = ly * canvas.height;
            ctx.beginPath();
            ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
            ctx.fillStyle = '#38bdf8';
            ctx.shadowColor = '#38bdf8';
            ctx.shadowBlur = 8;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(px, py, 6.0, 0, 2 * Math.PI);
            ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
            ctx.lineWidth = 1;
            ctx.stroke();
        });
    }

    // Floating Tag Above Face (Text rendered unmirrored from left to right)
    let tagText = `${emoji} ${expr} ${(conf * 100).toFixed(1)}%`;
    if (data.is_uncertain) {
        tagText = `❓ UNCERTAIN (${expr} ${(conf * 100).toFixed(1)}%)`;
    }
    ctx.font = 'bold 15px Inter, sans-serif';
    const textWidth = ctx.measureText(tagText).width;
    const tagH = 26;
    const tagY = Math.max(8, by - tagH - 8);

    ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
    ctx.fillRect(bx, tagY, textWidth + 18, tagH);
    ctx.strokeStyle = data.is_uncertain ? '#f59e0b' : color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(bx, tagY, textWidth + 18, tagH);

    ctx.fillStyle = '#ffffff';
    ctx.fillText(tagText, bx + 9, tagY + 18);
    ctx.restore();

    // Update Right-Hand Analytics Hero Card
    document.getElementById('webcam-emotion-name').textContent = expr;
    document.getElementById('webcam-emotion-name').style.color = color;
    document.getElementById('webcam-emotion-emoji').textContent = emoji;
    document.getElementById('webcam-confidence-val').textContent = (conf * 100).toFixed(1) + '%';
    document.getElementById('webcam-confidence-val').style.color = color;

    const tierBadge = document.getElementById('webcam-tier-badge');
    if (tierBadge) {
        if (data.is_uncertain) {
            tierBadge.textContent = 'UNCERTAIN / AMBIGUOUS';
            tierBadge.className = 'tier-badge tier-subtle';
        } else {
            const tier = data.confidence_tier || (conf >= 0.65 ? 'Robust' : conf >= 0.50 ? 'Moderate' : 'Subtle / Ambiguous');
            tierBadge.textContent = `${tier.toUpperCase()} CONFIDENCE`;
            if (conf >= 0.65) {
                tierBadge.className = 'tier-badge tier-robust';
            } else if (conf >= 0.50) {
                tierBadge.className = 'tier-badge tier-moderate';
            } else {
                tierBadge.className = 'tier-badge tier-subtle';
            }
        }
    }

    // Update Probabilities Bars
    if (data.probabilities) {
        renderWebcamProbBars(data.probabilities);
    }

    // Update Rolling Timeline Sparkline
    if (data.apparent_expression && conf > 0.15) {
        appendTimelineChip(emoji, expr, conf, color);
    }

    // Handle Warnings / Alerts
    if (data.warnings && data.warnings.length > 0 && alertBox) {
        const relevant = data.warnings.filter(w => !w.includes('No face'));
        if (relevant.length > 0) {
            alertBox.style.display = 'block';
            alertBox.className = 'diagnostics-disclaimer';
            alertBox.innerHTML = `⚠️ ${relevant.join(' • ')}`;
        } else {
            alertBox.style.display = 'none';
        }
    }
}

function renderWebcamProbBars(probs) {
    const container = document.getElementById('webcam-prob-bars');
    if (!container) return;

    const emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'];
    const maxProb = Math.max(...Object.values(probs));

    let html = '';
    emotions.forEach(emo => {
        const val = probs[emo] || 0.0;
        const pct = (val * 100).toFixed(1);
        const color = EMOTION_COLORS[emo] || '#fff';
        const isLeader = val === maxProb && val > 0.2;

        html += `
            <div class="prob-row" style="margin-bottom: 6px;">
                <div class="prob-label" style="font-size: 0.76rem; width: 85px; display: flex; align-items: center; gap: 4px;">
                    <span>${EMOTION_EMOJIS[emo]}</span>
                    <strong style="${isLeader ? 'color: white;' : 'color: #94a3b8;'}">${emo.toUpperCase()}</strong>
                </div>
                <div class="prob-bar-track" style="flex: 1; height: 10px; background: rgba(255,255,255,0.06); border-radius: 9999px; overflow: hidden; position: relative;">
                    <div class="prob-bar-fill" style="width: ${pct}%; background: ${color}; height: 100%; border-radius: 9999px; transition: width 0.2s ease;"></div>
                </div>
                <div class="prob-val" style="width: 48px; text-align: right; font-family: var(--font-mono); font-size: 0.74rem; font-weight: 700; ${isLeader ? 'color:' + color : 'color:#64748b;'}">
                    ${pct}%
                </div>
            </div>
        `;
    });
    container.innerHTML = html;
}

function appendTimelineChip(emoji, expr, conf, color) {
    const row = document.getElementById('webcam-timeline-chips');
    const countEl = document.getElementById('webcam-history-count');
    if (!row) return;

    if (row.children.length === 1 && row.children[0].textContent.includes('Awaiting')) {
        row.innerHTML = '';
    }

    const chip = document.createElement('div');
    chip.className = 'timeline-chip';
    chip.style.borderColor = color;
    chip.innerHTML = `<span>${emoji}</span> <span style="color:${color}; font-weight:700;">${expr}</span> <small style="color:#94a3b8;">${(conf * 100).toFixed(0)}%</small>`;
    
    row.appendChild(chip);
    if (row.children.length > 15) {
        row.removeChild(row.children[0]);
    }
    row.scrollLeft = row.scrollWidth;

    webcamState.history.push({ expr, conf, time: new Date().toLocaleTimeString() });
    if (webcamState.history.length > 50) webcamState.history.shift();

    if (countEl) countEl.textContent = `${webcamState.history.length} samples`;
}

function showWebcamAlert(msg, type = 'danger') {
    const alertBox = document.getElementById('webcam-alert-box');
    if (!alertBox) return;
    alertBox.style.display = 'block';
    alertBox.style.background = type === 'danger' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(245, 158, 11, 0.15)';
    alertBox.style.border = type === 'danger' ? '1px solid rgba(239, 68, 68, 0.4)' : '1px solid rgba(245, 158, 11, 0.4)';
    alertBox.style.color = type === 'danger' ? '#fca5a5' : '#fbbf24';
    alertBox.innerHTML = msg;
}

// ============================================================================
// REAL-TIME MULTIMODAL SYNCHRONIZED AUDIO & SPEECH FUSION
// ============================================================================
async function toggleWebcamMicrophone() {
    const toggle = document.getElementById('webcam-mic-toggle');
    webcamState.micEnabled = toggle.checked;

    const modeBadge = document.getElementById('webcam-mode-badge');
    const mmPanel = document.getElementById('webcam-mm-panel');

    if (webcamState.micEnabled) {
        if (modeBadge) {
            modeBadge.innerHTML = '<span>🔮</span> MODE: REAL-TIME MULTIMODAL AI (VIDEO + AUDIO + SPEECH)';
            modeBadge.style.color = '#38bdf8';
            modeBadge.style.borderColor = 'rgba(56, 189, 248, 0.4)';
        }
        if (mmPanel) mmPanel.style.display = 'flex';
        showToast('Microphone fusion enabled. Real-time audio will synchronize with webcam.', 'info');
        if (webcamState.isRunning) {
            startMicrophoneStream();
        }
    } else {
        if (modeBadge) {
            modeBadge.innerHTML = '<span>🎥</span> MODE: WEBCAM FACIAL EXPRESSION RECOGNITION (UNIMODAL VIDEO)';
            modeBadge.style.color = '#94a3b8';
            modeBadge.style.borderColor = 'var(--border-subtle)';
        }
        if (mmPanel) mmPanel.style.display = 'none';
        stopMicrophoneStream();
        showToast('Microphone fusion disabled. Operating in unimodal video facial mode.', 'info');
    }
}

async function startMicrophoneStream() {
    try {
        const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        webcamState.audioStream = audioStream;

        let options = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported('audio/webm')) {
            options = {};
        }

        webcamState.mediaRecorder = new MediaRecorder(audioStream, options);
        webcamState.audioChunks = [];

        webcamState.mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
                webcamState.audioChunks.push(e.data);
            }
        };

        webcamState.mediaRecorder.onstop = async () => {
            if (webcamState.audioChunks.length === 0 || !webcamState.isRunning) return;
            const audioBlob = new Blob(webcamState.audioChunks, { type: 'audio/webm' });
            webcamState.audioChunks = [];
            await sendMultimodalStreamChunk(audioBlob);
            if (webcamState.isRunning && webcamState.micEnabled && webcamState.mediaRecorder) {
                webcamState.mediaRecorder.start();
            }
        };

        webcamState.mediaRecorder.start();

        // Slice audio chunks every 2.8 seconds
        webcamState.audioTimerId = setInterval(() => {
            if (webcamState.mediaRecorder && webcamState.mediaRecorder.state === 'recording') {
                webcamState.mediaRecorder.stop();
            }
        }, 2800);

    } catch (err) {
        console.warn('Microphone permission error:', err);
        showToast('Microphone access denied or unavailable: ' + err.message, 'warning');
        document.getElementById('webcam-mic-toggle').checked = false;
        webcamState.micEnabled = false;
    }
}

function stopMicrophoneStream() {
    if (webcamState.audioTimerId) {
        clearInterval(webcamState.audioTimerId);
        webcamState.audioTimerId = null;
    }
    if (webcamState.audioStream) {
        webcamState.audioStream.getTracks().forEach(t => t.stop());
        webcamState.audioStream = null;
    }
    if (webcamState.mediaRecorder && webcamState.mediaRecorder.state !== 'inactive') {
        webcamState.mediaRecorder.stop();
        webcamState.mediaRecorder = null;
    }
}

async function sendMultimodalStreamChunk(audioBlob) {
    const video = document.getElementById('webcam-video');
    if (!video || video.paused || video.ended) return;

    try {
        const snap = document.createElement('canvas');
        snap.width = 320;
        snap.height = 180;
        const sctx = snap.getContext('2d');
        sctx.drawImage(video, 0, 0, 320, 180);
        
        snap.toBlob(async (frameBlob) => {
            if (!frameBlob) return;
            const fd = new FormData();
            fd.append('frame_file', frameBlob, 'frame.jpg');
            fd.append('audio_file', audioBlob, 'chunk.webm');
            fd.append('client_id', webcamState.clientId);

            try {
                const res = await fetch('/predict/webcam/multimodal', {
                    method: 'POST',
                    body: fd
                });
                if (!res.ok) return;
                const mmData = await res.json();
                renderMultimodalStreamResult(mmData);
            } catch (e) {
                console.warn('Multimodal stream send error:', e);
            }
        }, 'image/jpeg', 0.85);

    } catch (err) {
        console.warn('sendMultimodalStreamChunk error:', err);
    }
}

function renderMultimodalStreamResult(data) {
    const predEl = document.getElementById('webcam-mm-pred');
    const confEl = document.getElementById('webcam-mm-conf');
    const wvEl = document.getElementById('webcam-weight-v');
    const waEl = document.getElementById('webcam-weight-a');
    const wtEl = document.getElementById('webcam-weight-t');
    const transEl = document.getElementById('webcam-mm-transcript');
    const explEl = document.getElementById('webcam-mm-explanation');

    if (predEl && data.fused_predicted_class) {
        const color = EMOTION_COLORS[data.fused_predicted_class.toLowerCase()] || '#38bdf8';
        predEl.textContent = data.fused_predicted_class.toUpperCase();
        predEl.style.color = color;
    }
    if (confEl && data.fused_confidence) {
        confEl.textContent = (data.fused_confidence * 100).toFixed(1) + '%';
    }
    if (wvEl && data.modality_weights) {
        wvEl.textContent = ((data.modality_weights.video || 0) * 100).toFixed(0) + '%';
        waEl.textContent = ((data.modality_weights.audio || 0) * 100).toFixed(0) + '%';
        wtEl.textContent = ((data.modality_weights.text || 0) * 100).toFixed(0) + '%';
    }
    if (transEl) {
        transEl.textContent = data.transcript_detected ? `"${data.transcript_detected}"` : '(Acoustic vocal tone analyzed without lexical speech)';
    }
    if (explEl && data.unified_explanation) {
        explEl.textContent = data.unified_explanation;
    }
}

async function saveWebcamDebugFrames() {
    try {
        const res = await fetch('/predict/webcam/debug-capture?num_frames=150', { method: 'POST' });
        if (res.ok) {
            showToast('📸 Debug capture active: Saving next 150 frames to results/live_debug/', 'info');
        } else {
            showToast('Failed to trigger debug capture', 'danger');
        }
    } catch (e) {
        showToast('Error triggering debug capture: ' + e.message, 'danger');
    }
}

