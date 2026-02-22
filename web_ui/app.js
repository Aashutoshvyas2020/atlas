(() => {
  const timelineEl = document.getElementById('timeline');
  const liveLineEl = document.getElementById('liveLine');
  const connectionStateEl = document.getElementById('connectionState');
  const browserOverlayEl = document.getElementById('browserOverlay');

  const taskPanelEl = document.getElementById('taskPanel');
  const taskPulseEl = document.getElementById('taskPulse');
  const taskTitleEl = document.getElementById('taskTitle');
  const taskSubtitleEl = document.getElementById('taskSubtitle');
  const thinkingTextEl = document.getElementById('thinkingText');
  const actionTextEl = document.getElementById('actionText');
  const thinkingStepsEl = document.getElementById('thinkingSteps');

  const searchQueryEl = document.getElementById('searchQuery');
  const searchSourcesEl = document.getElementById('searchSources');
  const memoryListEl = document.getElementById('memoryList');

  const micOnBtn = document.getElementById('micOnBtn');
  const micOffBtn = document.getElementById('micOffBtn');
  const museCalBtn = document.getElementById('museCalBtn');
  const emergencyBtn = document.getElementById('emergencyBtn');
  const textForm = document.getElementById('textForm');
  const textInput = document.getElementById('textInput');

  const confirmCard = document.getElementById('confirmCard');
  const confirmSummary = document.getElementById('confirmSummary');

  let ws = null;
  let speakerOn = true;
  let isMicOn = false;
  let currentTaskId = '';
  let pendingConfirmTaskId = '';
  const runningTaskIds = new Set();

  let lkRoom = null;
  let lkConfig = null;
  let assistantAudioSeenAt = 0;
  const lkAudioElements = new Map();
  let micMonitorStream = null;
  let micMonitorAudioCtx = null;
  let micMonitorAnalyser = null;
  let micMonitorRaf = 0;
  let micMonitorBuf = null;
  let bargeInLastAt = 0;
  let bargeInHotFrames = 0;
  let restoreVolumeTimer = 0;
  let overlayPulseTimer = 0;
  let micCalibrateInterval = 0;
  let thinkingSeq = 0;

  const BARGE_IN_THRESHOLD_RMS = 0.028;
  const BARGE_IN_REQUIRED_FRAMES = 2;
  const BARGE_IN_COOLDOWN_MS = 1200;
  const BARGE_IN_DUCK_VOLUME = 0.2;
  const BARGE_IN_DUCK_MS = 160;
  const BARGE_IN_RESTORE_DELAY_MS = 750;
  const BARGE_IN_RESTORE_MS = 240;
  const MIC_CALIBRATION_FAKE_SECONDS = 3;
  const MAX_CHAT_BUBBLES = 8;
  const MAX_THINKING_STEPS = 8;
  const MAX_MEMORY_ITEMS = 8;

  function wsUrl() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${location.host}/ws`;
  }

  function humanizeAction(action) {
    const raw = String(action || '').trim();
    if (!raw) return '-';
    if (raw.startsWith('tool:')) return `Tool: ${raw.replace('tool:', '').replaceAll('_', ' ')}`;
    return raw.replaceAll('_', ' ');
  }

  function setConnection(text, online = false) {
    connectionStateEl.textContent = text;
    connectionStateEl.classList.toggle('online', !!online);
  }

  function setLiveLine(text, mode = 'idle') {
    liveLineEl.textContent = text;
    liveLineEl.classList.remove('idle', 'active', 'warning');
    liveLineEl.classList.add(mode);
  }

  function setBrowserListeningGlow(active) {
    document.body.classList.toggle('listening', !!active);
  }

  function pulseBrowserOverlay() {
    if (!browserOverlayEl) return;
    browserOverlayEl.classList.remove('pulse');
    void browserOverlayEl.offsetWidth;
    browserOverlayEl.classList.add('pulse');
    if (overlayPulseTimer) {
      clearTimeout(overlayPulseTimer);
      overlayPulseTimer = 0;
    }
    overlayPulseTimer = setTimeout(() => {
      browserOverlayEl.classList.remove('pulse');
      overlayPulseTimer = 0;
    }, 760);
  }

  function setMuseCalibrationButton(state, remainingS = 0) {
    if (!museCalBtn) return;
    const mode = String(state || 'idle').trim().toLowerCase();
    const remain = Math.max(0, Number(remainingS) || 0);

    museCalBtn.classList.remove('calibrating', 'ready', 'error');
    museCalBtn.disabled = false;

    if (mode === 'restarting') {
      museCalBtn.classList.add('calibrating');
      museCalBtn.disabled = true;
      museCalBtn.textContent = 'Starting Calibration...';
      return;
    }
    if (mode === 'countdown' || mode === 'starting') {
      museCalBtn.classList.add('calibrating');
      museCalBtn.disabled = true;
      const shown = Math.max(1, Math.ceil(remain));
      museCalBtn.textContent = `Calibrating ${shown}s`;
      return;
    }
    if (mode === 'ready') {
      museCalBtn.classList.add('ready');
      museCalBtn.textContent = 'Calibrated: Mic On';
      return;
    }
    if (mode === 'error') {
      museCalBtn.classList.add('error');
      museCalBtn.textContent = 'Calibration Failed';
      return;
    }
    museCalBtn.textContent = 'Run Calibration (Mic Off)';
  }

  function updateMicButtons() {
    if (micOnBtn) micOnBtn.disabled = !!isMicOn;
    if (micOffBtn) micOffBtn.disabled = !isMicOn;
  }

  function clearMicCalibrationTicker() {
    if (micCalibrateInterval) {
      clearInterval(micCalibrateInterval);
      micCalibrateInterval = 0;
    }
  }

  function runMicCalibrationVisual(seconds = MIC_CALIBRATION_FAKE_SECONDS) {
    clearMicCalibrationTicker();
    let remain = Math.max(1, Number(seconds) || MIC_CALIBRATION_FAKE_SECONDS);
    setMuseCalibrationButton('countdown', remain);
    micCalibrateInterval = setInterval(() => {
      remain -= 1;
      if (remain <= 0) {
        clearMicCalibrationTicker();
        if (isMicOn) {
          setMuseCalibrationButton('ready', 0);
        } else {
          setMuseCalibrationButton('idle', 0);
        }
        return;
      }
      setMuseCalibrationButton('countdown', remain);
    }, 1000);
  }

  function addBubble(role, text) {
    const body = String(text || '').trim();
    if (!body) return;
    if (role !== 'user' && role !== 'assistant') return;

    const last = timelineEl.lastElementChild;
    if (last && last.classList.contains('bubble') && last.classList.contains(role)) {
      const bodyEl = last.querySelector('p');
      if (bodyEl) {
        bodyEl.textContent = `${bodyEl.textContent}\n${body}`;
        return;
      }
    }

    const item = document.createElement('article');
    item.className = `bubble ${role}`;

    const roleTag = document.createElement('span');
    roleTag.className = 'role-tag';
    roleTag.textContent = role === 'user' ? 'You' : 'Atlas';

    const bodyEl = document.createElement('p');
    bodyEl.textContent = body;

    item.appendChild(roleTag);
    item.appendChild(bodyEl);
    timelineEl.appendChild(item);
    while (timelineEl.childElementCount > MAX_CHAT_BUBBLES) {
      timelineEl.removeChild(timelineEl.firstElementChild);
    }
  }

  function resetThinkingSteps() {
    thinkingSeq = 0;
    if (thinkingStepsEl) {
      thinkingStepsEl.innerHTML = '';
    }
  }

  function pushThinkingStep(text, status = 'neutral', explicitStep = null) {
    if (!thinkingStepsEl) return;
    const body = String(text || '').trim();
    if (!body) return;

    thinkingSeq += 1;
    const stepNum = explicitStep == null ? thinkingSeq : explicitStep;
    const li = document.createElement('li');
    li.className = `thinking-step ${status}`;
    li.innerHTML = `<span class="step-index">#${stepNum}</span><span class="step-text"></span>`;
    li.querySelector('.step-text').textContent = body;
    thinkingStepsEl.appendChild(li);

    while (thinkingStepsEl.childElementCount > MAX_THINKING_STEPS) {
      thinkingStepsEl.removeChild(thinkingStepsEl.firstElementChild);
    }
  }

  function setTaskVisual(title, subtitle, running) {
    taskTitleEl.textContent = title;
    taskSubtitleEl.textContent = subtitle;
    taskPanelEl.classList.toggle('running', !!running);
    taskPulseEl.classList.toggle('idle', !running);
    taskPulseEl.classList.toggle('running', !!running);
  }

  function setThinking(text) {
    const body = String(text || '').trim();
    thinkingTextEl.textContent = body || 'Waiting for instructions.';
  }

  function setAction(text) {
    const body = String(text || '').trim();
    actionTextEl.textContent = body || '-';
  }

  function renderSearchSources(sources) {
    searchSourcesEl.innerHTML = '';
    if (!Array.isArray(sources) || !sources.length) return;

    for (const src of sources.slice(0, 4)) {
      const li = document.createElement('li');
      const a = document.createElement('a');
      const url = String(src?.url || '').trim();
      const title = String(src?.title || '').trim() || url;
      if (!url) continue;
      a.href = url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.textContent = title;
      li.appendChild(a);
      searchSourcesEl.appendChild(li);
    }
  }

  function parseToolOutput(raw) {
    if (!raw) return {};
    if (typeof raw === 'object') return raw;

    const text = String(raw || '').trim();
    if (!text) return {};

    try {
      return JSON.parse(text);
    } catch {
      // no-op
    }

    try {
      const normalized = text
        .replace(/\bTrue\b/g, 'true')
        .replace(/\bFalse\b/g, 'false')
        .replace(/\bNone\b/g, 'null')
        .replace(/'/g, '"');
      return JSON.parse(normalized);
    } catch {
      // no-op
    }

    const queryMatch = text.match(/['"]?query['"]?\s*:\s*['"]([^'"]+)['"]/i);
    const answerMatch = text.match(/['"]?answer['"]?\s*:\s*['"]([^'"]+)['"]/i);
    return {
      _raw: text,
      query: queryMatch ? queryMatch[1] : '',
      answer: answerMatch ? answerMatch[1] : '',
    };
  }

  function updateSearchView(query, sources) {
    const q = String(query || '').trim();
    searchQueryEl.textContent = q ? `Searching: ${q}` : 'No active search';
    renderSearchSources(sources);
  }

  function setMemoryItems(entries, sourceLabel = 'memory') {
    if (!memoryListEl) return;
    memoryListEl.innerHTML = '';
    const rows = Array.isArray(entries) ? entries : [];
    const trimmed = rows.slice(-MAX_MEMORY_ITEMS);
    if (!trimmed.length) {
      const li = document.createElement('li');
      li.textContent = 'No saved memory yet';
      memoryListEl.appendChild(li);
      return;
    }
    for (const row of trimmed) {
      const li = createMemoryNode(row, sourceLabel);
      if (li) memoryListEl.appendChild(li);
    }
  }

  function addMemoryItem(entry, sourceLabel = 'memory') {
    if (!memoryListEl) return;
    const text = String(entry || '').trim();
    if (!text) return;
    if (memoryListEl.childElementCount === 1 && /no saved memory yet/i.test(memoryListEl.textContent || '')) {
      memoryListEl.innerHTML = '';
    }
    const li = createMemoryNode(text, sourceLabel);
    if (!li) return;
    memoryListEl.appendChild(li);
    while (memoryListEl.childElementCount > MAX_MEMORY_ITEMS) {
      memoryListEl.removeChild(memoryListEl.firstElementChild);
    }
  }

  function parseMemoryEntry(rawEntry) {
    const text = String(rawEntry || '').trim();
    if (!text) return null;
    const m = text.match(/^\-\s*\[([^\]]+)\]\s*([^:]+):\s*(.*?)(?:\s*\(source=([^)]+)\))?$/i);
    if (!m) {
      return {
        category: '',
        key: '',
        value: text,
        source: '',
      };
    }
    return {
      category: String(m[1] || '').trim(),
      key: String(m[2] || '').trim(),
      value: String(m[3] || '').trim(),
      source: String(m[4] || '').trim(),
    };
  }

  function createMemoryNode(entry, sourceLabel) {
    const parsed = parseMemoryEntry(entry);
    if (!parsed) return null;

    const li = document.createElement('li');
    li.className = 'memory-item';

    const head = document.createElement('div');
    head.className = 'memory-head';

    const src = document.createElement('span');
    src.className = 'memory-src';
    src.textContent = String(sourceLabel || 'memory');
    head.appendChild(src);

    const cat = document.createElement('span');
    cat.className = 'memory-cat';
    cat.textContent = parsed.category || 'general';
    head.appendChild(cat);
    li.appendChild(head);

    const text = document.createElement('div');
    text.className = 'memory-text';
    if (parsed.key) {
      text.innerHTML = `<strong></strong><span></span>`;
      text.querySelector('strong').textContent = `${parsed.key}: `;
      text.querySelector('span').textContent = parsed.value || '';
    } else {
      text.textContent = parsed.value || '';
    }
    li.appendChild(text);

    if (parsed.source) {
      const meta = document.createElement('div');
      meta.className = 'memory-meta';
      meta.textContent = `from ${parsed.source}`;
      li.appendChild(meta);
    }
    return li;
  }

  function send(type, payload = {}) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type, payload }));
  }

  function speak(text) {
    if (!speakerOn || !text || !('speechSynthesis' in window)) return;
    // When LiveKit voice is active for the session, avoid browser TTS duplication.
    if (lkConfig) return;
    if (Date.now() - assistantAudioSeenAt < 1200) return;

    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.06;
    u.pitch = 1.0;
    window.speechSynthesis.speak(u);
  }

  function muteAllRemoteAudio(muted) {
    for (const el of lkAudioElements.values()) {
      el.muted = !!muted;
      if (!muted) {
        el.volume = 1.0;
      }
      if (!muted) {
        const p = el.play();
        if (p && typeof p.catch === 'function') p.catch(() => {});
      }
    }
  }

  function tweenRemoteVolume(targetVolume, durationMs) {
    const els = Array.from(lkAudioElements.values()).filter((el) => el && !el.muted);
    if (!els.length) return;
    const target = Math.max(0, Math.min(1, Number(targetVolume || 0)));
    const duration = Math.max(1, Number(durationMs || 1));
    const starts = els.map((el) => Number.isFinite(el.volume) ? el.volume : 1.0);
    const started = performance.now();

    const step = (now) => {
      const t = Math.max(0, Math.min(1, (now - started) / duration));
      for (let i = 0; i < els.length; i += 1) {
        const el = els[i];
        const v0 = starts[i];
        el.volume = v0 + ((target - v0) * t);
      }
      if (t < 1) {
        requestAnimationFrame(step);
      }
    };
    requestAnimationFrame(step);
  }

  function duckAssistantAudio() {
    if (!speakerOn) return;
    tweenRemoteVolume(BARGE_IN_DUCK_VOLUME, BARGE_IN_DUCK_MS);
    if (restoreVolumeTimer) {
      clearTimeout(restoreVolumeTimer);
    }
    restoreVolumeTimer = setTimeout(() => {
      if (!speakerOn) return;
      tweenRemoteVolume(1.0, BARGE_IN_RESTORE_MS);
    }, BARGE_IN_RESTORE_DELAY_MS);
  }

  function onUserBargeInDetected() {
    const now = Date.now();
    if (now - bargeInLastAt < BARGE_IN_COOLDOWN_MS) return;
    bargeInLastAt = now;
    duckAssistantAudio();
    send('client.assistant.interrupt', { reason: 'voice_barge_in' });
  }

  function stopSpeechInterruptWatcher() {
    if (micMonitorRaf) {
      cancelAnimationFrame(micMonitorRaf);
      micMonitorRaf = 0;
    }
    bargeInHotFrames = 0;
    micMonitorBuf = null;
    micMonitorAnalyser = null;
    if (micMonitorAudioCtx) {
      try {
        micMonitorAudioCtx.close();
      } catch {
        // no-op
      }
      micMonitorAudioCtx = null;
    }
    if (micMonitorStream) {
      for (const tr of micMonitorStream.getTracks()) {
        try {
          tr.stop();
        } catch {
          // no-op
        }
      }
      micMonitorStream = null;
    }
  }

  async function startSpeechInterruptWatcher() {
    if (micMonitorRaf || !isMicOn) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;

    try {
      micMonitorStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      micMonitorAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const src = micMonitorAudioCtx.createMediaStreamSource(micMonitorStream);
      micMonitorAnalyser = micMonitorAudioCtx.createAnalyser();
      micMonitorAnalyser.fftSize = 1024;
      src.connect(micMonitorAnalyser);
      micMonitorBuf = new Float32Array(micMonitorAnalyser.fftSize);

      const tick = () => {
        if (!isMicOn || !micMonitorAnalyser || !micMonitorBuf) return;
        micMonitorAnalyser.getFloatTimeDomainData(micMonitorBuf);
        let sumSq = 0;
        for (let i = 0; i < micMonitorBuf.length; i += 1) {
          const v = micMonitorBuf[i];
          sumSq += v * v;
        }
        const rms = Math.sqrt(sumSq / micMonitorBuf.length);
        if (rms >= BARGE_IN_THRESHOLD_RMS) {
          bargeInHotFrames += 1;
        } else {
          bargeInHotFrames = Math.max(0, bargeInHotFrames - 1);
        }
        if (bargeInHotFrames >= BARGE_IN_REQUIRED_FRAMES) {
          bargeInHotFrames = 0;
          onUserBargeInDetected();
        }
        micMonitorRaf = requestAnimationFrame(tick);
      };

      micMonitorRaf = requestAnimationFrame(tick);
    } catch {
      stopSpeechInterruptWatcher();
    }
  }

  async function disconnectLiveKit() {
    if (lkRoom) {
      try {
        await lkRoom.disconnect();
      } catch {
        // no-op
      }
      lkRoom = null;
    }

    for (const [key, el] of lkAudioElements.entries()) {
      try {
        el.pause();
        el.remove();
      } catch {
        // no-op
      }
      lkAudioElements.delete(key);
    }
  }

  async function connectLiveKit(cfg) {
    lkConfig = cfg;

    if (!window.LivekitClient) {
      setLiveLine('LiveKit client not loaded', 'warning');
      return;
    }

    await disconnectLiveKit();

    const { Room, RoomEvent, Track } = window.LivekitClient;
    const room = new Room({ adaptiveStream: true, dynacast: true });

    room.on(RoomEvent.Connected, () => {
      setConnection('Connected', true);
      if (isMicOn) {
        room.localParticipant.setMicrophoneEnabled(true).catch(() => {
          setLiveLine('Microphone permission failed', 'warning');
        });
      }
    });

    room.on(RoomEvent.Disconnected, () => {
      setConnection('Disconnected', false);
      setLiveLine('Connection lost. Reconnecting...', 'warning');
    });

    room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
      if (track.kind !== Track.Kind.Audio) return;
      const identity = participant?.identity || '';
      if (identity && !identity.startsWith('atlas-agent-')) return;
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }

      const el = track.attach();
      el.autoplay = true;
      el.playsInline = true;
      el.style.display = 'none';
      el.muted = !speakerOn;
      el.volume = speakerOn ? 1.0 : 0.0;
      document.body.appendChild(el);

      const key = publication?.trackSid || `${identity}:${Date.now()}`;
      lkAudioElements.set(key, el);
      assistantAudioSeenAt = Date.now();

      const p = el.play();
      if (p && typeof p.catch === 'function') p.catch(() => {});
    });

    room.on(RoomEvent.TrackUnsubscribed, (_track, publication) => {
      const key = publication?.trackSid;
      if (!key) return;
      const el = lkAudioElements.get(key);
      if (!el) return;
      try {
        el.pause();
        el.remove();
      } catch {
        // no-op
      }
      lkAudioElements.delete(key);
    });

    try {
      await room.connect(cfg.url, cfg.token, { autoSubscribe: true });
      lkRoom = room;
    } catch {
      setLiveLine('LiveKit connection failed', 'warning');
      await disconnectLiveKit();
    }
  }

  function onTaskStarted(payload) {
    const taskId = String(payload.task_id || '');
    const hadNoRunningTasks = runningTaskIds.size === 0;
    currentTaskId = taskId;
    if (taskId) runningTaskIds.add(taskId);

    const count = runningTaskIds.size;
    const goal = String(payload.goal || '').trim();
    const subtitle = count > 1 ? `${count} tasks running` : (goal || 'Task started');

    setTaskVisual('Running', subtitle, true);
    setThinking('Planning and executing actions...');
    if (hadNoRunningTasks) {
      resetThinkingSteps();
    }
    pushThinkingStep(`Task started${goal ? `: ${goal}` : ''}`, 'neutral', payload.step ?? null);
  }

  function onTaskStep(payload) {
    const taskId = String(payload.task_id || '');
    const action = String(payload.action || '').trim();
    const status = String(payload.status || 'ok').trim();
    const detail = payload.detail || {};
    const hadActiveTasks = runningTaskIds.size > 0;

    if (taskId) {
      currentTaskId = taskId;
      if (!runningTaskIds.has(taskId) && status !== 'error') runningTaskIds.add(taskId);
    }

    const detailText = typeof detail === 'object' ? JSON.stringify(detail) : String(detail || '');

    if (action === 'emergency_stop') {
      pulseBrowserOverlay();
    }

    if (action === 'memory_snapshot') {
      const entries = Array.isArray(detail?.entries) ? detail.entries : [];
      setMemoryItems(entries, 'snapshot');
    } else if (action === 'memory_write') {
      addMemoryItem(detail?.entry || '', String(detail?.source || 'memory'));
    } else if (action === 'memory_auto_capture') {
      addMemoryItem(detail?.entry || '', 'auto');
    }
    if (action === 'memory_snapshot' || action === 'memory_write' || action === 'memory_auto_capture') {
      setAction('Memory updated');
      return;
    }

    if (action === 'model_text') {
      const thought = String(detail?.text || '').trim() || detailText;
      if (thought) {
        setThinking(thought);
        pushThinkingStep(thought, status === 'error' ? 'error' : 'thinking', payload.step ?? null);
      }
    } else if (action.startsWith('tool:')) {
      const toolName = action.replace('tool:', '');
      const parsed = parseToolOutput(detail?.output ?? detail);
      if (toolName === 'remember_user_profile') {
        const entry = String(parsed?.entry || detail?.entry || '').trim();
        if (entry) {
          addMemoryItem(entry, 'tool');
        }
      }
      if (toolName === 'read_user_profile') {
        const matches = Array.isArray(parsed?.matches) ? parsed.matches : [];
        if (matches.length) {
          setMemoryItems(matches, 'read');
        }
      }

      if (toolName === 'quick_web_search') {
        const query = String(parsed.query || '').trim();
        const sources = Array.isArray(parsed.sources) ? parsed.sources : [];
        updateSearchView(query || 'Web search in progress', sources);
        if (query) {
          setThinking('Running web search for a direct answer...');
        } else {
          setThinking('Running web search...');
        }
      }
      pushThinkingStep(`Tool call: ${toolName}`, status === 'error' ? 'error' : 'ok', payload.step ?? null);
    } else if (action) {
      setThinking('Executing computer actions...');
      pushThinkingStep(humanizeAction(action), status === 'error' ? 'error' : 'ok', payload.step ?? null);
    }

    setAction(`${humanizeAction(action)}${status === 'error' ? ' (issue)' : ''}`);

    const runCount = runningTaskIds.size;
    if (!taskId && !runCount) {
      if (hadActiveTasks) {
        setTaskVisual('Idle', 'No active tasks', false);
      }
      return;
    }
    const subtitle = status === 'error'
      ? 'Encountered an issue, attempting recovery'
      : (runCount > 1 ? `${runCount} tasks running` : (taskId ? `Task ${taskId}` : 'Task active'));
    setTaskVisual('Running', subtitle, true);
  }

  function onTaskCompleted(payload) {
    const taskId = String(payload.task_id || '');
    const result = String(payload.result || '').trim() || 'completed';
    const reason = String(payload.reason || '').trim();

    if (taskId) runningTaskIds.delete(taskId);
    if (taskId === currentTaskId) currentTaskId = '';

    if (runningTaskIds.size > 0) {
      setTaskVisual('Running', `${runningTaskIds.size} tasks running`, true);
      return;
    }

    const title = result === 'success' ? 'Complete' : (result === 'stopped' ? 'Stopped' : 'Ended');
    setTaskVisual(title, reason || 'No active tasks', false);
    setThinking('Waiting for instructions.');
    setAction('-');
    pushThinkingStep(`Task ${title.toLowerCase()}${reason ? `: ${reason}` : ''}`, result === 'success' ? 'ok' : 'error', payload.step ?? null);
  }

  function onMuseCalibration(payload) {
    const state = String(payload.state || '').trim().toLowerCase();
    const ready = !!payload.ready;
    const remaining = Number(payload.remaining_s || 0);
    if (ready || state === 'ready') {
      setMuseCalibrationButton(isMicOn ? 'ready' : 'idle', 0);
      setLiveLine(isMicOn ? 'Listening...' : 'Mic is off', isMicOn ? 'active' : 'idle');
      return;
    }
    if (state === 'error') {
      clearMicCalibrationTicker();
      setMuseCalibrationButton('error', 0);
      setLiveLine('Muse recalibration failed', 'warning');
      return;
    }
    if (state === 'restarting') {
      setMuseCalibrationButton('restarting', remaining || MIC_CALIBRATION_FAKE_SECONDS);
      setLiveLine('Recalibrating Muse...', 'warning');
      return;
    }
    if (state === 'starting' || state === 'countdown') {
      setMuseCalibrationButton('countdown', remaining || MIC_CALIBRATION_FAKE_SECONDS);
      setLiveLine(`Muse calibrating ${Math.max(1, Math.ceil(remaining || MIC_CALIBRATION_FAKE_SECONDS))}s...`, 'warning');
      return;
    }
  }

  function handleServerMessage(msg) {
    const type = msg.type || '';
    const payload = msg.payload || {};

    if (type === 'server.ready') {
      setConnection('Connected', true);
      send('client.session.start', { mode: 'always_listen' });
      return;
    }

    if (type === 'server.livekit.config') {
      void connectLiveKit(payload);
      return;
    }

    if (type === 'server.session') {
      const status = String(payload.status || '').trim();
      if (status === 'connected') {
        setConnection('Connected', true);
      } else if (status === 'recovering') {
        setConnection('Recovering model', false);
      } else if (status === 'closed') {
        setConnection('Session closed', false);
      }
      return;
    }

    if (type === 'server.transcript.partial') {
      const txt = String(payload.text || '').trim();
      setLiveLine(txt ? `Listening: ${txt}` : 'Listening...', 'active');
      return;
    }

    if (type === 'server.transcript.final') {
      const txt = String(payload.text || '').trim();
      if (txt) addBubble('user', txt);
      setLiveLine(isMicOn ? 'Listening...' : 'Mic is off', isMicOn ? 'active' : 'idle');
      return;
    }

    if (type === 'server.assistant.transcript.partial') {
      const txt = String(payload.text || '').trim();
      if (txt) setLiveLine(`Atlas: ${txt}`, 'active');
      return;
    }

    if (type === 'server.assistant.text') {
      const txt = String(payload.text || '').trim();
      if (txt) {
        addBubble('assistant', txt);
        speak(txt);
      }
      return;
    }

    if (type === 'server.assistant.voice_hint') {
      const txt = String(payload.text || '').trim();
      if (txt) speak(txt);
      return;
    }

    if (type === 'server.turn.complete') {
      setLiveLine(isMicOn ? 'Listening...' : 'Mic is off', isMicOn ? 'active' : 'idle');
      return;
    }

    if (type === 'server.task.started') {
      onTaskStarted(payload);
      return;
    }

    if (type === 'server.task.step') {
      onTaskStep(payload);
      if (String(payload.action || '') === 'confirmation_response') {
        confirmCard.classList.add('hidden');
      }
      return;
    }

    if (type === 'server.task.confirmation_required') {
      pendingConfirmTaskId = String(payload.task_id || '');
      confirmSummary.textContent = String(payload.summary || 'Approve this action?');
      confirmCard.classList.remove('hidden');
      setTaskVisual('Awaiting confirmation', 'Action paused for your decision', true);
      setThinking('Waiting for approval to continue.');
      return;
    }

    if (type === 'server.mic.toggle') {
      const targetActive = typeof payload.active === 'boolean' ? payload.active : null;
      if (targetActive === true && !isMicOn) {
        void startMic();
      } else if (targetActive === false && isMicOn) {
        void stopMic();
      } else if (targetActive === null && isMicOn) {
        void stopMic();
      } else if (targetActive === null) {
        void startMic();
      }
      return;
    }

    if (type === 'server.task.completed') {
      confirmCard.classList.add('hidden');
      pendingConfirmTaskId = '';
      onTaskCompleted(payload);
      return;
    }

    if (type === 'server.muse.calibration') {
      onMuseCalibration(payload);
      return;
    }

    if (type === 'server.error') {
      const scope = String(payload.scope || '').trim();
      const message = String(payload.message || '').trim();
      setLiveLine(message ? `${scope}: ${message}` : 'An error occurred', 'warning');
      return;
    }
  }

  function connectWs() {
    setConnection('Connecting...', false);
    ws = new WebSocket(wsUrl());

    ws.onopen = () => {
      setConnection('Connected', true);
      setLiveLine(isMicOn ? 'Listening...' : 'Mic is off', isMicOn ? 'active' : 'idle');
    };

    ws.onclose = () => {
      setConnection('Reconnecting...', false);
      setLiveLine('Connection lost. Reconnecting...', 'warning');
      setBrowserListeningGlow(false);
      clearMicCalibrationTicker();
      setMuseCalibrationButton('idle', 0);
      setTimeout(connectWs, 1000);
    };

    ws.onerror = () => {
      setConnection('Connection error', false);
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        handleServerMessage(msg);
      } catch {
        setLiveLine('Received invalid server message', 'warning');
      }
    };
  }

  async function startMic() {
    if (isMicOn) {
      if (lkRoom) {
        try {
          await lkRoom.localParticipant.setMicrophoneEnabled(true);
        } catch {
          setLiveLine('Microphone permission failed', 'warning');
        }
      }
      setLiveLine('Listening...', 'active');
      setBrowserListeningGlow(true);
      setMuseCalibrationButton('ready', 0);
      updateMicButtons();
      return;
    }

    isMicOn = true;
    setLiveLine('Listening...', 'active');
    setBrowserListeningGlow(true);
    runMicCalibrationVisual();
    updateMicButtons();

    if (lkRoom) {
      try {
        await lkRoom.localParticipant.setMicrophoneEnabled(true);
      } catch {
        setLiveLine('Microphone permission failed', 'warning');
      }
    }

    await startSpeechInterruptWatcher();
  }

  async function stopMic() {
    if (!isMicOn) return;
    isMicOn = false;
    setLiveLine('Mic is off', 'idle');
    setBrowserListeningGlow(false);
    clearMicCalibrationTicker();
    setMuseCalibrationButton('idle', 0);
    updateMicButtons();

    if (lkRoom) {
      try {
        await lkRoom.localParticipant.setMicrophoneEnabled(false);
      } catch {
        setLiveLine('Unable to disable mic', 'warning');
      }
    }

    stopSpeechInterruptWatcher();
  }

  emergencyBtn.addEventListener('click', () => {
    send('client.assistant.interrupt', { reason: 'emergency_kill_switch' });
    send('client.task.interrupt', { task_id: currentTaskId, reason: 'ui_interrupt' });
  });

  if (micOnBtn) {
    micOnBtn.addEventListener('click', () => {
      void startMic();
    });
  }

  if (micOffBtn) {
    micOffBtn.addEventListener('click', () => {
      void stopMic();
    });
  }

  if (museCalBtn) {
    museCalBtn.addEventListener('click', () => {
      void startMic();
    });
  }

  textForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = textInput.value.trim();
    if (!text) return;
    addBubble('user', text);
    send('client.text.input', { text });
    textInput.value = '';
  });

  window.addEventListener('beforeunload', () => {
    clearMicCalibrationTicker();
    stopSpeechInterruptWatcher();
    void disconnectLiveKit();
    setBrowserListeningGlow(false);
  });

  setConnection('Connecting...', false);
  setTaskVisual('Idle', 'No active tasks', false);
  updateSearchView('', []);
  setThinking('Waiting for instructions.');
  setAction('-');
  resetThinkingSteps();
  setMemoryItems([], 'memory');
  setBrowserListeningGlow(false);
  setMuseCalibrationButton('idle', 0);
  updateMicButtons();
  connectWs();
})();

