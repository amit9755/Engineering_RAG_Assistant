// ============================================================
// app.js - Engineering RAG Assistant Frontend Logic
//
// Sections:
//   1. HEALTH CHECK
//   2. MAIN TAB NAVIGATION (Chat / Knowledge Sources / Investigate)
//   3. FILE UPLOAD (chat sidebar + sources tab)
//   4. CHAT - send query + streaming
//   5. RAG INTERNALS DISPLAY
//   6. SOURCE MANAGEMENT - documents, bitbucket, jira
//   7. MODALS - add bitbucket, add jira, confirm delete
//   8. UTILITY FUNCTIONS
// ============================================================

const API_BASE = 'http://localhost:8000/api/v1';

// ===== Application State =====
const state = {
  sessionId: generateSessionId(),
  conversationHistory: [],
  isLoading: false,
  lastResult: null,
  // Source management state
  allSources: { documents: [], legacy: [], bitbucket: [], jira: [], confluence: [] },
  pickers: {},          // 'bb' / 'conf' -> { items, selected: Set } for the browse lists
  sourcePollTimer: null,
  progress: {},         // source id -> live indexing progress from /sources/progress
  chatMessages: [],     // this chat's messages with sources, saved to /chats after each answer
  msgCounter: 0,
  sourceStatuses: {},   // id -> last seen status, to announce finished indexing
  pendingDelete: null,  // { type, id, name }
  bbConnectionTested: false,
  jiraConnectionTested: false,
};

// ===== Initialisation =====
document.addEventListener('DOMContentLoaded', () => {
  applyTheme(document.documentElement.dataset.theme);
  document.getElementById('sessionInfo').textContent = 'Session: ' + state.sessionId.slice(-8);
  checkServerHealth();
  loadKBStats();
  loadAllSources();
  loadChatList();
  // Poll health every 30 seconds
  setInterval(checkServerHealth, 30000);
  setInterval(loadKBStats, 60000);
});

// ============================================================
// HEALTH CHECK
// Learning Note:
//   We call the /health/ready endpoint every 30s to show a
//   green/red indicator so users know if the server is running.
// ============================================================
async function checkServerHealth() {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  try {
    dot.className = 'status-dot loading';
    text.textContent = 'Connecting...';
    const res = await fetch(`${API_BASE}/health/ready`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    dot.className = 'status-dot online';
    text.textContent = `Server ready (${data.environment})`;
  } catch {
    dot.className = 'status-dot error';
    text.textContent = 'Server offline - start with: python -m uvicorn src.api.main:app';
  }
}

async function loadKBStats() {
  try {
    const res = await fetch(`${API_BASE}/health/stats`);
    const data = await res.json();
    const kb = data.knowledge_base || {};
    const chunkEl = document.getElementById('chunkCount');
    if (chunkEl) chunkEl.textContent = kb.total_chunks !== undefined ? kb.total_chunks : '-';
  } catch { /* silent fail */ }
}

// ============================================================
// FILE UPLOAD
// Learning Note:
//   We handle both drag-and-drop and click-to-browse.
//   Files are sent as multipart/form-data to the ingest endpoint.
//   We show progress per file in the upload queue.
// ============================================================
// Shared by the Knowledge Sources upload area.
function handleDragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
  e.currentTarget.classList.remove('drag-over');
}

// ============================================================
// CHAT - SEND QUERY
// Learning Note:
//   We support two modes:
//   1. Normal mode: POST /query -> wait for full response
//   2. Streaming mode: POST /query/stream -> Server-Sent Events
//   The streaming mode shows tokens as they arrive (like ChatGPT).
// ============================================================
async function sendQuery() {
  const input = document.getElementById('queryInput');
  const question = input.value.trim();
  if (!question || state.isLoading) return;

  const sourceFilter = getSourceFilter();
  if (sourceFilter && !sourceFilter.source_ids.length && !sourceFilter.legacy_files.length) {
    showToast('Select at least one source under "Search In"', 'error');
    return;
  }

  // Hide welcome message on first question
  const welcome = document.getElementById('welcomeMessage');
  if (welcome) welcome.style.display = 'none';

  // Add user message to chat
  addMessage('user', question);
  input.value = '';
  autoResize(input);

  // Update conversation history
  state.conversationHistory.push({ role: 'user', content: question });
  state.chatMessages.push({ role: 'user', content: question, sources: [] });
  document.querySelectorAll('.suggestions').forEach(el => el.remove());

  const useStreaming = document.getElementById('streamingMode').checked;

  setLoading(true);

  if (useStreaming) {
    await sendStreamingQuery(question, sourceFilter);
  } else {
    await sendNormalQuery(question, sourceFilter);
  }
}

async function sendNormalQuery(question, sourceFilter) {
  // Show typing indicator
  const typingId = addTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-User-ID': 'ui-user' },
      body: JSON.stringify({
        question,
        session_id: state.sessionId,
        conversation_history: state.conversationHistory.slice(-10), // last 5 turns
        ...(sourceFilter || {}),
      }),
    });

    removeTypingIndicator(typingId);

    if (!res.ok) {
      const err = await res.json();
      addMessage('assistant', `Error: ${err.detail || 'Request failed'}`, { error: true });
      return;
    }

    const data = await res.json();
    state.lastResult = data;

    // Add assistant message
    const msgId = addMessage('assistant', data.answer, {
      sources: data.sources,
      inputSafe: data.input_safe,
      outputSafe: data.output_safe,
      latencyMs: data.latency_ms,
      model: data.model_used,
    });

    // Update right panel
    updateRagInternals(data);

    // Update conversation history with assistant response
    state.conversationHistory.push({ role: 'assistant', content: data.answer });
    afterAnswer(msgId, question, data.answer, data.sources || [], sourceFilter);

  } catch (err) {
    removeTypingIndicator(typingId);
    addMessage('assistant', `Connection error: ${err.message}. Is the server running?`, { error: true });
  } finally {
    setLoading(false);
  }
}

async function sendStreamingQuery(question, sourceFilter) {
  // Create an empty assistant message that we'll fill in
  const msgId = addStreamingMessage();

  try {
    const res = await fetch(`${API_BASE}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-User-ID': 'ui-user' },
      body: JSON.stringify({
        question,
        session_id: state.sessionId,
        conversation_history: state.conversationHistory.slice(-10),
        ...(sourceFilter || {}),
      }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fullResponse = '';
    let pending = '';   // a long event can arrive split across network reads

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      pending += decoder.decode(value, { stream: true });
      const lines = pending.split('\n');
      pending = lines.pop();   // keep the incomplete last line for the next read

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));

          if (event.error) {
            updateStreamingMessage(msgId, `Error: ${event.error}`);
            break;
          }

          if (event.token) {
            fullResponse += event.token;
            updateStreamingMessage(msgId, fullResponse);
          }

          if (event.done && event.type === 'metadata') {
            // Add sources to the message
            addSourcesToMessage(msgId, event.sources || []);
            state.conversationHistory.push({ role: 'assistant', content: fullResponse });
            afterAnswer(msgId, question, fullResponse, event.sources || [], sourceFilter);
          }
        } catch { /* malformed event, skip */ }
      }
    }
  } catch (err) {
    updateStreamingMessage(msgId, `Connection error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

// ============================================================
// AFTER EACH ANSWER: save the chat, then suggest follow-up questions
// ============================================================

function afterAnswer(msgId, question, answer, sources, sourceFilter) {
  state.chatMessages.push({ role: 'assistant', content: answer, sources });
  saveCurrentChat();
  loadSuggestions(msgId, question, answer, sources, sourceFilter);
}

async function loadSuggestions(msgId, question, answer, sources, sourceFilter) {
  const msg = document.getElementById(msgId);
  if (!msg) return;
  const box = document.createElement('div');
  box.className = 'suggestions';
  box.innerHTML = '<span class="suggestions-label"><span class="spinner"></span> Thinking of follow-up questions...</span>';
  msg.querySelector('.message-content').appendChild(box);
  try {
    const res = await fetch(`${API_BASE}/query/suggestions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, answer, sources, ...(sourceFilter || {}) }),
    });
    const data = res.ok ? await res.json() : { suggestions: [] };
    if (!data.suggestions.length) { box.remove(); return; }
    box.innerHTML = '<span class="suggestions-label">You could also ask:</span>' +
      data.suggestions.map(q => `<button class="suggestion-chip">${escapeHtml(q)}</button>`).join('');
    box.querySelectorAll('.suggestion-chip').forEach((chip, i) => {
      chip.addEventListener('click', () => {
        if (state.isLoading) return;
        const input = document.getElementById('queryInput');
        input.value = data.suggestions[i];
        sendQuery();
      });
    });
    scrollToBottom();
  } catch {
    box.remove();
  }
}

// ============================================================
// SAVED CHATS (right panel): the server keeps the last 10
// ============================================================

function switchRightTab(tab) {
  ['chats', 'internals'].forEach(name => {
    document.getElementById(`rtab-${name}`).classList.toggle('active', name === tab);
    document.getElementById(`rpanel-${name}`).classList.toggle('active', name === tab);
  });
}

async function saveCurrentChat() {
  if (!state.chatMessages.length) return;
  try {
    await fetch(`${API_BASE}/chats/${encodeURIComponent(state.sessionId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: state.chatMessages }),
    });
    await loadChatList();
  } catch { /* saving is best effort; the chat itself still works */ }
}

async function loadChatList() {
  try {
    const res = await fetch(`${API_BASE}/chats`);
    state.chats = res.ok ? await res.json() : [];
  } catch {
    state.chats = [];
  }
  renderChatList();
}

function formatWhen(iso) {
  const date = new Date(iso);
  const minutes = Math.round((Date.now() - date) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 24 * 60) return `${Math.round(minutes / 60)} h ago`;
  return date.toLocaleDateString();
}

function renderChatList() {
  const container = document.getElementById('chatList');
  const chats = state.chats || [];
  if (!chats.length) {
    container.innerHTML = '<p class="source-hint">No saved chats yet. Your last 10 chats appear here.</p>';
    return;
  }
  container.innerHTML = chats.map(chat => `
    <div class="chat-item ${chat.id === state.sessionId ? 'active' : ''}" data-chat="${escapeHtml(chat.id)}">
      <div class="chat-item-text">
        <div class="chat-item-title">${escapeHtml(chat.title)}</div>
        <div class="chat-item-meta">${formatWhen(chat.updated_at)} &middot; ${chat.message_count} messages</div>
      </div>
      <button class="chat-item-delete" data-delete-chat="${escapeHtml(chat.id)}" title="Delete chat">&times;</button>
    </div>`).join('');
  container.querySelectorAll('.chat-item').forEach(item => {
    item.addEventListener('click', () => openChat(item.dataset.chat));
  });
  container.querySelectorAll('[data-delete-chat]').forEach(button => {
    button.addEventListener('click', event => {
      event.stopPropagation();
      deleteChat(button.dataset.deleteChat);
    });
  });
}

async function openChat(chatId) {
  if (state.isLoading || chatId === state.sessionId) return;
  try {
    const res = await fetch(`${API_BASE}/chats/${encodeURIComponent(chatId)}`);
    if (!res.ok) throw new Error('Chat not found');
    const chat = await res.json();
    document.getElementById('messagesContainer').innerHTML = '';
    document.getElementById('welcomeMessage').style.display = 'none';
    chat.messages.forEach(m => addMessage(m.role, m.content, { sources: m.sources }));
    state.sessionId = chat.id;
    state.chatMessages = chat.messages;
    state.conversationHistory = chat.messages.map(m => ({ role: m.role, content: m.content }));
    document.getElementById('sessionInfo').textContent = 'Session: ' + state.sessionId.slice(-8);
    renderChatList();
  } catch (err) {
    showToast(err.message, 'error');
    loadChatList();
  }
}

async function deleteChat(chatId) {
  try {
    await fetch(`${API_BASE}/chats/${encodeURIComponent(chatId)}`, { method: 'DELETE' });
  } finally {
    if (chatId === state.sessionId) newChat();
    loadChatList();
  }
}

// ============================================================
// MESSAGE RENDERING
// Learning Note:
//   We build HTML strings and use innerHTML.
//   In a production app you would sanitize HTML to prevent XSS.
//   For this learning project, we trust our own backend responses.
// ============================================================

function addMessage(role, content, meta = {}) {
  const container = document.getElementById('messagesContainer');
  const id = `msg-${Date.now()}-${++state.msgCounter}`;
  const avatar = role === 'user' ? '&#128100;' : '&#129302;';
  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  let sourcesHtml = '';
  if (meta.sources && meta.sources.length > 0) {
    const tags = meta.sources.map(s => `<span class="source-tag">&#128196; ${escapeHtml(s)}</span>`).join('');
    sourcesHtml = `<div class="message-sources">${tags}</div>`;
  }

  let safetyHtml = '';
  if (meta.inputSafe !== undefined) {
    safetyHtml = meta.inputSafe && meta.outputSafe
      ? '<span class="safety-badge safe">&#9989; Safe</span>'
      : '<span class="safety-badge unsafe">&#9888; Flagged</span>';
  }

  let metaHtml = '';
  if (meta.latencyMs) {
    metaHtml = `<span class="message-meta">${timeStr} &bull; ${meta.latencyMs}ms &bull; ${escapeHtml(meta.model || '')}${safetyHtml}</span>`;
  } else {
    metaHtml = `<span class="message-meta">${timeStr}</span>`;
  }

  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.id = id;
  div.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-content">
      <div class="message-bubble">${formatMessageContent(content)}</div>
      ${sourcesHtml}
      ${metaHtml}
    </div>`;

  container.appendChild(div);
  scrollToBottom();
  return id;
}

function addTypingIndicator() {
  const container = document.getElementById('messagesContainer');
  const id = 'typing-' + Date.now();
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.id = id;
  div.innerHTML = `
    <div class="message-avatar">&#129302;</div>
    <div class="message-content">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>`;
  container.appendChild(div);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function addStreamingMessage() {
  const container = document.getElementById('messagesContainer');
  const id = `stream-${Date.now()}-${++state.msgCounter}`;
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.id = id;
  div.innerHTML = `
    <div class="message-avatar">&#129302;</div>
    <div class="message-content">
      <div class="message-bubble" id="${id}-bubble"><span class="spinner"></span></div>
      <span class="message-meta">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
    </div>`;
  container.appendChild(div);
  scrollToBottom();
  return id;
}

function updateStreamingMessage(id, content) {
  const bubble = document.getElementById(`${id}-bubble`);
  if (bubble) {
    bubble.innerHTML = formatMessageContent(content);
    scrollToBottom();
  }
}

function addSourcesToMessage(id, sources) {
  const msg = document.getElementById(id);
  if (!msg || !sources.length) return;
  const content = msg.querySelector('.message-content');
  if (!content) return;
  const tags = sources.map(s => `<span class="source-tag">&#128196; ${escapeHtml(s)}</span>`).join('');
  const sourcesDiv = document.createElement('div');
  sourcesDiv.className = 'message-sources';
  sourcesDiv.innerHTML = tags;
  content.insertBefore(sourcesDiv, content.querySelector('.message-meta'));
}

// ============================================================
// RIGHT PANEL: RAG INTERNALS
// Learning Note:
//   This panel is the KEY learning feature of this UI.
//   It shows you EXACTLY what happened inside the RAG pipeline:
//   - Which chunks were retrieved and why
//   - Which were filtered as "Noisy Data"
//   - What the RAGAS evaluation scores are
//   - Which nodes executed in the LangGraph pipeline
// ============================================================
function updateRagInternals(data) {
  document.getElementById('internalsEmpty').style.display = 'none';

  // Show eval scores
  if (data.eval_metrics && data.eval_metrics.evaluated) {
    showEvalScores(data.eval_metrics);
  }

  // Show pipeline trace
  const showTrace = document.getElementById('showTrace').checked;
  if (showTrace && data.pipeline_steps && data.pipeline_steps.length) {
    showPipelineTrace(data.pipeline_steps, data.latency_ms);
  }

  // Show True Data chunks
  if (data.true_data_chunks && data.true_data_chunks.length) {
    showChunks('true', data.true_data_chunks);
  }

  // Show Noisy Data chunks
  const showNoisy = document.getElementById('showNoisy').checked;
  if (showNoisy && data.noisy_data_chunks && data.noisy_data_chunks.length) {
    showChunks('noisy', data.noisy_data_chunks);
  }
}

function showEvalScores(metrics) {
  const section = document.getElementById('evalSection');
  const container = document.getElementById('evalScores');
  section.style.display = 'block';

  const scores = [
    { label: 'Faithfulness', key: 'faithfulness', desc: 'Answer grounded in context' },
    { label: 'Answer Relevancy', key: 'answer_relevancy', desc: 'Answer addresses the question' },
    { label: 'Context Precision', key: 'context_precision', desc: 'Retrieved chunks are relevant' },
  ];

  container.innerHTML = scores.map(s => {
    const val = metrics[s.key] || 0;
    const pct = Math.round(val * 100);
    const cls = val >= 0.7 ? 'score-high' : val >= 0.4 ? 'score-med' : 'score-low';
    const color = val >= 0.7 ? '#22c55e' : val >= 0.4 ? '#f59e0b' : '#ef4444';
    return `
      <div class="eval-score-row">
        <div class="eval-score-header">
          <span class="eval-score-label">${s.label}</span>
          <span class="eval-score-value" style="color:${color}">${pct}%</span>
        </div>
        <div class="eval-score-bar">
          <div class="eval-score-fill ${cls}" style="width:${pct}%"></div>
        </div>
      </div>`;
  }).join('');
}

// Node icons for pipeline trace display
const NODE_ICONS = {
  input_guardrails: '&#128737;',
  query_planner: '&#129504;',
  history_rewriter: '&#128196;',
  retrieval: '&#128270;',
  context_assembly: '&#128218;',
  generation: '&#9889;',
  output_guardrails: '&#9989;',
  evaluation: '&#128200;',
  error_response: '&#10060;',
};

function showPipelineTrace(steps, totalMs) {
  const section = document.getElementById('traceSection');
  const container = document.getElementById('pipelineTrace');
  section.style.display = 'block';

  const avgMs = totalMs ? Math.round(totalMs / steps.length) : null;
  container.innerHTML = steps.map((step, i) => {
    const icon = NODE_ICONS[step] || '&#9679;';
    const label = step.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    return `
      <div class="trace-step">
        <span class="trace-step-icon">${icon}</span>
        <span class="trace-step-name">${label}</span>
        ${avgMs ? `<span class="trace-step-time">~${avgMs}ms</span>` : ''}
      </div>`;
  }).join('');
}

function showChunks(type, chunks) {
  const sectionId = type === 'true' ? 'trueDataSection' : 'noisyDataSection';
  const listId = type === 'true' ? 'trueDataChunks' : 'noisyDataChunks';
  const section = document.getElementById(sectionId);
  const list = document.getElementById(listId);
  section.style.display = 'block';

  list.innerHTML = chunks.map((chunk, i) => {
    const score = chunk.score !== undefined ? (chunk.score * 100).toFixed(0) + '%' : '';
    const source = chunk.source || 'unknown';
    const content = chunk.content || '';
    const isNoisy = type === 'noisy';
    return `
      <div class="chunk-card ${isNoisy ? 'noisy' : ''}" onclick="toggleChunk(this)">
        <div class="chunk-header">
          <span class="chunk-source">&#128196; ${escapeHtml(source)}</span>
          <span class="chunk-score">score: ${score}</span>
        </div>
        <div class="chunk-preview">${escapeHtml(content)}</div>
      </div>`;
  }).join('');
}

function toggleChunk(card) {
  card.classList.toggle('expanded');
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

// Inline markdown: **bold** and `code` (input is already HTML-escaped).
function formatInline(html) {
  return html
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

// Small markdown renderer for chat answers: headings, bullet / numbered lists,
// tables and paragraphs. Everything is escaped first, so model output can't inject HTML.
function formatMessageContent(text) {
  if (!text) return '';
  const lines = escapeHtml(text).split('\n');
  const out = [];
  let paragraph = [];
  const flush = () => {
    if (paragraph.length) out.push(`<p>${formatInline(paragraph.join('<br/>'))}</p>`);
    paragraph = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const heading = line.match(/^\s*(#{1,4})\s+(.*)$/);
    if (heading) {
      flush();
      out.push(`<h4 class="md-heading">${formatInline(heading[2])}</h4>`);
    } else if (/^\s*\|.*\|\s*$/.test(line)) {
      flush();
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++]);
      i--;
      const cells = row => row.trim().replace(/^\||\|$/g, '').split('|').map(c => formatInline(c.trim()));
      const isDivider = row => /^\s*\|[\s:|-]+\|\s*$/.test(row);
      const body = rows.filter(r => !isDivider(r));
      const header = rows.length > 1 && isDivider(rows[1]) ? cells(body.shift()) : null;
      out.push('<div class="md-table-wrap"><table class="md-table">' +
        (header ? `<thead><tr>${header.map(c => `<th>${c}</th>`).join('')}</tr></thead>` : '') +
        `<tbody>${body.map(r => `<tr>${cells(r).map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody>` +
        '</table></div>');
    } else if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      flush();
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
        items.push(`<li>${formatInline(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ''))}</li>`);
        i++;
      }
      i--;
      out.push(ordered ? `<ol>${items.join('')}</ol>` : `<ul>${items.join('')}</ul>`);
    } else if (!line.trim()) {
      flush();
    } else {
      paragraph.push(line);
    }
  }
  flush();
  return out.join('');
}

function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function scrollToBottom() {
  const container = document.getElementById('messagesContainer');
  container.scrollTop = container.scrollHeight;
}

function setLoading(loading) {
  state.isLoading = loading;
  const btn = document.getElementById('sendBtn');
  const icon = document.getElementById('sendIcon');
  btn.disabled = loading;
  icon.innerHTML = loading ? '<span class="spinner" style="width:14px;height:14px;border-width:2px"></span>' : '&#9658;';
}

function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function newChat() {
  if (state.isLoading) {
    showToast('Wait for the current answer to finish', 'error');
    return;
  }
  document.getElementById('messagesContainer').innerHTML = '';
  document.getElementById('welcomeMessage').style.display = 'flex';
  state.conversationHistory = [];
  state.chatMessages = [];
  state.sessionId = generateSessionId();
  document.getElementById('sessionInfo').textContent = 'Session: ' + state.sessionId.slice(-8);
  // Reset internals panel
  ['evalSection', 'traceSection', 'trueDataSection', 'noisyDataSection'].forEach(id => {
    document.getElementById(id).style.display = 'none';
  });
  document.getElementById('internalsEmpty').style.display = 'flex';
  renderChatList();
  document.getElementById('queryInput').focus();
}

function applyTheme(theme) {
  const dark = theme === 'dark';
  if (dark) document.documentElement.dataset.theme = 'dark';
  else delete document.documentElement.dataset.theme;
  const button = document.getElementById('themeToggle');
  button.innerHTML = dark ? '&#9728;' : '&#9790;';   // sun / moon
  button.title = dark ? 'Switch to light mode' : 'Switch to dark mode';
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  try { localStorage.setItem('theme', next); } catch { /* storage unavailable: theme lasts this visit */ }
}

function toggleSettings() {
  const panel = document.getElementById('settingsPanel');
  panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
}

function generateSessionId() {
  return 'sess-' + Math.random().toString(36).slice(2, 11);
}

function showToast(message, type = '') {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = `toast ${type} show`;
  setTimeout(() => { toast.className = 'toast'; }, 3500);
}

// ============================================================
// MAIN TAB NAVIGATION
// ============================================================

function switchMainTab(tab) {
  // Update tab buttons
  document.querySelectorAll('.main-tab').forEach(btn => btn.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');

  // Show correct panel
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`panel-${tab}`).classList.add('active');

  // Refresh source lists when switching to sources tab
  if (tab === 'sources') {
    loadAllSources();
  }
  if (tab === 'investigate') {
    renderInvestigateSourceList();
  }
}

function switchSourceTab(tab) {
  document.querySelectorAll('.source-tab').forEach(btn => btn.classList.remove('active'));
  document.getElementById(`stab-${tab}`).classList.add('active');

  document.querySelectorAll('.source-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`spanel-${tab}`).classList.add('active');
}

// ============================================================
// SOURCE MANAGEMENT - Load all sources
// ============================================================

async function loadAllSources() {
  try {
    const [docsRes, legacyRes, bbRes, jiraRes, confRes, progressRes] = await Promise.all([
      fetch(`${API_BASE}/sources/documents`),
      fetch(`${API_BASE}/sources/documents/legacy`),
      fetch(`${API_BASE}/sources/bitbucket`),
      fetch(`${API_BASE}/sources/jira`),
      fetch(`${API_BASE}/sources/confluence`),
      fetch(`${API_BASE}/sources/progress`),
    ]);
    state.allSources.confluence = confRes.ok ? await confRes.json() : [];
    state.progress = progressRes.ok ? await progressRes.json() : {};

    state.allSources.documents = docsRes.ok ? await docsRes.json() : [];
    state.allSources.legacy = legacyRes.ok ? await legacyRes.json() : [];
    state.allSources.bitbucket = bbRes.ok ? await bbRes.json() : [];
    state.allSources.jira = jiraRes.ok ? await jiraRes.json() : [];

    renderDocumentsList();
    renderLegacyDocuments();
    renderBitbucketList();
    renderJiraList();
    renderConfluenceList();
    renderChatSourceSelection();
    trackIndexingProgress();

  } catch (err) {
    // Server may not be up yet - silent fail
  }
}

// ============================================================
// DOCUMENTS - render list
// ============================================================

function renderDocumentsList() {
  const container = document.getElementById('documentsList');
  const docs = state.allSources.documents;

  if (!docs.length) {
    container.innerHTML = `
      <div class="sources-empty">
        <span class="sources-empty-icon">&#128196;</span>
        <p>No documents uploaded yet.<br/>Upload PDF, DOCX, TXT, or MD files to get started.</p>
      </div>`;
    return;
  }

  container.innerHTML = docs.map(doc => {
    const statusCls = `status-badge-${doc.status}`;
    const sizeKb = doc.size_bytes ? Math.round(doc.size_bytes / 1024) + ' KB' : '';
    return `
      <div class="source-card" id="doc-card-${doc.id}">
        <div class="source-card-icon">&#128196;</div>
        <div class="source-card-info">
          <div class="source-card-name">${escapeHtml(doc.filename)}</div>
          <div class="source-card-meta">
            <span class="source-badge ${statusCls}">${doc.status}</span>
            ${sizeKb ? `<span>${sizeKb}</span>` : ''}
            <span>${doc.chunk_count} chunks</span>
          </div>
          ${doc.error_message ? `<p class="status-err">${escapeHtml(doc.error_message)}</p>` : ''}
        </div>
        <div class="source-card-actions">
          <button class="btn-action" data-reindex="${doc.id}"
            ${doc.status === 'indexing' ? 'disabled' : ''}>Reindex</button>
          <button class="btn-action btn-danger-sm" data-delete-document="${doc.id}" title="Delete">
            &#128465;
          </button>
        </div>
      </div>`;
  }).join('');
  container.querySelectorAll('[data-reindex]').forEach(button => {
    button.addEventListener('click', () => reindexDocument(button.dataset.reindex, button));
  });
  container.querySelectorAll('[data-delete-document]').forEach(button => {
    button.addEventListener('click', () => {
      const doc = docs.find(item => item.id === button.dataset.deleteDocument);
      confirmDeleteSource('document', doc.id, doc.filename);
    });
  });
}

// ============================================================
// DOCUMENTS - older uploads (no registry entry; remove only)
// ============================================================

function renderLegacyDocuments() {
  const section = document.getElementById('legacyDocumentsSection');
  const container = document.getElementById('legacyDocumentsList');
  const files = state.allSources.legacy;
  section.style.display = files.length ? 'block' : 'none';

  container.innerHTML = files.map((file, i) => `
    <div class="source-card">
      <div class="source-card-icon">&#128196;</div>
      <div class="source-card-info">
        <div class="source-card-name">${escapeHtml(file.source_file)}</div>
        <div class="source-card-meta">
          <span class="source-badge status-badge-pending">older upload</span>
          <span>${file.chunk_count} chunks</span>
        </div>
        ${file.preview ? `<div class="source-card-commit legacy-preview">"${escapeHtml(file.preview)}..."</div>` : ''}
      </div>
      <div class="source-card-actions">
        <button class="btn-action btn-danger-sm" data-remove-legacy="${i}" title="Remove from knowledge base">
          &#128465; Remove
        </button>
      </div>
    </div>`).join('');

  container.querySelectorAll('[data-remove-legacy]').forEach(button => {
    button.addEventListener('click', () => {
      const file = files[Number(button.dataset.removeLegacy)];
      confirmDeleteSource('legacy', file.source_file, file.source_file);
    });
  });
}

// ============================================================
// INDEXING PROGRESS - poll while any source is indexing
// ============================================================

function trackIndexingProgress() {
  const all = [
    ...state.allSources.documents.map(s => ({ ...s, label: s.filename })),
    ...state.allSources.bitbucket.map(s => ({ ...s, label: `${s.workspace}/${s.repository}` })),
    ...state.allSources.jira.map(s => ({ ...s, label: s.project_key })),
    ...state.allSources.confluence.map(s => ({ ...s, label: `Confluence ${s.space_key}` })),
  ];
  for (const src of all) {
    const previous = state.sourceStatuses[src.id];
    if (previous === 'indexing' && src.status === 'ready') {
      showToast(`Indexed ${src.label}: ${src.chunk_count} chunks`, 'success');
      loadKBStats();
    } else if (previous === 'indexing' && src.status === 'error') {
      showToast(`Indexing ${src.label} failed: ${src.error_message || 'see Knowledge Sources'}`, 'error');
    }
    state.sourceStatuses[src.id] = src.status;
  }

  clearTimeout(state.sourcePollTimer);
  if (all.some(src => src.status === 'indexing' || src.status === 'syncing')) {
    state.sourcePollTimer = setTimeout(loadAllSources, 2000);
  }
}

async function startSourceJob(type, id, action) {
  try {
    const res = await fetch(`${API_BASE}/sources/${type}/${encodeURIComponent(id)}/${action}`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `${action} failed`);
    showToast(action === 'sync' ? 'Sync started' : 'Indexing started - this can take a few minutes', 'success');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    await loadAllSources();
  }
}

function formatElapsed(seconds) {
  const m = Math.floor(seconds / 60), s = seconds % 60;
  return m ? `${m}m ${s}s` : `${s}s`;
}

// Live stage, counts, bar and elapsed time for a source that is being indexed.
function indexProgressHtml(src) {
  if (src.status !== 'indexing' && src.status !== 'syncing') return '';
  const p = state.progress[src.id];
  if (!p) return '<div class="index-progress"><div class="index-progress-text">Starting...</div></div>';
  const fmt = n => Number(n).toLocaleString();
  let count = '';
  if (p.done !== null && p.done !== undefined) {
    count = ` &middot; ${fmt(p.done)}${p.total ? ' / ' + fmt(p.total) : ''} ${escapeHtml(p.unit || '')}`;
  }
  const pct = p.percent !== null && p.percent !== undefined ? ` (${p.percent}%)` : '';
  const bar = p.percent !== null && p.percent !== undefined
    ? `<div class="index-progress-bar"><div style="width:${p.percent}%"></div></div>`
    : '<div class="index-progress-bar indeterminate"><div></div></div>';
  return `<div class="index-progress">
      ${bar}
      <div class="index-progress-text">${escapeHtml(p.stage)}${count}${pct} &middot; ${formatElapsed(p.elapsed_seconds)}</div>
    </div>`;
}

function sourceJobButtons(type, src) {
  const busy = src.status === 'indexing' || src.status === 'syncing';
  const indexed = src.status === 'ready' || src.chunk_count > 0;
  return `
    <button class="btn-action" data-job="${type}" data-id="${src.id}" data-action="index" ${busy ? 'disabled' : ''}>
      ${busy ? '<span class="spinner"></span> Indexing...' : (indexed ? '&#128260; Reindex' : '&#128269; Index')}
    </button>
    ${indexed && !busy ? `<button class="btn-action" data-job="${type}" data-id="${src.id}" data-action="sync"
      title="${{ bitbucket: 'Re-index only if the branch has new commits', jira: 'Re-fetch all issues',
        confluence: 'Re-fetch all pages; unchanged pages are reused' }[type]}">&#8635; Sync</button>` : ''}`;
}

function bindSourceCardActions(container, type, items, labelOf) {
  container.querySelectorAll('[data-job]').forEach(button => {
    button.addEventListener('click', () => {
      button.disabled = true;
      startSourceJob(button.dataset.job, button.dataset.id, button.dataset.action);
    });
  });
  container.querySelectorAll('[data-delete-source]').forEach(button => {
    button.addEventListener('click', () => {
      const item = items.find(src => src.id === button.dataset.deleteSource);
      confirmDeleteSource(type, item.id, labelOf(item));
    });
  });
}

// ============================================================
// DOCUMENTS - upload from sources tab
// ============================================================

async function reindexDocument(id, button) {
  button.disabled = true;
  button.textContent = 'Reindexing...';
  try {
    const res = await fetch(`${API_BASE}/sources/documents/${encodeURIComponent(id)}/reindex`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Reindex failed');
    showToast(`Reindexed: ${data.chunks_added} chunks`, 'success');
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    await loadAllSources();
    loadKBStats();
  }
}

function handleDropSources(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('drag-over');
  const files = Array.from(e.dataTransfer.files);
  uploadFilesFromSourcesTab(files);
}

function handleFileSelectSources(e) {
  const files = Array.from(e.target.files);
  uploadFilesFromSourcesTab(files);
  e.target.value = '';
}

async function uploadFilesFromSourcesTab(files) {
  const allowed = ['.pdf', '.txt', '.docx', '.md'];
  const progressArea = document.getElementById('uploadProgressSources');
  progressArea.style.display = 'block';
  progressArea.innerHTML = '';

  for (const file of files) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
      showToast(`Skipped ${file.name}: unsupported type`, 'error');
      continue;
    }
    await uploadSingleFileFromSourcesTab(file, progressArea);
  }

  loadKBStats();
  await loadAllSources();
}

async function uploadSingleFileFromSourcesTab(file, progressArea) {
  const rowId = 'up-' + Date.now();
  const row = document.createElement('div');
  row.className = 'upload-progress-row';
  row.id = rowId;
  row.innerHTML = `
    <span class="upload-progress-name">${escapeHtml(file.name)}</span>
    <span class="upload-progress-status" id="${rowId}-status">
      <span class="spinner"></span> Uploading...
    </span>`;
  progressArea.appendChild(row);

  try {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/sources/documents`, {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();
    const statusEl = document.getElementById(`${rowId}-status`);

    if (res.ok && data.status === 'ready') {
      statusEl.innerHTML = `<span class="status-ok">&#10003; ${data.chunk_count} chunks</span>`;
      showToast(`Ingested "${file.name}"`, 'success');
    } else {
      statusEl.innerHTML = `<span class="status-err">&#10007; ${escapeHtml(data.detail || 'Failed')}</span>`;
      showToast(`Failed: ${file.name}`, 'error');
    }
  } catch (err) {
    const statusEl = document.getElementById(`${rowId}-status`);
    if (statusEl) statusEl.innerHTML = `<span class="status-err">&#10007; Error</span>`;
  }
}

// ============================================================
// BITBUCKET - render list
// ============================================================

function renderBitbucketList() {
  const container = document.getElementById('bitbucketList');
  const repos = state.allSources.bitbucket;

  if (!repos.length) {
    container.innerHTML = `
      <div class="sources-empty">
        <span class="sources-empty-icon">&#128230;</span>
        <p>No repositories added yet.<br/>Click "Add Repository" to connect a Bitbucket repository.</p>
      </div>`;
    return;
  }

  container.innerHTML = repos.map(repo => {
    const statusCls = `status-badge-${repo.status}`;
    return `
      <div class="source-card" id="bb-card-${repo.id}">
        <div class="source-card-icon">&#128230;</div>
        <div class="source-card-info">
          <div class="source-card-name">${escapeHtml(repo.workspace)}/${escapeHtml(repo.repository)}</div>
          <div class="source-card-meta">
            <span class="source-badge ${statusCls}">${repo.status}</span>
            <span>&#127807; ${escapeHtml(repo.branch)}</span>
            <span>${repo.server_url ? escapeHtml(new URL(repo.server_url).host) : 'bitbucket.org'}</span>
            ${repo.file_count ? `<span>${repo.file_count} files</span>` : ''}
            ${repo.chunk_count ? `<span>${repo.chunk_count} chunks</span>` : ''}
            <span class="cred-badge">&#128274; Token configured</span>
          </div>
          ${repo.last_commit ? `<div class="source-card-commit">Indexed commit: ${escapeHtml(repo.last_commit.slice(0, 12))}${repo.last_sync ? ' &middot; ' + new Date(repo.last_sync).toLocaleString() : ''}</div>` : ''}
          ${repo.status === 'pending' ? '<div class="source-card-commit">Not indexed yet. Click Index so chat can answer from this repository.</div>' : ''}
          ${indexProgressHtml(repo)}
          ${repo.error_message ? `<p class="status-err">${escapeHtml(repo.error_message)}</p>` : ''}
        </div>
        <div class="source-card-actions">
          ${sourceJobButtons('bitbucket', repo)}
          <button class="btn-action btn-danger-sm" title="Delete" data-delete-source="${repo.id}">&#128465;</button>
        </div>
      </div>`;
  }).join('');
  bindSourceCardActions(container, 'bitbucket', repos, r => `${r.workspace}/${r.repository}`);
}

// ============================================================
// JIRA - render list
// ============================================================

function renderJiraList() {
  const container = document.getElementById('jiraList');
  const projects = state.allSources.jira;

  if (!projects.length) {
    container.innerHTML = `
      <div class="sources-empty">
        <span class="sources-empty-icon">&#128203;</span>
        <p>No Jira projects added yet.<br/>Click "Add Jira Project" to connect a Jira project.</p>
      </div>`;
    return;
  }

  container.innerHTML = projects.map(proj => {
    const statusCls = `status-badge-${proj.status}`;
    const syncDate = proj.last_sync ? new Date(proj.last_sync).toLocaleDateString() : 'Never';
    return `
      <div class="source-card" id="jira-card-${proj.id}">
        <div class="source-card-icon">&#128203;</div>
        <div class="source-card-info">
          <div class="source-card-name">${escapeHtml(proj.project_key)}</div>
          <div class="source-card-meta">
            <span class="source-badge ${statusCls}">${proj.status}</span>
            <span>${escapeHtml(proj.base_url)}</span>
            ${proj.issue_count ? `<span>${proj.issue_count} issues</span>` : ''}
            ${proj.chunk_count ? `<span>${proj.chunk_count} chunks</span>` : ''}
            <span class="cred-badge">&#128274; Token configured</span>
          </div>
          <div class="source-card-commit">Last sync: ${syncDate}</div>
          ${proj.status === 'pending' ? '<div class="source-card-commit">Not indexed yet. Click Index so chat can answer from these issues.</div>' : ''}
          ${indexProgressHtml(proj)}
          ${proj.error_message ? `<p class="status-err">${escapeHtml(proj.error_message)}</p>` : ''}
        </div>
        <div class="source-card-actions">
          ${sourceJobButtons('jira', proj)}
          <button class="btn-action btn-danger-sm" title="Delete" data-delete-source="${proj.id}">&#128465;</button>
        </div>
      </div>`;
  }).join('');
  bindSourceCardActions(container, 'jira', projects, p => p.project_key);
}

// ============================================================
// CHAT SOURCE SELECTION - render checkboxes in chat sidebar
// ============================================================

function chatSelectableSources() {
  const statusNote = s => s.status === 'ready' ? '' : ` (${s.status === 'pending' ? 'not indexed' : s.status})`;
  return [
    ...state.allSources.documents.map(d => ({ id: d.id, label: d.filename, type: 'document', icon: '&#128196;' })),
    ...state.allSources.legacy.map(l => ({ id: l.source_file, label: `${l.source_file} (older upload)`, type: 'legacy', icon: '&#128196;' })),
    ...state.allSources.bitbucket.map(b => ({ id: b.id, label: `${b.workspace}/${b.repository}${statusNote(b)}`, type: 'bitbucket', icon: '&#128230;' })),
    ...state.allSources.jira.map(j => ({ id: j.id, label: `${j.project_key}${statusNote(j)}`, type: 'jira', icon: '&#128203;' })),
    ...state.allSources.confluence.map(c => ({ id: c.id, label: `Confluence ${c.space_key}${statusNote(c)}`, type: 'confluence', icon: '&#128216;' })),
  ];
}

function renderChatSourceSelection() {
  const container = document.getElementById('sourceSelectionList');
  const allSrc = chatSelectableSources();

  if (!allSrc.length) {
    container.innerHTML = '<p class="source-hint">Add sources in the Knowledge Sources tab</p>';
    return;
  }

  // Keep the user's unchecked choices across re-renders (sources are polled while indexing).
  const unchecked = new Set(
    Array.from(document.querySelectorAll('.source-select-cb:not(:checked)')).map(cb => cb.value));
  container.innerHTML = allSrc.map(s => `
    <label class="source-toggle source-toggle-sm">
      <input type="checkbox" class="source-select-cb" value="${escapeHtml(s.id)}" data-type="${s.type}"
        ${unchecked.has(s.id) ? '' : 'checked'} />
      <span>${s.icon} ${escapeHtml(s.label)}</span>
    </label>`).join('');

  // Mirror to investigate tab
  renderInvestigateSourceList();
}

function renderInvestigateSourceList() {
  const container = document.getElementById('investigateSourceList');
  const allSrc = [
    ...state.allSources.documents.map(d => ({ id: d.id, label: d.filename, type: 'document', icon: '&#128196;' })),
    ...state.allSources.bitbucket.map(b => ({ id: b.id, label: `${b.workspace}/${b.repository}`, type: 'bitbucket', icon: '&#128230;' })),
    ...state.allSources.jira.map(j => ({ id: j.id, label: j.project_key, type: 'jira', icon: '&#128203;' })),
  ];

  if (!allSrc.length) {
    container.innerHTML = '<p class="source-hint">Add sources in the Knowledge Sources tab</p>';
    return;
  }

  container.innerHTML = allSrc.map(s => `
    <label class="source-toggle source-toggle-sm">
      <input type="checkbox" class="inv-source-cb" value="${s.id}" data-type="${s.type}" checked />
      <span>${s.icon} ${escapeHtml(s.label)}</span>
    </label>`).join('');
}

const SOURCE_TYPE_TOGGLES = { document: 'searchDocs', legacy: 'searchDocs', bitbucket: 'searchBitbucket',
  jira: 'searchJira', confluence: 'searchConfluence' };

// Returns null to search everything (all boxes ticked), else { source_ids, legacy_files }.
// A source is searched only if both its own box and its type box ("Documents", ...) are ticked.
function getSourceFilter() {
  const boxes = Array.from(document.querySelectorAll('.source-select-cb'));
  const typeOn = type => document.getElementById(SOURCE_TYPE_TOGGLES[type])?.checked !== false;
  const selected = boxes.filter(cb => cb.checked && typeOn(cb.dataset.type));
  const allTypesOn = Object.values(SOURCE_TYPE_TOGGLES).every(id => document.getElementById(id)?.checked !== false);
  if (allTypesOn && selected.length === boxes.length) return null;
  return {
    source_ids: selected.filter(cb => cb.dataset.type !== 'legacy').map(cb => cb.value),
    legacy_files: selected.filter(cb => cb.dataset.type === 'legacy').map(cb => cb.value),
  };
}

// ============================================================
// MODALS - Bitbucket
// ============================================================

function showAddBitbucketModal() {
  // Reset form (keep the last chosen type and server URL for convenience)
  ['bbWorkspace', 'bbRepository', 'bbUsername', 'bbToken'].forEach(id => {
    document.getElementById(id).value = '';
  });
  document.getElementById('bbBranch').value = 'main';
  document.getElementById('bbTestResult').style.display = 'none';
  hidePicker('bb');
  document.getElementById('bbAddBtn').disabled = true;
  state.bbConnectionTested = false;
  updateBitbucketForm();
  document.getElementById('modalBitbucket').style.display = 'flex';
}

function isBitbucketServer() {
  return document.getElementById('bbType').value === 'server';
}

function updateBitbucketForm() {
  const server = isBitbucketServer();
  document.getElementById('bbServerUrlGroup').style.display = server ? 'block' : 'none';
  document.getElementById('bbWorkspaceLabel').textContent = server ? 'Project key' : 'Workspace';
  document.getElementById('bbWorkspace').placeholder = server ? 'PROJECTKEY' : 'my-workspace';
  document.getElementById('bbUsernameLabel').textContent = server ? 'Username (optional)' : 'Account email';
  document.getElementById('bbTokenLabel').textContent = server ? 'HTTP access token' : 'API token / App password';
  document.getElementById('bbTokenHint').textContent = server
    ? 'Create it in Bitbucket: your avatar > Manage account > HTTP access tokens, with Repository read permission.'
    : 'Create it at bitbucket.org: Personal settings > API tokens, with repository read scope.';
  // Changing the type invalidates an earlier connection test.
  document.getElementById('bbAddBtn').disabled = true;
  state.bbConnectionTested = false;
}

// Fill project key and repository from a pasted repository link, e.g.
// https://bitbucket.company.com/projects/KEY/repos/my_repo/browse
function fillFromServerUrl() {
  const url = document.getElementById('bbServerUrl').value.trim();
  const match = url.match(/^https:\/\/.+?\/(?:projects|users)\/([^/]+)\/repos\/([^/?#]+)/i);
  if (match) {
    document.getElementById('bbWorkspace').value = decodeURIComponent(match[1]);
    document.getElementById('bbRepository').value = decodeURIComponent(match[2]);
  }
}

function bitbucketFormValues() {
  const server = isBitbucketServer();
  return {
    server_url: server ? document.getElementById('bbServerUrl').value.trim() : null,
    workspace: document.getElementById('bbWorkspace').value.trim(),
    repository: document.getElementById('bbRepository').value.trim(),
    branch: document.getElementById('bbBranch').value.trim() || 'main',
    username: document.getElementById('bbUsername').value.trim(),
    token: document.getElementById('bbToken').value,
  };
}

function bitbucketFormError(v) {
  if (v.server_url !== null && !/^https:\/\//i.test(v.server_url)) return 'Enter the server URL starting with https://';
  if (!v.workspace) return isBitbucketServer() ? 'Enter the project key' : 'Enter the workspace';
  if (!v.repository) return 'Enter the repository';
  if (v.server_url === null && !v.username) return 'Enter your account email';
  if (!v.token) return 'Enter the token';
  return null;
}

async function testBitbucketConnection() {
  const v = bitbucketFormValues();
  const resultEl = document.getElementById('bbTestResult');
  const error = bitbucketFormError(v);
  if (error) {
    showTestResult(resultEl, false, error);
    return;
  }

  resultEl.style.display = 'block';
  resultEl.innerHTML = '<span class="spinner"></span> Testing connection...';

  try {
    const res = await fetch(`${API_BASE}/sources/bitbucket/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        server_url: v.server_url, workspace: v.workspace, repository: v.repository,
        username: v.username, token: v.token,
      }),
    });
    const data = await res.json();

    if (res.ok && data.success) {
      const repoInfo = data.repos_found !== null ? ` (${data.repos_found} repos visible)` : '';
      showTestResult(resultEl, true, `${data.message}${repoInfo}`);
      document.getElementById('bbAddBtn').disabled = false;
      state.bbConnectionTested = true;
    } else {
      showTestResult(resultEl, false, data.message || data.detail || 'Connection failed');
      document.getElementById('bbAddBtn').disabled = true;
    }
  } catch (err) {
    showTestResult(resultEl, false, `Error: ${err.message}`);
  }
}

async function addBitbucketSource() {
  const v = bitbucketFormValues();
  const picker = state.pickers.bb;
  if (picker && picker.selected.size) return addSelectedBitbucketRepos(v, picker);
  const error = bitbucketFormError(v);
  if (error) {
    showToast(error, 'error');
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/sources/bitbucket`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(v),
    });

    if (res.ok) {
      const source = await res.json();
      closeModal('modalBitbucket');
      // Clear the token field immediately after success
      document.getElementById('bbToken').value = '';
      // Start indexing straight away: an added but unindexed repository has nothing to search.
      showToast(`Repository ${v.workspace}/${v.repository} added - indexing started`, 'success');
      await startSourceJob('bitbucket', source.id, 'index');
    } else {
      const err = await res.json();
      showToast(`Error: ${err.detail || 'Failed to add repository'}`, 'error');
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

async function addSelectedBitbucketRepos(v, picker) {
  const chosen = picker.items.filter(i => picker.selected.has(i.key));
  try {
    const res = await fetch(`${API_BASE}/sources/bitbucket/bulk`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_url: v.server_url, username: v.username, token: v.token, branch: v.branch,
                             repositories: chosen.map(i => ({ workspace: i.workspace, slug: i.slug })) }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to add repositories');
    closeModal('modalBitbucket');
    document.getElementById('bbToken').value = '';
    showToast(`${data.added} repositories added - indexing queued${data.skipped.length ? ` (${data.skipped.length} already added)` : ''}`, 'success');
    await loadAllSources();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

// ============================================================
// MODALS - Jira
// ============================================================

function showAddJiraModal() {
  ['jiraBaseUrl', 'jiraProjectKey', 'jiraEmail', 'jiraToken'].forEach(id => {
    document.getElementById(id).value = '';
  });
  document.getElementById('jiraTestResult').style.display = 'none';
  document.getElementById('jiraAddBtn').disabled = true;
  state.jiraConnectionTested = false;
  document.getElementById('modalJira').style.display = 'flex';
}

async function testJiraConnection() {
  const baseUrl = document.getElementById('jiraBaseUrl').value.trim();
  const email = document.getElementById('jiraEmail').value.trim();
  const token = document.getElementById('jiraToken').value;
  const resultEl = document.getElementById('jiraTestResult');

  const cloud = /atlassian\.net/i.test(baseUrl);
  if (!baseUrl || !token || (cloud && !email)) {
    showTestResult(resultEl, false, cloud ? 'Please fill in URL, email, and API token' : 'Please fill in URL and token');
    return;
  }
  const projectKey = document.getElementById('jiraProjectKey').value.trim().toUpperCase();

  resultEl.style.display = 'block';
  resultEl.innerHTML = '<span class="spinner"></span> Testing connection...';

  try {
    const res = await fetch(`${API_BASE}/sources/jira/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_url: baseUrl, email, token, project_key: projectKey || null }),
    });
    const data = await res.json();

    if (data.success) {
      const info = data.display_name ? ` (${data.display_name})` : '';
      const projInfo = data.projects_found !== null ? `, ${data.projects_found} projects` : '';
      showTestResult(resultEl, true, `Connected${info}${projInfo}`);
      document.getElementById('jiraAddBtn').disabled = false;
      state.jiraConnectionTested = true;
    } else {
      showTestResult(resultEl, false, data.message);
      document.getElementById('jiraAddBtn').disabled = true;
    }
  } catch (err) {
    showTestResult(resultEl, false, `Error: ${err.message}`);
  }
}

async function addJiraSource() {
  const baseUrl = document.getElementById('jiraBaseUrl').value.trim();
  const projectKey = document.getElementById('jiraProjectKey').value.trim().toUpperCase();
  const email = document.getElementById('jiraEmail').value.trim();
  const token = document.getElementById('jiraToken').value;

  if (!baseUrl || !projectKey || !token || (/atlassian\.net/i.test(baseUrl) && !email)) {
    showToast('Please fill in all required fields', 'error');
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/sources/jira`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_url: baseUrl, project_key: projectKey, email, token }),
    });

    if (res.ok) {
      const source = await res.json();
      closeModal('modalJira');
      document.getElementById('jiraToken').value = '';
      showToast(`Jira project ${projectKey} added - indexing started`, 'success');
      await startSourceJob('jira', source.id, 'index');
    } else {
      const err = await res.json();
      showToast(`Error: ${err.detail || 'Failed to add project'}`, 'error');
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

// ============================================================
// CONFLUENCE - render list
// ============================================================

function renderConfluenceList() {
  const container = document.getElementById('confluenceList');
  const spaces = state.allSources.confluence;
  if (!spaces.length) {
    container.innerHTML = `
      <div class="sources-empty">
        <span class="sources-empty-icon">&#128216;</span>
        <p>No Confluence spaces added yet.<br/>Click "Add Confluence Space" to index every page of a space.</p>
      </div>`;
    return;
  }
  container.innerHTML = spaces.map(space => `
      <div class="source-card" id="conf-card-${space.id}">
        <div class="source-card-icon">&#128216;</div>
        <div class="source-card-info">
          <div class="source-card-name">${escapeHtml(space.name)}</div>
          <div class="source-card-meta">
            <span class="source-badge status-badge-${space.status}">${space.status}</span>
            <span>${escapeHtml(space.space_key)}</span>
            <span>${escapeHtml(new URL(space.base_url).host)}</span>
            ${space.page_count ? `<span>${space.page_count} pages</span>` : ''}
            ${space.chunk_count ? `<span>${space.chunk_count} chunks</span>` : ''}
            <span class="cred-badge">&#128274; Token configured</span>
          </div>
          ${space.last_sync ? `<div class="source-card-commit">Last sync: ${new Date(space.last_sync).toLocaleString()}</div>` : ''}
          ${space.error_message ? `<p class="status-err">${escapeHtml(space.error_message)}</p>` : ''}
          ${indexProgressHtml(space)}
        </div>
        <div class="source-card-actions">
          ${sourceJobButtons('confluence', space)}
          <button class="btn-action btn-danger-sm" title="Delete" data-delete-source="${space.id}">&#128465;</button>
        </div>
      </div>`).join('');
  bindSourceCardActions(container, 'confluence', spaces, s => `Confluence ${s.space_key}`);
}

// ============================================================
// BROWSE LISTS (pick several repositories / spaces at once)
// ============================================================

function showPicker(name, items, emptyText) {
  state.pickers[name] = { items, selected: new Set() };
  const box = document.getElementById(`${name}Picker`);
  box.style.display = 'block';
  box.innerHTML = `
    <div class="picker-head">
      <input type="text" class="form-input picker-filter" placeholder="Filter..." />
      <label class="picker-all"><input type="checkbox" class="picker-select-all" /> Select all</label>
    </div>
    <div class="picker-list"></div>
    <div class="picker-count"></div>`;
  box.querySelector('.picker-filter').addEventListener('input', () => renderPicker(name));
  box.querySelector('.picker-select-all').addEventListener('change', event => {
    visiblePickerItems(name).filter(i => !i.already_added)
      .forEach(i => event.target.checked ? state.pickers[name].selected.add(i.key) : state.pickers[name].selected.delete(i.key));
    renderPicker(name);
  });
  box.dataset.empty = emptyText;
  renderPicker(name);
}

function visiblePickerItems(name) {
  const filter = document.querySelector(`#${name}Picker .picker-filter`).value.trim().toLowerCase();
  return state.pickers[name].items.filter(i => !filter || `${i.key} ${i.label}`.toLowerCase().includes(filter));
}

function renderPicker(name) {
  const box = document.getElementById(`${name}Picker`);
  const picker = state.pickers[name];
  const items = visiblePickerItems(name);
  box.querySelector('.picker-list').innerHTML = items.length ? items.map(i => `
      <label class="picker-item ${i.already_added ? 'picker-done' : ''}">
        <input type="checkbox" value="${escapeHtml(i.key)}" ${picker.selected.has(i.key) ? 'checked' : ''}
          ${i.already_added ? 'disabled' : ''} />
        <span><b>${escapeHtml(i.key)}</b>${i.label && i.label !== i.key ? ' &middot; ' + escapeHtml(i.label) : ''}
          ${i.already_added ? '<em>(already added)</em>' : ''}</span>
      </label>`).join('') : `<p class="source-hint">${escapeHtml(box.dataset.empty)}</p>`;
  box.querySelectorAll('.picker-item input').forEach(cb => cb.addEventListener('change', () => {
    cb.checked ? picker.selected.add(cb.value) : picker.selected.delete(cb.value);
    updatePickerCount(name);
  }));
  updatePickerCount(name);
}

function updatePickerCount(name) {
  const picker = state.pickers[name];
  const box = document.getElementById(`${name}Picker`);
  box.querySelector('.picker-count').textContent =
    `${picker.selected.size} selected of ${picker.items.length}` +
    (picker.selected.size ? ' - click Add to index them (a few at a time)' : '');
  const addBtn = document.getElementById(name === 'bb' ? 'bbAddBtn' : 'confAddBtn');
  if (picker.selected.size) addBtn.disabled = false;
  addBtn.textContent = picker.selected.size ? `+ Add ${picker.selected.size} selected`
    : (name === 'bb' ? '+ Add Repository' : '+ Add Space');
}

function hidePicker(name) {
  delete state.pickers[name];
  const box = document.getElementById(`${name}Picker`);
  box.style.display = 'none';
  box.innerHTML = '';
}

async function browseBitbucketRepos() {
  const v = bitbucketFormValues();
  const resultEl = document.getElementById('bbTestResult');
  if (v.server_url !== null && !/^https:\/\//i.test(v.server_url)) return showTestResult(resultEl, false, 'Enter the server URL starting with https://');
  if (v.server_url === null && (!v.workspace || !v.username)) return showTestResult(resultEl, false, 'Enter the workspace and account email');
  if (!v.token) return showTestResult(resultEl, false, 'Enter the token');
  resultEl.style.display = 'block';
  resultEl.innerHTML = '<span class="spinner"></span> Loading repositories...';
  try {
    const res = await fetch(`${API_BASE}/sources/bitbucket/discover`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ server_url: v.server_url, workspace: v.workspace, username: v.username, token: v.token }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not list repositories');
    showTestResult(resultEl, true, `${data.repositories.length} repositories found${v.workspace ? ' in ' + v.workspace : ''}. Select the ones to add.`);
    showPicker('bb', data.repositories.map(r => ({ key: `${r.workspace}/${r.slug}`, label: r.name, already_added: r.already_added,
      workspace: r.workspace, slug: r.slug })), 'No repositories match.');
  } catch (err) {
    showTestResult(resultEl, false, err.message);
  }
}

// ============================================================
// MODALS - Confluence
// ============================================================

function showAddConfluenceModal() {
  ['confSpaceKey', 'confUsername', 'confToken'].forEach(id => { document.getElementById(id).value = ''; });
  document.getElementById('confTestResult').style.display = 'none';
  document.getElementById('confAddBtn').disabled = true;
  hidePicker('conf');
  updateConfAddLabel();
  document.getElementById('modalConfluence').style.display = 'flex';
}

function updateConfAddLabel() {
  document.getElementById('confAddBtn').textContent = '+ Add Space';
}

// Fill the space key from a pasted /display/KEY/... or /spaces/KEY/... link.
function fillConfluenceSpace() {
  const url = document.getElementById('confBaseUrl').value.trim();
  const match = url.match(/\/(?:display|spaces)\/([^/?#]+)/) || url.match(/[?&]spaceKey=([^&#]+)/);
  if (match) document.getElementById('confSpaceKey').value = decodeURIComponent(match[1]);
}

function confluenceFormValues() {
  return {
    base_url: document.getElementById('confBaseUrl').value.trim(),
    space_key: document.getElementById('confSpaceKey').value.trim(),
    username: document.getElementById('confUsername').value.trim(),
    token: document.getElementById('confToken').value,
  };
}

function confluenceFormError(v) {
  if (!/^https:\/\//i.test(v.base_url)) return 'Enter the Confluence URL starting with https://';
  if (/atlassian\.net/i.test(v.base_url) && !v.username) return 'Enter your account email (Confluence Cloud)';
  if (!v.token) return 'Enter the token';
  return null;
}

async function testConfluenceConnection() {
  const v = confluenceFormValues();
  const resultEl = document.getElementById('confTestResult');
  const error = confluenceFormError(v);
  if (error) return showTestResult(resultEl, false, error);
  resultEl.style.display = 'block';
  resultEl.innerHTML = '<span class="spinner"></span> Testing connection...';
  try {
    const res = await fetch(`${API_BASE}/sources/confluence/test`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(v),
    });
    const data = await res.json();
    showTestResult(resultEl, !!data.success, data.message || data.detail || 'Connection failed');
    if (data.success && v.space_key) document.getElementById('confAddBtn').disabled = false;
  } catch (err) {
    showTestResult(resultEl, false, `Error: ${err.message}`);
  }
}

async function browseConfluenceSpaces() {
  const v = confluenceFormValues();
  const resultEl = document.getElementById('confTestResult');
  const error = confluenceFormError(v);
  if (error) return showTestResult(resultEl, false, error);
  resultEl.style.display = 'block';
  resultEl.innerHTML = '<span class="spinner"></span> Loading spaces...';
  try {
    const res = await fetch(`${API_BASE}/sources/confluence/discover`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_url: v.base_url, username: v.username, token: v.token }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not list spaces');
    showTestResult(resultEl, true, `${data.spaces.length} spaces found. Select the ones to add.`);
    showPicker('conf', data.spaces.map(sp => ({ key: sp.key, label: sp.name, already_added: sp.already_added })),
      'No spaces match.');
  } catch (err) {
    showTestResult(resultEl, false, err.message);
  }
}

async function addConfluenceSource() {
  const v = confluenceFormValues();
  const error = confluenceFormError(v);
  if (error) return showToast(error, 'error');
  const picker = state.pickers.conf;
  const bulk = picker && picker.selected.size;
  if (!bulk && !v.space_key) return showToast('Enter a space key or select spaces from Browse spaces', 'error');
  try {
    const res = await fetch(`${API_BASE}/sources/confluence${bulk ? '/bulk' : ''}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(bulk ? { base_url: v.base_url, username: v.username, token: v.token,
                                    space_keys: [...picker.selected] } : v),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to add space');
    closeModal('modalConfluence');
    document.getElementById('confToken').value = '';
    showToast(bulk ? `${data.added} spaces added - indexing queued${data.skipped.length ? ` (${data.skipped.length} already added)` : ''}`
                   : `Space ${data.space_key} added - indexing started`, 'success');
    await loadAllSources();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

// ============================================================
// DELETE SOURCE
// ============================================================

function confirmDeleteSource(type, id, name) {
  state.pendingDelete = { type, id, name };
  document.getElementById('confirmDeleteMessage').textContent =
    `Delete "${name}"? This will remove all indexed chunks.`;
  document.getElementById('modalConfirmDelete').style.display = 'flex';
}

async function executeDelete() {
  if (!state.pendingDelete) return;
  const { type, id } = state.pendingDelete;

  const endpoints = {
    document: `${API_BASE}/sources/documents/${id}`,
    legacy: `${API_BASE}/sources/documents/legacy?source_file=${encodeURIComponent(id)}`,
    bitbucket: `${API_BASE}/sources/bitbucket/${id}`,
    jira: `${API_BASE}/sources/jira/${id}`,
    confluence: `${API_BASE}/sources/confluence/${id}`,
  };

  closeModal('modalConfirmDelete');

  try {
    const res = await fetch(endpoints[type], { method: 'DELETE' });
    if (res.ok) {
      showToast(`Source deleted`, 'success');
      await loadAllSources();
      loadKBStats();
    } else {
      const err = await res.json();
      showToast(`Delete failed: ${err.detail || 'Unknown error'}`, 'error');
    }
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    state.pendingDelete = null;
  }
}

// ============================================================
// INVESTIGATE (stub - Phase 11)
// ============================================================

function runInvestigation() {
  const log = document.getElementById('failureLogInput').value.trim();
  if (!log) {
    showToast('Please enter a failure log or error message', 'error');
    return;
  }

  const report = document.getElementById('investigateReport');
  report.innerHTML = `
    <div class="investigation-placeholder">
      <span class="sources-empty-icon">&#128269;</span>
      <h3>Investigation Mode</h3>
      <p>Full investigation pipeline will be available in Phase 11.</p>
      <p class="field-hint">The system will search selected sources for evidence related to this error and generate a structured root cause report.</p>
    </div>`;
  showToast('Investigation mode coming in Phase 11', '');
}

// ============================================================
// MODAL HELPERS
// ============================================================

function closeModal(id) {
  document.getElementById(id).style.display = 'none';
}

function showTestResult(el, success, message) {
  el.style.display = 'block';
  el.className = `connection-result ${success ? 'connection-ok' : 'connection-err'}`;
  el.innerHTML = success
    ? `<span>&#10003; ${escapeHtml(message)}</span>`
    : `<span>&#10007; ${escapeHtml(message)}</span>`;
}
