'use client';

import { useChat } from '@ai-sdk/react';
import { useEffect, useRef, useState } from 'react';

// ---- types -------------------------------------------------------------
type ModelInfo = { name: string; param_size?: string; size_gb?: number | null };
type ModelsResp = {
  ok: boolean;
  local: ModelInfo[];
  cloud: ModelInfo[];
  default?: string;
  host?: string;
  error?: string;
};
type TraceItem = { faculty: string; title: string; items: string[] };
type Session = {
  id: string;
  ts: number;
  title: string;
  model: string;
  messages: any[];
};

const DEFAULT_SETTINGS = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 40,
  repeat_penalty: 1.1,
  num_ctx: 4096,
  max_tokens: 2048,
  frequency_penalty: 0,
  presence_penalty: 0,
  seed: '' as number | '',
};
type Settings = typeof DEFAULT_SETTINGS;

const HISTORY_KEY = 'aglm.history';
const SETTINGS_KEY = 'aglm.settings';

// text of a UI message (concatenate text parts)
function msgText(m: any): string {
  if (!m?.parts) return '';
  return m.parts.filter((p: any) => p.type === 'text').map((p: any) => p.text).join('');
}
// reasoning trace emitted by reasoning models (qwen3, deepseek-r1, …)
function msgReasoning(m: any): string {
  if (!m?.parts) return '';
  return m.parts
    .filter((p: any) => p.type === 'reasoning')
    .map((p: any) => p.text || p.delta || '')
    .join('');
}

export default function Console() {
  const [tab, setTab] = useState<'chat' | 'advanced' | 'history' | 'faculties' | 'about'>('chat');

  // models + live status
  const [models, setModels] = useState<ModelsResp | null>(null);
  const [model, setModel] = useState<string>('gpt-oss:120b-cloud');
  const [augment, setAugment] = useState(true);

  // generation settings (persisted)
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);

  // latest reasoning trace + composer input
  const [trace, setTrace] = useState<TraceItem[] | null>(null);
  const [input, setInput] = useState('');

  // history
  const [sessions, setSessions] = useState<Session[]>([]);
  const activeId = useRef<string | null>(null);

  const { messages, sendMessage, status, stop, setMessages, error } = useChat();
  const logRef = useRef<HTMLDivElement>(null);

  // ---- boot: load persisted settings/history + models -----------------
  useEffect(() => {
    try {
      const s = localStorage.getItem(SETTINGS_KEY);
      if (s) setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(s) });
      const h = localStorage.getItem(HISTORY_KEY);
      if (h) setSessions(JSON.parse(h));
    } catch {}
    loadModels();
    const t = setInterval(loadModels, 8000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

  // autoscroll
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, status]);

  // persist a session when a turn completes
  useEffect(() => {
    if (status !== 'ready' || messages.length === 0) return;
    saveActiveSession(messages);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  async function loadModels() {
    try {
      const r = await fetch('/aglm/ollama/models');
      const j: ModelsResp = await r.json();
      setModels(j);
      if (j.default && (j.local.concat(j.cloud)).some((m) => m.name === j.default)) {
        setModel((cur) => (cur === 'gpt-oss:120b-cloud' ? j.default! : cur));
      }
    } catch {
      setModels({ ok: false, local: [], cloud: [], error: 'offline' });
    }
  }

  // ---- send -----------------------------------------------------------
  async function onSend() {
    const text = input.trim();
    if (!text || status === 'streaming' || status === 'submitted') return;
    setInput('');
    let system: string | undefined;
    if (augment) {
      try {
        const r = await fetch('/aglm/augment-prompt', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ query: text }),
        });
        const j = await r.json();
        system = j.system;
        setTrace(j.trace || null);
      } catch {
        setTrace(null);
      }
    } else {
      setTrace(null);
    }
    sendMessage({ text }, { body: { model, system, settings } });
  }

  // ---- history --------------------------------------------------------
  function persistSessions(next: Session[]) {
    setSessions(next);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(next.slice(0, 100)));
  }
  function saveActiveSession(msgs: any[]) {
    const firstUser = msgs.find((m) => m.role === 'user');
    const title = firstUser ? msgText(firstUser).slice(0, 80) : 'Conversation';
    setSessions((prev) => {
      let id = activeId.current;
      let next = [...prev];
      if (!id) {
        id = 'h_' + Date.now().toString(36) + Math.floor(performance.now()).toString(36);
        activeId.current = id;
        next.unshift({ id, ts: Date.now(), title, model, messages: msgs });
      } else {
        const i = next.findIndex((s) => s.id === id);
        if (i >= 0) next[i] = { ...next[i], ts: Date.now(), title, model, messages: msgs };
        else next.unshift({ id, ts: Date.now(), title, model, messages: msgs });
      }
      localStorage.setItem(HISTORY_KEY, JSON.stringify(next.slice(0, 100)));
      return next;
    });
  }
  function newChat() {
    activeId.current = null;
    setMessages([]);
    setTrace(null);
    setTab('chat');
  }
  function openSession(s: Session) {
    activeId.current = s.id;
    setMessages(s.messages);
    setModel(s.model);
    setTrace(null);
    setTab('chat');
  }
  function deleteSession(id: string) {
    persistSessions(sessions.filter((s) => s.id !== id));
    if (activeId.current === id) activeId.current = null;
  }
  async function archiveToMemory(s: Session) {
    const pairs: [string, string][] = [];
    let pendingUser = '';
    for (const m of s.messages) {
      if (m.role === 'user') pendingUser = msgText(m);
      else if (m.role === 'assistant') {
        pairs.push([pendingUser, msgText(m)]);
        pendingUser = '';
      }
    }
    try {
      const r = await fetch('/aglm/memory/save', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ pairs }),
      });
      const j = await r.json();
      alert(j.saved ? 'Archived to AGLM memory: ' + j.saved : 'Error: ' + (j.error || 'failed'));
    } catch {
      alert('Could not reach AGLM memory (Flask brain on :5000).');
    }
  }

  // ---- derived --------------------------------------------------------
  const live = !!models?.ok;
  const nModels = (models?.local.length || 0) + (models?.cloud.length || 0);
  const busy = status === 'streaming' || status === 'submitted';
  // token counter: per-message metadata + running session total
  const tok = (m: any) => (m?.metadata as any) || null;
  const sessionTokens = messages.reduce((sum, m) => sum + (tok(m)?.totalTokens || 0), 0);
  const lastAssistantId = [...messages].reverse().find((m) => m.role === 'assistant')?.id;

  const allModels = models ? [...models.local, ...models.cloud] : [];
  const isCloud = (n: string) => models?.cloud.some((m) => m.name === n);

  // ---- render ---------------------------------------------------------
  return (
    <div className="app">
      <aside className="side">
        <div className="brand">
          <div className="logo">a<b>GLM</b></div>
          <div className="sub">Console · AI SDK</div>
        </div>
        <nav className="nav">
          <div className="grp">Participant</div>
          <NavBtn id="chat" tab={tab} set={setTab} ico="◈" label="Chat" />
          <NavBtn id="advanced" tab={tab} set={setTab} ico="⚙" label="Advanced" />
          <NavBtn id="history" tab={tab} set={setTab} ico="▤" label=".history" />
          <div className="grp">AGLM brain</div>
          <NavBtn id="faculties" tab={tab} set={setTab} ico="⌬" label="Faculties" />
          <NavBtn id="about" tab={tab} set={setTab} ico="ⓘ" label="About" />
          <div style={{ padding: '12px 8px' }}>
            <button className="btn ghost sm" style={{ width: '100%' }} onClick={newChat}>+ New chat</button>
          </div>
        </nav>
        <div className="foot">MASTERMIND aGLM · RAGE<br />Vercel AI SDK v7</div>
      </aside>

      <main className="main">
        <div className="topbar">
          <span className="pill">
            <span className={'dot ' + (live ? 'ok pulse' : 'bad')} />
            {live ? `Ollama live · ${nModels}` : 'Ollama offline'}
          </span>
          <span className="dim small">model</span>
          <select value={model} onChange={(e) => setModel(e.target.value)} style={{ minWidth: 200 }}>
            {models?.local.length ? (
              <optgroup label="local">
                {models.local.map((m) => <option key={m.name} value={m.name}>▣ {m.name}{m.param_size ? ` (${m.param_size})` : ''}</option>)}
              </optgroup>
            ) : null}
            {models?.cloud.length ? (
              <optgroup label="cloud">
                {models.cloud.map((m) => <option key={m.name} value={m.name}>☁ {m.name}{m.param_size ? ` (${m.param_size})` : ''}</option>)}
              </optgroup>
            ) : null}
            {!allModels.length ? <option value={model}>{model}</option> : null}
          </select>
          <span className={'tag ' + (isCloud(model) ? 'cloud' : 'local')}>{isCloud(model) ? 'cloud' : 'local'}</span>
          <span className="pill" title="total tokens this conversation">🪙 {sessionTokens.toLocaleString()} tok</span>
          <div className="spacer" />
          <div className={'toggle' + (augment ? ' on' : '')} onClick={() => setAugment((v) => !v)} title="Run queries through the AGLM logic-augmentation layer">
            <span className="switch" />
            <span className="small">AGLM augmentation</span>
          </div>
        </div>

        <div className="content">
          {/* CHAT */}
          <section className={'view' + (tab === 'chat' ? ' active' : '')}>
            <div className="chatwrap">
              <div className="chatlog" ref={logRef}>
                {messages.length === 0 && (
                  <div className="empty">
                    Ask anything. {augment ? 'AGLM augments your query with Socratic, epistemic, nonmonotonic & BDI reasoning, then streams from ' : 'Streaming directly from '}
                    <b className="mono">{model}</b>.
                  </div>
                )}
                {messages.map((m) => (
                  <div key={m.id} className={'msg ' + m.role}>
                    <div className="av">{m.role === 'user' ? 'YOU' : 'aG'}</div>
                    <div className="bubble">
                      {m.role === 'assistant' && msgReasoning(m) && (
                        <details className="think" open={!msgText(m)}>
                          <summary>{msgText(m) ? 'thought process' : <span><span className="spin" /> thinking…</span>}</summary>
                          <div className="thinktext">{msgReasoning(m)}</div>
                        </details>
                      )}
                      {msgText(m) || (busy && m.role === 'assistant' && m.id === lastAssistantId && !msgReasoning(m) ? <span className="cursor" /> : '')}
                      {m.role === 'assistant' && tok(m)?.totalTokens != null && (
                        <div className="tokline">
                          🪙 {tok(m).totalTokens} tokens
                          {tok(m).inputTokens != null && tok(m).outputTokens != null
                            ? ` · ${tok(m).inputTokens} prompt + ${tok(m).outputTokens} completion`
                            : ''}
                        </div>
                      )}
                      {augment && trace && m.role === 'assistant' && m.id === lastAssistantId && (
                        <div className="trace">
                          <div className="small dim" style={{ marginBottom: 4 }}>AGLM reasoning trace</div>
                          {trace.map((t, i) => (
                            <div className="tline" key={i}>
                              <span className="fac">{t.faculty}</span>{t.title}: <span className="dim">{t.items.join(' · ')}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {error && <div className="msg assistant"><div className="av">!</div><div className="bubble" style={{ color: 'var(--bad)' }}>Stream error: {String(error.message || error)}. Is Ollama running and the model pulled?</div></div>}
              </div>
              <div className="composer">
                <textarea
                  value={input}
                  placeholder={`Message ${model}…  (Enter to send, Shift+Enter for newline)`}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); } }}
                  rows={1}
                />
                {busy
                  ? <button className="btn" onClick={() => stop()}>■ Stop</button>
                  : <button className="btn primary" onClick={onSend} disabled={!input.trim()}>Send →</button>}
              </div>
            </div>
          </section>

          {/* ADVANCED */}
          <section className={'view' + (tab === 'advanced' ? ' active' : '')}>
            <h1>Advanced — model state</h1>
            <p className="lead">Change the generation state passed to the model on every turn. Standard params
              (temperature, top-p, penalties, max tokens, seed) are honored by Ollama&apos;s OpenAI-compatible
              endpoint; <span className="mono">top_k / num_ctx / repeat_penalty</span> are sent best-effort as native Ollama options.</p>
            <div className="card">
              <Slider label="Temperature" desc="randomness / creativity" min={0} max={2} step={0.05} k="temperature" s={settings} set={setSettings} />
              <Slider label="Top-p" desc="nucleus sampling" min={0} max={1} step={0.01} k="top_p" s={settings} set={setSettings} />
              <Slider label="Top-k" desc="Ollama native" min={0} max={100} step={1} k="top_k" s={settings} set={setSettings} />
              <Slider label="Repeat penalty" desc="Ollama native" min={0.8} max={2} step={0.05} k="repeat_penalty" s={settings} set={setSettings} />
              <Slider label="Frequency penalty" desc="reduce repetition" min={-2} max={2} step={0.1} k="frequency_penalty" s={settings} set={setSettings} />
              <Slider label="Presence penalty" desc="encourage new topics" min={-2} max={2} step={0.1} k="presence_penalty" s={settings} set={setSettings} />
            </div>
            <div className="card">
              <div className="grid2">
                <NumField label="Max output tokens" k="max_tokens" s={settings} set={setSettings} />
                <NumField label="Context window (num_ctx)" k="num_ctx" s={settings} set={setSettings} />
                <NumField label="Seed (blank = random)" k="seed" s={settings} set={setSettings} allowBlank />
              </div>
              <div className="row" style={{ marginTop: 12 }}>
                <button className="btn ghost sm" onClick={() => setSettings(DEFAULT_SETTINGS)}>Reset to defaults</button>
                <span className="dim small">settings persist in this browser</span>
              </div>
            </div>
          </section>

          {/* HISTORY */}
          <section className={'view' + (tab === 'history' ? ' active' : '')}>
            <h1>.history</h1>
            <p className="lead">Every conversation is saved locally. Reopen one to continue it, archive it to the
              AGLM long-term memory store (<span className="mono">./memory/*.json</span>), or delete it.</p>
            <div className="card">
              <div className="row" style={{ marginBottom: 12 }}>
                <button className="btn primary sm" onClick={newChat}>+ New chat</button>
                <span className="dim small">{sessions.length} saved conversation{sessions.length === 1 ? '' : 's'}</span>
              </div>
              {sessions.length === 0 ? (
                <div className="dim small">No history yet — send a message in Chat.</div>
              ) : (
                <ul className="hlist">
                  {sessions.map((s) => (
                    <li key={s.id}>
                      <div className="meta" onClick={() => openSession(s)} style={{ cursor: 'pointer' }}>
                        <div className="ttl">{s.title || 'Conversation'}</div>
                        <div className="sub">{new Date(s.ts).toLocaleString()} · {s.model} · {s.messages.length} msgs{activeId.current === s.id ? ' · active' : ''}</div>
                      </div>
                      <button className="btn ghost sm" onClick={() => openSession(s)}>open</button>
                      <button className="btn ghost sm" onClick={() => archiveToMemory(s)}>→ memory</button>
                      <button className="btn ghost sm" style={{ color: 'var(--bad)' }} onClick={() => deleteSession(s.id)}>✕</button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>

          {/* FACULTIES (Flask console embedded) */}
          <section className={'view' + (tab === 'faculties' ? ' active' : '')}>
            <h1>AGLM Faculties</h1>
            <p className="lead">The full cognitive console — Socratic, Logic Tables, Nonmonotonic, Epistemic, BDI,
              MASTERMIND, Autonomize, Prediction & Memory — served by the AGLM Flask brain on :5000.</p>
            <iframe className="frame" src="http://localhost:5000/" title="AGLM Faculties" />
          </section>

          {/* ABOUT */}
          <section className={'view' + (tab === 'about' ? ' active' : '')}>
            <h1>About this console</h1>
            <p className="lead">A clean-house participant experience built on the <a href="https://ai-sdk.dev" target="_blank" rel="noreferrer">Vercel AI SDK v7</a>.
              AGLM is not a model — it augments one. The chat streams from Ollama (local or <span className="tag cloud">:cloud</span>)
              via the OpenAI-compatible endpoint, with each query optionally passed through the AGLM logic-augmentation layer.</p>
            <div className="card">
              <h3>How a turn flows</h3>
              <pre>{`1. You send a message.
2. (if AGLM augmentation ON) → POST /aglm/augment-prompt
     Socratic · Epistemic · Nonmonotonic · BDI  →  augmented system prompt + trace
3. useChat → POST /api/chat  (AI SDK route)
     streamText( ollama(model), system, messages, {temperature, top_p, …} )
4. Tokens stream back into the bubble; the reasoning trace is shown beneath.`}</pre>
            </div>
            <div className="card">
              <h3>Stack</h3>
              <div className="row">
                <span className="badge">ai@7</span><span className="badge">@ai-sdk/react@4</span>
                <span className="badge">@ai-sdk/openai-compatible@3</span><span className="badge">next@15</span>
                <span className="badge">Ollama</span><span className="badge">Flask AGLM brain</span>
              </div>
              <div className="small dim" style={{ marginTop: 10 }}>
                Brain: <span className="mono">{models?.host || 'http://localhost:11434'}</span> · models {nModels} · status {live ? 'live' : 'offline'}
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

function NavBtn({ id, tab, set, ico, label }: any) {
  return (
    <button className={tab === id ? 'active' : ''} onClick={() => set(id)}>
      <span className="ico">{ico}</span><span className="lbl">{label}</span>
    </button>
  );
}

function Slider({ label, desc, min, max, step, k, s, set }: any) {
  return (
    <div className="field">
      <div><label>{label}</label><div className="desc">{desc}</div></div>
      <input type="range" min={min} max={max} step={step} value={s[k]}
        onChange={(e) => set((p: any) => ({ ...p, [k]: Number(e.target.value) }))} />
      <div className="val">{s[k]}</div>
    </div>
  );
}

function NumField({ label, k, s, set, allowBlank }: any) {
  return (
    <div>
      <label className="small dim">{label}</label>
      <input type="number" style={{ width: '100%', marginTop: 4 }} value={s[k]}
        onChange={(e) => {
          const v = e.target.value;
          set((p: any) => ({ ...p, [k]: v === '' ? (allowBlank ? '' : 0) : Number(v) }));
        }} />
    </div>
  );
}
