"""
aglm_app.py  —  The AGLM Cognitive Console (web UI).

A single self-contained Flask app that exposes every aspect of the AGLM
prototype through one UI that *actually loads*. It has no heavy ML dependency:
the cognitive faculties are pure Python, and the language model is an optional
Ollama backend (local small models for modest hardware, ':cloud' models for
larger ones). With no model attached it still runs in augmentation-only mode.

Run:
    python3 aglm_app.py
Then open http://localhost:5000
"""

import os
import json
from flask import Flask, request, jsonify, render_template_string

import aglm_engine as E

app = Flask(__name__)

# Long-lived faculty instances (single-user prototype).
SOCRATIC = E.SocraticFaculty()
LOGIC = E.LogicFaculty()
NONMONO = E.NonmonotonicFaculty()
EPISTEMIC = E.EpistemicFaculty()
BDI = E.BDIFaculty()
AUTONOMIZE = E.AutonomizeFaculty()
MASTERMIND_F = E.MastermindFaculty()
MEMORY_F = E.MemoryFaculty()
PREDICTION = E.PredictionFaculty()
AUGMENT = E.AugmentationEngine()

# Running chat transcript (list of [user, assistant]) for the Memory faculty.
TRANSCRIPT = []


# --------------------------------------------------------------------------- #
#  API
# --------------------------------------------------------------------------- #
@app.route("/api/status")
def api_status():
    return jsonify(E.system_status())


@app.route("/api/ollama/models")
def api_ollama_models():
    return jsonify(E.ModelBackends.ollama_models())


@app.route("/api/ollama/search")
def api_ollama_search():
    """Search the ollama.com cloud library. ?q=<query> filters; cloud-capable
    models are returned with their ':cloud' pull target. Marks already-pulled."""
    import re
    q = (request.args.get("q") or "").strip()
    url = "https://ollama.com/search?c=cloud"
    if q:
        url += "&q=" + q
    pulled = set()
    try:
        oll = E.ModelBackends.ollama_models()
        pulled = {m["name"] for m in (oll.get("cloud", []) + oll.get("local", []))}
    except Exception:
        pass
    try:
        import requests
        r = requests.get(url, timeout=12, headers={"User-Agent": "aglm-console"})
        r.raise_for_status()
        names = sorted(set(re.findall(r'href="/library/([^"/]+)"', r.text)))
        results = []
        for n in names:
            target = n + ":cloud"
            results.append({"name": n, "pull_target": target,
                            "pulled": target in pulled or n in pulled})
        return jsonify({"ok": True, "query": q, "results": results, "source": url})
    except Exception as ex:
        return jsonify({"ok": False, "error": str(ex), "results": []}), 502


@app.route("/api/ollama/pull", methods=["POST"])
def api_ollama_pull():
    """Pull an Ollama model (e.g. a cloud model like glm-5.2:cloud)."""
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "no model name"}), 400
    try:
        import requests
        # Non-streaming pull; may take a while for large models.
        r = requests.post(f"{E.OLLAMA_HOST}/api/pull",
                          json={"name": name, "stream": False}, timeout=1800)
        r.raise_for_status()
        return jsonify({"ok": True, "status": r.json().get("status", "success"), "model": name})
    except Exception as ex:
        return jsonify({"ok": False, "error": str(ex)}), 502


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    backend = data.get("backend", "augment")
    model = data.get("model")
    if not query:
        return jsonify({"error": "empty query"}), 400
    result = AUGMENT.augment(query, backend=backend, model=model)
    TRANSCRIPT.append([query, result["response"]])
    return jsonify(result)


@app.route("/api/augment-prompt", methods=["POST"])
def api_augment_prompt():
    """Return the AGLM system-level augmentation + reasoning trace for a query,
    without calling any model. The AI SDK console uses this as the system prompt
    and then streams the conversation from Ollama itself."""
    data = request.json or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400
    return jsonify(AUGMENT.augment_system(query))


@app.route("/api/socratic", methods=["POST"])
def api_socratic():
    d = request.json or {}
    action = d.get("action")
    premise = d.get("premise", "")
    if action == "add":
        return jsonify(SOCRATIC.add(premise))
    if action == "challenge":
        return jsonify(SOCRATIC.challenge(premise))
    if action == "conclude":
        return jsonify(SOCRATIC.conclude())
    if action == "questions":
        return jsonify({"questions": SOCRATIC.questions(premise), "premises": SOCRATIC.premises})
    if action == "reset":
        SOCRATIC.reset()
        return jsonify(SOCRATIC.snapshot())
    return jsonify({"error": "unknown action"}), 400


@app.route("/api/logic", methods=["POST"])
def api_logic():
    d = request.json or {}
    return jsonify(LOGIC.evaluate(d.get("variables", []), d.get("expressions", [])))


@app.route("/api/nonmonotonic", methods=["POST"])
def api_nonmono():
    d = request.json or {}
    return jsonify(NONMONO.evaluate(d.get("rules", []), d.get("defaults", []), d.get("query", "")))


@app.route("/api/epistemic", methods=["POST"])
def api_epistemic():
    d = request.json or {}
    action = d.get("action")
    if action == "init":
        return jsonify(EPISTEMIC.init(d.get("beliefs", {})))
    if action == "add":
        return jsonify(EPISTEMIC.add(d.get("new_info", {})))
    if action == "revise":
        return jsonify(EPISTEMIC.revise())
    return jsonify({"error": "unknown action"}), 400


@app.route("/api/bdi", methods=["POST"])
def api_bdi():
    d = request.json or {}
    action = d.get("action")
    text = d.get("text", "")
    if action == "belief":
        return jsonify(BDI.add_belief(text))
    if action == "desire":
        return jsonify(BDI.add_desire(text))
    if action == "intention":
        return jsonify(BDI.add_intention(text))
    if action == "execute":
        return jsonify(BDI.execute())
    if action == "reset":
        return jsonify(BDI.reset())
    return jsonify({"error": "unknown action"}), 400


@app.route("/api/autonomize", methods=["POST"])
def api_autonomize():
    return jsonify(AUTONOMIZE.run())


@app.route("/api/mastermind")
def api_mastermind_status():
    return jsonify(MASTERMIND_F.status())


@app.route("/api/mastermind/run", methods=["POST"])
def api_mastermind_run():
    return jsonify(MASTERMIND_F.run())


@app.route("/api/memory")
def api_memory():
    return jsonify(MEMORY_F.list())


@app.route("/api/memory/save", methods=["POST"])
def api_memory_save():
    # Accept an explicit [[user, assistant], ...] list (used by the AI SDK
    # console's .history tab) or fall back to the server-side transcript.
    pairs = (request.json or {}).get("pairs")
    if pairs:
        clean = [[str(p[0]), str(p[1])] for p in pairs if isinstance(p, (list, tuple)) and len(p) >= 2]
        if not clean:
            return jsonify({"error": "no valid pairs"}), 400
        return jsonify(MEMORY_F.save(clean))
    if not TRANSCRIPT:
        return jsonify({"error": "no conversation to save yet"}), 400
    return jsonify(MEMORY_F.save(TRANSCRIPT))


@app.route("/api/prediction")
def api_prediction():
    return jsonify(PREDICTION.status())


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


# --------------------------------------------------------------------------- #
#  Frontend (self-contained: no external CDNs, so it loads fully offline)
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AGLM // Cognitive Console</title>
<style>
:root{
  --bg:#0a0e14; --panel:#0f1620; --panel2:#131c28; --line:#1e2c3a;
  --ink:#cdd9e5; --dim:#6b7e92; --accent:#39d3c7; --accent2:#7c5cff;
  --warn:#ffb454; --bad:#ff5c7a; --good:#54e09a; --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:radial-gradient(1200px 600px at 70% -10%,#11202c 0,var(--bg) 60%);
  color:var(--ink);font-family:Inter,system-ui,Segoe UI,Roboto,sans-serif;font-size:14px;line-height:1.5}
a{color:var(--accent)}
.app{display:grid;grid-template-columns:240px 1fr;height:100vh}
/* sidebar */
.side{background:linear-gradient(180deg,#0c131c,#0a0e14);border-right:1px solid var(--line);
  display:flex;flex-direction:column;overflow:hidden}
.brand{padding:18px 18px 12px}
.brand .logo{font-family:var(--mono);font-weight:700;font-size:20px;letter-spacing:.06em;color:#fff}
.brand .logo b{color:var(--accent)}
.brand .sub{color:var(--dim);font-size:11px;letter-spacing:.14em;text-transform:uppercase;margin-top:2px}
.nav{padding:8px;overflow:auto;flex:1}
.nav .grp{color:var(--dim);font-size:10px;letter-spacing:.18em;text-transform:uppercase;margin:14px 12px 6px}
.nav button{width:100%;text-align:left;background:transparent;border:0;color:var(--ink);
  padding:9px 12px;border-radius:8px;cursor:pointer;display:flex;gap:10px;align-items:center;font-size:13px}
.nav button:hover{background:#121d29}
.nav button.active{background:linear-gradient(90deg,#13283042,#7c5cff14);color:#fff;
  box-shadow:inset 2px 0 0 var(--accent)}
.nav .ico{width:18px;text-align:center;opacity:.85}
.side .foot{padding:12px 16px;border-top:1px solid var(--line);font-size:11px;color:var(--dim)}
/* main */
.main{display:flex;flex-direction:column;overflow:hidden}
.topbar{display:flex;align-items:center;gap:14px;padding:10px 18px;border-bottom:1px solid var(--line);
  background:#0b1219;flex-wrap:wrap}
.pill{display:inline-flex;align-items:center;gap:7px;background:#101a25;border:1px solid var(--line);
  padding:5px 11px;border-radius:20px;font-size:12px;color:var(--dim)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--dim);box-shadow:0 0 8px currentColor}
.dot.ok{background:var(--good);color:var(--good)} .dot.warn{background:var(--warn);color:var(--warn)}
.dot.bad{background:var(--bad);color:var(--bad)}
.topbar .spacer{flex:1}
select,input,textarea,button.btn{font-family:inherit;color:var(--ink)}
select,input[type=text],textarea{background:#0b1219;border:1px solid var(--line);border-radius:8px;
  padding:8px 10px;color:var(--ink);outline:none}
select:focus,input:focus,textarea:focus{border-color:var(--accent)}
.btn{background:#15212e;border:1px solid var(--line);border-radius:8px;padding:8px 14px;cursor:pointer;
  color:var(--ink);font-size:13px}
.btn:hover{border-color:var(--accent);color:#fff}
.btn.primary{background:linear-gradient(90deg,var(--accent),#2bb6 );background:linear-gradient(90deg,#16a89e,#2f6df0);
  border:0;color:#06131a;font-weight:600}
.btn.primary:hover{filter:brightness(1.08)}
.btn.ghost{background:transparent}
.content{flex:1;overflow:auto;padding:22px 26px}
.view{display:none;max-width:1080px;margin:0 auto}
.view.active{display:block}
h1{font-size:20px;margin:0 0 2px} .lead{color:var(--dim);margin:0 0 18px;font-size:13px;max-width:70ch}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:16px}
.card h3{margin:0 0 4px;font-size:14px} .card .hint{color:var(--dim);font-size:12px;margin-bottom:12px}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.row.tight{gap:6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}.app{grid-template-columns:64px 1fr}.brand .sub,.nav .grp,.nav button span.lbl{display:none}}
.mono{font-family:var(--mono)} .small{font-size:12px} .dim{color:var(--dim)}
.tag{font-family:var(--mono);font-size:11px;padding:2px 8px;border-radius:6px;border:1px solid var(--line);color:var(--dim)}
.tag.local{color:var(--good);border-color:#1d3a2c} .tag.cloud{color:var(--accent2);border-color:#2a2350}
pre{background:#070b10;border:1px solid var(--line);border-radius:10px;padding:14px;overflow:auto;
  font-family:var(--mono);font-size:12.5px;color:#bfe9e4;white-space:pre-wrap;word-break:break-word;margin:0}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:12.5px}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:center}
th{background:#0b1622;color:var(--accent)}
td.T{color:var(--good)} td.F{color:var(--bad)}
.list{list-style:none;margin:0;padding:0}
.list li{padding:8px 10px;border:1px solid var(--line);border-radius:8px;margin-bottom:6px;background:#0c141d;
  display:flex;justify-content:space-between;gap:10px;align-items:center}
.list li .x{color:var(--bad);cursor:pointer;opacity:.7} .list li .x:hover{opacity:1}
.chip{display:inline-flex;gap:6px;align-items:center;background:#0c141d;border:1px solid var(--line);
  border-radius:18px;padding:4px 10px;font-size:12px;margin:0 6px 6px 0}
/* chat */
.chat{display:flex;flex-direction:column;gap:14px}
.msg{display:flex;gap:12px} .msg .av{width:30px;height:30px;border-radius:8px;flex:none;display:grid;place-items:center;
  font-family:var(--mono);font-weight:700;font-size:12px}
.msg.user .av{background:#10212e;color:var(--accent)} .msg.ai .av{background:#1a1640;color:#b6a6ff}
.bubble{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:12px 14px;flex:1;white-space:pre-wrap}
.trace{margin-top:10px;border-top:1px dashed var(--line);padding-top:10px}
.trace .tline{font-size:12.5px;margin:3px 0}
.trace .fac{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;color:var(--accent);margin-right:6px}
.kbd{font-family:var(--mono);background:#0b1219;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:11px}
.spin{display:inline-block;width:13px;height:13px;border:2px solid var(--line);border-top-color:var(--accent);
  border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px}
@keyframes s{to{transform:rotate(360deg)}}
.facgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
.facgrid .f{border:1px solid var(--line);border-radius:10px;padding:11px;background:#0c141d}
.facgrid .f .n{font-family:var(--mono);font-size:12px} .facgrid .f .d{color:var(--dim);font-size:11px;margin-top:3px;word-break:break-all}
.notice{background:#10202d;border:1px solid #1d3a44;border-radius:10px;padding:10px 12px;color:#9fd;font-size:12.5px;margin-bottom:12px}
.livebar{display:flex;align-items:center;gap:12px;background:var(--panel);border:1px solid var(--line);
  border-radius:12px;padding:12px 16px;margin-bottom:16px}
.livebar.live{border-color:#1d4a36;box-shadow:0 0 0 1px #1d4a3666,0 0 24px -10px var(--good)}
.livebar.dead{border-color:#4a1d2a}
.dot.lg{width:12px;height:12px}
.dot.pulse{animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 currentColor}50%{box-shadow:0 0 0 5px transparent}}
.mrow{display:flex;align-items:center;gap:10px;padding:8px 10px;border:1px solid var(--line);border-radius:9px;
  margin-bottom:7px;background:#0c141d}
.mrow b{font-family:var(--mono);font-size:12.5px}
.mrow .spacer{flex:1}
.badge{font-size:10.5px;font-family:var(--mono);padding:2px 7px;border-radius:5px;border:1px solid var(--line);color:var(--dim)}
.badge.ok{color:var(--good);border-color:#1d3a2c} .badge.pull{color:var(--accent2);border-color:#2a2350}
</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <div class="brand">
      <div class="logo">a<b>GLM</b></div>
      <div class="sub">Cognitive Console</div>
    </div>
    <nav class="nav" id="nav">
      <div class="grp">Augmentation</div>
      <button data-v="chat" class="active"><span class="ico">◈</span><span class="lbl">Augment / Chat</span></button>
      <button data-v="models"><span class="ico">▣</span><span class="lbl">Models (Ollama)</span></button>
      <div class="grp">Cognitive Faculties</div>
      <button data-v="socratic"><span class="ico">?</span><span class="lbl">Socratic</span></button>
      <button data-v="logic"><span class="ico">⊤</span><span class="lbl">Logic Tables</span></button>
      <button data-v="nonmono"><span class="ico">∴</span><span class="lbl">Nonmonotonic</span></button>
      <button data-v="epistemic"><span class="ico">◉</span><span class="lbl">Epistemic</span></button>
      <button data-v="bdi"><span class="ico">⟁</span><span class="lbl">BDI Agent</span></button>
      <div class="grp">System</div>
      <button data-v="mastermind"><span class="ico">⌬</span><span class="lbl">MASTERMIND</span></button>
      <button data-v="autonomize"><span class="ico">↻</span><span class="lbl">Autonomize</span></button>
      <button data-v="prediction"><span class="ico">∿</span><span class="lbl">Prediction</span></button>
      <button data-v="memory"><span class="ico">▤</span><span class="lbl">Memory</span></button>
      <button data-v="about"><span class="ico">ⓘ</span><span class="lbl">About AGLM</span></button>
    </nav>
    <div class="foot">MASTERMIND aGLM · RAGE<br>codephreak · GPLv3 2024</div>
  </aside>

  <main class="main">
    <div class="topbar">
      <span class="pill"><span class="dot" id="d-fac"></span><span id="t-fac">faculties…</span></span>
      <span class="pill"><span class="dot" id="d-oll"></span><span id="t-oll">ollama…</span></span>
      <span class="pill"><span class="dot ok"></span><span id="t-sys">cpu –</span></span>
      <div class="spacer"></div>
      <span class="dim small">backend</span>
      <select id="backend" style="min-width:150px">
        <option value="augment">Augmentation only</option>
        <option value="ollama">Ollama model</option>
      </select>
      <select id="model" style="min-width:190px;display:none"></select>
    </div>

    <div class="content">
      <!-- CHAT -->
      <section class="view active" id="v-chat">
        <h1>Logic-Augmented Chat</h1>
        <p class="lead">AGLM is not a model — it augments one. Your query is run through every cognitive
          faculty to build a logic-augmented prompt, which is then handed to the selected model
          (or shown directly in augmentation-only mode).</p>
        <div class="card">
          <div class="chat" id="chatlog"></div>
          <div class="row" style="margin-top:14px">
            <input id="q" type="text" placeholder="Ask anything — e.g. 'Is P implies Q the same as not P or Q?'" style="flex:1">
            <button class="btn primary" id="send">Augment →</button>
          </div>
          <div class="small dim" style="margin-top:8px">Press <span class="kbd">Enter</span> to send · current backend shown top-right</div>
        </div>
        <div class="card" id="promptcard" style="display:none">
          <h3>Logic-augmented prompt <span class="dim small">(what AGLM feeds the model)</span></h3>
          <pre id="augprompt"></pre>
        </div>
      </section>

      <!-- MODELS -->
      <section class="view" id="v-models">
        <h1>Ollama Models</h1>
        <p class="lead">Small local models run on this modest hardware for testing; <span class="tag cloud">:cloud</span>
          models are proxied by the local Ollama daemon to ollama.com for larger reasoning. Pick one as the chat backend.</p>
        <div class="livebar" id="livebar">
          <span class="dot" id="d-live"></span>
          <b id="live-txt">checking Ollama…</b>
          <span class="dim small" id="live-meta"></span>
          <span class="spacer"></span>
          <button class="btn ghost" id="reloadmodels">↻ refresh</button>
        </div>
        <div class="grid2">
          <div class="card">
            <h3>Local <span class="dim small">on-device · quick tests</span></h3>
            <div id="localmodels" style="margin-top:10px"></div>
          </div>
          <div class="card">
            <h3>Cloud <span class="dim small">pulled · proxied to ollama.com</span></h3>
            <div id="cloudmodels" style="margin-top:10px"></div>
          </div>
        </div>
        <div class="card">
          <h3>Cloud library search <span class="dim small">browse available models on ollama.com</span></h3>
          <div class="hint">Cloud models need an Ollama account (<span class="mono">ollama signin</span>). Click <b>Pull</b> to auto-download the <span class="mono">:cloud</span> tag, then <b>Use</b>.</div>
          <div class="row"><input id="searchq" type="text" placeholder="search e.g. glm, deepseek, qwen, gpt-oss …" style="flex:1">
            <button class="btn primary" id="searchbtn">Search cloud</button>
            <button class="btn ghost" id="browseall">Browse all cloud</button></div>
          <div id="searchout" style="margin-top:14px"></div>
        </div>
      </section>

      <!-- SOCRATIC -->
      <section class="view" id="v-socratic">
        <h1>Socratic Reasoning</h1>
        <p class="lead">Manage premises, challenge them, and draw conclusions — the original
          <span class="mono">reasoning.SocraticReasoning</span> engine. Also generates probing questions.</p>
        <div class="grid2">
          <div class="card">
            <h3>Premises</h3>
            <div class="row"><input id="soc-premise" type="text" placeholder="Add a premise…" style="flex:1">
              <button class="btn" onclick="soc('add')">Add</button></div>
            <ul class="list" id="soc-list" style="margin-top:12px"></ul>
            <div class="row">
              <button class="btn primary" onclick="soc('conclude')">Draw conclusion</button>
              <button class="btn" onclick="soc('questions')">Probing questions</button>
              <button class="btn ghost" onclick="soc('reset')">Reset</button>
            </div>
          </div>
          <div class="card"><h3>Engine log</h3><pre id="soc-log">—</pre></div>
        </div>
      </section>

      <!-- LOGIC -->
      <section class="view" id="v-logic">
        <h1>Logic Tables</h1>
        <p class="lead">Generate truth tables over boolean variables — the original
          <span class="mono">logic.LogicTables</span>. Use Python boolean syntax: <span class="mono">and or not</span>.</p>
        <div class="card">
          <div class="row">
            <label class="dim small">Variables</label>
            <input id="lt-vars" type="text" value="P, Q" style="flex:1">
          </div>
          <div class="row" style="margin-top:10px">
            <label class="dim small">Expressions</label>
            <input id="lt-exprs" type="text" value="P and Q, P or Q, not P" style="flex:1">
          </div>
          <div class="row" style="margin-top:12px"><button class="btn primary" onclick="logic()">Generate table</button></div>
          <div id="lt-out" style="margin-top:14px;overflow:auto"></div>
        </div>
      </section>

      <!-- NONMONOTONIC -->
      <section class="view" id="v-nonmono">
        <h1>Nonmonotonic / Default Logic</h1>
        <p class="lead">Conclusions hold by default and retract under contradicting evidence —
          <span class="mono">nonmonotonic.DefaultLogic</span>. Rules fire when conditions are present; defaults
          fire when their conditions are <i>absent</i>.</p>
        <div class="grid2">
          <div class="card">
            <h3>Rules <span class="dim small">conditions ⇒ conclusions</span></h3>
            <div class="row tight"><input id="nm-rc" type="text" placeholder="conditions: A,B" style="flex:1">
              <input id="nm-rk" type="text" placeholder="conclusions: C" style="flex:1">
              <button class="btn" onclick="addRule()">+</button></div>
            <ul class="list" id="nm-rules" style="margin-top:10px"></ul>
            <h3 style="margin-top:14px">Defaults <span class="dim small">fire if conditions absent</span></h3>
            <div class="row tight"><input id="nm-dc" type="text" placeholder="conditions: D" style="flex:1">
              <input id="nm-dk" type="text" placeholder="conclusions: E" style="flex:1">
              <button class="btn" onclick="addDef()">+</button></div>
            <ul class="list" id="nm-defs" style="margin-top:10px"></ul>
          </div>
          <div class="card">
            <h3>Query</h3>
            <div class="row"><input id="nm-q" type="text" placeholder="Is this entailed? e.g. C" style="flex:1">
              <button class="btn primary" onclick="nonmono()">Evaluate</button></div>
            <div id="nm-out" style="margin-top:14px"></div>
          </div>
        </div>
      </section>

      <!-- EPISTEMIC -->
      <section class="view" id="v-epistemic">
        <h1>Epistemic Belief Revision</h1>
        <p class="lead">Track beliefs and revise them under new information —
          <span class="mono">epistemic.AutoepistemicAgent</span>.</p>
        <div class="grid2">
          <div class="card">
            <h3>Beliefs</h3>
            <div class="row tight"><input id="ep-k" type="text" placeholder="proposition" style="flex:1">
              <select id="ep-v"><option value="true">True</option><option value="false">False</option></select>
              <button class="btn" onclick="epAdd()">Set</button></div>
            <div class="row" style="margin-top:12px">
              <button class="btn" onclick="epRevise()">Revise (retract contradictions)</button>
              <button class="btn ghost" onclick="epReset()">Reset</button>
            </div>
          </div>
          <div class="card"><h3>Belief state</h3><pre id="ep-out">{}</pre><div id="ep-log" class="small dim" style="margin-top:8px"></div></div>
        </div>
      </section>

      <!-- BDI -->
      <section class="view" id="v-bdi">
        <h1>BDI Agent</h1>
        <p class="lead">Beliefs · Desires · Intentions — the cognitive structure of an agent
          (<span class="mono">bdi.py</span>). Add elements then execute intentions.</p>
        <div class="card">
          <div class="row">
            <select id="bdi-type"><option value="belief">Belief</option><option value="desire">Desire</option><option value="intention">Intention</option></select>
            <input id="bdi-text" type="text" placeholder="text…" style="flex:1">
            <button class="btn" onclick="bdiAdd()">Add</button>
            <button class="btn primary" onclick="bdiExec()">Execute intentions</button>
            <button class="btn ghost" onclick="bdiReset()">Reset</button>
          </div>
          <div class="grid2" style="margin-top:16px">
            <div><h3>Beliefs</h3><ul class="list" id="bdi-b"></ul></div>
            <div><h3>Desires</h3><ul class="list" id="bdi-d"></ul></div>
          </div>
          <div style="margin-top:8px"><h3>Intentions</h3><ul class="list" id="bdi-i"></ul></div>
          <pre id="bdi-log" style="margin-top:8px;display:none"></pre>
        </div>
      </section>

      <!-- MASTERMIND -->
      <section class="view" id="v-mastermind">
        <h1>MASTERMIND Controller</h1>
        <p class="lead">The agent creator and control layer — loads, validates, and executes agents
          concurrently, accumulating their data (<span class="mono">MASTERMIND.py</span>).</p>
        <div class="card">
          <h3>Authorized agents <span class="dim small">(config.json)</span></h3>
          <div id="mm-agents" style="margin-top:8px"></div>
          <div class="row" style="margin-top:12px"><button class="btn primary" onclick="mmRun()">Run agent cycle</button></div>
          <div id="mm-out" style="margin-top:14px"></div>
        </div>
      </section>

      <!-- AUTONOMIZE -->
      <section class="view" id="v-autonomize">
        <h1>Autonomize / Self-Healing</h1>
        <p class="lead">Resilient task execution with exponential backoff and a self-healing recovery
          procedure (<span class="mono">autonomize.Autonomizer</span>).</p>
        <div class="card">
          <button class="btn primary" onclick="autoRun()">Run resilient cycle</button>
          <pre id="auto-out" style="margin-top:14px">—</pre>
        </div>
      </section>

      <!-- PREDICTION -->
      <section class="view" id="v-prediction">
        <h1>Prediction</h1>
        <p class="lead">Optional ML inference layer (<span class="mono">prediction.Predictor</span>).
          Drop a trained <span class="mono">trained_model.pkl</span> into <span class="mono">./models</span> to enable it.</p>
        <div class="card"><div id="pred-out">…</div></div>
      </section>

      <!-- MEMORY -->
      <section class="view" id="v-memory">
        <h1>Memory</h1>
        <p class="lead">Persistent conversation memory written to <span class="mono">./memory/*.json</span>
          (<span class="mono">memory.py</span>). This is the long-term context AGLM builds on.</p>
        <div class="card">
          <div class="row"><button class="btn primary" onclick="memSave()">Save current chat to memory</button>
            <button class="btn ghost" onclick="memList()">↻ refresh</button>
            <span id="mem-msg" class="small dim"></span></div>
          <div id="mem-out" style="margin-top:14px"></div>
        </div>
      </section>

      <!-- ABOUT -->
      <section class="view" id="v-about">
        <h1>About AGLM</h1>
        <p class="lead">Autonomous General Learning Model — a hybridization of MASTERMIND aGLM with RAGE.
          AGLM is <b>not</b> a model; it is a logic layer that augments any existing model with Socratic,
          epistemic, nonmonotonic, and BDI reasoning. This console is the original Mastermind prototype,
          made to actually load.</p>
        <div class="card">
          <h3>Faculty health</h3>
          <div class="facgrid" id="fac-grid"></div>
        </div>
        <div class="card">
          <h3>Augmentation system prompt <span class="dim small">(Professor Codephreak persona)</span></h3>
          <pre id="sysprompt" style="max-height:240px"></pre>
        </div>
      </section>
    </div>
  </main>
</div>

<script>
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const api=async(u,b)=>{const r=await fetch(u,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{});return r.json()};
const esc=s=>(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// nav
$$('#nav button').forEach(b=>b.onclick=()=>{
  $$('#nav button').forEach(x=>x.classList.remove('active'));b.classList.add('active');
  $$('.view').forEach(v=>v.classList.remove('active'));
  $('#v-'+b.dataset.v).classList.add('active');
  const v=b.dataset.v;
  if(v==='models')loadModels(); if(v==='mastermind')mmStatus();
  if(v==='prediction')predStatus(); if(v==='memory')memList(); if(v==='about')aboutLoad();
});

// status bar
async function status(){
  const s=await api('/api/status');
  const fok=s.faculties_ok, ftot=s.faculties_total;
  $('#t-fac').textContent=`${fok}/${ftot} faculties`;
  $('#d-fac').className='dot '+(fok===ftot?'ok':fok>0?'warn':'bad');
  $('#t-sys').textContent=`cpu ${s.cpu??'–'}% · mem ${s.mem??'–'}%`;
  window._sys=s;
}
async function ollStatus(){
  const o=await api('/api/ollama/models');
  const n=(o.local||[]).length+(o.cloud||[]).length;
  $('#t-oll').textContent=o.ok?`ollama · ${n} models`:'ollama offline';
  $('#d-oll').className='dot '+(o.ok&&n?'ok':o.ok?'warn':'bad');
  window._oll=o;
  fillModelSelect(o);
}
function fillModelSelect(o){
  const sel=$('#model'); sel.innerHTML='';
  const add=(m,cloud)=>{const op=document.createElement('option');op.value=m.name;
    op.textContent=(cloud?'☁ ':'▣ ')+m.name+(m.param_size?` (${m.param_size})`:'');sel.appendChild(op)};
  (o.local||[]).forEach(m=>add(m,false));(o.cloud||[]).forEach(m=>add(m,true));
  if(o.default&&[...sel.options].some(x=>x.value===o.default))sel.value=o.default;
}
$('#backend').onchange=e=>{$('#model').style.display=e.target.value==='ollama'?'':'none'};

// ---- CHAT ----
function pushMsg(role,txt){const log=$('#chatlog');const d=document.createElement('div');
  d.className='msg '+role;d.innerHTML=`<div class="av">${role==='user'?'YOU':'aG'}</div><div class="bubble">${esc(txt)}</div>`;
  log.appendChild(d);d.scrollIntoView({behavior:'smooth',block:'end'});return d}
async function send(){
  const q=$('#q').value.trim(); if(!q)return; $('#q').value='';
  pushMsg('user',q);
  const backend=$('#backend').value, model=$('#model').value;
  const aiEl=pushMsg('ai',''); aiEl.querySelector('.bubble').innerHTML='<span class="spin"></span> augmenting…';
  const r=await api('/api/chat',{query:q,backend,model});
  const bub=aiEl.querySelector('.bubble');
  bub.innerHTML=esc(r.response);
  // trace
  const tr=document.createElement('div');tr.className='trace';
  tr.innerHTML='<div class="small dim" style="margin-bottom:4px">backend: <b>'+esc(r.backend||'')+'</b> · reasoning trace</div>'+
    (r.trace||[]).map(t=>`<div class="tline"><span class="fac">${esc(t.faculty)}</span>${esc(t.title)}: <span class="dim">${esc((t.items||[]).join(' · '))}</span></div>`).join('');
  bub.appendChild(tr);
  $('#promptcard').style.display='block'; $('#augprompt').textContent=r.augmented_prompt||'';
}
$('#send').onclick=send; $('#q').addEventListener('keydown',e=>{if(e.key==='Enter')send()});

// ---- MODELS ----
async function loadModels(){
  const o=await api('/api/ollama/models'); window._oll=o; fillModelSelect(o);
  // live light
  const n=(o.local||[]).length+(o.cloud||[]).length;
  const lb=$('#livebar');
  if(o.ok){lb.className='livebar live';$('#d-live').className='dot lg pulse ok';
    $('#live-txt').textContent='Ollama is LIVE';
    $('#live-meta').textContent=`${o.host} · ${(o.local||[]).length} local + ${(o.cloud||[]).length} cloud`;}
  else{lb.className='livebar dead';$('#d-live').className='dot lg bad';
    $('#live-txt').textContent='Ollama offline';$('#live-meta').textContent=o.error||'start with: ollama serve';}
  const row=(m,cloud)=>`<div class="mrow"><span class="tag ${cloud?'cloud':'local'}">${cloud?'☁ cloud':'▣ local'}</span>
    <b>${esc(m.name)}</b> <span class="dim small">${m.param_size||(m.size_gb?m.size_gb+'GB':'')||''}</span>
    <span class="spacer"></span><button class="btn ghost small" style="padding:3px 10px" onclick="useModel('${esc(m.name)}')">use →</button></div>`;
  $('#localmodels').innerHTML=(o.local||[]).map(m=>row(m,false)).join('')||'<span class="dim small">none</span>';
  $('#cloudmodels').innerHTML=(o.cloud||[]).map(m=>row(m,true)).join('')||'<span class="dim small">none pulled — search below</span>';
}
function useModel(n){$('#backend').value='ollama';$('#backend').dispatchEvent(new Event('change'));
  if(![...$('#model').options].some(o=>o.value===n)){const op=document.createElement('option');op.value=n;op.textContent=n;$('#model').appendChild(op)}
  $('#model').value=n; $$('#nav button').forEach(x=>x.classList.remove('active'));
  $$('#nav button').forEach(b=>{if(b.dataset.v==='chat')b.classList.add('active')});
  $$('.view').forEach(v=>v.classList.remove('active'));$('#v-chat').classList.add('active');}
$('#reloadmodels').onclick=async()=>{await ollStatus();loadModels()};

// cloud library search + auto-pull
async function cloudSearch(q){
  $('#searchout').innerHTML='<span class="spin"></span> searching ollama.com…';
  const r=await api('/api/ollama/search'+(q?('?q='+encodeURIComponent(q)):''));
  if(!r.ok){$('#searchout').innerHTML='<span class="dim">'+esc(r.error||'search failed')+'</span>';return}
  if(!r.results.length){$('#searchout').innerHTML='<span class="dim small">no cloud models found</span>';return}
  $('#searchout').innerHTML=r.results.map(m=>`<div class="mrow" id="sr-${esc(m.name)}">
     <span class="tag cloud">☁ cloud</span><b>${esc(m.pull_target)}</b><span class="spacer"></span>
     ${m.pulled
        ? '<span class="badge ok">✓ pulled</span><button class="btn ghost small" style="padding:3px 10px" onclick="useModel(\''+esc(m.pull_target)+'\')">use →</button>'
        : '<button class="btn small" style="padding:3px 12px" onclick="autoPull(\''+esc(m.pull_target)+'\')">⬇ pull</button>'}
   </div>`).join('');
}
async function autoPull(target){
  const el=$('#sr-'+CSS.escape(target.split(':')[0]));
  if(el)el.querySelector('.spacer').nextElementSibling.outerHTML='<span class="badge pull"><span class="spin"></span> pulling… large models take a while</span>';
  const r=await api('/api/ollama/pull',{name:target});
  await ollStatus(); await loadModels();
  if(r.ok){cloudSearch($('#searchq').value.trim());}
  else if(el){el.innerHTML+='<span class="dim small">✗ '+esc(r.error||'failed')+' (try: ollama signin)</span>';}
}
$('#searchbtn').onclick=()=>cloudSearch($('#searchq').value.trim());
$('#searchq').addEventListener('keydown',e=>{if(e.key==='Enter')cloudSearch($('#searchq').value.trim())});
$('#browseall').onclick=()=>{$('#searchq').value='';cloudSearch('')};

// ---- SOCRATIC ----
async function soc(action){
  const premise=$('#soc-premise').value.trim();
  const r=await api('/api/socratic',{action,premise});
  if(action==='add')$('#soc-premise').value='';
  if(r.premises)$('#soc-list').innerHTML=r.premises.map(p=>`<li><span>${esc(p)}</span><span class="x" onclick="socChal('${esc(p).replace(/'/g,"\\'")}')">challenge ✕</span></li>`).join('')||'<li class="dim">no premises</li>';
  if(r.questions)$('#soc-log').textContent='Probing questions:\n'+r.questions.map(q=>' • '+q).join('\n');
  else if(r.log)$('#soc-log').textContent=(r.log||[]).join('\n')||'—';
}
async function socChal(p){const r=await api('/api/socratic',{action:'challenge',premise:p});
  $('#soc-list').innerHTML=(r.premises||[]).map(x=>`<li><span>${esc(x)}</span><span class="x" onclick="socChal('${esc(x).replace(/'/g,"\\'")}')">challenge ✕</span></li>`).join('')||'<li class="dim">no premises</li>';
  $('#soc-log').textContent=(r.log||[]).join('\n');}

// ---- LOGIC ----
async function logic(){
  const variables=$('#lt-vars').value.split(',').map(s=>s.trim()).filter(Boolean);
  const expressions=$('#lt-exprs').value.split(',').map(s=>s.trim()).filter(Boolean);
  const r=await api('/api/logic',{variables,expressions});
  if(r.error){$('#lt-out').innerHTML='<span class="dim">'+esc(r.error)+'</span>';return}
  let h='<table><tr>'+r.headers.map(x=>`<th>${esc(x)}</th>`).join('')+'</tr>';
  r.rows.forEach(row=>{h+='<tr>'+row.map(c=>{const t=c===true?'T':c===false?'F':'';return `<td class="${t}">${c===true?'T':c===false?'F':esc(String(c))}</td>`}).join('')+'</tr>'});
  $('#lt-out').innerHTML=h+'</table>';
}

// ---- NONMONOTONIC ----
let RULES=[],DEFS=[];
const splitc=s=>s.split(',').map(x=>x.trim()).filter(Boolean);
function renderNm(){
  $('#nm-rules').innerHTML=RULES.map((r,i)=>`<li><span class="mono">{${r.conditions.join(',')}} ⇒ {${r.conclusions.join(',')}}</span><span class="x" onclick="RULES.splice(${i},1);renderNm()">✕</span></li>`).join('')||'<li class="dim">no rules</li>';
  $('#nm-defs').innerHTML=DEFS.map((d,i)=>`<li><span class="mono">absent{${d.conditions.join(',')}} ⇒ {${d.conclusions.join(',')}}</span><span class="x" onclick="DEFS.splice(${i},1);renderNm()">✕</span></li>`).join('')||'<li class="dim">no defaults</li>';
}
function addRule(){const c=splitc($('#nm-rc').value),k=splitc($('#nm-rk').value);if(!c.length||!k.length)return;RULES.push({conditions:c,conclusions:k});$('#nm-rc').value='';$('#nm-rk').value='';renderNm()}
function addDef(){const c=splitc($('#nm-dc').value),k=splitc($('#nm-dk').value);if(!c.length||!k.length)return;DEFS.push({conditions:c,conclusions:k});$('#nm-dc').value='';$('#nm-dk').value='';renderNm()}
async function nonmono(){
  const query=$('#nm-q').value.trim();
  const r=await api('/api/nonmonotonic',{rules:RULES,defaults:DEFS,query});
  if(r.error){$('#nm-out').innerHTML='<span class="dim">'+esc(r.error)+'</span>';return}
  $('#nm-out').innerHTML=`<div class="notice">Derived belief set: <b class="mono">{${(r.beliefs||[]).join(', ')}}</b></div>`+
    (query?`<div class="row"><span class="tag">${esc(query)}</span> is <b style="color:${r.entailed?'var(--good)':'var(--bad)'}">${r.entailed?'ENTAILED':'NOT entailed'}</b></div>`:'');
}

// ---- EPISTEMIC ----
let EP_INIT=false;
async function epAdd(){const k=$('#ep-k').value.trim();if(!k)return;const v=$('#ep-v').value==='true';
  const body={};body[k]=v;
  const r=await api('/api/epistemic',{action:EP_INIT?'add':'init',[EP_INIT?'new_info':'beliefs']:body});
  EP_INIT=true;$('#ep-k').value='';renderEp(r)}
async function epRevise(){renderEp(await api('/api/epistemic',{action:'revise'}))}
function epReset(){EP_INIT=false;$('#ep-out').textContent='{}';$('#ep-log').textContent=''}
function renderEp(r){if(r.error){$('#ep-log').textContent=r.error;return}
  $('#ep-out').textContent=JSON.stringify(r.beliefs,null,2);
  $('#ep-log').textContent=(r.action?('→ '+r.action+'  '):'')+(r.log||[]).join(' · ')}

// ---- BDI ----
async function bdiAdd(){const type=$('#bdi-type').value,text=$('#bdi-text').value.trim();if(!text)return;
  const r=await api('/api/bdi',{action:type,text});$('#bdi-text').value='';renderBdi(r)}
async function bdiExec(){const r=await api('/api/bdi',{action:'execute'});renderBdi(r);
  $('#bdi-log').style.display='block';$('#bdi-log').textContent=(r.log||[]).join('\n')||'(no intentions)'}
async function bdiReset(){renderBdi(await api('/api/bdi',{action:'reset'}));$('#bdi-log').style.display='none'}
function renderBdi(r){const li=a=>(a||[]).map(x=>`<li><span>${esc(x)}</span></li>`).join('')||'<li class="dim">—</li>';
  $('#bdi-b').innerHTML=li(r.beliefs);$('#bdi-d').innerHTML=li(r.desires);$('#bdi-i').innerHTML=li(r.intentions)}

// ---- MASTERMIND ----
async function mmStatus(){const s=await api('/api/mastermind');
  $('#mm-agents').innerHTML=(s.allowed_agents||[]).map(a=>`<span class="chip"><span class="tag">agent</span><span class="mono">${esc(a)}</span></span>`).join('')}
async function mmRun(){$('#mm-out').innerHTML='<span class="spin"></span> running…';
  const r=await api('/api/mastermind/run',{});
  if(r.error){$('#mm-out').innerHTML='<span class="dim">'+esc(r.error)+'</span>';return}
  $('#mm-out').innerHTML=`<div class="notice">Executed agents: <b>${(r.loaded_agents||[]).join(', ')||'none'}</b></div><h3>Data store</h3><pre>${esc(JSON.stringify(r.data_store,null,2))}</pre>`}

// ---- AUTONOMIZE ----
async function autoRun(){$('#auto-out').innerHTML='<span class="spin"></span> running resilient cycle…';
  const r=await api('/api/autonomize',{});
  $('#auto-out').textContent=r.error?r.error:('attempts allowed: '+r.attempts+'\n\n'+(r.log||[]).join('\n'))}

// ---- PREDICTION ----
async function predStatus(){const r=await api('/api/prediction');
  $('#pred-out').innerHTML=`<div class="row"><span class="tag">Predictor import</span> <b style="color:${r.predictor_importable?'var(--good)':'var(--bad)'}">${r.predictor_importable?'ok':'missing'}</b></div>
   <div class="row" style="margin-top:6px"><span class="tag">model file</span> <b style="color:${r.model_present?'var(--good)':'var(--warn)'}">${r.model_present?'present':'not found'}</b> <span class="dim mono">${esc(r.expected_path)}</span></div>
   <div class="notice" style="margin-top:12px">${esc(r.note)}</div>`}

// ---- MEMORY ----
async function memSave(){const r=await api('/api/memory/save',{});
  $('#mem-msg').textContent=r.error?r.error:('✓ saved '+r.saved);memList()}
async function memList(){const r=await api('/api/memory');
  if(!r.files||!r.files.length){$('#mem-out').innerHTML='<span class="dim small">No memory files yet in '+esc(r.folder)+'. Chat, then Save.</span>';return}
  $('#mem-out').innerHTML='<table><tr><th>file</th><th>turns</th><th>first instruction</th></tr>'+
    r.files.map(f=>`<tr><td class="mono">${esc(f.file)}</td><td>${f.turns}</td><td style="text-align:left">${esc((f.preview&&(f.preview.instruction||f.preview[0]))||'')||'—'}</td></tr>`).join('')+'</table>'}

// ---- ABOUT ----
async function aboutLoad(){const s=window._sys||await api('/api/status');
  $('#sysprompt').textContent=s.system_prompt||'';
  $('#fac-grid').innerHTML=Object.entries(s.faculties||{}).map(([k,v])=>
    `<div class="f"><div class="n">${k} <span class="dot ${v.ok?'ok':'bad'}" style="display:inline-block"></span></div><div class="d">${esc(v.detail)}</div></div>`).join('')}

// boot
status(); ollStatus(); renderNm(); setInterval(status,5000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print("=" * 60)
    print("  AGLM // Cognitive Console")
    print("  Autonomous General Learning Model — augmentation layer")
    print(f"  http://localhost:{port}")
    print("=" * 60)
    s = E.system_status()
    print(f"  Faculties online: {s['faculties_ok']}/{s['faculties_total']}")
    oll = E.ModelBackends.ollama_models()
    if oll.get("ok"):
        nloc = len(oll["local"]); ncl = len(oll["cloud"])
        print(f"  Ollama: {nloc} local + {ncl} cloud model(s)  [default: {oll['default']}]")
    else:
        print("  Ollama: offline (augmentation-only mode still works)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
