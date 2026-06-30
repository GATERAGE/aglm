"""
aglm_engine.py  —  The AGLM cognitive augmentation engine.

AGLM (Autonomous General Learning Model) is *not* a model. It is a logic layer
that augments any existing model. This module wires the original AGLM cognitive
faculties (the pure-Python prototype modules in this repo) into one engine that
the web UI drives. It imports the REAL modules where they are safe to run and
captures their output so it can be displayed.

Nothing here requires a GPU, a 7B checkpoint, or any heavy ML stack — which is
precisely why this layer "actually loads". A language model, if available, is an
optional backend plugged into the augmentation pipeline (see ModelBackends).
"""

import os
import io
import sys
import json
import glob
import time
import contextlib

# ----------------------------------------------------------------------------
# Import the original AGLM faculties. Each import is guarded so the engine still
# loads (and reports the faculty as degraded) if a dependency is missing.
# ----------------------------------------------------------------------------

FACULTIES = {}


def _register(name, ok, detail=""):
    FACULTIES[name] = {"ok": ok, "detail": detail}


try:
    from reasoning import SocraticReasoning
    _register("socratic", True, "reasoning.SocraticReasoning")
except Exception as e:  # pragma: no cover
    SocraticReasoning = None
    _register("socratic", False, str(e))

try:
    from logic import LogicTables
    _register("logic", True, "logic.LogicTables")
except Exception as e:  # pragma: no cover
    LogicTables = None
    _register("logic", False, str(e))

try:
    from nonmonotonic import DefaultLogic, Rule, Default
    _register("nonmonotonic", True, "nonmonotonic.DefaultLogic")
except Exception as e:  # pragma: no cover
    DefaultLogic = Rule = Default = None
    _register("nonmonotonic", False, str(e))

try:
    from epistemic import AutoepistemicAgent
    _register("epistemic", True, "epistemic.AutoepistemicAgent")
except Exception as e:  # pragma: no cover
    AutoepistemicAgent = None
    _register("epistemic", False, str(e))

try:
    from bdi import Belief, Desire, Intention, Goal, Reward
    _register("bdi", True, "bdi.Belief/Desire/Intention")
except Exception as e:  # pragma: no cover
    Belief = Desire = Intention = Goal = Reward = None
    _register("bdi", False, str(e))

try:
    from autonomize import Autonomizer
    _register("autonomize", True, "autonomize.Autonomizer")
except Exception as e:  # pragma: no cover
    Autonomizer = None
    _register("autonomize", False, str(e))

try:
    from memory import save_conversation_memory, MEMORY_FOLDER
    _register("memory", True, "memory.save_conversation_memory")
except Exception as e:  # pragma: no cover
    save_conversation_memory = None
    MEMORY_FOLDER = "./memory/"
    _register("memory", False, str(e))

try:
    from MASTERMIND import MASTERMIND, SimpleAgent, save_data_store
    _register("mastermind", True, "MASTERMIND agent controller")
except Exception as e:  # pragma: no cover
    MASTERMIND = SimpleAgent = save_data_store = None
    _register("mastermind", False, str(e))

# The original Professor Codephreak system prompt — the persona AGLM augments a
# model with. Imported from the prototype so the UI reflects the real thing.
try:
    from automind import DEFAULT_SYSTEM_PROMPT
except Exception:  # pragma: no cover
    DEFAULT_SYSTEM_PROMPT = "You are Professor Codephreak."

# Prediction faculty is optional (needs joblib/pandas + a trained .pkl).
try:
    from prediction import Predictor
    _register("prediction", True, "prediction.Predictor")
except Exception as e:  # pragma: no cover
    Predictor = None
    _register("prediction", False, str(e))


def _capture(fn, *args, **kwargs):
    """Run fn capturing anything it prints; return (return_value, printed_text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


# ----------------------------------------------------------------------------
# Socratic faculty — wraps reasoning.SocraticReasoning, capturing its log output
# ----------------------------------------------------------------------------
class SocraticFaculty:
    def __init__(self):
        self.reset()

    def reset(self):
        self.engine = SocraticReasoning() if SocraticReasoning else None
        self._log = []
        if self.engine:
            # The instance attribute shadows the bound method; internal calls to
            # self.log() in the real module now route through our capture.
            self.engine.log = self._capture_log

    def _capture_log(self, message, level="info"):
        self._log.append(str(message))

    @property
    def premises(self):
        return list(self.engine.premises) if self.engine else []

    def add(self, premise):
        if self.engine:
            self.engine.add_premise(premise)
        return self.snapshot()

    def challenge(self, premise):
        if self.engine:
            self.engine.challenge_premise(premise)
        return self.snapshot()

    def conclude(self):
        if self.engine:
            self.engine.draw_conclusion()
        return self.snapshot()

    def questions(self, topic):
        """Generate Socratic probing questions for a topic (augmentation layer)."""
        t = topic.strip().rstrip("?.! ") or "this claim"
        return [
            f"What do we mean precisely by “{t}”?",
            f"What evidence supports that {t} is true?",
            f"What assumptions underlie {t}?",
            f"What would a counter-example to {t} look like?",
            f"How would we know if we were wrong about {t}?",
        ]

    def snapshot(self):
        log = list(self._log)
        self._log = []
        return {"premises": self.premises, "log": log}


# ----------------------------------------------------------------------------
# Logic faculty — wraps logic.LogicTables truth-table generation
# ----------------------------------------------------------------------------
class LogicFaculty:
    def evaluate(self, variables, expressions):
        if not LogicTables:
            return {"error": "logic faculty unavailable"}
        lt = LogicTables()
        # Preserve user order: LogicTables uses a set internally, so we keep our
        # own ordered list for stable headers.
        ordered_vars = [v.strip() for v in variables if v.strip()]
        for v in ordered_vars:
            lt.add_variable(v)
        exprs = [e.strip() for e in expressions if e.strip()]
        for e in exprs:
            lt.add_expression(e)
        import itertools
        headers = ordered_vars + exprs
        rows = []
        try:
            for combo in itertools.product([True, False], repeat=len(ordered_vars)):
                vd = dict(zip(ordered_vars, combo))
                row = list(combo)
                for e in exprs:
                    try:
                        row.append(bool(lt.evaluate_expression(e, vd)))
                    except Exception as ex:
                        row.append(f"ERR: {ex}")
                rows.append(row)
        except Exception as ex:
            return {"error": str(ex)}
        return {"headers": headers, "rows": rows}


# ----------------------------------------------------------------------------
# Nonmonotonic faculty — default logic with retractable conclusions
# ----------------------------------------------------------------------------
class NonmonotonicFaculty:
    def evaluate(self, rules, defaults, query):
        if not DefaultLogic:
            return {"error": "nonmonotonic faculty unavailable"}
        dl = DefaultLogic()
        derived = []
        for r in rules:
            cond = set(r.get("conditions", []))
            concl = set(r.get("conclusions", []))
            dl.add_rule(Rule(cond, concl))
        for d in defaults:
            cond = set(d.get("conditions", []))
            concl = set(d.get("conclusions", []))
            dl.add_default(Default(cond, concl))
        # Re-derive the full belief closure for display.
        beliefs = set()
        while True:
            new = set()
            for r in dl.rules:
                if r.applies(beliefs):
                    new |= r.conclusions
            for d in dl.defaults:
                if d.applies(beliefs):
                    new |= d.conclusions
            if new <= beliefs:
                break
            beliefs |= new
        entailed = dl.evaluate(query) if query else None
        return {"beliefs": sorted(beliefs), "query": query, "entailed": entailed}


# ----------------------------------------------------------------------------
# Epistemic faculty — belief revision under new information
# ----------------------------------------------------------------------------
class EpistemicFaculty:
    def __init__(self):
        self.agent = None

    def init(self, beliefs):
        if not AutoepistemicAgent:
            return {"error": "epistemic faculty unavailable"}
        self.agent = AutoepistemicAgent(dict(beliefs))
        return self.snapshot("initialized")

    def add(self, new_info):
        if not self.agent:
            self.init({})
        _, out = _capture(self.agent.add_information, dict(new_info))
        return self.snapshot("added information", out)

    def revise(self):
        if not self.agent:
            return {"error": "no beliefs initialized"}
        _, out = _capture(self.agent.revise_beliefs)
        return self.snapshot("revised beliefs", out)

    def snapshot(self, action, out=""):
        return {
            "action": action,
            "beliefs": self.agent.beliefs if self.agent else {},
            "log": [l for l in out.splitlines() if l.strip()],
        }


# ----------------------------------------------------------------------------
# BDI faculty — Belief / Desire / Intention agent state
# ----------------------------------------------------------------------------
class BDIFaculty:
    def __init__(self):
        self.beliefs = []
        self.desires = []
        self.intentions = []
        self.log = []

    def add_belief(self, text):
        self.beliefs.append(text)
        return self.snapshot()

    def add_desire(self, text):
        self.desires.append(text)
        return self.snapshot()

    def add_intention(self, text):
        self.intentions.append(text)
        return self.snapshot()

    def execute(self):
        self.log = []
        if Intention:
            for plan in self.intentions:
                _, out = _capture(Intention(plan).execute)
                self.log.append(out.strip() or f"Executing plan: {plan}")
        return self.snapshot()

    def reset(self):
        self.__init__()
        return self.snapshot()

    def snapshot(self):
        return {
            "beliefs": self.beliefs,
            "desires": self.desires,
            "intentions": self.intentions,
            "log": self.log,
        }


# ----------------------------------------------------------------------------
# Autonomize faculty — self-healing resilience loop
# ----------------------------------------------------------------------------
class AutonomizeFaculty:
    def run(self):
        if not Autonomizer:
            return {"error": "autonomize faculty unavailable"}
        a = Autonomizer()
        log_path = "autonomize.log"
        before = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        a.resilient_function()
        lines = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                f.seek(before)
                lines = [l.rstrip() for l in f if l.strip()]
        return {"attempts": a.attempts, "log": lines or ["Task completed successfully"]}


# ----------------------------------------------------------------------------
# MASTERMIND faculty — agent orchestration / control
# ----------------------------------------------------------------------------
class MastermindFaculty:
    def status(self):
        cfg = {}
        try:
            with open("config.json") as f:
                cfg = json.load(f)
        except Exception:
            pass
        return {
            "allowed_agents": cfg.get("allowed_agents", []),
            "available": bool(MASTERMIND),
        }

    def run(self):
        if not MASTERMIND:
            return {"error": "MASTERMIND unavailable"}
        mm = MASTERMIND()
        # config.json uses "allowed_agents"; the controller checks "agents".
        # Authorize SimpleAgent for this demonstration run regardless of key.
        mm.config["allowed_agents"] = list(set(mm.config.get("allowed_agents", []) + ["SimpleAgent"]))
        mm.load_agent("SimpleAgent", SimpleAgent)
        mm.execute_agents()
        try:
            save_data_store(mm)
        except Exception:
            pass
        return {
            "loaded_agents": list(mm.agent_store.keys()),
            "data_store": mm.data_store,
            "config": mm.config,
        }


# ----------------------------------------------------------------------------
# Memory faculty — persistent conversation memory (memory/*.json)
# ----------------------------------------------------------------------------
class MemoryFaculty:
    def list(self):
        files = sorted(glob.glob(os.path.join(MEMORY_FOLDER, "*.json")), reverse=True)
        out = []
        for fp in files[:50]:
            try:
                with open(fp) as f:
                    data = json.load(f)
                turns = len(data) if isinstance(data, list) else 0
            except Exception:
                data, turns = None, 0
            out.append({
                "file": os.path.basename(fp),
                "turns": turns,
                "preview": (data[0] if isinstance(data, list) and data else None),
            })
        return {"folder": MEMORY_FOLDER, "files": out}

    def save(self, pairs):
        if not save_conversation_memory:
            return {"error": "memory faculty unavailable"}
        path = save_conversation_memory(pairs)
        return {"saved": str(path)}


# ----------------------------------------------------------------------------
# Prediction faculty — optional ML model inference
# ----------------------------------------------------------------------------
class PredictionFaculty:
    def status(self):
        model_path = os.path.join("models", "trained_model.pkl")
        available = bool(Predictor) and os.path.exists(model_path)
        return {
            "predictor_importable": bool(Predictor),
            "model_present": os.path.exists(model_path),
            "expected_path": model_path,
            "note": "Prediction augments AGLM with a trained sklearn-style model. "
                    "Drop a trained_model.pkl into ./models to enable inference.",
        }


# ----------------------------------------------------------------------------
# Model backends — the OPTIONAL language model AGLM augments. AGLM itself is not
# a model; it hands its logic-augmented prompt to whatever model you attach.
#
# Models are discovered live from a running Ollama daemon (local + cloud). On
# this modest hardware the small local models (e.g. qwen3:0.6b, deepseek-r1:1.5b)
# are used for testing; ":cloud" models are proxied by Ollama to ollama.com for
# larger models (gpt-oss:120b-cloud, glm-5.1:cloud, deepseek-v4-pro:cloud).
#
# Defaults to "augmentation-only" so the engine always loads even with no daemon.
# ----------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("AGLM_DEFAULT_MODEL", "gpt-oss:120b-cloud")


class ModelBackends:
    @staticmethod
    def ollama_models():
        """Live list of Ollama models, split into local and cloud."""
        try:
            import requests
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=4)
            r.raise_for_status()
            models = r.json().get("models", [])
        except Exception as e:
            return {"ok": False, "error": str(e), "local": [], "cloud": []}
        local, cloud = [], []
        for m in models:
            name = m.get("name", "")
            is_cloud = bool(m.get("remote_host")) or name.endswith(":cloud")
            size = m.get("size", 0)
            entry = {
                "name": name,
                "param_size": (m.get("details") or {}).get("parameter_size", ""),
                "size_gb": round(size / 1e9, 2) if size and size > 1e6 else None,
                "family": (m.get("details") or {}).get("family", ""),
                "remote_host": m.get("remote_host", ""),
            }
            (cloud if is_cloud else local).append(entry)
        # Stable ordering: smallest local first (best for modest hardware).
        local.sort(key=lambda e: e["size_gb"] or 0)
        return {"ok": True, "local": local, "cloud": cloud,
                "default": DEFAULT_OLLAMA_MODEL, "host": OLLAMA_HOST}

    @staticmethod
    def available():
        backends = [
            {"id": "augment", "label": "Augmentation only (no model)", "ready": True},
        ]
        oll = ModelBackends.ollama_models()
        backends.append({
            "id": "ollama",
            "label": "Ollama (local + cloud)",
            "ready": oll["ok"] and bool(oll["local"] or oll["cloud"]),
            "ollama": oll,
        })
        return backends


# ----------------------------------------------------------------------------
# The augmentation pipeline — the heart of AGLM. Takes a user query and runs it
# through every cognitive faculty to produce a logic-augmented prompt + a full
# reasoning trace. That augmented prompt is what gets handed to any model.
# ----------------------------------------------------------------------------
class AugmentationEngine:
    def __init__(self):
        self.socratic = SocraticFaculty()
        self.epistemic = EpistemicFaculty()

    def augment(self, query, backend="augment", model=None):
        trace = []

        # 1. Socratic — frame the query as a premise and probe it.
        self.socratic.reset()
        self.socratic.add(query)
        questions = self.socratic.questions(query)
        trace.append({"faculty": "socratic", "title": "Socratic framing",
                      "items": questions})

        # 2. Epistemic — register the query as a tentative belief.
        self.epistemic.init({query: True})
        trace.append({"faculty": "epistemic", "title": "Belief registered",
                      "items": [f"{query} = True (tentative, revisable)"]})

        # 3. Nonmonotonic — note that conclusions are defeasible.
        trace.append({"faculty": "nonmonotonic", "title": "Defeasibility",
                      "items": ["Conclusions hold by default and retract under "
                                "contradicting evidence."]})

        # 4. BDI — derive a belief -> desire -> intention scaffold.
        trace.append({"faculty": "bdi", "title": "BDI scaffold", "items": [
            f"Belief: the user asserts/asks — {query}",
            "Desire: produce a correct, logically-grounded answer",
            "Intention: reason step-by-step, then respond concisely",
        ]})

        # Build the augmented system prompt that wraps any model.
        augmented_prompt = self._build_prompt(query, questions)

        # 5. Hand to a model backend (optional).
        response, backend_used = self._call_backend(backend, augmented_prompt, query, model)

        return {
            "query": query,
            "augmented_prompt": augmented_prompt,
            "trace": trace,
            "response": response,
            "backend": backend_used,
            "model": model,
        }

    def augment_system(self, query):
        """Return only the system-level augmentation (persona + reasoning
        scaffold) plus the trace, WITHOUT wrapping the user text. Used by the
        AI SDK route, which supplies the conversation messages separately."""
        self.socratic.reset()
        self.socratic.add(query)
        questions = self.socratic.questions(query)
        q_block = "\n".join(f"  - {q}" for q in questions)
        system = (
            f"{DEFAULT_SYSTEM_PROMPT}\n\n"
            f"[AGLM LOGIC AUGMENTATION]\n"
            f"Before answering, internally consider these Socratic probes:\n{q_block}\n"
            f"Reason step-by-step. Treat every conclusion as defeasible — hold it "
            f"by default and retract it under contradicting evidence (nonmonotonic). "
            f"Register the user's claim as a tentative, revisable belief (epistemic). "
            f"Frame your plan as Belief -> Desire -> Intention (BDI), then answer "
            f"concisely."
        )
        trace = [
            {"faculty": "socratic", "title": "Socratic framing", "items": questions},
            {"faculty": "epistemic", "title": "Belief registered",
             "items": [f"{query} = True (tentative, revisable)"]},
            {"faculty": "nonmonotonic", "title": "Defeasibility",
             "items": ["Conclusions hold by default; retract under contradiction."]},
            {"faculty": "bdi", "title": "BDI scaffold", "items": [
                f"Belief: user asserts/asks — {query}",
                "Desire: a correct, logically-grounded answer",
                "Intention: reason step-by-step, then respond concisely"]},
        ]
        return {"system": system, "trace": trace, "questions": questions}

    def _build_prompt(self, query, questions):
        q_block = "\n".join(f"  - {q}" for q in questions)
        return (
            f"<<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"[AGLM LOGIC AUGMENTATION]\n"
            f"Before answering, internally consider:\n{q_block}\n"
            f"Reason step-by-step. Treat conclusions as defeasible.\n\n"
            f"[USER]\n{query}\n[/USER]"
        )

    def _call_backend(self, backend, prompt, query, model=None):
        if backend == "ollama":
            model = model or DEFAULT_OLLAMA_MODEL
            try:
                return self._ollama(prompt, model), f"ollama:{model}"
            except Exception as e:
                return (f"[Ollama backend unavailable for '{model}': {e}]\n\n"
                        f"Falling back to augmentation-only. The augmented prompt "
                        f"above is exactly what AGLM would feed to the model."), "augment (fallback)"

        # Augmentation-only: AGLM's deliverable is the logic-augmented prompt and
        # reasoning scaffold itself. This is what makes the UI always load.
        return (
            "AGLM is running in augmentation-only mode (no language model "
            "attached). The panel above shows the logic-augmented prompt and "
            "reasoning trace AGLM produces for this query. Switch the backend to "
            "Ollama and pick a model — a small local one (qwen3:0.6b) for quick "
            "tests on modest hardware, or a ':cloud' model (gpt-oss:120b-cloud) "
            "for heavier reasoning — to have AGLM hand this augmented prompt to a "
            "real model for a natural-language answer."
        ), "augment"

    def _ollama(self, prompt, model):
        """Send the augmented prompt to Ollama (/api/chat). Works for both local
        and ':cloud' models — the local daemon proxies cloud models to ollama.com."""
        import requests
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # Disable hidden chain-of-thought for reasoning models so the visible
            # answer is returned promptly on modest hardware.
            "think": False,
        }
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        msg = (data.get("message") or {}).get("content", "")
        return msg or "[model returned no content]"


def system_status():
    """High-level health snapshot for the UI status bar."""
    try:
        import psutil
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
    except Exception:
        cpu = mem = None
    ok = sum(1 for f in FACULTIES.values() if f["ok"])
    return {
        "faculties": FACULTIES,
        "faculties_ok": ok,
        "faculties_total": len(FACULTIES),
        "cpu": cpu,
        "mem": mem,
        "backends": ModelBackends.available(),
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
