# app_eval_viewer.py
# Streamlit LLM Evaluation Viewer
# - Reads step2_scores_auto.csv for scores + notes
# - Select model + molecule
# - Loads JSON output from outputs/<model>/*.json (supports fuzzy names like *_gem.json)
# - Loads explanation from:
#     1) "explanation" field inside JSON, OR
#     2) sidecar: outputs/<model>/*_explanation.txt (fuzzy matched), OR
#     3) PART B: inside combined txt/md (fallback)
# - Makes the "REVIEW: OK; ..." line more intuitive (badge + chips + raw expander)
# - Renders explanation as bullet points (auto-sentence split if needed)
# - Shows citation count (and clarifies citation_integrity is quality-based)

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="LLM Evaluation Viewer", layout="wide")

ROOT = Path(__file__).resolve().parent
SCORES_PATH = ROOT / "step2_scores_auto_new_can.csv"
OUTPUTS_DIR = ROOT / "outputs"


# ---------------------------
# Helpers: safe I/O
# ---------------------------
def safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def norm(s: str) -> str:
    """Normalize for matching (case/underscores/punct)."""
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


# ---------------------------
# Helpers: JSON extraction from txt
# ---------------------------
def extract_json_from_text(raw: str):
    """
    Tries to find the first JSON object in a text blob.
    Works for "PART A: { ... }" style outputs.
    """
    if not raw:
        return None, "Empty text."

    raw_stripped = raw.strip()

    # If whole blob is JSON
    if raw_stripped.startswith("{") and raw_stripped.endswith("}"):
        try:
            return json.loads(raw_stripped), None
        except Exception:
            pass

    # Heuristic: first '{' to last '}' slice
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = raw[first : last + 1]
        try:
            return json.loads(candidate), None
        except Exception as e:
            return None, f"Found braces but JSON parse failed: {e}"

    return None, "No JSON object found in text."


# ---------------------------
# Helpers: Review parsing & display
# ---------------------------
def parse_review_line(note: str):
    """
    Parses:
    'REVIEW: OK; Top3 overlap=1; Temporal match 0/3; Uncertainty stated; Citations present (review)'
    """
    out = {
        "status": None,
        "top3": None,        # int
        "temporal": None,    # (k,total)
        "uncertainty": None, # bool
        "citations": None,   # bool
        "raw": note or "",
    }
    if not note:
        return out

    m = re.search(r"REVIEW:\s*([A-Za-z]+)", note)
    if m:
        out["status"] = m.group(1).upper()

    m = re.search(r"Top3 overlap\s*=\s*(\d+)", note)
    if m:
        out["top3"] = int(m.group(1))

    m = re.search(r"Temporal match\s*(\d+)\s*/\s*(\d+)", note)
    if m:
        out["temporal"] = (int(m.group(1)), int(m.group(2)))

    out["uncertainty"] = bool(re.search(r"Uncertainty\s+stated", note, re.IGNORECASE))
    out["citations"] = bool(re.search(r"Citations\s+present", note, re.IGNORECASE))
    return out


def review_badge(status: str):
    if status == "OK":
        return "🟢 Review: OK"
    if status in ("WARN", "WARNING"):
        return "🟡 Review: WARN"
    if status in ("FAIL", "BAD", "ERROR"):
        return "🔴 Review: FAIL"
    return "🟦 Review"


# ---------------------------
# Helpers: find best matching files
# ---------------------------
def find_best_json_file(model_dir: Path, molecule: str, model_name: str):
    """
    Finds best matching json even if named like:
      01_acetaminophen_gem.json  vs molecule=01_Acetaminophen
    """
    target = norm(molecule)
    json_files = list(model_dir.glob("*.json"))
    if not json_files:
        return None

    best = None
    best_score = -10

    for p in json_files:
        stem_n = norm(p.stem)

        # Skip "explanation" JSON files if any appear
        if "explanation" in stem_n:
            continue

        score = 0
        if target and target in stem_n:
            score += 5
        if norm(model_name) and norm(model_name) in stem_n:
            score += 1

        # Bonus if stem starts with same prefix number like 06_
        mol_prefix = re.match(r"^\s*(\d+)", str(molecule))
        if mol_prefix:
            prefix = mol_prefix.group(1)
            if stem_n.startswith(prefix):
                score += 2

        if score > best_score:
            best_score = score
            best = p

    return best if best_score >= 3 else None


def find_sidecar_explanation(model_dir: Path, molecule: str):
    """
    Finds *explanation*.txt like:
      06_Metformin_explanation.txt
      01_acetaminophen_explanation.txt
    """
    target = norm(molecule)
    txts = list(model_dir.glob("*explanation*.txt"))
    if not txts:
        return None

    best = None
    best_score = -10

    for p in txts:
        s = norm(p.stem)
        score = 0
        if target and target in s:
            score += 5
        if "explanation" in s:
            score += 1

        mol_prefix = re.match(r"^\s*(\d+)", str(molecule))
        if mol_prefix:
            prefix = mol_prefix.group(1)
            if s.startswith(prefix):
                score += 2

        if score > best_score:
            best_score = score
            best = p

    return best if best_score >= 3 else None


# ---------------------------
# Helpers: explanation bullets
# ---------------------------
def render_explanation_bullets(expl: str):
    expl = (expl or "").strip()
    if not expl:
        st.info("No explanation found.")
        return

    # If already bullet-form, keep it
    bullet_lines = []
    for line in expl.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(\-|\*|\d+\.)\s+", line):
            bullet_lines.append(re.sub(r"^(\-|\*|\d+\.)\s+", "", line).strip())

    if bullet_lines:
        st.markdown("\n".join([f"- {b}" for b in bullet_lines]))
        return

    # Otherwise, split into sentences and bullet them
    sentences = re.split(r"(?<=[.!?])\s+", expl)
    sentences = [s.strip() for s in sentences if s.strip()]
    st.markdown("\n".join([f"- {s}" for s in sentences]))


# ---------------------------
# Helpers: citation counting (from schema JSON)
# ---------------------------
def count_citations_in_schema(schema_json: dict) -> int:
    """
    Counts citations inside dominant_mechanisms[*].citation
    Adjust this if your schema stores citations elsewhere.
    """
    if not isinstance(schema_json, dict):
        return 0

    # Most likely: schema_json["dominant_mechanisms"] is list of dicts with "citation"
    mechs = schema_json.get("dominant_mechanisms", [])
    if not isinstance(mechs, list):
        return 0

    n = 0
    for m in mechs:
        if not isinstance(m, dict):
            continue
        c = m.get("citation", None)
        if isinstance(c, str) and c.strip() and c.strip().lower() != "citation unavailable":
            n += 1
    return n


# ---------------------------
# Loader: JSON + explanation
# ---------------------------
def load_output_for(model: str, molecule: str):
    """
    Returns: (schema_json, explanation_text, raw_text, path_used, error)
    """
    base = OUTPUTS_DIR / model
    if not base.exists():
        return None, None, None, None, f"Model folder not found: {base}"

    # 1) Prefer exact match JSON
    direct = base / f"{molecule}.json"
    if direct.exists():
        json_path = direct
    else:
        # 2) Fuzzy match JSON filename
        json_path = find_best_json_file(base, molecule, model)

    if json_path and json_path.exists():
        try:
            data = json.loads(safe_read_text(json_path))
        except Exception as e:
            return None, None, None, json_path, f"Could not parse JSON file: {e}"

        schema_json = data.get("json", data if isinstance(data, dict) else {})
        explanation = data.get("explanation", "")

        # If explanation missing inside JSON, try sidecar *_explanation.txt
        if not explanation:
            sidecar = find_sidecar_explanation(base, molecule)
            if sidecar and sidecar.exists():
                explanation = safe_read_text(sidecar).strip()

        return schema_json, explanation, None, json_path, None

    # 3) Fallback: combined txt/md with embedded JSON + PART B
    for ext in (".txt", ".md"):
        p = base / f"{molecule}{ext}"
        if p.exists():
            raw = safe_read_text(p)
            m = re.search(r"PART\s*B\s*:(.*)$", raw, flags=re.IGNORECASE | re.DOTALL)
            explanation = m.group(1).strip() if m else ""
            schema_json, err = extract_json_from_text(raw)
            return schema_json, explanation, raw, p, err

    return None, None, None, None, "No output file found for this selection."


# ---------------------------
# App UI
# ---------------------------
st.title("LLM Evaluation Viewer")
st.caption("Select a model + chemical to inspect score panel, JSON output, and explanation.")

if not SCORES_PATH.exists():
    st.error(f"Missing scores file: {SCORES_PATH.name} (expected at {SCORES_PATH})")
    st.stop()

scores_df = pd.read_csv(SCORES_PATH)

# Required columns
required_cols = {"model", "molecule"}
missing = required_cols - set(scores_df.columns)
if missing:
    st.error(f"CSV missing required columns: {sorted(missing)}. Found: {list(scores_df.columns)}")
    st.stop()

# Score columns (optional but expected)
score_cols = [
    "json_compliance",
    "mechanism_plausibility",
    "temporal_plausibility",
    "uncertainty_honesty",
    "citation_integrity",
]
for c in score_cols:
    if c not in scores_df.columns:
        scores_df[c] = 0

if "notes" not in scores_df.columns:
    scores_df["notes"] = ""

models = sorted(scores_df["model"].dropna().unique().tolist())

left, right = st.columns([1, 2])

with left:
    model = st.selectbox("Select LLM / Model", models)

    molecules = (
        scores_df.loc[scores_df["model"] == model, "molecule"]
        .dropna()
        .unique()
        .tolist()
    )
    molecules = sorted(molecules)
    molecule = st.selectbox("Select Molecule / Chemical", molecules)

# Pull row for selected pair
sel = scores_df[(scores_df["model"] == model) & (scores_df["molecule"] == molecule)]
if sel.empty:
    st.error("No matching row found in CSV for selected model+molecule.")
    st.stop()

row = sel.iloc[0]

# Load outputs early so we can show citation count in score panel
schema_json, explanation, raw_text, used_path, load_err = load_output_for(model, molecule)
cit_count = count_citations_in_schema(schema_json) if schema_json else 0

with right:
    st.subheader("Score panel (from step2_scores_new_can.csv)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("JSON compliance", int(row.get("json_compliance", 0)))
    c2.metric("Mechanism plausibility", int(row.get("mechanism_plausibility", 0)))
    c3.metric("Temporal plausibility", int(row.get("temporal_plausibility", 0)))
    c4.metric("Uncertainty honesty", int(row.get("uncertainty_honesty", 0)))
    c5.metric("Citation integrity", int(row.get("citation_integrity", 0)))

    # Explain the "why score=1 when citations exist" confusion
    st.caption(f"Citations found in JSON: **{cit_count}** (citation_integrity scores *quality*, not count)")

    note = str(row.get("notes", "")).strip()
    st.caption("Notes")
    if note:
        parsed = parse_review_line(note)

        st.markdown(f"**{review_badge(parsed['status'])}**")

        chips = []
        if parsed["top3"] is not None:
            chips.append(f"Top-3 overlap: **{parsed['top3']} / 3**")
        if parsed["temporal"] is not None:
            k, t = parsed["temporal"]
            chips.append(f"Temporal match: **{k} / {t}**")

        chips.append(f"Uncertainty: **{'stated' if parsed['uncertainty'] else 'missing'}**")
        chips.append(f"Citations: **{'present' if parsed['citations'] else 'missing'}**")

        st.write(" • ".join(chips))

        with st.expander("Raw review text"):
            st.write(parsed["raw"])
    else:
        st.info("No notes found for this row.")

st.divider()
st.subheader("LLM Output Viewer")

uploaded = st.file_uploader(
    "If the output file isn't in outputs/, upload it here (.json, .txt, .md)",
    type=["json", "txt", "md"],
)

# If user uploads, override loaded content
if uploaded is not None:
    content = uploaded.read().decode("utf-8", errors="replace")
    used_path = Path(uploaded.name)
    load_err = None

    if uploaded.name.lower().endswith(".json"):
        try:
            data = json.loads(content)
            schema_json = data.get("json", data if isinstance(data, dict) else {})
            explanation = data.get("explanation", "")
            raw_text = None
        except Exception as e:
            schema_json = None
            explanation = ""
            raw_text = content
            load_err = f"Uploaded JSON parse failed: {e}"
    else:
        schema_json, err = extract_json_from_text(content)
        load_err = err
        m = re.search(r"PART\s*B\s*:(.*)$", content, flags=re.IGNORECASE | re.DOTALL)
        explanation = m.group(1).strip() if m else ""
        raw_text = content

# Display path/error
if used_path:
    st.caption(f"Loaded from: `{used_path}`")
if load_err:
    st.warning(load_err)

colA, colB = st.columns(2)

with colA:
    st.markdown("### JSON output")
    if schema_json is not None:
        st.json(schema_json)
    else:
        st.info("No JSON available for this selection.")

with colB:
    st.markdown("### Explanation")
    render_explanation_bullets(explanation)

if raw_text:
    st.markdown("### Raw text (debug)")
    st.code(raw_text, language="text")
