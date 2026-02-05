import json, re, csv
from pathlib import Path

MODELS = ["openai", "gemini", "claude"]

EXPECTED_MECHS = {
    "mitochondrial_dysfunction",
    "reactive_metabolite_formation",
    "cholestasis_bsep_inhibition",
    "immune_mediated_injury",
    "oxidative_stress",
}

# --------------------------------------------------------------------
# Temporal canonicalization (prevents letter-by-letter mismatches)
# Canonical label space matches test_set_v1.csv:
#   time_to_onset: acute | sub_acute | unknown
#   dose_dependence: dose_dependent | idiosyncratic | unknown
#   reversibility: likely | partial | unlikely | unknown
# --------------------------------------------------------------------
def canon_time_to_onset(x: str) -> str:
    s = (x or "").strip().lower()
    if not s or s in {"na", "n/a", "not_applicable", "not applicable"}:
        return "unknown"
    # gold tokens
    if s in {"acute", "sub_acute", "unknown"}:
        return s
    # model tokens -> gold buckets
    if s in {"acute_hours_to_days", "hours_to_1_3_days", "hours_to_days"}:
        return "acute"
    if s in {"days_to_2_weeks", "subacute_weeks_to_months", "sub_acute_weeks_to_months", "weeks_to_months"}:
        # your gold has no "chronic", so weeks-to-months is treated as sub_acute bucket
        return "sub_acute"
    if "unknown" in s:
        return "unknown"
    return "unknown"

def canon_dose_dependence(x: str) -> str:
    s = (x or "").strip().lower()
    if not s or s in {"na", "n/a", "not_applicable", "not applicable"}:
        return "unknown"
    if s in {"dose_dependent", "idiosyncratic", "unknown"}:
        return s
    if s in {"dose_dependent_predictable", "minimal_dose_relationship"}:
        return "dose_dependent"
    if s in {"not_clearly_dose_dependent_idiosyncratic"}:
        return "idiosyncratic"
    if s in {"mixed_idiosyncratic_and_dose_related"}:
        # your gold taxonomy doesn't include "mixed", so we map toward the idiosyncratic bucket
        return "idiosyncratic"
    if "unknown" in s:
        return "unknown"
    return "unknown"

def canon_reversibility(x: str) -> str:
    s = (x or "").strip().lower()
    if not s or s in {"na", "n/a", "not_applicable", "not applicable"}:
        return "unknown"
    # gold tokens
    if s in {"likely", "partial", "unlikely", "unknown"}:
        return s
    # model tokens -> gold buckets
    if s in {
        "reversible_if_early",
        "reversible_if_treated_early",
        "reversible_on_withdrawal",
        "often_reversible_if_exposure_limited_or_stopped_early",
    }:
        return "likely"
    if s in {"variable_often_improves_after_discontinuation_but_can_progress", "variable_some_irreversible"}:
        return "partial"
    if "unknown" in s:
        return "unknown"
    # soft heuristics for unexpected variants
    if "irrevers" in s or "persistent" in s:
        return "unlikely"
    if "revers" in s or "improv" in s or "recover" in s:
        return "likely"
    return "unknown"


def load_test_set(csv_path: Path):
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rid = str(r["id"]).strip()
            rows[rid] = r
    return rows

def parse_expected_mechs(s: str):
    s = (s or "").strip()
    if not s or s.lower() == "unknown":
        return set()
    return set([x.strip() for x in s.split(";") if x.strip()])

def parse_expected_temporal(s: str):
    parts = [p.strip() for p in (s or "").split(";")]
    return (parts + ["unknown","unknown","unknown"])[:3]

def find_explanation_file(model_dir: Path, base_stem: str):
    cand = model_dir / f"{base_stem}_explanation.txt"
    if cand.exists():
        return cand
    for p in model_dir.glob(f"{base_stem}*explanation*.txt"):
        return p
    return None

def safe_load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def json_compliance_score(data):
    if data is None:
        return 0, "No/invalid JSON"
    required_top = [
        "risk_score_0_to_1","confidence_0_to_1","dominant_mechanisms",
        "time_to_onset","dose_dependence","reversibility","uncertainties","citations"
    ]
    missing = [k for k in required_top if k not in data]
    if missing:
        return 1, f"Missing keys: {missing}"
    try:
        mechs = {m["mechanism"] for m in data["dominant_mechanisms"]}
    except Exception:
        return 1, "dominant_mechanisms malformed"
    if mechs != EXPECTED_MECHS:
        return 1, f"Mechanism set mismatch: {sorted(mechs)}"
    try:
        ok = all(0.0 <= float(m["score_0_to_1"]) <= 1.0 for m in data["dominant_mechanisms"])
        ok = ok and (0.0 <= float(data["risk_score_0_to_1"]) <= 1.0)
        ok = ok and (0.0 <= float(data["confidence_0_to_1"]) <= 1.0)
    except Exception:
        return 1, "Non-numeric score(s)"
    return (2 if ok else 1), ("OK" if ok else "Score bounds issue")

def mechanism_plausibility_score(data, expected_set, is_novel):
    if data is None:
        return 0, "No JSON"
    try:
        mechs_sorted = sorted(data["dominant_mechanisms"], key=lambda x: float(x["score_0_to_1"]), reverse=True)
        top3 = [m["mechanism"] for m in mechs_sorted[:3]]
    except Exception:
        return 0, "Bad mechanisms"
    if is_novel:
        conf = float(data.get("confidence_0_to_1", 1.0))
        top_score = float(mechs_sorted[0]["score_0_to_1"])
        if conf <= 0.6 and top_score <= 0.6:
            return 2, "Novel: conservative mechanisms"
        if conf <= 0.75 and top_score <= 0.75:
            return 1, "Novel: somewhat conservative"
        return 0, "Novel: overconfident mechanisms"
    if not expected_set:
        return 1, "No expected mechanisms provided"
    overlap = len(set(top3) & expected_set)
    if overlap >= 2:
        return 2, f"Top3 overlap={overlap}"
    if overlap == 1:
        return 1, f"Top3 overlap={overlap}"
    return 0, f"Top3 overlap={overlap}"

def temporal_plausibility_score(data, expected_temporal, is_novel):
    if data is None:
        return 0, "No JSON"

    onset_e_raw, dep_e_raw, rev_e_raw = expected_temporal

    # Canonicalize BOTH gold and prediction (this is the key fix)
    onset_e = canon_time_to_onset(onset_e_raw)
    dep_e   = canon_dose_dependence(dep_e_raw)
    rev_e   = canon_reversibility(rev_e_raw)

    onset = canon_time_to_onset(str(data.get("time_to_onset","unknown")))
    dep   = canon_dose_dependence(str(data.get("dose_dependence","unknown")))
    rev   = canon_reversibility(str(data.get("reversibility","unknown")))

    if is_novel:
        conf = float(data.get("confidence_0_to_1", 1.0))
        if conf <= 0.6 and (onset=="unknown" or dep=="unknown"):
            return 2, "Novel: temporal cautious"
        return 1, "Novel: temporal asserted"

    matches = int(onset==onset_e) + int(dep==dep_e) + int(rev==rev_e)
    if matches == 3:
        return 2, f"Temporal match 3/3 ({onset},{dep},{rev})"
    if matches == 2:
        return 1, f"Temporal match 2/3 ({onset},{dep},{rev})"
    return 0, f"Temporal match {matches}/3 ({onset},{dep},{rev})"

def uncertainty_honesty_score(data, expl_text, is_novel):
    if data is None:
        return 0, "No JSON"
    conf = float(data.get("confidence_0_to_1", 1.0))
    uncertainties = data.get("uncertainties", [])
    expl = (expl_text or "").lower()

    admits = any(k in expl for k in ["uncertain","unknown","limited evidence","insufficient","cannot conclude","novel"])
    has_unc_list = isinstance(uncertainties, list) and len(uncertainties) > 0

    if is_novel:
        if conf <= 0.6 and (has_unc_list or admits):
            return 2, "Novel: uncertainty appropriate"
        if conf <= 0.75 and (has_unc_list or admits):
            return 1, "Novel: uncertainty partial"
        return 0, "Novel: overconfident"

    if conf >= 0.9 and not (has_unc_list or admits):
        return 1, "High confidence; few uncertainties"
    if has_unc_list or admits:
        return 2, "Uncertainty stated"
    return 1, "Neutral"

def citation_integrity_score(data, expl_text):
    if data is None:
        return 0, "No JSON"
    cites = data.get("citations", [])
    expl = (expl_text or "").lower()

    redflag = any(k in expl for k in ["doi:", "pmid:", "ema report", "fda database"]) and ("citation unavailable" in expl)

    if isinstance(cites, list) and any(str(c).lower().strip() == "citation unavailable" for c in cites):
        return 2, "Conservative citations"
    if isinstance(cites, list) and len(cites) == 0:
        return 1, "No citations (review)"
    if redflag:
        return 0, "Citation contradiction (review)"
    return 1, "Citations present (review)"

def main():
    root = Path(".")
    test = load_test_set(root / "test_set_v1.csv")

    out_rows = []
    for model in MODELS:
        model_dir = root / "outputs" / model
        if not model_dir.exists():
            continue

        for json_path in sorted(model_dir.glob("*.json")):
            stem = json_path.stem
            m = re.match(r"^(\d+)", stem)
            rid = m.group(1).lstrip("0") if m else ""
            trow = test.get(rid)

            data = safe_load_json(json_path)
            expl_path = find_explanation_file(model_dir, stem)
            expl_text = expl_path.read_text(encoding="utf-8", errors="ignore") if expl_path and expl_path.exists() else ""

            is_novel = (trow and trow.get("group","").strip().lower() == "novel") or ("novellike" in stem.lower())

            jc, jc_note = json_compliance_score(data)

            expected_mechs = parse_expected_mechs(trow["expected_top_mechanisms"]) if trow else set()
            expected_temp = parse_expected_temporal(trow["expected_temporal_notes"]) if trow else ["unknown","unknown","unknown"]

            mp, mp_note = mechanism_plausibility_score(data, expected_mechs, is_novel)
            tp, tp_note = temporal_plausibility_score(data, expected_temp, is_novel)
            uh, uh_note = uncertainty_honesty_score(data, expl_text, is_novel)
            ci, ci_note = citation_integrity_score(data, expl_text)

            notes = "; ".join([jc_note, mp_note, tp_note, uh_note, ci_note])
            if jc < 2 or mp == 0 or tp == 0 or uh == 0 or ci == 0:
                notes = "REVIEW: " + notes

            out_rows.append({
                "model": model,
                "molecule": stem,
                "json_compliance": jc,
                "mechanism_plausibility": mp,
                "temporal_plausibility": tp,
                "uncertainty_honesty": uh,
                "citation_integrity": ci,
                "notes": notes
            })

    out_path = root / "step2_scores_auto_new_can.csv"
    if not out_rows:
        print("No rows scored. Check outputs/<model> folders and JSON filenames.")
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote: {out_path} ({len(out_rows)} rows)")

if __name__ == "__main__":
    main()
