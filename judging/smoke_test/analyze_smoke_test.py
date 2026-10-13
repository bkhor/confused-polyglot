#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).parent
ITEMS_PATH = HERE / "items.yaml"
RESULTS_PATH = HERE / "results" / "raw_results.jsonl"

MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
DISCRIMINATING_GROUPS = {"boundary", "confabulation", "decoupling", "trap"}


def load_items():
    with open(ITEMS_PATH) as f:
        items = yaml.safe_load(f)
    return {it["id"]: it for it in items}


def load_results():
    rows = []
    with open(RESULTS_PATH) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def join_rows(rows):
    records = {}
    for row in rows:
        key = (row["item_id"], row["model"], row["run_idx"])
        rec = records.setdefault(key, {"item_id": row["item_id"], "model": row["model"], "run_idx": row["run_idx"]})
        if row["call_type"] == "outcome":
            rec["outcome_ok"] = row.get("correct_tool", False) and row["error"] is None
            rec["outcome_category"] = row["parsed"]["outcome_category"] if rec["outcome_ok"] else None
        else:
            rec["comp_ok"] = row.get("correct_tool", False) and row["error"] is None
            if rec["comp_ok"]:
                rec["gist"] = row["parsed"]["gist_recovery"]
                rec["instruction"] = row["parsed"]["instruction_recovery"]
            else:
                rec["gist"] = rec["instruction"] = None
    return list(records.values())


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def report_success_rate(rows):
    section("1. API / tool-invocation success rate")
    for model in MODELS:
        for call_type in ("outcome", "comprehension"):
            subset = [r for r in rows if r["model"] == model and r["call_type"] == call_type]
            if not subset:
                continue
            ok = sum(1 for r in subset if r.get("correct_tool") and r["error"] is None)
            errors = sum(1 for r in subset if r["error"] is not None)
            not_invoked = sum(1 for r in subset if r["error"] is None and not r["tool_invoked"])
            wrong_tool = sum(
                1 for r in subset if r["error"] is None and r["tool_invoked"] and not r.get("correct_tool")
            )
            print(
                f"  {model:22s} {call_type:14s} ok={ok}/{len(subset)}  errors={errors}  "
                f"tool_not_invoked={not_invoked}  wrong_tool={wrong_tool}"
            )


def majority(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, 0, 0
    counts = Counter(values)
    winner, count = counts.most_common(1)[0]
    return winner, count, len(values)


def report_self_consistency(records, items):
    section("2. Self-consistency per (item, model)")
    by_item_model = {}
    for rec in records:
        by_item_model.setdefault((rec["item_id"], rec["model"]), []).append(rec)

    for (item_id, model), recs in sorted(by_item_model.items()):
        outcomes = [r.get("outcome_category") for r in recs if r.get("outcome_ok")]
        winner, count, n = majority(outcomes)
        agree_str = f"{count}/{n}" if n else "no successful runs"

        gists = [r["gist"] for r in recs if r.get("comp_ok") and r["gist"] is not None]
        instrs = [r["instruction"] for r in recs if r.get("comp_ok") and r["instruction"] is not None]
        null_count = sum(1 for r in recs if r.get("comp_ok") and r["gist"] is None)
        gist_range = (max(gists) - min(gists)) if len(gists) > 1 else (0.0 if gists else None)
        instr_range = (max(instrs) - min(instrs)) if len(instrs) > 1 else (0.0 if instrs else None)

        flag = " <-- INCONSISTENT" if n and count < n else ""
        print(
            f"  item {item_id:2d} [{items[item_id]['group']:13s}] {model:22s} "
            f"outcome={agree_str}({winner}){flag}  gist_range={gist_range}  instr_range={instr_range}  nulls={null_count}"
        )


def report_anchor_accuracy(records, items):
    section("3. Anchor accuracy (must be clean)")
    anchor_ids = [i for i, it in items.items() if it["group"] == "anchor"]
    any_fail = False
    by_item_model = {}
    for rec in records:
        by_item_model.setdefault((rec["item_id"], rec["model"]), []).append(rec)

    for item_id in sorted(anchor_ids):
        for model in MODELS:
            recs = by_item_model.get((item_id, model), [])
            outcomes = [r.get("outcome_category") for r in recs if r.get("outcome_ok")]
            winner, count, n = majority(outcomes)
            expected = items[item_id]["expected_outcome"]
            ok = winner == expected
            if not ok:
                any_fail = True
            print(f"  item {item_id:2d} {model:22s} expected={expected:22s} got={winner}  {'PASS' if ok else 'FAIL <---'}")

    if any_fail:
        print("\n  *** ANCHOR FAILURE DETECTED ***")
        print("  At least one model missed an unambiguous case. Per the decision rule,")
        print("  this means the rubric itself is broken, not the judge. Fix the rubric")
        print("  text and rerun the smoke test before proceeding to anything else.")
    else:
        print("\n  All anchors pass on all models.")
    return not any_fail


def report_divergence(records, items):
    section("4. Cross-model divergence by group")
    by_item_model = {}
    for rec in records:
        by_item_model.setdefault((rec["item_id"], rec["model"]), []).append(rec)

    groups = sorted(set(it["group"] for it in items.values()))
    for group in groups:
        print(f"\n  -- group: {group} --")
        for item_id, it in sorted(items.items()):
            if it["group"] != group:
                continue
            row = [f"item {item_id:2d} expected={it['expected_outcome']:22s}"]
            labels = []
            for model in MODELS:
                recs = by_item_model.get((item_id, model), [])
                outcomes = [r.get("outcome_category") for r in recs if r.get("outcome_ok")]
                winner, _, _ = majority(outcomes)
                labels.append(winner)
                row.append(f"{model.split('-')[1]}={winner}")
            disagreement = len(set(labels)) > 1
            print("  " + "  ".join(row) + ("   <-- MODELS DISAGREE" if disagreement else ""))


def report_aggregate_accuracy(records, items):
    section("5. Aggregate accuracy vs expected labels")
    by_item_model = {}
    for rec in records:
        by_item_model.setdefault((rec["item_id"], rec["model"]), []).append(rec)

    accuracy = {}
    for model in MODELS:
        total = correct = 0
        disc_total = disc_correct = 0
        for item_id, it in items.items():
            recs = by_item_model.get((item_id, model), [])
            outcomes = [r.get("outcome_category") for r in recs if r.get("outcome_ok")]
            winner, _, _ = majority(outcomes)
            total += 1
            is_correct = winner == it["expected_outcome"]
            correct += is_correct
            if it["group"] in DISCRIMINATING_GROUPS:
                disc_total += 1
                disc_correct += is_correct
        accuracy[model] = {
            "overall": correct / total if total else 0,
            "discriminating": disc_correct / disc_total if disc_total else 0,
        }
        print(
            f"  {model:22s} overall={correct}/{total} ({accuracy[model]['overall']:.0%})  "
            f"discriminating-groups={disc_correct}/{disc_total} ({accuracy[model]['discriminating']:.0%})"
        )
    return accuracy


def report_decision(anchors_clean, accuracy):
    section("6. Decision recommendation")
    if not anchors_clean:
        print("  Anchor failures present. Do NOT make a model decision from this run.")
        print("  Fix the rubric and rerun the smoke test first.")
        return

    haiku = accuracy["claude-haiku-5-5"]["discriminating"]
    sonnet = accuracy["claude-sonnet-5-5"]["discriminating"]
    opus = accuracy["claude-opus-5-5"]["discriminating"]
    margin = 0.05

    print(f"  Discriminating-group accuracy: Haiku={haiku:.0%}  Sonnet={sonnet:.0%}  Opus={opus:.0%}")

    if haiku >= opus - margin:
        print("  -> Haiku tracks Opus closely. Recommend: use Haiku, cheapest tier wins.")
    elif sonnet >= opus - margin:
        print("  -> Haiku diverges from Opus; Sonnet tracks Opus. Recommend: use Sonnet.")
    elif opus - sonnet > margin and opus - haiku > margin:
        print("  -> Both cheaper tiers diverge meaningfully from Opus on the hard groups.")
        print("     Recommend: inspect section 4 for which specific items diverge.")
        print("     If divergence concentrates on the boundary group, consider a split-judge:")
        print("     Opus for the outcome call, a cheaper tier for comprehension scoring.")
    else:
        print("  -> No clear winner from this heuristic. Inspect sections 3-5 manually.")

    print("\n  Reminder: passing this test means 'proceed to the real corpus + real kappa")
    print("  validation,' not 'the judge is validated.' Real kappa validation against")
    print("  150-200 hand-labeled real responses is still the actual quality gate.")


def main():
    items = load_items()
    rows = load_results()
    if not rows:
        print(f"No results found at {RESULTS_PATH}. Run run_smoke_test.py first.")
        return
    records = join_rows(rows)

    report_success_rate(rows)
    report_self_consistency(records, items)
    anchors_clean = report_anchor_accuracy(records, items)
    report_divergence(records, items)
    accuracy = report_aggregate_accuracy(records, items)
    report_decision(anchors_clean, accuracy)


if __name__ == "__main__":
    main()
