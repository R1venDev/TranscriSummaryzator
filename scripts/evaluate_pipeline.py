#!/usr/bin/env python3
"""Evaluate text, speakers and meeting meaning against human gold data."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from benchmark import read_rttm, score as diarization_score
except ModuleNotFoundError:
    from scripts.benchmark import read_rttm, score as diarization_score


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for index, a in enumerate(left, 1):
        current = [index]
        for column, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (a != b)))
        previous = current
    return previous[-1]


def normalize_text(value):
    return re.sub(r"\s+", " ", re.sub(r"[^\w%]+", " ", str(value).casefold())).strip()


def transcript_text(path):
    path = Path(path)
    if path.suffix.casefold() != ".json":
        return path.read_text(encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("words"):
        return " ".join(str(item.get("text", "")) for item in payload["words"])
    return " ".join(str(item.get("text", "")) for item in payload.get("utterances", []))


def text_score(reference, hypothesis):
    ref = normalize_text(transcript_text(reference))
    hyp = normalize_text(transcript_text(hypothesis))
    ref_words, hyp_words = ref.split(), hyp.split()
    def class_error(pattern):
        expected = pattern.findall(ref)
        actual = pattern.findall(hyp)
        return round(edit_distance(expected, actual) / max(1, len(expected)), 6)
    return {
        "WER": round(edit_distance(ref_words, hyp_words) / max(1, len(ref_words)), 6),
        "CER": round(edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", ""))) / max(1, len(ref.replace(" ", ""))), 6),
        "reference_words": len(ref_words),
        "hypothesis_words": len(hyp_words),
        "number_error_rate": class_error(re.compile(r"\d+(?:[.,:]\d+)*%?")),
        "negation_error_rate": class_error(re.compile(r"\b(?:не|нет|нельзя|никогда|без)\b")),
        "technical_term_error_rate": class_error(re.compile(r"\b(?:bos|fbos|smc|order block|tradingview|binance|[a-zа-я]+\d+)\b")),
    }


def load_records(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("records", payload.get("facts", []))
    return {str(item.get("record_id", item.get("fact_id"))): item for item in values}


def f1(tp, fp, fn):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return precision, recall, 2 * precision * recall / max(1e-12, precision + recall)


def semantic_score(reference_path, hypothesis_path, alignment_path):
    reference, hypothesis = load_records(reference_path), load_records(hypothesis_path)
    reference_document = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    hypothesis_document = json.loads(Path(hypothesis_path).read_text(encoding="utf-8"))
    alignment = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
    matches = [item for item in alignment.get("matches", []) if item.get("supported") is True]
    matched_ref = {item["reference_id"] for item in matches if item.get("reference_id") in reference}
    matched_hyp = {item["hypothesis_id"] for item in matches if item.get("hypothesis_id") in hypothesis}
    claim_precision = len(matched_hyp) / max(1, len(hypothesis))
    claim_recall = len(matched_ref) / max(1, len(reference))
    claim_f1 = 2 * claim_precision * claim_recall / max(1e-12, claim_precision + claim_recall)
    type_hits = 0
    owner_tp = owner_fp = owner_fn = condition_tp = condition_fp = condition_fn = 0
    decision_tp = decision_fp = action_tp = action_fp = 0
    deadline_hits = deadline_total = number_hits = number_total = negation_hits = negation_total = 0
    question_hits = question_total = citation_tp = citation_fp = 0
    for match in matches:
        gold = reference.get(match.get("reference_id"))
        predicted = hypothesis.get(match.get("hypothesis_id"))
        if not gold or not predicted:
            continue
        type_hits += gold.get("kind", gold.get("type")) == predicted.get("kind", predicted.get("type"))
        gold_kind, predicted_kind = gold.get("kind", gold.get("type")), predicted.get("kind", predicted.get("type"))
        decision_tp += gold_kind == predicted_kind == "decision"
        decision_fp += predicted_kind == "decision" and gold_kind != "decision"
        action_tp += gold_kind == predicted_kind == "action"
        action_fp += predicted_kind == "action" and gold_kind != "action"
        gold_owners, predicted_owners = set(gold.get("assignees", [])), set(predicted.get("assignees", []))
        owner_tp += len(gold_owners & predicted_owners)
        owner_fp += len(predicted_owners - gold_owners)
        owner_fn += len(gold_owners - predicted_owners)
        condition_text = lambda item: normalize_text(item.get("text", item) if isinstance(item, dict) else item)
        gold_conditions = {condition_text(item) for item in gold.get("conditions", [])}
        predicted_conditions = {condition_text(item) for item in predicted.get("conditions", [])}
        condition_tp += len(gold_conditions & predicted_conditions)
        condition_fp += len(predicted_conditions - gold_conditions)
        condition_fn += len(gold_conditions - predicted_conditions)
        if gold.get("time_expression") is not None or predicted.get("time_expression") is not None:
            deadline_total += 1
            deadline_hits += normalize_text(gold.get("time_expression")) == normalize_text(predicted.get("time_expression"))
        gold_numbers = set(re.findall(r"\d+(?:[.,:]\d+)*", str(gold.get("statement", ""))))
        predicted_numbers = set(re.findall(r"\d+(?:[.,:]\d+)*", str(predicted.get("statement", ""))))
        if gold_numbers or predicted_numbers:
            number_total += 1; number_hits += gold_numbers == predicted_numbers
        gold_neg = bool(re.search(r"(?iu)(?:^|\W)(?:не|нет|нельзя|никогда|без)(?:\W|$)", str(gold.get("statement", ""))))
        predicted_neg = bool(re.search(r"(?iu)(?:^|\W)(?:не|нет|нельзя|никогда|без)(?:\W|$)", str(predicted.get("statement", ""))))
        negation_total += 1; negation_hits += gold_neg == predicted_neg
        if gold_kind == "question":
            question_total += 1; question_hits += gold.get("question_status") == predicted.get("question_status")
        gold_citations, predicted_citations = set(gold.get("evidence_ids", [])), set(predicted.get("evidence_ids", []))
        citation_tp += len(gold_citations & predicted_citations)
        citation_fp += len(predicted_citations - gold_citations)
    op, ore, of = f1(owner_tp, owner_fp, owner_fn)
    cp, cr, cf = f1(condition_tp, condition_fp, condition_fn)
    relation_key = lambda item: (item.get("relation"), item.get("source_record_id", item.get("source_event")), item.get("target_record_id", item.get("target_event")))
    gold_relations = {relation_key(item) for item in reference_document.get("relations", [])}
    predicted_relations = {relation_key(item) for item in hypothesis_document.get("relations", [])}
    unsupported_relations = predicted_relations - gold_relations
    return {
        "claim_precision": round(claim_precision, 6),
        "claim_recall": round(claim_recall, 6),
        "claim_F1": round(claim_f1, 6),
        "type_accuracy": round(type_hits / max(1, len(matches)), 6),
        "assignee_precision": round(op, 6),
        "assignee_recall": round(ore, 6),
        "assignee_F1": round(of, 6),
        "condition_precision": round(cp, 6),
        "condition_recall": round(cr, 6),
        "condition_F1": round(cf, 6),
        "decision_precision": round(decision_tp / max(1, decision_tp + decision_fp), 6),
        "action_precision": round(action_tp / max(1, action_tp + action_fp), 6),
        "deadline_accuracy": round(deadline_hits / max(1, deadline_total), 6),
        "number_accuracy": round(number_hits / max(1, number_total), 6),
        "negation_accuracy": round(negation_hits / max(1, negation_total), 6),
        "question_resolution_accuracy": round(question_hits / max(1, question_total), 6),
        "citation_precision": round(citation_tp / max(1, citation_tp + citation_fp), 6),
        "omission_rate": round(1 - claim_recall, 6),
        "unsupported_relation_rate": round(len(unsupported_relations) / max(1, len(predicted_relations)), 6),
        "gold_records": len(reference),
        "hypothesis_records": len(hypothesis),
        "supported_matches": len(matches),
    }


def resolve(base, value):
    return str((base / value).resolve()) if value else None


def evaluate(manifest):
    manifest_path = Path(manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    results = {"schema_version": 1, "dataset": payload.get("name"), "cases": {}, "systems": {}}
    if not payload.get("cases"):
        raise ValueError("В manifest нет эталонных случаев")
    totals = {}
    for case in payload["cases"]:
        case_id = case["id"]
        if case.get("reference_status") != "gold":
            raise ValueError(f"{case_id}: reference_status должен быть gold; автоматический черновик нельзя использовать как эталон")
        reference = case["reference"]
        case_result = {}
        for system_name, system in case.get("systems", {}).items():
            metrics = {
                "text": text_score(resolve(base, reference["transcript"]), resolve(base, system["transcript"])),
                "speaker": diarization_score(read_rttm(resolve(base, reference["rttm"])), read_rttm(resolve(base, system["rttm"]))),
                "meaning": semantic_score(
                    resolve(base, reference["semantics"]), resolve(base, system["semantics"]),
                    resolve(base, system["semantic_alignment"]),
                ),
            }
            metrics["error_attribution"] = {
                "asr_correct_summary_wrong": metrics["text"]["WER"] == 0 and metrics["meaning"]["claim_F1"] < 1,
                "asr_error_present": metrics["text"]["WER"] > 0,
                "speaker_error_with_assignee_error": metrics["speaker"]["DER"] > 0 and metrics["meaning"]["assignee_F1"] < 1,
                "summary_semantic_error": metrics["meaning"]["claim_F1"] < 1 or metrics["meaning"]["unsupported_relation_rate"] > 0,
            }
            case_result[system_name] = metrics
            totals.setdefault(system_name, []).append(metrics)
        results["cases"][case_id] = case_result
    for name, values in totals.items():
        results["systems"][name] = {
            "cases": len(values),
            "WER_macro": round(sum(item["text"]["WER"] for item in values) / len(values), 6),
            "CER_macro": round(sum(item["text"]["CER"] for item in values) / len(values), 6),
            "number_error_rate_macro": round(sum(item["text"]["number_error_rate"] for item in values) / len(values), 6),
            "negation_error_rate_macro": round(sum(item["text"]["negation_error_rate"] for item in values) / len(values), 6),
            "technical_term_error_rate_macro": round(sum(item["text"]["technical_term_error_rate"] for item in values) / len(values), 6),
            "DER_macro": round(sum(item["speaker"]["DER"] for item in values) / len(values), 6),
            "missed_speech_macro": round(sum(item["speaker"]["missed_speech"] for item in values) / len(values), 6),
            "false_alarm_macro": round(sum(item["speaker"]["false_alarm"] for item in values) / len(values), 6),
            "speaker_confusion_macro": round(sum(item["speaker"]["speaker_confusion"] for item in values) / len(values), 6),
            "claim_F1_macro": round(sum(item["meaning"]["claim_F1"] for item in values) / len(values), 6),
            "type_accuracy_macro": round(sum(item["meaning"]["type_accuracy"] for item in values) / len(values), 6),
            "assignee_F1_macro": round(sum(item["meaning"]["assignee_F1"] for item in values) / len(values), 6),
            "condition_F1_macro": round(sum(item["meaning"]["condition_F1"] for item in values) / len(values), 6),
            **{metric + "_macro": round(sum(item["meaning"][metric] for item in values) / len(values), 6) for metric in (
                "decision_precision", "action_precision", "deadline_accuracy", "number_accuracy",
                "negation_accuracy", "question_resolution_accuracy", "citation_precision", "omission_rate", "unsupported_relation_rate",
            )},
        }
    baseline, candidate = payload.get("baseline"), payload.get("candidate")
    if baseline in results["systems"] and candidate in results["systems"]:
        a, b = results["systems"][baseline], results["systems"][candidate]
        results["comparison"] = {
            "baseline": baseline, "candidate": candidate,
            "WER_delta": round(b["WER_macro"] - a["WER_macro"], 6),
            "CER_delta": round(b["CER_macro"] - a["CER_macro"], 6),
            "DER_delta": round(b["DER_macro"] - a["DER_macro"], 6),
            "claim_F1_delta": round(b["claim_F1_macro"] - a["claim_F1_macro"], 6),
            "assignee_F1_delta": round(b["assignee_F1_macro"] - a["assignee_F1_macro"], 6),
            "condition_F1_delta": round(b["condition_F1_macro"] - a["condition_F1_macro"], 6),
        }
        regressions = []
        for metric in ("WER_delta", "CER_delta", "DER_delta"):
            if results["comparison"][metric] > 0:
                regressions.append(metric)
        for metric in ("claim_F1_delta", "assignee_F1_delta", "condition_F1_delta"):
            if results["comparison"][metric] < 0:
                regressions.append(metric)
        results["comparison"]["regressions"] = regressions
        results["comparison"]["passed"] = not regressions
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.manifest)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.fail_on_regression and not result.get("comparison", {}).get("passed", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
