"""REPORT.md, the committed output of the full run, against the method's purpose."""

from __future__ import annotations

import re


def row(report: str, scale: str) -> dict:
    match = re.search(rf"\| scale_{re.escape(scale)} \| ([\d.]+)% \| ([\d.]+)% \| ([\d.]+)% \| ([\d.]+)% \|", report)
    assert match, f"scale_{scale} row missing from REPORT.md"
    t_acc, _, s_acc, _ = (float(g) for g in match.groups())
    return {"teacher": t_acc, "student": s_acc}


def test_recogniser_works_at_full_resolution(report, targets):
    assert row(report, "1.0")["teacher"] >= targets["clean_lfw_accuracy_floor"]


def test_student_is_resolution_invariant(report, targets):
    """Title: the distilled student keeps its accuracy at low resolution."""
    assert row(report, "0.25")["student"] >= row(report, "1.0")["student"] - targets["invariance_tolerance_pp"]


def test_student_beats_the_teacher_on_low_resolution_faces(report):
    """The point of distilling to an HR/LR student: it handles LR faces better than the HR teacher."""
    for scale in ("0.5", "0.25"):
        r = row(report, scale)
        assert r["student"] > r["teacher"]


def test_student_keeps_high_resolution_accuracy(report, targets):
    r = row(report, "1.0")
    assert r["student"] >= r["teacher"] - targets["invariance_tolerance_pp"]
