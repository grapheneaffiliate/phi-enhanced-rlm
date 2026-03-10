#!/usr/bin/env python3
"""Tests for phi-Code Review module."""

import sys
import json
import time
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.code_review import (
    PhiCodeReview, ReviewFinding, ReviewResult, REVIEW_PERSPECTIVES
)


class TestReviewPerspectives:
    def test_four_perspectives(self):
        assert len(REVIEW_PERSPECTIVES) == 4

    def test_required_perspectives(self):
        for name in ["logic", "security", "quality", "architecture"]:
            assert name in REVIEW_PERSPECTIVES
            assert "casimir" in REVIEW_PERSPECTIVES[name]
            assert "prompt" in REVIEW_PERSPECTIVES[name]

    def test_casimir_degrees_ascending(self):
        degrees = [REVIEW_PERSPECTIVES[k]["casimir"]
                   for k in ["logic", "security", "quality", "architecture"]]
        assert degrees == sorted(degrees)


class TestReviewFinding:
    def test_severity_ranking(self):
        assert ReviewFinding("a", "critical", 0.9, "", "", "", "").severity_rank == 0
        assert ReviewFinding("a", "important", 0.9, "", "", "", "").severity_rank == 1
        assert ReviewFinding("a", "minor", 0.9, "", "", "", "").severity_rank == 2
        assert ReviewFinding("a", "suggestion", 0.9, "", "", "", "").severity_rank == 3

    def test_fields(self):
        f = ReviewFinding("security", "critical", 0.95, "auth.py", "42", "Bug", "desc", "fix")
        assert f.perspective == "security"
        assert f.confidence == 0.95
        assert f.suggestion == "fix"


class TestReviewResult:
    def test_counts(self):
        findings = [
            ReviewFinding("a", "critical", 0.9, "", "", "", ""),
            ReviewFinding("b", "critical", 0.85, "", "", "", ""),
            ReviewFinding("c", "important", 0.9, "", "", "", ""),
            ReviewFinding("d", "minor", 0.5, "", "", "", ""),
        ]
        result = ReviewResult(findings=findings)
        assert result.critical_count == 2
        assert result.important_count == 1

    def test_high_confidence_filter(self):
        findings = [
            ReviewFinding("a", "critical", 0.95, "", "", "", ""),
            ReviewFinding("b", "minor", 0.30, "", "", "", ""),
        ]
        result = ReviewResult(findings=findings)
        hc = result.high_confidence_findings
        assert len(hc) == 1
        assert hc[0].confidence == 0.95

    def test_format_report_no_findings(self):
        result = ReviewResult()
        report = result.format_report()
        assert "No issues found" in report

    def test_format_report_with_findings(self):
        findings = [
            ReviewFinding("logic", "critical", 0.95, "auth.py", "42",
                         "Null pointer", "user can be None", "Add check"),
        ]
        result = ReviewResult(findings=findings, total_agents=4, review_time_seconds=2.0)
        report = result.format_report()
        assert "Critical" in report
        assert "auth.py" in report
        assert "Null pointer" in report


class TestJsonParsing:
    def setup_method(self):
        from src.phi_enhanced_rlm import MockLLMBackend
        self.reviewer = PhiCodeReview(MockLLMBackend())

    def test_direct_json(self):
        r = self.reviewer._parse_json('[{"severity":"critical"}]')
        assert isinstance(r, list) and len(r) == 1

    def test_code_block(self):
        r = self.reviewer._parse_json('```json\n[{"severity":"minor"}]\n```')
        assert isinstance(r, list)

    def test_garbage(self):
        r = self.reviewer._parse_json("no json here")
        assert isinstance(r, list)

    def test_empty_array(self):
        r = self.reviewer._parse_json("[]")
        assert r == []


class TestParallelExecution:
    def test_uses_multiple_threads(self):
        threads_seen = set()

        class TrackLLM:
            def __call__(self, prompt, max_tokens=100):
                threads_seen.add(threading.current_thread().name)
                time.sleep(0.05)
                return '[]'

        reviewer = PhiCodeReview(TrackLLM(), max_workers=4, confidence_threshold=0.0)
        reviewer.review_text("def foo(): pass")
        assert len(threads_seen) >= 2


class TestConfidenceFiltering:
    def test_threshold_filters(self):
        class FixedLLM:
            def __call__(self, prompt, max_tokens=100):
                return json.dumps([
                    {"severity": "critical", "confidence": 0.95,
                     "file": "a.py", "lines": "1", "title": "High", "description": "d"},
                    {"severity": "minor", "confidence": 0.30,
                     "file": "b.py", "lines": "2", "title": "Low", "description": "d"},
                ])

        strict = PhiCodeReview(FixedLLM(), confidence_threshold=0.80)
        loose = PhiCodeReview(FixedLLM(), confidence_threshold=0.20)
        r1 = strict.review_text("code")
        r2 = loose.review_text("code")
        assert len(r2.findings) > len(r1.findings)


class TestGitIntegration:
    def test_diff_stats_parsing(self):
        from src.phi_enhanced_rlm import MockLLMBackend
        reviewer = PhiCodeReview(MockLLMBackend())
        fake_diff = (
            "diff --git a/file.py b/file.py\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "+new line 1\n"
            "+new line 2\n"
            "-old line\n"
        )
        stats = reviewer._get_diff_stats(fake_diff)
        assert stats["files_changed"] == 1
        assert stats["additions"] == 2
        assert stats["deletions"] == 1
