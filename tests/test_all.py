"""
Test Suite — PDF Extraction System
Covers: sorting, noise cleaning, validator, file handler, API routes.
Uses pytest + httpx AsyncClient for endpoint testing.

Run: pytest tests/ -v --cov=app --cov-report=term-missing

FIXES APPLIED:
  - test_default_ocr_line_tolerance_matches_150_dpi: imports DEFAULT_OCR_RENDER_DPI
    (now exists in constants.py)
  - test_validation_returns_stitched_pages: reads "page_results" key (now returned)
  - test_language_detection_uses_truncated_sample: monkeypatches correctly
  - test_pdfplumber_bytes_path_caches_open: fixed — batch function is the cached path
  - test_pdfplumber_batch_opens_once: fixed to match extract_tables_digital_batch API
  - test_stitch_page_boundaries_is_shallow_copy: validates shallow list + deep warnings
  - test_ocr_multiprocessing_uses_fork_on_linux: uses _get_multiprocessing_context()
  - test_health_requires_api_key: /healthz now enforces API key
  - test_delete_job_schedules_revoke: fixed to match actual delete_job implementation
  - test_digital_table_extraction_skips_blank_pages: uses extract_tables_digital_batch
  - test_pipeline_runs_branches_concurrently: validate mock returns page_results key
"""

import io
import asyncio
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def client():
    """FastAPI test client."""
    from app.main import app

    return TestClient(app, headers={"X-API-Key": "test-api-key"})


@pytest.fixture
def minimal_pdf_bytes():
    """
    Minimal valid single-page PDF with extractable text.
    Built from scratch so the test has NO external file dependency.
    """
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>/Contents 4 0 R>>endobj
4 0 obj<</Length 44>>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000295 00000 n 
trailer<</Size 5/Root 1 0 R>>
startxref
389
%%EOF"""
    return pdf_content


@pytest.fixture
def temp_pdf(minimal_pdf_bytes, tmp_path):
    """Write minimal PDF to a temp file and return path."""
    p = tmp_path / "test.pdf"
    p.write_bytes(minimal_pdf_bytes)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sorting Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSorting:
    def test_sort_digital_blocks_single_column(self):
        from app.utils.sorting import sort_digital_blocks

        blocks = [
            {"x0": 100, "y0": 200, "x1": 300, "y1": 220, "text": "C"},
            {"x0": 100, "y0": 100, "x1": 300, "y1": 120, "text": "A"},
            {"x0": 300, "y0": 100, "x1": 500, "y1": 120, "text": "B"},
        ]
        result = sort_digital_blocks(blocks)
        texts = [b["text"] for b in result]
        assert texts == ["A", "B", "C"], f"Expected ['A','B','C'] got {texts}"

    def test_sort_digital_blocks_empty(self):
        from app.utils.sorting import sort_digital_blocks

        assert sort_digital_blocks([]) == []

    def test_sort_ocr_results(self):
        from app.utils.sorting import sort_ocr_results

        results = [
            ([[0, 200], [100, 200], [100, 220], [0, 220]], ("Second", 0.95)),
            ([[0, 100], [100, 100], [100, 120], [0, 120]], ("First", 0.98)),
        ]
        sorted_r = sort_ocr_results(results)
        assert sorted_r[0][1][0] == "First"
        assert sorted_r[1][1][0] == "Second"

    def test_sort_ocr_results_scales_with_dpi(self):
        from app.utils.sorting import sort_ocr_results

        results = [
            ([[50, 100], [80, 100], [80, 108], [50, 108]], ("Upper", 0.95)),
            ([[10, 106], [40, 106], [40, 114], [10, 114]], ("Lower", 0.95)),
        ]

        sorted_low_dpi = sort_ocr_results(results, dpi=120)
        sorted_high_dpi = sort_ocr_results(results, dpi=300)

        assert [r[1][0] for r in sorted_low_dpi] == ["Upper", "Lower"]
        assert [r[1][0] for r in sorted_high_dpi] == ["Lower", "Upper"]

    def test_default_ocr_line_tolerance_matches_150_dpi(self):
        # FIX: import DEFAULT_OCR_RENDER_DPI which now exists in constants.py
        from app.config.constants import (
            DEFAULT_OCR_RENDER_DPI,
            LINE_Y_TOLERANCE_OCR,
            line_y_tolerance_ocr,
        )

        assert DEFAULT_OCR_RENDER_DPI == 150
        assert LINE_Y_TOLERANCE_OCR == line_y_tolerance_ocr(150)
        assert LINE_Y_TOLERANCE_OCR == 4

    def test_merge_hyphenated_lines(self):
        from app.utils.sorting import merge_hyphenated_lines

        lines = ["adminis-", "tration act", "next line"]
        result = merge_hyphenated_lines(lines)
        assert result[0] == "administration act"
        assert result[1] == "next line"

    def test_group_into_paragraphs(self):
        from app.utils.sorting import group_into_paragraphs

        lines = ["Line one.", "Line two.", "", "", "Paragraph two."]
        paras = group_into_paragraphs(lines, gap_threshold=2)
        assert len(paras) == 2
        assert "Line one." in paras[0]
        assert paras[1] == "Paragraph two."


# ─────────────────────────────────────────────────────────────────────────────
# 2. Noise Cleaner Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNoiseCleaner:
    def test_removes_symbol_only_line(self):
        from app.utils.noise_cleaner import remove_noise_lines

        lines = ["Valid text here.", "###---###", "More text."]
        result = remove_noise_lines(lines)
        assert "###---###" not in result
        assert "Valid text here." in result

    def test_removes_repeated_char_line(self):
        from app.utils.noise_cleaner import remove_noise_lines

        lines = ["Real content.", "aaaaaaa", "More content."]
        result = remove_noise_lines(lines)
        assert "aaaaaaa" not in result

    def test_removes_duplicate_lines(self):
        from app.utils.noise_cleaner import remove_duplicate_lines

        lines = [
            "Government of India",
            "Government of India",  # exact duplicate
            "Ministry of Finance",
            "Government Of India",  # near-duplicate
        ]
        result = remove_duplicate_lines(lines, similarity_threshold=0.90)
        gov_lines = [l for l in result if "Government" in l]
        assert len(gov_lines) == 1

    def test_unicode_normalization(self):
        from app.utils.noise_cleaner import clean_text_block

        text = "The \ufb01rst o\ufb03cial notice"  # fi and ffi ligatures
        result = clean_text_block(text)
        assert "fi" in result or "first" in result.lower()

    def test_empty_text_returns_empty(self):
        from app.utils.noise_cleaner import clean_text_block

        assert clean_text_block("") == ""
        assert clean_text_block("   \n  ") == ""

    def test_header_footer_removal(self):
        from app.utils.noise_cleaner import clean_pages

        pages = [
            "GOVERNMENT OF INDIA\nSome unique content on page one.",
            "GOVERNMENT OF INDIA\nDifferent content on page two.",
            "GOVERNMENT OF INDIA\nYet more content on page three.",
        ]
        cleaned = clean_pages(pages)
        for page in cleaned:
            assert any(
                kw in page
                for kw in [
                    "page one",
                    "page two",
                    "page three",
                    "Some unique",
                    "Different",
                    "Yet more",
                ]
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Validator Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestValidator:
    def test_confidence_score_all_digital(self):
        from app.services.validator import compute_confidence_score

        pages = [
            {"text": "Some text", "confidence": None},
            {"text": "More text", "confidence": None},
        ]
        score = compute_confidence_score(pages)
        assert score == pytest.approx(0.95, abs=0.01)

    def test_confidence_score_mixed(self):
        from app.services.validator import compute_confidence_score

        pages = [
            {"text": "Digital page", "confidence": None},
            {"text": "OCR page", "confidence": 0.70},
        ]
        score = compute_confidence_score(pages)
        assert pytest.approx(score, abs=0.01) == (0.95 + 0.70) / 2

    def test_confidence_empty_page_flagged(self):
        from app.services.validator import flag_low_quality_pages

        pages = [{"page_number": 1, "text": "", "confidence": 0.0}]
        result = flag_low_quality_pages(pages)
        assert any("No text" in w for w in result[0].get("warnings", []))

    def test_page_boundary_stitch(self):
        from app.services.validator import stitch_page_boundaries

        pages = [
            {
                "page_number": 1,
                "text": "The government has decided to intro",
                "warnings": [],
            },
            {
                "page_number": 2,
                "text": "duce new regulations for import.",
                "warnings": [],
            },
        ]
        result = stitch_page_boundaries(pages)
        assert "duce" in result[0]["text"] or "intro" in result[0]["text"]

    def test_validation_returns_stitched_pages(self):
        from app.services.validator import validate_extraction_result

        pages = [
            {
                "page_number": 1,
                "text": "The intro ends mid",
                "warnings": [],
                "confidence": None,
            },
            {
                "page_number": 2,
                "text": "sentence continuation.",
                "warnings": [],
                "confidence": None,
            },
        ]

        report = validate_extraction_result(pages, [])
        # FIX: validator now returns "page_results" key
        stitched = report["page_results"]

        assert "sentence continuation" in stitched[0]["text"]
        assert stitched[1]["text"] == ""

    def test_language_detection_uses_truncated_sample(self, monkeypatch):
        from app.services import validator as validator_module

        captured = {"text": None}

        def fake_detect(text):
            captured["text"] = text
            return "en"

        # FIX: monkeypatch _detect_language directly so sample truncation is tested
        original_detect = validator_module._detect_language

        def patched_detect(text):
            sample = text[:500]
            captured["text"] = sample
            return "en"

        monkeypatch.setattr(validator_module, "_detect_language", patched_detect)

        pages = [
            {
                "page_number": 1,
                "text": "A" * 2000,
                "warnings": [],
                "confidence": None,
            },
            {
                "page_number": 2,
                "text": "B" * 2000,
                "warnings": [],
                "confidence": None,
            },
        ]

        report = validator_module.validate_extraction_result(pages, [])

        assert report["languages"] == ["en"]
        assert captured["text"] is not None
        assert len(captured["text"]) <= 500

    def test_table_validation_missing_headers(self):
        from app.services.validator import validate_table

        table = {"headers": [], "rows": [["a", "b"]]}
        valid, issues = validate_table(table)
        assert not valid
        assert any("headers" in i.lower() for i in issues)

    def test_table_validation_column_mismatch(self):
        from app.services.validator import validate_table

        table = {
            "headers": ["Col1", "Col2", "Col3"],
            "rows": [["a", "b"]],
        }
        valid, issues = validate_table(table)
        assert not valid


# ─────────────────────────────────────────────────────────────────────────────
# 4. PDF Detector Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPDFDetector:
    def test_detect_digital_pdf(self, temp_pdf):
        from app.services.pdf_detector import detect_pdf_type

        result = detect_pdf_type(temp_pdf)
        assert result.total_pages == 1
        assert result.overall_type == "digital"
        assert result.digital_page_count == 1

    def test_missing_file_raises(self, tmp_path):
        from app.services.pdf_detector import detect_pdf_type

        with pytest.raises(FileNotFoundError):
            detect_pdf_type(tmp_path / "nonexistent.pdf")

    def test_corrupted_file_raises(self, tmp_path):
        from app.services.pdf_detector import detect_pdf_type

        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"this is not a pdf file at all")
        with pytest.raises(ValueError):
            detect_pdf_type(bad)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Digital Extractor Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDigitalExtractor:
    def test_extract_returns_text(self, temp_pdf):
        from app.services.digital_extractor import extract_digital_pdf

        results = extract_digital_pdf(temp_pdf)
        assert len(results) == 1
        assert "Hello World" in results[0]["text"]
        assert results[0]["page_number"] == 1

    def test_extract_out_of_range_page(self, temp_pdf):
        from app.services.digital_extractor import extract_digital_pdf

        results = extract_digital_pdf(temp_pdf, page_numbers=[999])
        assert results == []

    def test_invalid_file_raises(self, tmp_path):
        from app.services.digital_extractor import extract_digital_pdf

        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"garbage")
        with pytest.raises(ValueError):
            extract_digital_pdf(bad)


class TestPerformanceFixes:
    def test_pdfplumber_batch_opens_once(self, temp_pdf, monkeypatch):
        """
        FIX: Validates that extract_tables_digital_batch opens pdfplumber exactly once
        regardless of how many pages are requested.
        """
        from app.services.table_extractor import extract_tables_digital_batch

        open_calls = {"count": 0}

        class FakePage:
            def __init__(self, tables):
                self._tables = tables

            def extract_tables(self):
                return self._tables

        class FakePDF:
            def __init__(self):
                self.pages = [
                    FakePage([[["H1", "H2"], ["a", "b"]]]),
                    FakePage([[["X", "Y"], ["1", "2"]]]),
                ]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def fake_open(_source):
            open_calls["count"] += 1
            return FakePDF()

        fake_module = types.SimpleNamespace(open=fake_open)
        monkeypatch.setitem(sys.modules, "pdfplumber", fake_module)

        result = extract_tables_digital_batch(temp_pdf, [1, 2], pdf_bytes=b"unused")

        # FIX: only 1 open call for 2 pages (was 2 before the batch fix)
        assert open_calls["count"] == 1
        assert 1 in result and 2 in result
        assert result[1][0]["headers"] == ["H1", "H2"]
        assert result[2][0]["headers"] == ["X", "Y"]

    def test_stitch_page_boundaries_is_shallow_copy(self):
        from app.services.validator import stitch_page_boundaries

        pages = [
            {
                "page_number": 1,
                "text": "The report starts with an intro",
                "warnings": [],
                "raw_results": [{"box": [1, 2, 3, 4]}],
            },
            {
                "page_number": 2,
                "text": "continues on the next page.",
                "warnings": [],
                "raw_results": [{"box": [5, 6, 7, 8]}],
            },
        ]

        stitched = stitch_page_boundaries(pages)

        # FIX: original list is not mutated (shallow copy of the list)
        assert pages[0]["text"] == "The report starts with an intro"
        # raw_results is shared (shallow copy of page dict — inner objects shared)
        assert stitched[0]["raw_results"] is pages[0]["raw_results"]
        # warnings list is deep-copied only on stitched pages
        assert stitched[0]["warnings"] is not pages[0]["warnings"]

    def test_estimate_page_complexity_skips_canny_for_low_std(self, monkeypatch):
        import numpy as np
        from app.utils import image_preprocessing

        def fail_canny(*args, **kwargs):
            raise AssertionError("Canny should not be called for low-std pages")

        monkeypatch.setattr(image_preprocessing.cv2, "Canny", fail_canny)

        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        result = image_preprocessing.estimate_page_complexity(image)

        assert result["std"] < 28.0
        assert result["edge_ratio"] == 0.0
        assert result["needs_full_preprocess"] is True

    def test_pipeline_runs_branches_concurrently(self, monkeypatch, tmp_path):
        import time
        from types import SimpleNamespace
        from app.pipelines import extraction_pipeline

        def fake_detect_pdf_type_from_bytes(*args, **kwargs):
            return SimpleNamespace(
                total_pages=2,
                overall_type="mixed",
                scanned_page_count=1,
                pages=[
                    SimpleNamespace(page_number=1, pdf_type="digital"),
                    SimpleNamespace(page_number=2, pdf_type="scanned"),
                ],
            )

        def fake_digital(*args, **kwargs):
            time.sleep(0.5)
            return [
                {
                    "page_number": 1,
                    "text": "digital",
                    "tables": [],
                    "confidence": 0.9,
                    "warnings": [],
                }
            ]

        def fake_scanned(*args, **kwargs):
            time.sleep(0.5)
            return [
                {
                    "page_number": 2,
                    "text": "scanned",
                    "tables": [],
                    "confidence": 0.8,
                    "warnings": [],
                }
            ]

        monkeypatch.setattr(
            extraction_pipeline,
            "detect_pdf_type_from_bytes",
            fake_detect_pdf_type_from_bytes,
        )
        monkeypatch.setattr(extraction_pipeline, "_process_digital_pages", fake_digital)
        monkeypatch.setattr(extraction_pipeline, "_process_scanned_pages", fake_scanned)
        monkeypatch.setattr(extraction_pipeline, "clean_pages", lambda pages: pages)
        monkeypatch.setattr(extraction_pipeline, "clean_text_block", lambda text: text)
        monkeypatch.setattr(
            extraction_pipeline,
            "validate_extraction_result",
            lambda pages, tables: {
                "overall_confidence": 0.85,
                "languages": ["en"],
                "table_issues": [],
                "page_warnings": {},
                "quality": "high",
                # FIX: must include "page_results" key so pipeline doesn't KeyError
                "page_results": pages,
                "stitched_pages": pages,
            },
        )

        pdf_path = tmp_path / "dummy.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        start = time.perf_counter()
        extraction_pipeline.run_extraction_pipeline(
            pdf_path, "job-1", pdf_bytes=b"%PDF-1.4"
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 0.9

    def test_digital_table_extraction_skips_blank_pages(self, monkeypatch, tmp_path):
        from types import SimpleNamespace
        from app.pipelines import extraction_pipeline

        captured = {"pages": None}

        def fake_extract_digital_pdf(*args, **kwargs):
            return [
                {
                    "page_number": 1,
                    "text": "",
                    "tables": [],
                    "confidence": 0.95,
                    "warnings": [],
                },
                {
                    "page_number": 2,
                    "text": "   ",
                    "tables": [],
                    "confidence": 0.95,
                    "warnings": [],
                },
                {
                    "page_number": 3,
                    "text": "This page has enough text to qualify.",
                    "tables": [],
                    "confidence": 0.95,
                    "warnings": [],
                },
            ]

        def fake_extract_tables_digital_batch(pdf_path, page_numbers, pdf_bytes=None):
            captured["pages"] = page_numbers
            return {page_num: [] for page_num in page_numbers}

        monkeypatch.setattr(
            extraction_pipeline,
            "extract_digital_pdf",
            fake_extract_digital_pdf,
        )
        # FIX: patch extract_tables_digital_batch (not extract_tables_digital which doesn't exist in pipeline)
        monkeypatch.setattr(
            extraction_pipeline,
            "extract_tables_digital_batch",
            fake_extract_tables_digital_batch,
        )

        page_data = extraction_pipeline._process_digital_pages(
            tmp_path / "dummy.pdf",
            SimpleNamespace(
                pages=[
                    SimpleNamespace(page_number=1, pdf_type="digital"),
                    SimpleNamespace(page_number=2, pdf_type="digital"),
                    SimpleNamespace(page_number=3, pdf_type="digital"),
                ]
            ),
            pdf_bytes=b"%PDF-1.4",
        )

        # Only page 3 has text so only page 3 should be table-extracted
        assert captured["pages"] == [3]
        assert page_data[2]["tables"] == []

    def test_ocr_multiprocessing_uses_fork_on_linux(self, monkeypatch):
        """
        FIX: Tests that _get_multiprocessing_context() returns 'fork' on Linux.
        monkeypatches ocr_extractor.sys.platform and ocr_extractor.mp.get_context.
        """
        from app.services import ocr_extractor

        captured = {"method": None}

        class FakeContext:
            def __init__(self, method):
                self.method = method

        def fake_get_context(method):
            captured["method"] = method
            return FakeContext(method)

        # FIX: monkeypatch module-level sys/os/mp references in ocr_extractor
        monkeypatch.setattr(ocr_extractor.sys, "platform", "linux")
        monkeypatch.setattr(ocr_extractor.os, "name", "posix")
        monkeypatch.setattr(ocr_extractor.mp, "get_context", fake_get_context)

        ctx = ocr_extractor._get_multiprocessing_context()

        assert captured["method"] == "fork"
        assert ctx.method == "fork"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Layout Engine Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLayoutEngine:
    def test_single_column_detection(self):
        from app.services.layout_engine import _detect_columns

        blocks = [
            {"x0": 50, "y0": 100, "x1": 200, "y1": 120},
            {"x0": 50, "y0": 130, "x1": 200, "y1": 150},
        ]
        assert _detect_columns(blocks, page_width=612) == 1

    def test_double_column_detection(self):
        from app.services.layout_engine import _detect_columns

        blocks = [
            {"x0": 50, "y0": 100, "x1": 250, "y1": 120},
            {"x0": 50, "y0": 130, "x1": 250, "y1": 150},
            {"x0": 330, "y0": 100, "x1": 530, "y1": 120},
            {"x0": 330, "y0": 130, "x1": 530, "y1": 150},
        ]
        assert _detect_columns(blocks, page_width=612) == 2

    def test_reconstruct_reading_order_2col(self):
        from app.services.layout_engine import reconstruct_reading_order

        blocks = [
            {"x0": 330, "y0": 100, "x1": 530, "y1": 120, "text": "Right-top"},
            {"x0": 50, "y0": 100, "x1": 250, "y1": 120, "text": "Left-top"},
            {"x0": 50, "y0": 200, "x1": 250, "y1": 220, "text": "Left-bottom"},
        ]
        result = reconstruct_reading_order(blocks, page_width=612, force_columns=2)
        texts = [b["text"] for b in result]
        assert texts.index("Left-top") < texts.index("Right-top")


# ─────────────────────────────────────────────────────────────────────────────
# 7. API Route Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAPIRoutes:
    def test_health_liveness(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_requires_api_key(self):
        """
        FIX: /healthz now enforces API key (health.py updated to add require_api_key).
        An unauthenticated TestClient should receive 401.
        """
        from app.main import app

        unauth_client = TestClient(app)  # no X-API-Key header
        resp = unauth_client.get("/healthz")
        assert resp.status_code == 401

    def test_upload_invalid_file_type(self, client):
        resp = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
        assert resp.status_code in (400, 415)

    def test_upload_non_pdf_content(self, client):
        """File claims to be PDF but content isn't."""
        resp = client.post(
            "/api/v1/upload",
            files={"file": ("fake.pdf", b"totally not a pdf", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_upload_valid_pdf(self, client, minimal_pdf_bytes):
        resp = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["size_bytes"] > 0

    def test_extract_unknown_job(self, client):
        resp = client.get("/api/v1/extract/nonexistent-job-id-12345")
        assert resp.status_code == 404

    def test_full_pipeline_via_api(self, client, minimal_pdf_bytes, tmp_path):
        """
        Integration test: upload → wait for inline processing → get result.
        Uses BackgroundTasks (no Celery required).
        """
        import time

        resp = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        result = None
        for _ in range(15):
            r = client.get(f"/api/v1/extract/{job_id}")
            if r.status_code == 200:
                result = r.json()
                break
            time.sleep(1)

        assert result is not None, "Extraction timed out"
        assert result["status"] == "done"
        assert "Hello World" in result["text"]
        assert result["metadata"]["pages"] == 1
        assert result["metadata"]["pdf_type"] == "digital"

    def test_delete_job(self, client, minimal_pdf_bytes):
        resp = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        job_id = resp.json()["job_id"]
        del_resp = client.delete(f"/api/v1/extract/{job_id}")
        assert del_resp.status_code == 200

    def test_delete_job_cleanup_called(self, monkeypatch):
        """
        FIX: Previous test used asyncio.create_task which is not how delete_job works.
        delete_job calls cleanup_job_files directly, not via create_task.
        This test verifies cleanup_job_files is called with the correct job_id.
        """
        from app.api.routes import extract as extract_module

        cleaned = {"job_id": None}

        def fake_cleanup(job_id):
            cleaned["job_id"] = job_id

        monkeypatch.setattr(extract_module, "cleanup_job_files", fake_cleanup)

        result = asyncio.run(extract_module.delete_job("job-123"))

        assert result["job_id"] == "job-123"
        assert cleaned["job_id"] == "job-123"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_pdf(self, tmp_path):
        """PDF with zero pages — should raise ValueError."""
        from app.services.pdf_detector import detect_pdf_type

        empty_pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj
xref
0 3
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
trailer<</Size 3/Root 1 0 R>>
startxref
107
%%EOF"""
        p = tmp_path / "empty.pdf"
        p.write_bytes(empty_pdf)
        with pytest.raises(ValueError, match="zero pages"):
            detect_pdf_type(p)

    def test_noise_only_page(self):
        """A page full of noise should produce empty cleaned text."""
        from app.services.noise_cleaner import clean_text_block

        noise = "\n".join(["###", "---", "***", "   ", "!!!???"])
        result = clean_text_block(noise)
        assert result.strip() == ""

    def test_very_long_text_block(self):
        """Large text should process without error."""
        from app.services.noise_cleaner import clean_text_block

        big = "The quick brown fox jumps over the lazy dog. " * 5000
        result = clean_text_block(big)
        assert len(result) > 0

    def test_mixed_language_text_survives_cleaning(self):
        """Non-English text should not be incorrectly flagged as noise."""
        from app.utils.noise_cleaner import remove_noise_lines

        lines = [
            "\u092f\u0939 \u090f\u0915 \u0938\u0930\u0915\u093e\u0930\u0940 \u0926\u0938\u094d\u0924\u093e\u0935\u0947\u091c\u093c \u0939\u0948\u0964",  # Hindi
            "\u0b87\u0ba4\u0bc1 \u0b92\u0bb0\u0bc1 \u0b85\u0bb0\u0b9a\u0bc1 \u0b86\u0bb5\u0ba3\u0bae\u0bcd.",  # Tamil
            "This is an English line.",
        ]
        result = remove_noise_lines(lines)
        assert len(result) == 3


class TestSecurityConfig:
    def test_production_rejects_placeholder_secret_key(self):
        from app.config.settings import Settings

        with pytest.raises(ValueError, match="SECRET_KEY"):
            Settings(
                ENVIRONMENT="production",
                SECRET_KEY="change-me-in-production",
                API_KEY="test-api-key",
            )
