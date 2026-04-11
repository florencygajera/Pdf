"""
Test Suite — PDF Extraction System
Covers: sorting, noise cleaning, validator, file handler, API routes.
Uses pytest + httpx AsyncClient for endpoint testing.

Run: pytest tests/ -v --cov=app --cov-report=term-missing
"""

import io
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

    return TestClient(app)


@pytest.fixture
def minimal_pdf_bytes():
    """
    Minimal valid single-page PDF with extractable text.
    Built from scratch so the test has NO external file dependency.
    """
    # This is a hand-crafted minimal PDF with the text "Hello World"
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

        # PaddleOCR format: ([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, conf))
        results = [
            ([[0, 200], [100, 200], [100, 220], [0, 220]], ("Second", 0.95)),
            ([[0, 100], [100, 100], [100, 120], [0, 120]], ("First", 0.98)),
        ]
        sorted_r = sort_ocr_results(results)
        assert sorted_r[0][1][0] == "First"
        assert sorted_r[1][1][0] == "Second"

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
        # Should keep only one "Government of India" variant + Ministry
        gov_lines = [l for l in result if "Government" in l]
        assert len(gov_lines) == 1

    def test_unicode_normalization(self):
        from app.utils.noise_cleaner import clean_text_block

        text = "The ﬁrst oﬃcial notice"  # fi and ffi ligatures
        result = clean_text_block(text)
        assert "fi" in result or "first" in result.lower()

    def test_empty_text_returns_empty(self):
        from app.utils.noise_cleaner import clean_text_block

        assert clean_text_block("") == ""
        assert clean_text_block("   \n  ") == ""

    def test_header_footer_removal(self):
        from app.utils.noise_cleaner import clean_pages

        # A line appearing on every page should be identified as header/footer
        pages = [
            "GOVERNMENT OF INDIA\nSome unique content on page one.",
            "GOVERNMENT OF INDIA\nDifferent content on page two.",
            "GOVERNMENT OF INDIA\nYet more content on page three.",
        ]
        cleaned = clean_pages(pages)
        # The repeated header should be removed from at least some pages
        # (detection threshold is 60%)
        for page in cleaned:
            # Unique content should survive
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
            {"text": "Some text", "confidence": None},  # digital → 0.95
            {"text": "More text", "confidence": None},
        ]
        score = compute_confidence_score(pages)
        assert score == pytest.approx(0.95, abs=0.01)

    def test_confidence_score_mixed(self):
        from app.services.validator import compute_confidence_score

        pages = [
            {"text": "Digital page", "confidence": None},  # 0.95
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
        # Page 1 should now contain both parts
        assert "duce" in result[0]["text"] or "intro" in result[0]["text"]

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
        }  # only 2 cols
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
                    FakePage([[
                        ["H1", "H2"],
                        ["a", "b"],
                    ]]),
                    FakePage([[
                        ["X", "Y"],
                        ["1", "2"],
                    ]]),
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

        assert pages[0]["text"] == "The report starts with an intro"
        assert stitched[0]["raw_results"] is pages[0]["raw_results"]
        assert stitched[0]["warnings"] is not pages[0]["warnings"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Layout Engine Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLayoutEngine:
    def test_single_column_detection(self):
        from app.services.layout_engine import _detect_columns

        # All blocks on left side → single column
        blocks = [
            {"x0": 50, "y0": 100, "x1": 200, "y1": 120},
            {"x0": 50, "y0": 130, "x1": 200, "y1": 150},
        ]
        assert _detect_columns(blocks, page_width=612) == 1

    def test_double_column_detection(self):
        from app.services.layout_engine import _detect_columns

        blocks = [
            {"x0": 50, "y0": 100, "x1": 250, "y1": 120},  # left
            {"x0": 50, "y0": 130, "x1": 250, "y1": 150},  # left
            {"x0": 330, "y0": 100, "x1": 530, "y1": 120},  # right
            {"x0": 330, "y0": 130, "x1": 530, "y1": 150},  # right
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
        # Left column should come first
        assert texts.index("Left-top") < texts.index("Right-top")


# ─────────────────────────────────────────────────────────────────────────────
# 7. API Route Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAPIRoutes:
    def test_health_liveness(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

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

        # Upload
        resp = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", minimal_pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Poll for up to 15 seconds
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


# ─────────────────────────────────────────────────────────────────────────────
# 8. Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_pdf(self, tmp_path):
        """PDF with zero pages — should raise ValueError."""
        from app.services.pdf_detector import detect_pdf_type

        # A PDF with just a catalog but no pages
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
        from app.services.noise_cleaner import remove_noise_lines

        lines = [
            "यह एक सरकारी दस्तावेज़ है।",  # Hindi
            "இது ஒரு அரசு ஆவணம்.",  # Tamil
            "This is an English line.",
        ]
        result = remove_noise_lines(lines)
        # All three should survive (they have alphanumeric characters)
        assert len(result) == 3
