"""The pre-submission scan must catch identifying text in every form it appears."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_submission_copy import DEFAULT_TERMS, load_private_terms, scan  # noqa: E402

# Made-up school/city standing in for the real ones, which live only in the
# git-ignored private/forbidden_terms.txt
FAKE_PRIVATE = [r"Example High School", r"\bSpringfield\b"]
PATTERNS = [re.compile(t, re.IGNORECASE) for t in DEFAULT_TERMS + FAKE_PRIVATE]


@pytest.mark.parametrize("text", [
    r"C:\Users\someone\PycharmProjects",           # Windows path in a log
    r'"root": "C:\\Users\\Some One\\data"',        # same path inside JSON
    "C:/Users/someone/data",                       # forward slashes
    "A. Student · Example High School, Springfield",
    "Co-Authored-By: someone",
])
def test_identifying_text_is_caught(text):
    assert any(p.search(text) for p in PATTERNS)


@pytest.mark.parametrize("text", [
    r"../datasets/NetworkList\Network_8\CI\base-1.2.csv",
    "L-TOWN (Network 3), KY15 (Network 6), Richmond (Network 8)",
])
def test_ordinary_text_is_not_flagged(text):
    assert not any(p.search(text) for p in PATTERNS)


def test_scan_streams_large_files_and_caps_hits(tmp_path):
    big = tmp_path / "big.csv"
    big.write_text("ok\n" * 200_000 + "Springfield\n" * 8, encoding="utf-8")
    hits = scan(tmp_path, PATTERNS)
    assert len(hits) == 6                           # 5 shown + 1 "... and 3 more" line
    assert hits[-1][2].startswith("... and 3 more")


def test_redaction_replaces_user_paths_in_the_copy(tmp_path):
    from make_submission_copy import redact_user_paths
    f = tmp_path / "run.json"
    json_line = r'{"root": "C:\\Users\\Some One\\PycharmProjects\\SJWP_Research\\datasets\\x"}'
    log_line = r"Saved C:\Users\someone\PycharmProjects\SJWP_Research\plots\a.png"
    f.write_text(json_line + "\n" + log_line + "\n", encoding="utf-8")
    assert redact_user_paths(tmp_path) == 1
    text = f.read_text(encoding="utf-8")
    assert "Users" not in text and text.count("<repo>") == 2
    assert not any(p.search(text) for p in PATTERNS)


def test_new_run_records_store_repo_relative_paths():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from _common import REPO_ROOT, _jsonable
    out = _jsonable({"root": str(REPO_ROOT / "datasets" / "x"), "other": "C:/elsewhere"})
    assert out["root"] == "<repo>/datasets/x" and out["other"] == "C:/elsewhere"


def test_private_terms_file_is_read_and_comments_skipped(tmp_path):
    f = tmp_path / "terms.txt"
    f.write_text("# comment\n\nExample High School\n  \\bSpringfield\\b  \n", encoding="utf-8")
    assert load_private_terms(f) == ["Example High School", r"\bSpringfield\b"]
    assert load_private_terms(tmp_path / "missing.txt") == []


def test_script_itself_names_no_place():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "make_submission_copy.py").read_text(encoding="utf-8")
    assert not any(re.search(t, src, re.IGNORECASE) for t in load_private_terms())
