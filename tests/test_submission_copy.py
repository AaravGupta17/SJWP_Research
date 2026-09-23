"""The pre-submission scan must catch identifying text in every form it appears."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_submission_copy import DEFAULT_TERMS, scan  # noqa: E402

PATTERNS = [re.compile(t, re.IGNORECASE) for t in DEFAULT_TERMS]


@pytest.mark.parametrize("text", [
    r"C:\Users\someone\PycharmProjects",           # Windows path in a log
    r'"root": "C:\\Users\\Some One\\data"',        # same path inside JSON
    "C:/Users/someone/data",                       # forward slashes
    "Aarav Gupta & Armaan Guha · Delhi Public School Noida",
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
    big.write_text("ok\n" * 200_000 + "Noida\n" * 8, encoding="utf-8")
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
