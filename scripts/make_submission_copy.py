"""
make_submission_copy.py — build a history-free copy of the repo and check it for identifying text
===================================================================================================
IRIS rules: nothing submitted may name the school, city or state. This
repository's git HISTORY still contains such text (older docs), so a link to
it cannot be submitted as-is. This script:

  1. copies every git-tracked file (current contents, LFS files included)
     into a NEW folder outside the repo, WITHOUT the .git history;
     archive/ and private/ are left out;
  2. scans every text file in the copy for forbidden terms and prints each
     hit with file and line number;
  3. exits with status 1 if anything was found.

It never creates a repository or pushes anything. If the scan is clean,
create a new repository from the copy yourselves.

    python scripts/make_submission_copy.py                      # -> ../SJWP_submission_copy
    python scripts/make_submission_copy.py --out D:/copy --extra-terms "Sector 30"

The school/city/state terms go in private/forbidden_terms.txt (git-ignored),
one regex per line, so that this script does not itself contain them.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = ("archive/", "private/")
TEXT_EXT = {".py", ".md", ".txt", ".csv", ".json", ".ps1", ".m", ".yml", ".yaml", ".toml",
            ".cfg", ".ini", ".gitignore", ".gitattributes", ".inp", ""}

# Case-insensitive patterns: AI-tool names per AGENTS.md and personal Windows
# user paths. School/city/state are NOT written here, since this file is
# submitted too; they are read from the git-ignored PRIVATE_TERMS file.
DEFAULT_TERMS = [
    r"\bClaude\b", r"\bAnthropic\b", r"Co-Authored-By", r"Generated with",
    r"[A-Za-z]:[\\/]+Users[\\/]+\w",       # C:\Users\name, C:\\Users\\name (JSON), C:/Users/name
]
PRIVATE_TERMS = REPO / "private" / "forbidden_terms.txt"


def load_private_terms(path: Path = PRIVATE_TERMS):
    """One regex per line (e.g. the school name, \\bCityName\\b); '#' lines are comments."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [t.strip() for t in lines if t.strip() and not t.lstrip().startswith("#")]


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True)
    return [f for f in out.stdout.decode("utf-8").split("\0")
            if f and not f.startswith(EXCLUDE_DIRS)]


def scan(root: Path, patterns):
    hits = []
    for p in root.rglob("*"):
        if not p.is_file() or (p.suffix.lower() not in TEXT_EXT and p.name not in TEXT_EXT):
            continue
        n_hits = 0
        with open(p, encoding="utf-8", errors="replace") as fh:   # streamed, so large CSVs too
            for i, line in enumerate(fh, 1):
                for pat in patterns:
                    if pat.search(line):
                        n_hits += 1
                        if n_hits <= 5:
                            hits.append((p, i, line.strip()[:120]))
                        break
        if n_hits > 5:
            hits.append((p, 0, f"... and {n_hits - 5} more lines in this file"))
    return hits


USER_PATH = re.compile(r"[A-Za-z]:(?:\\\\|\\|/)+Users(?:\\\\|\\|/)+[^\\/\"]+(?:(?:\\\\|\\|/)+[^\\/\"]+)*?"
                       r"(?:\\\\|\\|/)+SJWP_Research")


def redact_user_paths(root: Path) -> int:
    """In the COPY only: replace '<drive>:/Users/<name>/.../SJWP_Research' with '<repo>'.
    The originals in the repository are evidence and are not modified."""
    changed = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".json", ".txt", ".md", ".csv"} and p.stat().st_size < 20_000_000:
            text = p.read_text(encoding="utf-8", errors="replace")
            new = USER_PATH.sub("<repo>", text)
            if new != text:
                p.write_text(new, encoding="utf-8")
                changed += 1
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO.parent / "SJWP_submission_copy")
    ap.add_argument("--extra-terms", nargs="*", default=[],
                    help="more words to forbid (e.g. the school's area or teacher names)")
    ap.add_argument("--redact-paths", action="store_true",
                    help="in the copy, replace user-folder paths to the repo with <repo>")
    args = ap.parse_args()

    out = args.out.resolve()
    if REPO in out.parents or out == REPO:
        sys.exit("--out must be outside the repository")
    if out.exists():
        sys.exit(f"{out} already exists; delete it or choose another --out")

    files = tracked_files()
    for f in files:
        src, dst = REPO / f, out / f
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print(f"Copied {len(files)} tracked files (no git history, without {', '.join(EXCLUDE_DIRS)})"
          f" to {out}")
    if args.redact_paths:
        print(f"Redacted user-folder paths in {redact_user_paths(out)} file(s) of the copy")

    private = load_private_terms()
    if not private:
        print(f"WARNING: {PRIVATE_TERMS.relative_to(REPO)} is missing or empty, so the school/city "
              "are NOT being checked. Create it with one term per line.")
    patterns = [re.compile(t, re.IGNORECASE) for t in DEFAULT_TERMS + private]
    patterns += [re.compile(re.escape(t), re.IGNORECASE) for t in args.extra_terms]
    hits = scan(out, patterns)
    if not hits:
        print("Scan clean: no forbidden terms found.")
        print("Also check by hand: PDF/Word metadata, images with text, and the new repo's name/description.")
        return
    print(f"\n{len(hits)} line(s) to fix before submitting:")
    for p, i, text in hits:
        print(f"  {p.relative_to(out)}:{i}: {text}")
    sys.exit(1)


if __name__ == "__main__":
    main()
