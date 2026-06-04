#!/usr/bin/env python3
"""
IKON SKILLS™ — MCQ Question Bank Generator (v2.0)
====================================================
Architecture:
  Domain → MicroCredential → Competency → FA1 / FA2 / FA3 (40 MCQs each)

  Per CC: 3 FAs × 40 MCQs = 120 MCQs
  Per MC: 10 CCs × 120     = 1,200 MCQs
  Total (184 MCs):           220,800 MCQs

Question type: 4-option MCQ only. NO scenario, NO short-answer, NO SA.

Batch strategy:
  10 MCQs per Claude API call (4 batches per FA × 3 FAs × 10 CCs = 120 batches/MC).

Cost (Sonnet 4.6 @ $3/$15 per MTok):
  Per batch:  ~$0.017
  Per MC:     ~$2.00 (120 batches)
  All 184:    ~$375

Usage:
  python generate_banks.py                                  # all 184 MCs (~$375) — bare run = full catalog
  python generate_banks.py --all                            # same as bare run
  python generate_banks.py --mc "AI Prompt Engineer"        # single MC (~$2)
  python generate_banks.py --category "AI"                  # one domain
  python generate_banks.py --dry-run                        # no API calls (full catalog dry)

Output:
  output/qbank/{MC_safe_name}_qbank.csv  — one CSV per MC, 1,200 rows each
  output/generation_manifest.json        — run log
"""

import os
import json
import csv
import time
import random
import argparse
import urllib.request
import urllib.error
import anthropic
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()  # ANTHROPIC_API_KEY + IKON_API_TOKEN


# ── CONFIG ─────────────────────────────────────────────────────────────────────
PIPELINE_VERSION = "v2.0.0"
MODEL            = "claude-sonnet-4-6"
MAX_TOKENS       = 4096
BATCH_SIZE       = 10                          # MCQs per Claude call
QUESTIONS_PER_FA = 40
BATCHES_PER_FA   = QUESTIONS_PER_FA // BATCH_SIZE  # 4
ASSESSMENTS      = ["FA1", "FA2", "FA3"]
RETRY_LIMIT      = 3
RETRY_DELAY      = 5
RATE_LIMIT_DELAY = 0.5

OUTPUT_DIR = Path("output")
CSV_DIR    = OUTPUT_DIR / "qbank"
CSV_DIR.mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "mc_id", "mc_name",
    "domain_id", "domain_name",
    "level", "source",
    "cc_id", "cc_code", "cc_title", "cc_description",
    "assessment_type", "question_number",
    "question",
    "option_a", "option_b", "option_c", "option_d",
    "correct_answer", "explanation",
    "difficulty", "estimated_time_minutes",
    "generated_at", "pipeline_version",
]

DIFFICULTY_BY_FA = {
    "FA1": "foundational",
    "FA2": "intermediate",
    "FA3": "advanced",
}


def _balanced_answer_dist(n: int = BATCH_SIZE) -> list[str]:
    """
    Return n answer letters (A/B/C/D) with balanced distribution.
    Each letter appears floor(n/4) or ceil(n/4) times, randomly shuffled.
    For n=10: each appears 2 or 3 times (never 0 or 4+).
    """
    per_letter, remainder = divmod(n, 4)
    pool = ["A", "B", "C", "D"] * per_letter + random.sample(["A", "B", "C", "D"], remainder)
    random.shuffle(pool)
    return pool

ASSESSMENT_SPECS = {
    "FA1": (
        "Formative Assessment 1 (FA1) — Recall and Comprehension. "
        "Test definition recall, terminology matching, and basic conceptual understanding. "
        "Foundational difficulty."
    ),
    "FA2": (
        "Formative Assessment 2 (FA2) — Application and Understanding. "
        "Test the ability to apply concepts in a straightforward professional context. "
        "Intermediate difficulty."
    ),
    "FA3": (
        "Formative Assessment 3 (FA3) — Analysis and Synthesis. "
        "Test analytical reasoning, comparison between concepts/options, and reasoned judgement. "
        "Advanced difficulty (still knowledge-based, NOT a scenario simulation)."
    ),
}


# ── ANTHROPIC CLIENT ───────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_mcq_batch_prompt(mc: dict, cc: dict, assessment_type: str, batch_num: int,
                           previous_questions: list[str] | None = None,
                           answer_dist: list[str] | None = None) -> str:
    """Build prompt for one batch of BATCH_SIZE MCQs.

    previous_questions: question texts already generated for this (CC, FA) — avoids repeats.
    answer_dist: pre-generated balanced list of correct answer letters (A/B/C/D) for each
                 question position, e.g. ['A','C','B','A','D',...]. Prevents Claude's known
                 bias toward B as the correct answer.
    """
    all_cc_titles = [c["cc_title"] for c in mc["competencies"]]
    start_q = batch_num * BATCH_SIZE + 1
    end_q   = start_q + BATCH_SIZE - 1

    prev_section = ""
    if previous_questions:
        lines = "\n".join(f"{i+1}. {q}" for i, q in enumerate(previous_questions))
        prev_section = f"""
QUESTIONS ALREADY WRITTEN FOR {assessment_type} of this competency — do NOT repeat or closely paraphrase any of these:
{lines}

"""

    dist = answer_dist or _balanced_answer_dist(BATCH_SIZE)
    dist_line = "  ".join(f"Q{i+1}→{letter}" for i, letter in enumerate(dist))
    dist_section = f"""
REQUIRED CORRECT ANSWER DISTRIBUTION (strictly follow — write each question so the specified option is genuinely correct):
  {dist_line}
Count: A×{dist.count('A')}, B×{dist.count('B')}, C×{dist.count('C')}, D×{dist.count('D')}
"""

    return f"""You are building the MCQ Question Bank for IKON SKILLS™, a professional micro-credentialing platform.

MICRO-CREDENTIAL: {mc['mc_name']}
DOMAIN: {mc['domain_name']}
LEVEL: {mc.get('level', 'EQF 6')}
CORE COMPETENCY {cc['cc_code']}: {cc['cc_title']}
DESCRIPTION: {cc['cc_description']}

ALL 10 COMPETENCIES IN THIS MC (for context only — do NOT test on the other 9):
{chr(10).join(f"{i+1}. {t}" for i, t in enumerate(all_cc_titles))}

YOUR TASK:
Generate exactly {BATCH_SIZE} unique MCQ questions for {assessment_type} of THIS competency.
This is BATCH {batch_num + 1} of {BATCHES_PER_FA} — questions {start_q} through {end_q} (out of {QUESTIONS_PER_FA} total for this FA).

ASSESSMENT SPECIFICATION:
{ASSESSMENT_SPECS[assessment_type]}
{prev_section}{dist_section}
CRITICAL RULES:
- EVERY question must be MCQ with exactly 4 options (A, B, C, D). NO short-answer. NO scenario simulations.
- Each question is standalone — NO "imagine you are…" or workplace role-play framings.
- Direct, simple, universally understood English. No Western idioms or culturally specific references.
- Write for working professionals in our 13 target markets: India, Bangladesh, Pakistan, Sri Lanka, Nepal, Philippines, Malaysia, Indonesia, Vietnam, Thailand, Nigeria, Kenya, South Africa.
- Focus ENTIRELY on: {cc['cc_description']}
- Ensure the {BATCH_SIZE} questions in this batch are distinct from each other AND from the already-written questions above.
- Mark exactly ONE correct answer per question. The other 3 options must be plausible distractors (not obviously wrong).
- Provide a one-sentence explanation for why the correct answer is correct (used for learner feedback).
- estimated_time_minutes: realistic time to answer (typically 1-3 minutes).

OUTPUT FORMAT — Return ONLY a JSON array of exactly {BATCH_SIZE} MCQ objects, nothing else (no commentary, no markdown):
[
  {{
    "question": "<MCQ text>",
    "option_a": "<text>",
    "option_b": "<text>",
    "option_c": "<text>",
    "option_d": "<text>",
    "correct_answer": "<A | B | C | D>",
    "explanation": "<one sentence>",
    "estimated_time_minutes": <int>
  }},
  ... ({BATCH_SIZE} total)
]"""


# ═══════════════════════════════════════════════════════════════════════════════
# CORE API CALLER
# ═══════════════════════════════════════════════════════════════════════════════

def call_claude_array(prompt: str, expected_len: int, dry_run: bool = False) -> list | None:
    """
    Call Claude and parse a JSON array response of length >= expected_len.
    Truncates to expected_len if longer. Returns None on total failure.
    """
    if dry_run:
        return [{
            "question":               f"DRY RUN MCQ #{i+1}",
            "option_a":               "Option A",
            "option_b":               "Option B",
            "option_c":               "Option C",
            "option_d":               "Option D",
            "correct_answer":         "A",
            "explanation":            "Dry run — no API call made.",
            "estimated_time_minutes": 1,
        } for i in range(expected_len)]

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw
                raw = raw.rsplit("```", 1)[0].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError(f"expected JSON array, got {type(parsed).__name__}")
            if len(parsed) < expected_len:
                raise ValueError(f"expected >= {expected_len} items, got {len(parsed)}")

            time.sleep(RATE_LIMIT_DELAY)
            return parsed[:expected_len]  # truncate if model returned extras

        except (json.JSONDecodeError, ValueError) as e:
            print(f"    ⚠ Parse error (attempt {attempt}/{RETRY_LIMIT}): {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
        except anthropic.RateLimitError:
            print(f"    ⚠ Rate limit hit (attempt {attempt}/{RETRY_LIMIT}), waiting 30s...")
            time.sleep(30)
        except anthropic.APIError as e:
            print(f"    ⚠ API error (attempt {attempt}/{RETRY_LIMIT}): {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)

    print(f"    ✗ FAILED after {RETRY_LIMIT} attempts")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CSV I/O & RESUME
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_name(mc_name: str) -> str:
    return mc_name.replace("/", "-").replace(" ", "_")


def _csv_path(mc_name: str, domain_name: str = "") -> Path:
    folder = CSV_DIR / _safe_name(domain_name) if domain_name else CSV_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{_safe_name(mc_name)}_qbank.csv"


def _save_csv(path: Path, rows: list[dict]):
    """Atomically write the full row set to CSV (replaces existing)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    os.replace(tmp, path)


def _load_partial_csv(mc_name: str, domain_name: str = "") -> tuple[list[dict], set]:
    """
    Load any previously saved CSV rows for resume.
    Drops partial batches (< BATCH_SIZE rows) and rewrites CSV without them.
    Returns (clean_rows, done_batches) where done_batches = {(cc_code, fa, batch_num)}.
    """
    path = _csv_path(mc_name, domain_name)
    if not path.exists():
        return [], set()

    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Group rows by (cc_code, assessment_type, batch_num)
    by_batch: dict[tuple, list[dict]] = {}
    for r in rows:
        try:
            cc_code = int(r["cc_code"])
            qn      = int(r["question_number"])
        except (KeyError, ValueError):
            continue
        bnum = (qn - 1) // BATCH_SIZE
        by_batch.setdefault((cc_code, r["assessment_type"], bnum), []).append(r)

    done_batches: set = set()
    clean_rows:   list[dict] = []
    dropped = 0
    for key, batch_rows in by_batch.items():
        if len(batch_rows) >= BATCH_SIZE:
            done_batches.add(key)
            clean_rows.extend(batch_rows)
        else:
            dropped += len(batch_rows)

    if dropped:
        _save_csv(path, clean_rows)
        print(f"  ↳ Dropped {dropped} orphaned row(s) from partial batches.")

    return clean_rows, done_batches


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE MC GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_single_mc(mc: dict, dry_run: bool = False, verbose: bool = True) -> dict:
    """
    Generate the full MCQ bank (1,200 MCQs) for one MC.

    `mc` must be a dict shaped like what fetch_catalog_from_api() returns:
      {
        "mc_id":int, "mc_name":str,
        "domain_id":int, "domain_name":str,
        "level":str, "source":str,
        "competencies":[{"cc_id":int, "cc_code":int,
                         "cc_title":str, "cc_description":str}, ...]
      }
    """
    path = _csv_path(mc["mc_name"], mc.get("domain_name", ""))
    all_rows, done_batches = _load_partial_csv(mc["mc_name"], mc.get("domain_name", ""))

    expected_total = len(mc["competencies"]) * len(ASSESSMENTS) * QUESTIONS_PER_FA
    expected_batches = len(mc["competencies"]) * len(ASSESSMENTS) * BATCHES_PER_FA

    if verbose:
        print(f"\n{'='*60}")
        print(f"  MC: {mc['mc_name']}")
        print(f"  Domain: {mc['domain_name']}")
        print(f"  Output: {path}")
        print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE — Claude Sonnet 4.6'}")
        if all_rows:
            print(f"  Resuming: {len(all_rows)} MCQs saved "
                  f"({len(done_batches)}/{expected_batches} batches done)")
        print(f"{'='*60}")

    total_calls  = 0
    failed_calls = 0
    errors:      list[str] = []

    # Seed prev_questions from already-saved rows (for resume continuity)
    prev_questions: dict[tuple, list[str]] = {}
    for row in all_rows:
        key = (int(row["cc_code"]), row["assessment_type"])
        prev_questions.setdefault(key, []).append(row.get("question", ""))

    for cc in mc["competencies"]:
        cc_code = cc["cc_code"]
        if verbose:
            print(f"\n  CC {cc_code:02d}: {cc['cc_title'][:60]}")

        for fa in ASSESSMENTS:
            for batch_num in range(BATCHES_PER_FA):
                key = (cc_code, fa, batch_num)
                if key in done_batches:
                    if verbose:
                        print(f"    {fa}/B{batch_num+1}  ⏭ (saved)")
                    continue

                if verbose:
                    print(f"    {fa}/B{batch_num+1}  ", end="", flush=True)

                prev_q   = prev_questions.get((cc_code, fa), [])
                ans_dist = _balanced_answer_dist(BATCH_SIZE)
                prompt   = build_mcq_batch_prompt(mc, cc, fa, batch_num, prev_q, ans_dist)
                mcqs   = call_claude_array(prompt, expected_len=BATCH_SIZE, dry_run=dry_run)
                total_calls += 1

                if mcqs is None:
                    failed_calls += 1
                    errors.append(f"CC{cc_code}/{fa}/B{batch_num+1}")
                    if verbose: print("✗ FAILED")
                    continue

                now = datetime.now(timezone.utc).isoformat()
                new_rows = []
                for i, mcq in enumerate(mcqs):
                    qn = batch_num * BATCH_SIZE + i + 1
                    new_rows.append({
                        "mc_id":            mc.get("mc_id", ""),
                        "mc_name":          mc["mc_name"],
                        "domain_id":        mc.get("domain_id", ""),
                        "domain_name":      mc.get("domain_name", ""),
                        "level":            mc.get("level", ""),
                        "source":           mc.get("source", ""),
                        "cc_id":            cc.get("cc_id", ""),
                        "cc_code":          cc_code,
                        "cc_title":         cc["cc_title"],
                        "cc_description":   cc["cc_description"],
                        "assessment_type":  fa,
                        "question_number":  qn,
                        "question":         mcq.get("question", ""),
                        "option_a":         mcq.get("option_a", ""),
                        "option_b":         mcq.get("option_b", ""),
                        "option_c":         mcq.get("option_c", ""),
                        "option_d":         mcq.get("option_d", ""),
                        "correct_answer":   mcq.get("correct_answer", ""),
                        "explanation":      mcq.get("explanation", ""),
                        "difficulty":       mcq.get("difficulty", DIFFICULTY_BY_FA[fa]),
                        "estimated_time_minutes": mcq.get("estimated_time_minutes", ""),
                        "generated_at":     now,
                        "pipeline_version": PIPELINE_VERSION,
                    })
                all_rows.extend(new_rows)
                _save_csv(path, all_rows)  # atomic full rewrite
                done_batches.add(key)
                # Track questions for next-batch dedup context
                prev_questions.setdefault((cc_code, fa), []).extend(
                    mcq.get("question", "") for mcq in mcqs
                )
                if verbose: print(f"✓ (+{BATCH_SIZE}, total {len(all_rows)})")

    # ── FINAL STATUS ──────────────────────────────────────────────────────────
    if len(all_rows) == expected_total:
        status = "COMPLETE"
    elif total_calls > 0 and failed_calls == total_calls:
        status = "FAILED"
    else:
        status = "PARTIAL"

    if verbose:
        print(f"\n  ── Summary ──────────────────────────────────────")
        print(f"  MCQs:       {len(all_rows)}/{expected_total}")
        print(f"  This run:   {total_calls} calls, {failed_calls} failed")
        print(f"  Status:     {status}")

    return {
        "mc_name":          mc["mc_name"],
        "status":           status,
        "rows_count":       len(all_rows),
        "errors":           errors,
        "csv_path":         str(path),
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CATALOG LOADER (API)
# ═══════════════════════════════════════════════════════════════════════════════

IKON_API_URL   = os.getenv("IKON_API_URL", "https://api.ikonskills.ac/lesson/competencies/")
IKON_API_TOKEN = os.getenv("IKON_API_TOKEN")


def _is_real_cc(cc: dict) -> bool:
    """Filter out placeholder/test competencies ('TEST ONLY', 'Lorem Ipsum')."""
    title = (cc.get("title") or "").strip().lower()
    desc  = (cc.get("description") or "").strip().lower()
    if not title and not desc:
        return False
    if "test only" in title or "test only" in desc:
        return False
    if desc.startswith("lorem ipsum"):
        return False
    return True


def fetch_catalog_from_api(url: str = None, token: str = None,
                           verbose: bool = True) -> list[dict]:
    """
    Fetch the MC catalog from /lesson/competencies/ and transform it for v2 output.
    Includes full metadata (mc_id, domain_id, cc_id, level, source) needed by CSV.
    """
    url   = url   or IKON_API_URL
    token = token or IKON_API_TOKEN
    if not token:
        raise RuntimeError(
            "No IKON API token found. Set IKON_API_TOKEN in your .env file "
            "or pass --token on the CLI."
        )

    if verbose:
        print(f"\n→ Fetching catalog from {url}")

    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept":        "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"IKON API HTTP {e.code}: {body}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"IKON API connection failed: {e.reason}") from None

    if not payload.get("success"):
        raise RuntimeError(f"IKON API returned error: {payload.get('message')}")

    catalog = []
    skipped = []
    for dom in payload["data"]["domains"]:
        for mc in dom.get("micro_credentials", []):
            mc_name = (mc.get("micro_credential") or "").strip()
            if not mc_name:
                continue

            real_ccs = [c for c in mc.get("competencies", []) if _is_real_cc(c)]
            if len(real_ccs) != 10:
                skipped.append((dom.get("name"), mc_name, len(real_ccs)))
                continue

            competencies = []
            for c in real_ccs:
                competencies.append({
                    "cc_id":          c.get("id"),
                    "cc_code":        c.get("code"),
                    "cc_title":       (c.get("title") or "").strip(),
                    "cc_description": (c.get("description") or "").strip(),
                })

            catalog.append({
                "mc_id":        mc.get("id"),
                "mc_name":      mc_name,
                "domain_id":    dom.get("id"),
                "domain_name":  dom.get("name", "Unknown"),
                "level":        mc.get("level", ""),
                "source":       mc.get("source", ""),
                "competencies": competencies,
            })

    if verbose:
        print(f"  ✓ Loaded {len(catalog)} MCs (10 CCs each)")
        if skipped:
            print(f"  ⚠ Skipped {len(skipped)} MCs with != 10 real CCs:")
            for cat, name, n in skipped:
                print(f"      - [{cat}] {name} ({n} CCs)")

    return catalog


def load_catalog(catalog_path: str = None, from_api: bool = True,
                 api_url: str = None, api_token: str = None) -> list[dict]:
    """Load the catalog. Default source is the IKON API; pass a path to use a local JSON file."""
    if catalog_path and not from_api:
        with open(catalog_path, encoding="utf-8") as f:
            return json.load(f)
    return fetch_catalog_from_api(url=api_url, token=api_token)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_batch(catalog: list[dict], filter_mc: str = None, filter_cat: str = None,
              dry_run: bool = False):
    """Generate banks for all MCs in the catalog (or a filtered subset)."""
    if filter_mc:
        catalog = [c for c in catalog if c["mc_name"].lower() == filter_mc.lower()]
    elif filter_cat:
        catalog = [c for c in catalog if filter_cat.lower() in c["domain_name"].lower()]

    if not catalog:
        print("No MCs matched the filter. Exiting.")
        return

    total        = len(catalog)
    completed    = 0
    failed_mcs:  list[str] = []
    manifest:    list[dict] = []
    calls_per_mc = 10 * len(ASSESSMENTS) * BATCHES_PER_FA   # 120
    mcqs_per_mc  = 10 * len(ASSESSMENTS) * QUESTIONS_PER_FA  # 1,200
    cost_per_mc  = 2.00

    print(f"\nIKON SKILLS™ MCQ Bank Generator ({PIPELINE_VERSION})")
    print(f"Model: {MODEL} | {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"MCs to process:       {total}")
    print(f"Estimated API calls:  {total * calls_per_mc:,} ({calls_per_mc} per MC)")
    print(f"Estimated MCQs:       {total * mcqs_per_mc:,} ({mcqs_per_mc:,} per MC)")
    print(f"Estimated cost:       ~${total * cost_per_mc:.2f}")
    print("=" * 60)

    start = time.time()
    for i, mc in enumerate(catalog, 1):
        print(f"\n[{i}/{total}] {mc['mc_name']}")

        path = _csv_path(mc["mc_name"], mc.get("domain_name", ""))
        if path.exists() and not dry_run:
            existing, _ = _load_partial_csv(mc["mc_name"], mc.get("domain_name", ""))
            if len(existing) >= mcqs_per_mc:
                print("  ↳ Already complete, skipping.")
                manifest.append({"mc": mc["mc_name"], "status": "SKIPPED"})
                continue
            print(f"  ↳ Partial ({len(existing)}/{mcqs_per_mc} MCQs) — resuming...")

        result = generate_single_mc(mc, dry_run=dry_run, verbose=True)

        if result["status"] == "COMPLETE":
            print(f"  ✓ Wrote {result['rows_count']} MCQs → {Path(result['csv_path']).name}")
            completed += 1
            manifest.append({"mc": mc["mc_name"], "status": "COMPLETE",
                             "csv": result["csv_path"], "rows": result["rows_count"]})
        elif result["status"] == "PARTIAL":
            print(f"  ⚠ PARTIAL ({result['rows_count']}/{mcqs_per_mc}) "
                  f"— {len(result['errors'])} batch errors: {result['errors'][:3]}")
            failed_mcs.append(mc["mc_name"])
            manifest.append({"mc": mc["mc_name"], "status": "PARTIAL",
                             "rows": result["rows_count"], "errors": result["errors"]})
        else:
            print(f"  ✗ FAILED")
            failed_mcs.append(mc["mc_name"])
            manifest.append({"mc": mc["mc_name"], "status": "FAILED"})

    manifest_path = OUTPUT_DIR / "generation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_at":     datetime.now(timezone.utc).isoformat(),
            "model":      MODEL,
            "pipeline":   PIPELINE_VERSION,
            "dry_run":    dry_run,
            "total_mcs":  total,
            "completed":  completed,
            "failed_mcs": failed_mcs,
            "elapsed_s":  round(time.time() - start, 1),
            "manifest":   manifest,
        }, f, indent=2)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"  Completed: {completed}/{total}")
    print(f"  Failed:    {len(failed_mcs)}")
    print(f"  Elapsed:   {elapsed/60:.1f} minutes")
    print(f"  Manifest:  {manifest_path}")
    if failed_mcs:
        print(f"\n  Failed MCs (re-run with --mc to retry):")
        for mc in failed_mcs:
            print(f"    - {mc}")


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL WEBHOOK
# ═══════════════════════════════════════════════════════════════════════════════

def admin_panel_publish_webhook(mc: dict) -> dict:
    """
    Entry point for backend admin panel; mc must have full API metadata
    (mc_id, mc_name, domain_id, domain_name, level, source, competencies).
    Call this when mc_status transitions to PUBLISHED.
    Runs in a separate worker — NEVER in the live practitioner session handler.
    """
    return generate_single_mc(mc, dry_run=False, verbose=False)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IKON SKILLS™ — MCQ Question Bank Generator (v2)"
    )
    parser.add_argument("--all",      action="store_true",
                        help="Generate banks for all 184 MCs (same as bare invocation)")
    parser.add_argument("--mc",       type=str, default=None,
                        help='Generate for one specific MC, e.g. --mc "AI Prompt Engineer"')
    parser.add_argument("--category", type=str, default=None,
                        help='Generate for one domain, e.g. --category "AI"')
    parser.add_argument("--dry-run",  action="store_true",
                        help="Test run without making API calls (writes dummy MCQs)")
    parser.add_argument("--catalog",  type=str, default=None,
                        help="Path to local catalog JSON (overrides API fetch)")
    parser.add_argument("--api-url",  type=str, default=None,
                        help="IKON API URL (default: from IKON_API_URL env)")
    parser.add_argument("--token",    type=str, default=None,
                        help="IKON API Bearer token (default: from IKON_API_TOKEN env)")
    args = parser.parse_args()

    # Bare `python generate_banks.py` (no filter) → run full catalog (same as --all).
    # --mc / --category still work as filters.

    catalog = load_catalog(
        catalog_path=args.catalog,
        from_api=(args.catalog is None),
        api_url=args.api_url,
        api_token=args.token,
    )
    run_batch(catalog=catalog, filter_mc=args.mc,
              filter_cat=args.category, dry_run=args.dry_run)
