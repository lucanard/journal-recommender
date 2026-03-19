#!/usr/bin/env python3
"""
Master Enrichment Pipeline Runner
====================================
Runs all enrichment steps in sequence with progress tracking and resume support.

Usage:
    python run_enrichment.py              # Run all steps
    python run_enrichment.py --step 2     # Run from step 2 (OpenAlex)
    python run_enrichment.py --step 4     # Run only final merge/export
    python run_enrichment.py --reset      # Clear all progress and start fresh
    python run_enrichment.py --status     # Show current progress without running
"""

import subprocess
import sys
import os
import json
import argparse
from datetime import datetime


STEPS = [
    {
        "num": 1,
        "name": "DOAJ API Enrichment",
        "script": "01_enrich_doaj.py",
        "output": "enriched_doaj.json",
        "progress": "doaj_progress.json",
        "description": "Fetches aims & scope, subjects, APC, and OA details from DOAJ",
        "est_time": "~90 min for 9k journals (0.6s/request)",
    },
    {
        "num": 2,
        "name": "OpenAlex Enrichment",
        "script": "02_enrich_openalex.py",
        "output": "enriched_openalex.json",
        "progress": "openalex_progress.json",
        "description": "Fetches topics, impact metrics, publisher info from OpenAlex",
        "est_time": "~25 min with polite pool, ~170 min without",
    },
    {
        "num": 3,
        "name": "CrossRef Enrichment",
        "script": "03_enrich_crossref.py",
        "output": "enriched_crossref.json",
        "progress": "crossref_progress.json",
        "description": "Fetches subject areas, publisher, activity status from CrossRef",
        "est_time": "~15 min with polite pool",
    },
    {
        "num": 4,
        "name": "Merge & Export",
        "script": "04_merge_and_export.py",
        "output": "journal_database_final.json",
        "progress": None,
        "description": "Deduplicates, scores completeness, exports all formats",
        "est_time": "<1 min",
    },
]


def show_status():
    """Show current progress of each step."""
    print("\n" + "=" * 60)
    print(" ENRICHMENT PIPELINE STATUS")
    print("=" * 60)
    
    for step in STEPS:
        output_exists = os.path.exists(step["output"]) if step["output"] else False
        progress_exists = os.path.exists(step["progress"]) if step["progress"] else False
        
        status = "Not started"
        details = ""
        
        if output_exists:
            status = "✅ Complete"
            size = os.path.getsize(step["output"])
            status += f" ({size/1024/1024:.1f} MB)"
        elif progress_exists:
            status = "🔄 In progress"
            try:
                with open(step["progress"], encoding="utf-8") as f:
                    prog = json.load(f)
                details = f"  → {len(prog)} journals processed so far"
            except:
                pass
        else:
            status = "⬚ Not started"
        
        print(f"\n  Step {step['num']}: {step['name']}")
        print(f"  Status: {status}")
        if details:
            print(details)
        print(f"  Est. time: {step['est_time']}")
    
    print("\n" + "=" * 60)
    
    # Check for required input
    if not os.path.exists("journal_database.json"):
        print("\n⚠  WARNING: journal_database.json not found!")
        print("  Copy it to this directory before running the pipeline.")
    else:
        with open("journal_database.json", encoding="utf-8") as f:
            count = len(json.load(f))
        print(f"\n  Base database: {count:,} journals ready")
    
    print()


def reset_progress():
    """Clear all progress files to start fresh."""
    progress_files = [
        "doaj_progress.json",
        "openalex_progress.json",
        "crossref_progress.json",
    ]
    output_files = [
        "enriched_doaj.json",
        "enriched_openalex.json",
        "enriched_crossref.json",
        "journal_database_final.json",
        "journal_embeddings_input.jsonl",
        "journal_database_enriched.xlsx",
        "enrichment_report.txt",
    ]
    
    deleted = 0
    for f in progress_files + output_files:
        if os.path.exists(f):
            os.remove(f)
            deleted += 1
    
    print(f"Reset complete: {deleted} files removed.")


def run_step(step):
    """Run a single enrichment step."""
    print(f"\n{'='*60}")
    print(f" STEP {step['num']}: {step['name']}")
    print(f" {step['description']}")
    print(f" Estimated time: {step['est_time']}")
    print(f"{'='*60}\n")
    
    script = step["script"]
    if not os.path.exists(script):
        print(f"ERROR: Script not found: {script}")
        return False
    
    start = datetime.now()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = datetime.now() - start
    
    if result.returncode != 0:
        print(f"\n❌ Step {step['num']} failed with return code {result.returncode}")
        return False
    
    print(f"\n✅ Step {step['num']} completed in {elapsed}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Journal Enrichment Pipeline")
    parser.add_argument("--step", type=int, help="Start from this step number (1-4)")
    parser.add_argument("--reset", action="store_true", help="Clear all progress")
    parser.add_argument("--status", action="store_true", help="Show current status")
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return
    
    if args.reset:
        reset_progress()
        return
    
    start_step = args.step or 1
    
    print("\n" + "=" * 60)
    print(" AI JOURNAL RECOMMENDATION — ENRICHMENT PIPELINE")
    print(f" Starting from step {start_step}")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Pre-flight checks
    if not os.path.exists("journal_database.json"):
        print("\n❌ journal_database.json not found in current directory!")
        print("   Copy it here before running the pipeline.")
        sys.exit(1)
    
    if start_step == 1:
        print("\n⏱  Full pipeline estimated time: ~2-3 hours")
        print("   (Each step saves progress — you can interrupt and resume)\n")
    
    # Run steps
    for step in STEPS:
        if step["num"] < start_step:
            continue
        
        success = run_step(step)
        if not success:
            print(f"\nPipeline stopped at step {step['num']}.")
            print(f"Fix the issue and resume with: python run_enrichment.py --step {step['num']}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print(" 🎉 ENRICHMENT PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nProduced files:")
    print("  journal_database_final.json    — Production database")
    print("  journal_embeddings_input.jsonl  — Ready for embedding generation")
    print("  journal_database_enriched.xlsx  — Excel export")
    print("  enrichment_report.txt           — Quality report")
    print("\nNext: Generate embeddings and set up vector search!")
    print()


if __name__ == "__main__":
    main()
