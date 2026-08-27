"""
sanity_check.py
----------------
Run this BEFORE embedding anything. Confirms the corpus is actually usable:
files exist, aren't empty/corrupt, and each tier has enough documents to be
meaningful. Cheap to run (seconds, no models loaded) — catches problems that
would otherwise surface confusingly deep inside the embedding step.

Usage:
    python sanity_check.py --corpus ./corpus
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def sanity_check(corpus_dir: str = "./corpus"):
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ No manifest.json found in {corpus_dir} — run corpus_selection.py first.")
        return

    manifest = json.loads(manifest_path.read_text())
    print(f"Total records in manifest: {len(manifest)}")

    tier_counts = Counter(r["tier"] for r in manifest)
    modality_counts = Counter(r["modality"] for r in manifest)
    print("\nPer-tier counts:")
    for tier, n in sorted(tier_counts.items()):
        print(f"  {tier:25s} {n}")
    print("\nPer-modality counts:", dict(modality_counts))

    missing, empty_text, bad_images = [], [], []
    for r in manifest:
        p = Path(r["path"])
        if not p.exists():
            missing.append(r["path"])
            continue
        if r["modality"] == "text":
            if p.stat().st_size == 0:
                empty_text.append(r["path"])
        elif r["modality"] == "image":
            try:
                from PIL import Image
                Image.open(p).verify()
            except Exception:
                bad_images.append(r["path"])

    print(f"\nMissing files     : {len(missing)}")
    print(f"Empty text files  : {len(empty_text)}")
    print(f"Unreadable images : {len(bad_images)}")

    if missing or empty_text or bad_images:
        print("\nNote: broken records are skipped automatically during embedding,")
        print("but if a tier's usable count looks too low, re-run corpus_selection.py")
        print("for just that tier.")
    else:
        print("\nAll records present and readable.")

    print("\nMinimum-size check (rule of thumb: >=20 per tier to be meaningful):")
    for tier, n in sorted(tier_counts.items()):
        flag = "OK" if n >= 20 else "LOW — consider re-running this tier with a higher n_docs"
        print(f"  {tier:25s} {n:4d}  {flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="./corpus")
    args = parser.parse_args()
    sanity_check(args.corpus)
