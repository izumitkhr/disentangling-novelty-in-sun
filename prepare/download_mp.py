"""Download the Materials Project reference: every non-deprecated material
(experimental and theoretical) and its GGA/GGA+U thermo record, in resumable
batches.

Needs MP_API_KEY. Writes data/input/mp_gga/:
  summary.jsonl.gz          material ID, theoretical flag, chemsys, ICSD IDs
  thermo_batches/*.jsonl.gz raw MP thermo documents (entries with the relaxed
                            structures and the MP2020 adjustments)
  entries.jsonl.gz          compact entries (composition, corrected energy)
  manifest.json             query, database version, counts, checksums
Resume an interrupted download with --resume.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from mp_api.client import MPRester
from pymatgen.core.entries import ComputedEntry

ROOT = Path(__file__).resolve().parents[1]
SCHEME = "GGA_GGA+U"
SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "nelements",
    "theoretical",
    "deprecated",
    "database_IDs",
]
THERMO_FIELDS = [
    "material_id",
    "thermo_type",
    "deprecated",
    "entries",
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "decomposes_to",
    "decomposition_enthalpy",
    "decomposition_enthalpy_decomposes_to",
    "origins",
    "last_updated",
]


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def write_jsonl(path, records):
    temp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=1) as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    temp.replace(path)


def read_jsonl(path):
    with gzip.open(path, "rt") as f:
        yield from (json.loads(line) for line in f)


def normalize(doc):
    """Preserve MP corrections once; convert to compact, locally usable records."""
    if doc["thermo_type"] != SCHEME or doc.get("deprecated"):
        raise ValueError("Unexpected thermo type or deprecated thermo record")
    records = []
    for key, raw in doc["entries"].items():
        if raw is None:
            continue
        entry = ComputedEntry.from_dict(
            {
                k: v
                for k, v in raw.items()
                if k not in {"structure", "@class", "@module"}
            }
        )
        if not math.isfinite(entry.energy) or entry.composition.num_atoms <= 0:
            raise ValueError("Invalid entry energy/composition")
        records.append(
            {
                "entry_id": f"{doc['material_id']}::{key}",
                "material_id": doc["material_id"],
                "entry_type": key,
                "thermo_type": SCHEME,
                "composition": entry.composition.as_dict(),
                "energy": entry.energy,
                "energy_per_atom": entry.energy_per_atom,
                "uncorrected_energy": entry.uncorrected_energy,
                "correction": entry.correction,
                "mp_energy_above_hull": doc.get("energy_above_hull"),
                "mp_energy_per_atom": doc.get("energy_per_atom"),
                "mp_decomposition_enthalpy": doc.get("decomposition_enthalpy"),
                "task_id": raw.get("data", {}).get("task_id"),
            }
        )
    if not records:
        raise ValueError("Thermo document contains no entries")
    if not math.isclose(
        min(r["energy_per_atom"] for r in records),
        doc["energy_per_atom"],
        abs_tol=1e-6,
        rel_tol=0,
    ):
        raise ValueError("Entry energy does not reproduce MP corrected energy")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/input/mp_gga",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("MP_API_KEY"):
        parser.error("Set MP_API_KEY")
    if args.batch_size < 1 or args.batch_size > 1000:
        parser.error("--batch-size must be between 1 and 1000")
    output = args.output_dir
    path = output / "manifest.json"
    if output.exists() and not args.resume:
        parser.error("Output exists; use --resume or a new output directory")
    if args.resume and not path.exists():
        parser.error("No manifest to resume")
    output.mkdir(parents=True, exist_ok=True)
    (output / "thermo_batches").mkdir(exist_ok=True)
    manifest = (
        json.loads(path.read_text())
        if args.resume
        else {
            "status": "in_progress",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "query": {"deprecated": False},
            "thermo_type": SCHEME,
            "batch_size": args.batch_size,
            "summary_fields": SUMMARY_FIELDS,
            "thermo_fields": THERMO_FIELDS,
            "packages": {
                name: version(name) for name in ["mp-api", "pymatgen", "emmet-core"]
            },
            "energy_convention": "energy is MP-corrected total eV; do not apply corrections again",
        }
    )
    if manifest["batch_size"] != args.batch_size:
        parser.error("Resume requires the original --batch-size")
    write_json(path, manifest)
    try:
        with MPRester(
            os.environ["MP_API_KEY"], use_document_model=False, mute_progress_bars=True
        ) as mpr:
            if args.resume and manifest.get("database_version") not in {
                None,
                mpr.db_version,
            }:
                raise ValueError("Database version changed; use a new output directory")
            manifest.update(status="in_progress", database_version=mpr.db_version)
            write_json(path, manifest)
            summary_path = output / "summary.jsonl.gz"
            if not summary_path.exists():
                print(
                    "Fetching all non-deprecated material IDs (experimental and theoretical)...",
                    flush=True,
                )
                docs = mpr.materials.summary.search(
                    deprecated=False, all_fields=False, fields=SUMMARY_FIELDS
                )
                docs.sort(key=lambda d: d["material_id"])
                write_jsonl(summary_path, docs)
                del docs
            summary = list(read_jsonl(summary_path))
            ids = [d["material_id"] for d in summary]
            if (
                not ids
                or len(set(ids)) != len(ids)
                or any(d["deprecated"] for d in summary)
            ):
                raise ValueError("Invalid summary cohort")
            manifest.update(
                summary_count=len(ids),
                theoretical_counts=dict(
                    Counter(str(d["theoretical"]) for d in summary)
                ),
            )
            write_json(path, manifest)
            print(f"Reference cohort: {len(ids):,} materials", flush=True)
            for start in range(0, len(ids), args.batch_size):
                shard = output / "thermo_batches" / f"{start:07d}.jsonl.gz"
                if not shard.exists():
                    batch_ids = ids[start : start + args.batch_size]
                    docs = mpr.materials.thermo.search(
                        material_ids=batch_ids,
                        thermo_types=[SCHEME],
                        all_fields=False,
                        fields=THERMO_FIELDS,
                    )
                    seen = set()
                    for doc in docs:
                        if (
                            doc["material_id"] not in batch_ids
                            or doc["material_id"] in seen
                        ):
                            raise ValueError("Unexpected/duplicate thermo material ID")
                        seen.add(doc["material_id"])
                        normalize(doc)
                    write_jsonl(shard, docs)
                manifest["materials_queried"] = min(start + args.batch_size, len(ids))
                write_json(path, manifest)
                print(
                    f"Thermo: {manifest['materials_queried']:,}/{len(ids):,} materials",
                    flush=True,
                )
        metadata = {d["material_id"]: d for d in summary}
        covered, entry_ids = set(), set()

        def records():
            for shard in sorted((output / "thermo_batches").glob("*.jsonl.gz")):
                for doc in read_jsonl(shard):
                    if doc["material_id"] in covered:
                        raise ValueError("Duplicate material across batches")
                    covered.add(doc["material_id"])
                    for row in normalize(doc):
                        if row["entry_id"] in entry_ids:
                            raise ValueError("Duplicate entry ID")
                        entry_ids.add(row["entry_id"])
                        row["theoretical"] = metadata[row["material_id"]]["theoretical"]
                        yield row

        entry_path = output / "entries.jsonl.gz"
        write_jsonl(entry_path, records())
        with entry_path.open("rb") as f:
            checksum = hashlib.file_digest(f, "sha256").hexdigest()
        manifest.update(
            status="complete",
            finished_at_utc=datetime.now(UTC).isoformat(),
            thermo_count=len(covered),
            entry_count=len(entry_ids),
            materials_without_gga_thermo=sorted(set(ids) - covered),
            entries_sha256=checksum,
        )
        write_json(path, manifest)
        print(
            f"Complete: {len(entry_ids):,} entries from {len(covered):,} materials; {output}",
            flush=True,
        )
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__)
        write_json(path, manifest)
        print(
            f"Failed ({type(exc).__name__}); completed batches retained for --resume.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
