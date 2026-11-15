"""Single-point MLIP energies for every MP reference entry, with MP's MP2020 adjustments.

Evaluates the MP relaxed structure of every entry of data/input/mp_gga
(prepare/download_mp.py) with MACE-OMAT-0 and writes a reference directory
with the same layout (entries.jsonl.gz, summary.jsonl.gz, manifest.json) to
data/input/mp_mace_omat_0_medium. The analyses replace the MP2020
corrections by their OMat24 refit on loading (scripts/analysis/corrections).
Resume an interrupted run with --resume.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor

ROOT = Path(__file__).resolve().parents[1]
LABEL = "MACE-OMAT-0-medium"  # thermo_type stored with every entry


def relative_to_repo(path):
    """Path relative to the repository (symbolic links are not followed)."""
    path = Path(os.path.abspath(path))
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def load_summary(reference_dir):
    theoretical = {}
    with gzip.open(reference_dir / "summary.jsonl.gz", "rt") as f:
        for line in f:
            d = json.loads(line)
            theoretical[d["material_id"]] = bool(d["theoretical"])
    return theoretical


def make_calculator(model_path, device, dtype):
    from mace.calculators import MACECalculator

    return MACECalculator(
        model_paths=[str(model_path)], device=device, default_dtype=dtype
    )


def mlip_energy(calc, structure):
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calc
    return float(atoms.get_potential_energy())


def corrected_entry(structure, raw_energy, original):
    """Re-create the MP entry with the MLIP energy and MP's own MP2020 adjustments."""
    return ComputedStructureEntry(
        structure,
        raw_energy,
        energy_adjustments=list(original.energy_adjustments),
        parameters=original.parameters,
        data=original.data,
        entry_id=original.entry_id,
    )


def process_batch(path, calc, theoretical):
    records = []
    with gzip.open(path, "rt") as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("deprecated"):
                raise ValueError("Deprecated thermo record")
            for key, raw in doc["entries"].items():
                original = ComputedStructureEntry.from_dict(raw)
                structure = original.structure
                energy = mlip_energy(calc, structure)
                if not math.isfinite(energy):
                    raise ValueError(f"Non-finite MLIP energy for {doc['material_id']}")
                entry = corrected_entry(structure, energy, original)
                if abs(entry.uncorrected_energy - energy) > 1e-8:
                    raise ValueError("Uncorrected energy changed during processing")
                records.append(
                    {
                        "entry_id": f"{doc['material_id']}::{key}",
                        "material_id": doc["material_id"],
                        "entry_type": key,
                        "thermo_type": LABEL,
                        "composition": entry.composition.as_dict(),
                        "energy": entry.energy,
                        "energy_per_atom": entry.energy_per_atom,
                        "uncorrected_energy": entry.uncorrected_energy,
                        "correction": entry.correction,
                        "energy_adjustments": [
                            {"name": a.name, "value": a.value}
                            for a in entry.energy_adjustments
                        ],
                        "nsites": len(structure),
                        "dft_uncorrected_energy": original.uncorrected_energy,
                        "dft_correction": original.correction,
                        "dft_energy": original.energy,
                        "mp_energy_above_hull": doc.get("energy_above_hull"),
                        "mp_energy_per_atom": doc.get("energy_per_atom"),
                        "mp_decomposition_enthalpy": doc.get("decomposition_enthalpy"),
                        "task_id": raw.get("data", {}).get("task_id"),
                        "theoretical": theoretical[doc["material_id"]],
                    }
                )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "data/input/mp_gga",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "data/models/mace-omat-0-medium.model",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/input/mp_mace_omat_0_medium"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit-batches", type=int, help="Process only the first N thermo batches"
    )
    args = parser.parse_args()
    output = args.output_dir
    source = json.loads((args.reference_dir / "manifest.json").read_text())
    if source["status"] != "complete":
        raise ValueError("Reference download is not complete")
    batch_dir = output / "mlip_batches"
    if output.exists() and not args.resume:
        parser.error("Output directory exists; use --resume or a new directory")
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = {
        "status": "in_progress",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "source_reference_dir": relative_to_repo(args.reference_dir),
        "source_entries_sha256": source["entries_sha256"],
        "database_version": source["database_version"],
        "thermo_type": LABEL,
        "energy_source": "MLIP single point on the MP relaxed structure of each entry",
        "correction_source": (
            "energy_adjustments stored by MP for the same entry (not recomputed)"
        ),
        "model_path": relative_to_repo(args.model),
        "model_sha256": sha256(args.model),
        "device": args.device,
        "dtype": args.dtype,
        "mp2020_applied": True,
        "energy_convention": (
            "energy is MLIP total eV with MP2020 corrections; do not apply "
            "corrections again"
        ),
        "packages": {
            name: version(name) for name in ["mace-torch", "torch", "pymatgen", "ase"]
        },
        "python": sys.version.split()[0],
        "script_sha256": sha256(Path(__file__)),
    }
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ["model_sha256", "dtype", "mp2020_applied", "thermo_type"]:
            if previous.get(key) != manifest[key]:
                raise ValueError(f"Resume settings differ from the previous run: {key}")
        manifest["started_at_utc"] = previous["started_at_utc"]
    write_json(manifest_path, manifest)

    theoretical = load_summary(args.reference_dir)
    calc = make_calculator(args.model, args.device, args.dtype)
    batches = sorted((args.reference_dir / "thermo_batches").glob("*.jsonl.gz"))
    if args.limit_batches:
        batches = batches[: args.limit_batches]
    for i, path in enumerate(batches, 1):
        target = batch_dir / path.name
        if args.resume and target.exists():
            continue
        records = process_batch(path, calc, theoretical)
        temp = target.with_suffix(".tmp")
        with gzip.open(temp, "wt") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        temp.replace(target)
        print(f"Batch {i}/{len(batches)} done ({len(records)} entries)", flush=True)

    entries_path = output / "entries.jsonl.gz"
    count = 0
    with gzip.open(entries_path, "wt") as out:
        for path in batches:
            with gzip.open(batch_dir / path.name, "rt") as f:
                for line in f:
                    out.write(line)
                    count += 1
    shutil.copyfile(
        args.reference_dir / "summary.jsonl.gz", output / "summary.jsonl.gz"
    )
    manifest.update(
        status="complete",
        finished_at_utc=datetime.now(UTC).isoformat(),
        n_batches=len(batches),
        entry_count=count,
        entries_sha256=sha256(entries_path),
        summary_count=source["summary_count"],
        summary_sha256=sha256(output / "summary.jsonl.gz"),
        source_entry_count=source["entry_count"],
    )
    write_json(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ["status", "entry_count", "n_batches"]}))


if __name__ == "__main__":
    main()
