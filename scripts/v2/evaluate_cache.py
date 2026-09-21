"""Evaluate a newly fitted HTAN cache without repeating either alignment."""

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np

from real_data import metrics
from scdebussy.align._api import _source_fingerprint


def main():
    """Check cache identity/provenance and report the frozen real-data metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adata', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contract', type=Path, default=Path(__file__).with_name('contract_v21.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('choose a new output path; prior reports are not overwritten')
    data = ad.read_h5ad(args.adata)
    expected_patients = {"RU1083", "RU1303", "RU1444", "RU151", "RU263", "RU581", "RU831", "RU942"}
    observed_patients = set(data.obs.patient.astype(str).unique())
    if observed_patients != expected_patients:
        raise ValueError(
            "the HTAN cache evaluator requires the eight-patient RU942-updated cohort "
            f"without RU1646; observed={sorted(observed_patients)}"
        )
    cache = np.load(args.cache, allow_pickle=False)
    if not np.array_equal(cache['cell_ids'], np.asarray(data.obs_names, str)):
        raise ValueError('cached cell IDs/order differ from input')
    if not np.array_equal(cache['v2_genes'], np.asarray(data.var_names, str)):
        raise ValueError('cached gene IDs/order differ from input')
    provenance = json.loads(str(cache['v2_provenance']))
    if provenance['source_sha256'] != _source_fingerprint():
        raise ValueError('cached fit used different source')
    config = json.loads(str(cache['v2_config']))
    contract = json.loads(args.contract.read_text())
    data.obs['cached_v1'] = cache['v1_pos']
    data.obs['cached_v2'] = cache['v2_pos']
    baseline = metrics(data, 'cached_v1', 100)
    candidate = metrics(data, 'cached_v2', 100)
    data.obs['cached_supported'] = np.where(cache['v2_status'] == 'supported', cache['v2_pos'], np.nan)
    supported = metrics(data, 'cached_supported', 100)
    patients = list(dict.fromkeys(cache['cell_patient']))
    offsets = cache['v2_landmark_offsets']
    descriptions = {}
    for k, patient in enumerate(patients):
        selected = cache['cell_patient'] == patient
        positions = cache['v2_pos'][selected]
        finite = positions[np.isfinite(positions)]
        descriptions[patient] = dict(
            landmarks=cache['v2_landmark_coordinates'][offsets[k]:offsets[k+1]].tolist(),
            landmark_count=int(offsets[k+1]-offsets[k]),
            position_min=float(finite.min()) if len(finite) else None,
            position_max=float(finite.max()) if len(finite) else None,
            retained_mass=float(np.mean(1-cache['v2_unprojected'][selected])),
        )
    limits = contract['thresholds']
    reproduced = (abs(baseline['coverage']-limits['baseline_coverage_reference']) <= limits['real_coverage_tolerance']
                  and abs(baseline['purity']-limits['baseline_purity_reference']) <= 0.02)
    passed = (reproduced and bool(cache['v2_converged'])
              and abs(candidate['coverage']-baseline['coverage']) <= limits['real_coverage_tolerance']
              and candidate['purity'] >= limits['real_purity_min'])
    report = dict(
        status='completed', input=str(args.adata), cache=str(args.cache),
        n_cells=data.n_obs, n_genes=data.n_vars, config=config,
        baseline=baseline, candidate=candidate, candidate_supported_only=supported,
        patients=descriptions, converged=bool(cache['v2_converged']), iterations=int(cache['v2_iterations']),
        consensus_coordinates=cache['v2_coordinates'].tolist(), provenance=provenance,
        config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        contract_sha256=hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        baseline_reproduced=reproduced, gate_d_numeric_pass=passed, replacement_recommended=False,
        note='Unit/smoke success does not establish scientific replacement; full revised suites remain outstanding.',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(baseline=baseline, candidate=candidate, gate_d_numeric_pass=passed), indent=2))


if __name__ == '__main__':
    main()
