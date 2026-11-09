"""Snapshot of Codex's strict convergence monitor (v1), frozen for the benchmark.

Source: /home/wangm10/scdebussy-span-test/01_simulation/sim_v2_convergence_core.py
sha256 of the source: 6d5dded1e54edee1fdf6aeaa7c358525c4706db60354ff42eb594dd2f1802b67
Copied 2026-10-26 unchanged apart from this header.
"""
import json
from pathlib import Path
import numpy as np

OBJECTIVE_TOL=1e-6
BARYCENTER_TOL=1e-5
STATIONARITY_TOL=1e-5
COORDINATE_TOL=1e-4
MASS_TOL=1e-4
INCREASE_TOL=1e-8
PATIENCE=5

def penalized_objective(posterior,barycenter,difference,gamma,beta):
    nll=-sum(p['log_z'] for p in posterior.values())
    penalty=beta*gamma/barycenter.shape[1]*np.sum((difference@barycenter)**2)
    return float(nll+penalty),float(nll),float(penalty)

def stationarity(expression,posterior,barycenter,patients,difference,gamma):
    numerator=sum(posterior[p]['matrix'].T@expression[p] for p in patients)
    weights=sum(posterior[p]['matrix'].sum(axis=0) for p in patients)
    residual=weights[:,None]*barycenter+gamma*difference.T@(difference@barycenter)-numerator
    return float(np.linalg.norm(residual)/max(1.,np.linalg.norm(numerator)))

def stable_update(relative_objective,relative_barycenter,residual,coordinate_change,mass_change,objective_increase):
    return bool(np.isfinite([relative_objective,relative_barycenter,residual,coordinate_change,mass_change,objective_increase]).all()
        and abs(relative_objective)<=OBJECTIVE_TOL and relative_barycenter<=BARYCENTER_TOL
        and residual<=STATIONARITY_TOL and coordinate_change<=COORDINATE_TOL
        and mass_change<=MASS_TOL and objective_increase<=INCREASE_TOL)

def fit_until_stable(expression,graphs,barycenter,posterior,patients,all_patients,
        local,observed,gamma,beta,difference,expectation,update,project,
        max_iter,checkpoint_dir,min_iter=100):
    checkpoint_dir=Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    history=[]
    stable=legacy_stable=0
    converged=False
    legacy_iteration=first_strict_iteration=None
    projected,_,_=project(posterior,local,observed,all_patients)
    def save(name,iteration):
        np.savez_compressed(checkpoint_dir/name,barycenter=barycenter,coordinate=projected,
            patient_order=np.asarray(patients),posterior_matrix=np.stack([posterior[p]['matrix'] for p in patients]),
            history=json.dumps(history),iteration=iteration,strict_converged=converged,
            legacy_iteration=-1 if legacy_iteration is None else legacy_iteration)
    for iteration in range(1,max_iter+1):
        before,nll_before,_=penalized_objective(posterior,barycenter,difference,gamma,beta)
        updated,support=update(expression,posterior,barycenter,patients,gamma)
        following=expectation(expression,updated,graphs,patients)
        after,nll_after,penalty=penalized_objective(following,updated,difference,gamma,beta)
        next_projected,_,_=project(following,local,observed,all_patients)
        same_mask=np.array_equal(np.isfinite(projected),np.isfinite(next_projected))
        common=np.isfinite(projected)&np.isfinite(next_projected)
        coord_change=float(np.max(np.abs(projected[common]-next_projected[common]))) if same_mask and common.any() else float('inf')
        mass_change=max(float(np.max(np.abs(following[p]['matrix'].sum(axis=1)-posterior[p]['matrix'].sum(axis=1)))) for p in patients)
        relative=(after-before)/max(1.,abs(before))
        relative_barycenter=float(np.linalg.norm(updated-barycenter)/max(1.,np.linalg.norm(barycenter)))
        residual=stationarity(expression,following,updated,patients,difference,gamma)
        valid=stable_update(relative,relative_barycenter,residual,coord_change,mass_change,max(0.,relative))
        stable=stable+1 if valid else 0
        legacy_relative=(nll_before-nll_after)/max(1.,abs(nll_before))
        legacy_stable=legacy_stable+1 if legacy_relative<=1e-6 else 0
        legacy_now=legacy_iteration is None and legacy_stable>=3
        if legacy_now:legacy_iteration=iteration
        converged=stable>=PATIENCE
        if converged and first_strict_iteration is None:first_strict_iteration=iteration
        history.append({'iteration':iteration,'penalized_objective':after,'negative_log_partition':nll_after,
            'smoothing_penalty':penalty,'relative_objective_change':relative,
            'legacy_relative_improvement':legacy_relative,'relative_barycenter_change':relative_barycenter,
            'barycenter_max_change':float(np.max(np.abs(updated-barycenter))),
            'stationarity_residual':residual,'maximum_coordinate_change':coord_change,
            'maximum_match_mass_change':mass_change,'min_consensus_match_support':float(support.min()),
            'strict_stable_updates':stable,'legacy_would_stop':legacy_now})
        barycenter,posterior,projected=updated,following,next_projected
        if legacy_now:save('legacy_stop.npz',iteration)
        if iteration in (100,200,400,800,1600):
            save(f'iteration_{iteration}.npz',iteration)
            print(f'CHECKPOINT iteration={iteration} objective={after:.8g} stationarity={residual:.3g} coordinate_change={coord_change:.3g} stable={stable}',flush=True)
        if converged and iteration>=min_iter:break
    save('final_checkpoint.npz',iteration)
    (checkpoint_dir/'termination.json').write_text(json.dumps({'converged':converged,
        'iterations':iteration,'reason':'strict_convergence' if converged else 'iteration_cap',
        'first_strict_iteration':first_strict_iteration,'legacy_iteration':legacy_iteration,
        'objective_increase_count':sum(h['relative_objective_change']>INCREASE_TOL for h in history)},indent=2)+'\n')
    return barycenter,posterior,history,converged
