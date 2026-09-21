"""RU831 diagnostics, part 1: raw cost-surface geometry.

For RU831, RU1303, and a control patient, on both the pre-joint
(trajectory_250_unit) and one joint-warp candidate's cost matrices.

Extends validation/v21/experiment_round/cost_diagnostics.py's visualization
(np.log1p(cost/median) convention, row-argmin "best correspondence" overlay) to
an arbitrary patient list, adds a raw-scale figure per patient, and adds a
qualitative valley-shape characterization (single monotone valley vs multiple
comparable valleys vs diffuse low-cost region vs abrupt jumps vs early
termination), following JOINT_WARP_CONTRACT / RU831 diagnostic task instructions.

Does not modify or re-derive cost matrices: reuses cached `cost_matrices` from
existing .npz fit caches (match costs do not depend on omega_warp/
lambda_terminal_gap, only which edges are cheap to traverse -- this script
confirms that, it does not assume it).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    """Load an .npz fit cache without pickling."""
    return np.load(path, allow_pickle=False)


def patient_slice(data, patient):
    """Return (cost, local_coords, matchable_mask, consensus_coords) for one patient."""
    names = data["patient_order"].tolist()
    p = names.index(patient)
    offsets = data["landmark_offsets"]
    start, stop = offsets[p], offsets[p + 1]
    cost = data["cost_matrices"][p]
    local = data["landmark_coordinates"][start:stop]
    valid = data["landmark_matchable"][start:stop]
    consensus = data["consensus_coordinates"]
    return cost, local, valid, consensus


def valley_characterization(cost, local, valid, consensus, row_gap_factor=3.0, competing_frac=1.10):
    """Qualitative + quantitative description of the row-argmin path shape.

    - "jump": |consecutive argmin consensus position deltas| for matchable rows
      compared to the median local landmark spacing; rows whose jump exceeds
      row_gap_factor times the median matchable-row spacing count as abrupt.
    - "competing minima": for each matchable row, count consensus positions
      whose cost is within `competing_frac` of the row minimum (i.e. cost <=
      competing_frac * min(cost[row])) and are NOT contiguous with the argmin
      (more than 2 landmark-cells away) -- evidence of a second, comparably
      cheap valley for that row.
    - "early termination": whether the row-argmin path for the *last* 10% of
      matchable rows stays close to a smooth continuation of the rest of the
      path (residual from a linear fit on the first 80% of matchable rows,
      extrapolated), or departs from it.
    """
    valid_ix = np.flatnonzero(valid)
    if len(valid_ix) < 5:
        return {"n_matchable": int(len(valid_ix))}
    best = np.argmin(cost, axis=1)
    best_pos = consensus[best]
    row_min = cost[np.arange(len(cost)), best]

    # abrupt jumps in the row-argmin path (matchable rows only, in landmark order)
    ordered = valid_ix  # landmark index order == local pseudotime order (fixed landmarks)
    local_spacing = np.median(np.diff(local[ordered])) if len(ordered) > 1 else np.nan
    jumps = np.abs(np.diff(best_pos[ordered]))
    jump_threshold = row_gap_factor * max(local_spacing, 1e-6)
    n_abrupt = int(np.sum(jumps > jump_threshold))

    # competing minima: count of well-separated near-minimal consensus cells per row
    n_multi = 0
    multi_rows = []
    for i in ordered:
        row = cost[i]
        thresh = row_min[i] * competing_frac if row_min[i] > 0 else row_min[i] + 1e-9
        near = np.flatnonzero(row <= thresh)
        # cluster near-minimal indices into contiguous runs; count runs beyond the argmin's run
        if len(near) < 2:
            continue
        splits = np.flatnonzero(np.diff(near) > 2)
        runs = np.split(near, splits + 1)
        if len(runs) > 1:
            n_multi += 1
            multi_rows.append(int(i))

    # early termination: compare last 10% matchable rows' argmin path to a
    # linear extrapolation fit on the first 80% of matchable rows
    n_ord = len(ordered)
    fit_ix = ordered[: max(2, int(0.8 * n_ord))]
    tail_ix = ordered[-max(1, int(0.1 * n_ord)) :]
    if len(fit_ix) >= 2:
        coeffs = np.polyfit(local[fit_ix], best_pos[fit_ix], 1)
        predicted_tail = np.polyval(coeffs, local[tail_ix])
        tail_residual = float(np.mean(np.abs(best_pos[tail_ix] - predicted_tail)))
    else:
        tail_residual = float("nan")

    return {
        "n_matchable": int(len(valid_ix)),
        "local_spacing_median": float(local_spacing),
        "jump_threshold": float(jump_threshold),
        "n_abrupt_jumps": n_abrupt,
        "max_jump": float(jumps.max()) if len(jumps) else float("nan"),
        "n_rows_with_competing_minima": n_multi,
        "frac_rows_with_competing_minima": float(n_multi / n_ord) if n_ord else float("nan"),
        "competing_minima_rows": multi_rows,
        "tail_residual_from_linear_fit": tail_residual,
        "row_min_cost_median": float(np.median(row_min[ordered])),
        "row_min_cost_p90": float(np.percentile(row_min[ordered], 90)),
    }


def plot_patient(ax_log, ax_raw, cost, local, valid, consensus, title):
    """Draw the log-scale and raw-scale cost-surface panels for one patient."""
    for ax, transform, cmap, label in (
        (ax_log, lambda c: np.log1p(c / max(np.median(c), 1e-12)), "viridis", "log1p(cost/median)"),
        (ax_raw, lambda c: c, "magma", "raw cost"),
    ):
        image = transform(cost)
        im = ax.imshow(image, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap=cmap)
        ax.plot(local, local, color="white", lw=1, ls="--", label="identity")
        best = np.argmin(cost, axis=1)
        colors = np.where(valid, "#ff6b6b", "#888888")
        ax.scatter(consensus[best], local, c=colors, s=4, label="row minimum")
        ax.set_title(f"{title} ({label})", fontsize=9)
        ax.set_xlabel("consensus position")
        ax.set_ylabel("patient landmark position")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    """CLI entry point: render cost-geometry figures and write summary CSV/JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pre-joint", type=Path, required=True, help="trajectory_250_unit.npz (pre-joint cost matrices)"
    )
    parser.add_argument("--joint-candidate", type=Path, required=True, help="one joint_warp_*.npz")
    parser.add_argument("--patients", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pre = load(args.pre_joint)
    joint = load(args.joint_candidate)

    rows = []
    # confirm cost matrices are (numerically) identical between pre-joint and
    # joint-candidate fits, as the task instructs to check rather than assume
    identity_check = {}
    for patient in args.patients:
        cost_pre, local_pre, valid_pre, cons_pre = patient_slice(pre, patient)
        cost_joint, local_joint, valid_joint, cons_joint = patient_slice(joint, patient)
        same_shape = cost_pre.shape == cost_joint.shape
        max_abs_diff = float(np.max(np.abs(cost_pre - cost_joint))) if same_shape else float("nan")
        rel_diff = max_abs_diff / max(float(np.median(cost_pre)), 1e-12)
        identity_check[patient] = {
            "same_shape": bool(same_shape),
            "max_abs_diff": max_abs_diff,
            "max_abs_diff_relative_to_median_cost": rel_diff,
        }

    fig, axes = plt.subplots(len(args.patients), 4, figsize=(20, 5 * len(args.patients)), dpi=150)
    if len(args.patients) == 1:
        axes = axes[None, :]
    for row, patient in enumerate(args.patients):
        cost_pre, local_pre, valid_pre, cons_pre = patient_slice(pre, patient)
        cost_joint, local_joint, valid_joint, cons_joint = patient_slice(joint, patient)

        plot_patient(
            axes[row, 0],
            axes[row, 1],
            cost_pre,
            local_pre,
            valid_pre,
            cons_pre,
            f"{patient}: pre-joint (trajectory_250_unit)",
        )
        plot_patient(
            axes[row, 2],
            axes[row, 3],
            cost_joint,
            local_joint,
            valid_joint,
            cons_joint,
            f"{patient}: joint candidate ({args.joint_candidate.stem})",
        )

        char_pre = valley_characterization(cost_pre, local_pre, valid_pre, cons_pre)
        char_joint = valley_characterization(cost_joint, local_joint, valid_joint, cons_joint)
        rows.append({"patient": patient, "source": "pre_joint", **char_pre})
        rows.append({"patient": patient, "source": "joint_candidate", **char_joint})

    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "cost_geometry_grid.png", bbox_inches="tight")
    fig.savefig(args.output_dir / "cost_geometry_grid.pdf", bbox_inches="tight")
    plt.close(fig)

    # per-patient single-panel raw-scale figures (explicit raw-scale requirement)
    for patient in args.patients:
        cost_pre, local_pre, valid_pre, cons_pre = patient_slice(pre, patient)
        fig, ax = plt.subplots(figsize=(6, 5.5), dpi=170)
        im = ax.imshow(cost_pre, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="magma")
        ax.plot(local_pre, local_pre, "w--", lw=1, label="identity")
        best = np.argmin(cost_pre, axis=1)
        ax.scatter(cons_pre[best], local_pre, c=np.where(valid_pre, "#00e5ff", "#888888"), s=5, label="row minimum")
        ax.set(
            xlabel="consensus position",
            ylabel=f"{patient} local landmark",
            title=f"{patient}: raw-scale cost (pre-joint)",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{patient}_raw_scale_cost.png", bbox_inches="tight")
        plt.close(fig)

    fieldnames = sorted({k for r in rows for k in r})
    with (args.output_dir / "cost_geometry_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r = dict(r)
            r["competing_minima_rows"] = ";".join(map(str, r.get("competing_minima_rows", [])))
            writer.writerow(r)

    (args.output_dir / "cost_matrix_identity_check.json").write_text(json.dumps(identity_check, indent=2) + "\n")
    print(json.dumps(identity_check, indent=2))
    print("wrote:", args.output_dir / "cost_geometry_summary.csv")


if __name__ == "__main__":
    main()
