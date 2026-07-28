#!/bin/bash
#SBATCH --job-name=hvg_sweep
#SBATCH --output=temp/logs/hvg_sweep_%A_%a.out
#SBATCH --error=temp/logs/hvg_sweep_%A_%a.err
#SBATCH --array=0-6        # one task per HVG value (indices into HVG_GRID)
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu      # adjust to your cluster partition

# ---------------------------------------------------------------------------
# Grid of HVG_TOP_PER_PATIENT values to sweep.
# To change the grid, edit this array and update --array to 0-<N-1>.
# ---------------------------------------------------------------------------
HVG_GRID=(500 1000 1500 2000 3000 4000 5000)

HVG=${HVG_GRID[$SLURM_ARRAY_TASK_ID]}

NOTEBOOK_DIR="/data1/chanj3/wangm10/scDeBussy/docs/scDeBussy-old/docs/notebooks"
TEMPLATE="${NOTEBOOK_DIR}/HTAN_demo_template.ipynb"
OUTPUT_DIR="temp/results/notebooks"
LOG_DIR="temp/logs"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

OUTPUT="${OUTPUT_DIR}/HTAN_demo_hvg${HVG}.ipynb"

CONDA_ENV="/data1/chanj3/wangm10/conda_envs/scdebussy-py311"
PAPERMILL="${CONDA_ENV}/bin/papermill"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: HVG_TOP_PER_PATIENT=${HVG}"
echo "  template : ${TEMPLATE}"
echo "  output   : ${OUTPUT}"

"${PAPERMILL}" \
    "${TEMPLATE}" \
    "${OUTPUT}" \
    -p HVG_TOP_PER_PATIENT "${HVG}" \
    --execution-timeout 21600 \
    --log-output

echo "[$(date)] Done: HVG_TOP_PER_PATIENT=${HVG}"
