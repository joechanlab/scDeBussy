from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def load_from_manifest(manifest_path: Path) -> ad.AnnData:
    """Load and concatenate per-patient AnnData paths listed in a manifest file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as fh:
        paths = [line.strip() for line in fh if line.strip()]
    if not paths:
        raise ValueError("No h5ad paths found in manifest.")

    adatas = [ad.read_h5ad(path) for path in paths]
    adata = ad.concat(adatas, join="inner", merge="same")
    adata.obs_names_make_unique()
    return adata


def matrix_from_layer(adata: ad.AnnData, layer_name: str | None = "counts") -> np.ndarray:
    """Return dense float matrix from a layer when available, otherwise from ``X``."""
    if layer_name is not None and layer_name in adata.layers:
        matrix = adata.layers[layer_name]
    else:
        matrix = adata.X

    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float)


def filter_informative_genes(
    adata: ad.AnnData,
    *,
    species: str = "hsapiens_gene_ensembl",
    host: str = "http://www.ensembl.org",
) -> ad.AnnData:
    """Keep informative genes using Ensembl protein-coding annotation plus regex filters."""
    gene_names = pd.Index(adata.var_names.astype(str))
    upper_names = gene_names.str.upper()
    regex_exclude = np.asarray(
        upper_names.str.startswith(("MT-", "RPS", "RPL")) | upper_names.str.contains("ORF", regex=False),
        dtype=bool,
    )

    biotype_keep = pd.Series(True, index=gene_names)
    annotation_source = "regex_only"
    try:
        from pybiomart import Dataset

        dataset = Dataset(name=species, host=host)
        annotations = dataset.query(attributes=["external_gene_name", "gene_biotype"])
        annotations = annotations.rename(columns={"Gene name": "external_gene_name", "Gene type": "gene_biotype"})

        needed = {"external_gene_name", "gene_biotype"}
        if not needed.issubset(annotations.columns):
            raise ValueError(f"Annotation query returned columns {annotations.columns.tolist()}")

        annotations = annotations.dropna(subset=["external_gene_name", "gene_biotype"]).copy()
        annotations["external_gene_name"] = annotations["external_gene_name"].astype(str)
        annotations = annotations.drop_duplicates(subset=["external_gene_name"])
        protein_coding = annotations.loc[annotations["gene_biotype"] == "protein_coding", "external_gene_name"]
        protein_coding_set = set(protein_coding.astype(str))
        biotype_keep = pd.Series(gene_names.isin(protein_coding_set), index=gene_names)
        annotation_source = "ensembl_protein_coding_plus_regex"
    except (ImportError, AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Gene annotation lookup failed; falling back to regex-only informative filter: {exc}")

    keep_mask = np.asarray(biotype_keep, dtype=bool) & (~regex_exclude)
    if not np.any(keep_mask):
        raise ValueError("Informative-gene filter removed all genes.")

    filtered = adata[:, keep_mask].copy()
    filtered.var["informative_gene"] = True
    filtered.uns["informative_gene_filter"] = {
        "annotation_source": annotation_source,
        "species": species,
        "host": host,
        "n_genes_before": int(adata.n_vars),
        "n_genes_after": int(filtered.n_vars),
    }
    return filtered


def filter_recurrent_hvgs(
    adata: ad.AnnData,
    *,
    patient_key: str,
    top_per_patient: int = 3000,
    min_patients: int | None = None,
    layer_name: str = "counts",
) -> tuple[ad.AnnData, pd.Series]:
    """Keep genes that are highly variable in at least ``min_patients`` samples."""
    patient_ids = adata.obs[patient_key].astype(str).unique().tolist()
    n_patients = len(patient_ids)
    if n_patients < 2:
        raise ValueError("Recurrent HVG filtering requires at least 2 patients.")

    if min_patients is None:
        min_patients = max(2, int(np.ceil(n_patients / 2)))
    min_patients = max(1, min(int(min_patients), n_patients))

    gene_counts = pd.Series(0, index=adata.var_names.astype(str), dtype=int)
    for patient_id in patient_ids:
        mask = (adata.obs[patient_key].astype(str) == patient_id).to_numpy()
        patient_matrix = matrix_from_layer(adata[mask], layer_name=layer_name)
        if patient_matrix.ndim != 2 or patient_matrix.shape[1] == 0:
            continue

        mean = np.mean(patient_matrix, axis=0)
        var = np.var(patient_matrix, axis=0)
        dispersion = var / (mean + 1e-8)
        dispersion = np.nan_to_num(dispersion, nan=0.0, posinf=0.0, neginf=0.0)

        n_top = max(1, min(int(top_per_patient), patient_matrix.shape[1]))
        top_idx = np.argpartition(-dispersion, kth=n_top - 1)[:n_top]
        top_genes = adata.var_names.astype(str)[top_idx]
        gene_counts.loc[top_genes] += 1

    recurrent_mask = gene_counts >= min_patients
    if not recurrent_mask.any():
        raise ValueError(
            f"Recurrent HVG filter selected zero genes with top_per_patient={top_per_patient} "
            f"and min_patients={min_patients}."
        )

    filtered = adata[:, recurrent_mask.to_numpy()].copy()
    filtered.var["recurrent_hvg"] = True
    filtered.uns["recurrent_hvg_filter"] = {
        "top_per_patient": int(top_per_patient),
        "min_patients": int(min_patients),
        "n_patients": int(n_patients),
        "n_genes_selected": int(filtered.n_vars),
        "source_layer": layer_name if layer_name in adata.layers else "X",
    }
    return filtered, gene_counts.sort_values(ascending=False)


S_GENES_TIROSH = [
    "MCM5",
    "PCNA",
    "TYMS",
    "FEN1",
    "MCM2",
    "MCM4",
    "RRM1",
    "UNG",
    "GINS2",
    "MCM6",
    "CDCA7",
    "DTL",
    "PRIM1",
    "UHRF1",
    "MLF1IP",
    "HELLS",
    "RFC2",
    "RPA2",
    "NASP",
    "RAD51AP1",
    "GMNN",
    "WDR76",
    "SLBP",
    "CCNE2",
    "UBR7",
    "POLD3",
    "MSH2",
    "ATAD2",
    "RAD51",
    "RRM2",
    "CDC45",
    "CDC6",
    "EXO1",
    "TIPIN",
    "DSCC1",
    "BLM",
    "CASP8AP2",
    "USP1",
    "CLSPN",
    "CHAF1B",
    "BRIP1",
    "E2F8",
]
G2M_GENES_TIROSH = [
    "HMGB2",
    "CDK1",
    "NUSAP1",
    "UBE2C",
    "BIRC5",
    "TPX2",
    "TOP2A",
    "NDC80",
    "CKS2",
    "NUF2",
    "CKS1B",
    "PCLAF",
    "TACC3",
    "CENPA",
    "SMC4",
    "CCNB2",
    "CKAP2L",
    "CKAP2",
    "AURKB",
    "BUB1",
    "KIF11",
    "ANP32E",
    "TUBB4B",
    "GTSE1",
    "KIF20B",
    "HJURP",
    "CDCA3",
    "HN1",
    "CDC20",
    "TTK",
    "CDC25C",
    "KIF2C",
    "RANGAP1",
    "NCAPD2",
    "DLGAP5",
    "CDCA2",
    "CDCA8",
    "ECT2",
    "KIF23",
    "HMMR",
    "AURKA",
    "PSRC1",
    "ANLN",
    "LBR",
    "CKAP5",
    "CENPE",
    "CTCF",
    "NEK2",
    "G2E3",
    "GAS2L3",
    "CBX5",
    "CENPF",
]


def remove_cell_cycle_genes(gene_set_dict, extra_genes_to_remove=None):
    """
    Remove cell cycle genes

    Removes proliferation and cell-cycle genes from the gene sets
    to prevent false positive recurrence hits.
    """
    # Standard S-phase and G2M-phase genes (shortened list for brevity)

    cell_cycle_genes = set(S_GENES_TIROSH + G2M_GENES_TIROSH)
    if extra_genes_to_remove:
        cell_cycle_genes.update(extra_genes_to_remove)

    cleaned_dict = {}
    for set_name, genes in gene_set_dict.items():
        # Keep genes only if they aren't in the cell cycle list
        cleaned_genes = [g for g in genes if g.upper() not in cell_cycle_genes]

        # Only keep the set if it still has a reasonable number of genes
        if len(cleaned_genes) >= 5:
            cleaned_dict[set_name] = cleaned_genes

    return cleaned_dict


__all__ = [
    "filter_informative_genes",
    "filter_recurrent_hvgs",
    "load_from_manifest",
    "matrix_from_layer",
    "remove_cell_cycle_genes",
]
