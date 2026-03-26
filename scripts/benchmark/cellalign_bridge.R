#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  # Keep this script dependency-light for cluster environments.
  library(cellAlign)
}))

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  get_arg <- function(name, default = NULL) {
    idx <- match(name, args)
    if (is.na(idx) || idx == length(args)) {
      return(default)
    }
    args[[idx + 1]]
  }
  list(
    input_dir = get_arg("--input_dir"),
    output_dir = get_arg("--output_dir"),
    mode = get_arg("--mode", "single_axis"),
    source_patient = get_arg("--source_patient"),
    target_patient = get_arg("--target_patient"),
    num_pts = as.integer(get_arg("--num_pts", "200")),
    win_sz = as.numeric(get_arg("--win_sz", "0.1")),
    dist_method = get_arg("--dist_method", "Euclidean")
  )
}

read_required_inputs <- function(input_dir) {
  expr_path <- file.path(input_dir, "expression.csv")
  obs_path <- file.path(input_dir, "obs.csv")
  config_path <- file.path(input_dir, "config.json")

  missing <- c()
  for (p in c(expr_path, obs_path, config_path)) {
    if (!file.exists(p)) {
      missing <- c(missing, p)
    }
  }
  if (length(missing) > 0) {
    stop(sprintf("Missing required input files: %s", paste(missing, collapse = ", ")))
  }

  expr <- read.csv(expr_path, stringsAsFactors = FALSE, check.names = FALSE)
  obs <- read.csv(obs_path, stringsAsFactors = FALSE)

  required_obs <- c("cell_id", "patient", "s_local")
  if (!all(required_obs %in% colnames(obs))) {
    stop(sprintf("obs.csv is missing required columns: %s", paste(setdiff(required_obs, colnames(obs)), collapse = ", ")))
  }
  if (!("cell_id" %in% colnames(expr))) {
    stop("expression.csv is missing required column: cell_id")
  }

  obs$cell_id <- as.integer(obs$cell_id)
  expr$cell_id <- as.integer(expr$cell_id)

  obs <- obs[order(obs$cell_id), , drop = FALSE]
  expr <- expr[order(expr$cell_id), , drop = FALSE]

  if (nrow(obs) != nrow(expr)) {
    stop(sprintf("obs/expression row mismatch: %d vs %d", nrow(obs), nrow(expr)))
  }
  if (!all(obs$cell_id == expr$cell_id)) {
    stop("obs.csv and expression.csv cell_id columns are not aligned")
  }

  gene_cols <- setdiff(colnames(expr), "cell_id")
  if (length(gene_cols) == 0) {
    stop("expression.csv must contain at least one gene column")
  }

  expr_mat <- t(as.matrix(expr[, gene_cols, drop = FALSE]))
  storage.mode(expr_mat) <- "numeric"
  colnames(expr_mat) <- as.character(expr$cell_id)

  traj <- as.numeric(obs$s_local)
  names(traj) <- as.character(obs$cell_id)

  list(obs = obs, expr = expr_mat, traj = traj)
}

select_pair <- function(obs, mode, source_patient = NULL, target_patient = NULL) {
  patients <- sort(unique(as.character(obs$patient)))
  if (length(patients) < 2) {
    stop("cellAlign bridge expects at least two patient labels in obs.csv")
  }

  if (is.null(source_patient) || is.null(target_patient)) {
    if (mode == "single_axis") {
      source_patient <- patients[[1]]
      target_patient <- patients[[2]]
      warning(sprintf(
        "single_axis mode without explicit source/target; defaulting to source=%s target=%s",
        source_patient,
        target_patient
      ))
    } else {
      stop("source_patient and target_patient are required for pairwise modes")
    }
  }

  if (!(source_patient %in% patients)) {
    stop(sprintf("Unknown source_patient=%s", source_patient))
  }
  if (!(target_patient %in% patients)) {
    stop(sprintf("Unknown target_patient=%s", target_patient))
  }
  if (identical(source_patient, target_patient)) {
    stop("source_patient and target_patient must be different")
  }

  list(source_patient = source_patient, target_patient = target_patient)
}

map_query_to_reference <- function(
  expr_mat,
  traj,
  obs,
  source_patient,
  target_patient,
  num_pts,
  win_sz,
  dist_method
) {
  source_ids <- as.character(obs$cell_id[as.character(obs$patient) == source_patient])
  target_ids <- as.character(obs$cell_id[as.character(obs$patient) == target_patient])

  if (length(source_ids) < 3 || length(target_ids) < 3) {
    stop("Each patient needs at least 3 cells for stable interpolation/alignment")
  }

  source_expr <- expr_mat[, source_ids, drop = FALSE]
  target_expr <- expr_mat[, target_ids, drop = FALSE]
  source_traj <- traj[source_ids]
  target_traj <- traj[target_ids]

  inter_source <- interWeights(
    expDataBatch = source_expr,
    trajCond = source_traj,
    winSz = win_sz,
    numPts = num_pts
  )
  inter_target <- interWeights(
    expDataBatch = target_expr,
    trajCond = target_traj,
    winSz = win_sz,
    numPts = num_pts
  )
  scaled_source <- scaleInterpolate(inter_source)
  scaled_target <- scaleInterpolate(inter_target)

  alignment <- globalAlign(
    scaled_source$scaledData,
    scaled_target$scaledData,
    dist.method = dist_method,
    normDist = TRUE,
    verbose = FALSE
  )

  mapping <- mapRealDataGlobal(
    alignment,
    intTrajQuery = scaled_source$traj,
    realTrajQuery = source_traj,
    intTrajRef = scaled_target$traj,
    realTrajRef = target_traj
  )

  # Build metanode->reference pseudotime map.
  query_meta_to_pt_ref <- setNames(mapping$metaNodesPt$ptRef, mapping$metaNodesPt$metaNodeQuery)

  query_aligned <- setNames(rep(NA_real_, length(source_ids)), source_ids)
  for (meta_node in names(mapping$queryAssign)) {
    pt_ref <- query_meta_to_pt_ref[[meta_node]]
    if (is.null(pt_ref) || !is.finite(pt_ref)) {
      next
    }
    cells <- as.character(mapping$queryAssign[[meta_node]])
    cells <- cells[cells %in% names(query_aligned)]
    if (length(cells) > 0) {
      query_aligned[cells] <- as.numeric(pt_ref)
    }
  }

  # Fallback for any unmapped cells using interpolation-path monotone approximation.
  missing_ids <- names(query_aligned)[!is.finite(query_aligned)]
  if (length(missing_ids) > 0) {
    q_path <- scaled_source$traj[alignment$align[[1]]$index1]
    r_path <- scaled_target$traj[alignment$align[[1]]$index2]
    # Aggregate duplicate q_path values before approx.
    ord <- order(q_path, r_path)
    q_path <- q_path[ord]
    r_path <- r_path[ord]
    q_unique <- unique(q_path)
    r_agg <- sapply(q_unique, function(qv) mean(r_path[q_path == qv]))
    query_aligned[missing_ids] <- approx(
      x = q_unique,
      y = r_agg,
      xout = source_traj[missing_ids],
      rule = 2,
      ties = "ordered"
    )$y
  }

  aligned <- setNames(rep(NA_real_, nrow(obs)), as.character(obs$cell_id))
  aligned[target_ids] <- as.numeric(target_traj[target_ids])
  aligned[names(query_aligned)] <- as.numeric(query_aligned)

  # For any remaining cells (unexpected in pairwise input), keep native pseudotime.
  unresolved <- names(aligned)[!is.finite(aligned)]
  if (length(unresolved) > 0) {
    aligned[unresolved] <- as.numeric(traj[unresolved])
  }

  list(
    aligned = aligned,
    diagnostics = list(
      distance = as.numeric(alignment$distance),
      normalized_distance = as.numeric(alignment$normalizedDistance),
      n_query_cells = length(source_ids),
      n_ref_cells = length(target_ids),
      n_unresolved_query = as.integer(sum(!is.finite(query_aligned)))
    )
  )
}

write_outputs <- function(
  output_dir,
  obs,
  aligned,
  mode,
  source_patient = NULL,
  target_patient = NULL,
  diagnostics = list()
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  aligned <- data.frame(
    cell_id = as.integer(obs$cell_id),
    aligned_pseudotime = as.numeric(aligned[as.character(obs$cell_id)])
  )

  aligned_path <- file.path(output_dir, "aligned.csv")
  write.csv(aligned, aligned_path, row.names = FALSE, quote = FALSE)

  metrics_path <- file.path(output_dir, "metrics.json")
  distance_val <- diagnostics$distance
  norm_distance_val <- diagnostics$normalized_distance
  n_query_cells <- diagnostics$n_query_cells
  n_ref_cells <- diagnostics$n_ref_cells
  n_unresolved_query <- diagnostics$n_unresolved_query
  if (is.null(distance_val) || !is.finite(distance_val)) distance_val <- "null" else distance_val <- as.character(distance_val)
  if (is.null(norm_distance_val) || !is.finite(norm_distance_val)) norm_distance_val <- "null" else norm_distance_val <- as.character(norm_distance_val)
  if (is.null(n_query_cells)) n_query_cells <- "null" else n_query_cells <- as.character(n_query_cells)
  if (is.null(n_ref_cells)) n_ref_cells <- "null" else n_ref_cells <- as.character(n_ref_cells)
  if (is.null(n_unresolved_query)) n_unresolved_query <- "null" else n_unresolved_query <- as.character(n_unresolved_query)

  metrics <- paste0(
    "{\n",
    "  \"bridge_status\": \"ok\",\n",
    "  \"mode\": \"", mode, "\",\n",
    "  \"source_patient\": ", if (is.null(source_patient)) "null" else paste0("\"", source_patient, "\""), ",\n",
    "  \"target_patient\": ", if (is.null(target_patient)) "null" else paste0("\"", target_patient, "\""), ",\n",
    "  \"distance\": ", distance_val, ",\n",
    "  \"normalized_distance\": ", norm_distance_val, ",\n",
    "  \"n_query_cells\": ", n_query_cells, ",\n",
    "  \"n_ref_cells\": ", n_ref_cells, ",\n",
    "  \"n_unresolved_query\": ", n_unresolved_query, ",\n",
    "  \"n_cells\": ", nrow(obs), "\n",
    "}\n"
  )
  writeLines(metrics, metrics_path)
}

main <- function() {
  parsed <- parse_args()
  if (is.null(parsed$input_dir) || is.null(parsed$output_dir)) {
    stop("Usage: cellalign_bridge.R --input_dir <dir> --output_dir <dir> [--mode <name>] [--source_patient <id>] [--target_patient <id>]")
  }

  payload <- read_required_inputs(parsed$input_dir)
  pair <- select_pair(
    payload$obs,
    mode = parsed$mode,
    source_patient = parsed$source_patient,
    target_patient = parsed$target_patient
  )

  fit <- map_query_to_reference(
    expr_mat = payload$expr,
    traj = payload$traj,
    obs = payload$obs,
    source_patient = pair$source_patient,
    target_patient = pair$target_patient,
    num_pts = parsed$num_pts,
    win_sz = parsed$win_sz,
    dist_method = parsed$dist_method
  )

  write_outputs(
    parsed$output_dir,
    payload$obs,
    fit$aligned,
    parsed$mode,
    source_patient = pair$source_patient,
    target_patient = pair$target_patient,
    diagnostics = fit$diagnostics
  )
  message(sprintf(
    "[cellalign_bridge] Wrote aligned.csv and metrics.json (mode=%s source=%s target=%s dist=%.6f)",
    parsed$mode,
    pair$source_patient,
    pair$target_patient,
    as.numeric(fit$diagnostics$distance)
  ))
}

main()
