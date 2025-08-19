import argparse


def add_arguments_infer(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to model checkpoint (.ckpt). If not provided, uses model_dir/checkpoints/final.ckpt",
    )
    parser.add_argument("--adata", type=str, required=True, help="Path to input AnnData file (.h5ad)")
    parser.add_argument(
        "--embed-key",
        type=str,
        default=None,
        help="Key in adata.obsm for input features (if absent or None, uses adata.X)",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default="drugname_drugconc",
        help="Column in adata.obs containing perturbation labels",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to output AnnData file (.h5ad)")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help=(
            "Path to the run directory that contains training artifacts: "
            "config.yaml, var_dims.pkl, pert_onehot_map.pt, batch_onehot_map.pkl, checkpoints/"
        ),
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        default=None,
        help=(
            "Column in adata.obs to GROUP BY (defaults to data.kwargs.cell_type_key from config, "
            "finally falls back to 'cell_type')."
        ),
    )
    parser.add_argument(
        "--celltypes",
        type=str,
        default=None,
        help="Optional comma-separated allowlist of cell types to include after grouping",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        default=None,
        help=(
            "Optional override for the batch column name in adata.obs. "
            "If not provided, uses data.kwargs.batch_col from config when available."
        ),
    )

def run_tx_infer(args):
    import logging
    import os
    import pickle

    import numpy as np
    import scanpy as sc
    import torch
    import yaml
    from tqdm import tqdm

    from ...tx.models.state_transition import StateTransitionPerturbationModel

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ---------- Utils ----------
    def load_config(cfg_path: str) -> dict:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)

    def as_numpy(x):
        try:
            return x.toarray()
        except Exception:
            return np.asarray(x)

    def to_str_array(series):
        # robustly convert categories/objects/numbers to string keys
        return np.asarray(series.astype(str).values)

    # ---------- Load training artifacts ----------
    config_path = os.path.join(args.model_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Figure out default columns from training config
    cfg_data_kwargs = cfg.get("data", {}).get("kwargs", {}) if isinstance(cfg, dict) else {}
    default_celltype_col = cfg_data_kwargs.get("cell_type_key", None)
    default_batch_col = cfg_data_kwargs.get("batch_col", None)

    celltype_col = args.celltype_col or default_celltype_col or "cell_type"
    batch_col = args.batch_col or default_batch_col

    # Choose checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(args.model_dir, "checkpoints", "final.ckpt")
        logger.info(f"No --checkpoint provided, using default: {checkpoint_path}")

    # var_dims is only needed for some shapes; we can proceed without it but keep for safety
    var_dims_path = os.path.join(args.model_dir, "var_dims.pkl")
    if os.path.exists(var_dims_path):
        with open(var_dims_path, "rb") as f:
            var_dims = pickle.load(f)
        logger.info(f"Loaded var_dims from {var_dims_path}")
        pert_dim = var_dims.get("pert_dim", None)
    else:
        logger.warning(f"var_dims.pkl not found at {var_dims_path}; will infer dims from checkpoint and data.")
        var_dims = {}
        pert_dim = None

    # Load mappings saved by training
    pert_onehot_map_path = os.path.join(args.model_dir, "pert_onehot_map.pt")
    if not os.path.exists(pert_onehot_map_path):
        raise FileNotFoundError(f"Missing pert onehot map at {pert_onehot_map_path}")
    pert_onehot_map = torch.load(pert_onehot_map_path, map_location="cpu", weights_only=False)

    batch_onehot_map_path = os.path.join(args.model_dir, "batch_onehot_map.pkl")
    batch_onehot_map = None
    if os.path.exists(batch_onehot_map_path):
        with open(batch_onehot_map_path, "rb") as f:
            batch_onehot_map = pickle.load(f)
        logger.info(f"Loaded batch_onehot_map from {batch_onehot_map_path} (size={len(batch_onehot_map)})")
    else:
        logger.warning(f"batch_onehot_map.pkl not found at {batch_onehot_map_path}; will fallback to zeros if needed.")

    # Control perturbation fallback
    control_pert = cfg_data_kwargs.get("control_pert", "non-targeting")
    if args.pert_col == "drugname_drugconc":  # Tahoe special-case retained for backward-compat
        control_pert = "[('DMSO_TF', 0.0, 'uM')]"

    # ---------- Load model ----------
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = next(model.parameters()).device
    cell_set_len = getattr(model, "cell_sentence_len", 256)

    logger.info(
        f"Model ready on device={device}; cell_set_len={cell_set_len}; "
        f"batch_encoder={'yes' if getattr(model, 'batch_encoder', None) is not None else 'no'}"
    )

    # ---------- Load AnnData ----------
    logger.info(f"Loading AnnData from: {args.adata}")
    adata = sc.read_h5ad(args.adata)
    n = adata.n_obs

    # Optional cell type filtering (before grouping)
    if args.celltypes is not None:
        keep = set([s.strip() for s in args.celltypes.split(",") if s.strip()])
        if celltype_col not in adata.obs:
            raise ValueError(f"Column '{celltype_col}' not found in adata.obs.")
        initial = adata.n_obs
        adata = adata[adata.obs[celltype_col].astype(str).isin(keep)].copy()
        logger.info(f"Filtered to {adata.n_obs}/{initial} cells with {celltype_col} in {sorted(keep)}")

    # Features: use .obsm[embed_key] if present, else .X
    using_obsm = args.embed_key is not None and args.embed_key in adata.obsm
    if using_obsm:
        X_all = np.asarray(adata.obsm[args.embed_key])
        logger.info(f"Using adata.obsm['{args.embed_key}'] as input features: shape {X_all.shape}")
    else:
        X_all = as_numpy(adata.X)
        logger.info(f"Using adata.X as input features: shape {X_all.shape}")

    # Perturbation values per row
    if args.pert_col not in adata.obs:
        raise ValueError(f"Column '{args.pert_col}' not found in adata.obs.")
    pert_names_all = to_str_array(adata.obs[args.pert_col])

    # Batch column (optional but recommended when model has batch_encoder)
    if getattr(model, "batch_encoder", None) is not None:
        if batch_col is None:
            logger.warning(
                "Model has a batch_encoder but no batch_col provided/available in config; "
                "falling back to zeros for 'batch' indices."
            )
        elif batch_col not in adata.obs:
            logger.warning(
                f"Model has a batch_encoder but '{batch_col}' not found in adata.obs; "
                "falling back to zeros for 'batch' indices."
            )

    # Grouping column (cell type by default)
    if celltype_col not in adata.obs:
        logger.warning(
            f"Grouping column '{celltype_col}' not in adata.obs; proceeding without grouping (single group of all cells)."
        )
        groups = {"__ALL__": np.arange(adata.n_obs)}
    else:
        groups = {}
        for ct, idx in adata.obs.groupby(celltype_col).indices.items():
            groups[str(ct)] = np.asarray(idx, dtype=np.int64)
        logger.info(f"Grouping by '{celltype_col}' into {len(groups)} group(s).")

    # ---------- Helper mappers ----------
    # infer pert_dim if not present
    if pert_dim is None:
        any_vec = next(iter(pert_onehot_map.values()))
        pert_dim = int(any_vec.numel())

    def encode_pert_window(names: np.ndarray) -> torch.Tensor:
        """Map array[str] -> [L, pert_dim] one-hot (torch.float32, device)."""
        out = torch.zeros((len(names), pert_dim), dtype=torch.float32, device=device)
        matched = 0
        control_vec = None
        if control_pert in pert_onehot_map:
            control_vec = pert_onehot_map[control_pert].to(device)

        for i, name in enumerate(names):
            vec = pert_onehot_map.get(name, None)
            if vec is None:
                if control_vec is not None:
                    out[i] = control_vec
                else:
                    # fall back to the first available vector
                    first_k = next(iter(pert_onehot_map.keys()))
                    out[i] = pert_onehot_map[first_k].to(device)
            else:
                out[i] = vec.to(device)
                matched += 1
        return out

    def encode_batch_indices_window(batch_vals: np.ndarray, seq_len: int) -> torch.Tensor:
        """
        Map array[str] -> [seq_len] Long indices for batch_encoder embedding.
        Accepts dict values from training that could be ints or one-hot vectors.
        """
        if getattr(model, "batch_encoder", None) is None:
            return torch.zeros((seq_len,), dtype=torch.long, device=device)

        if batch_col is None or batch_onehot_map is None or batch_col not in adata.obs:
            return torch.zeros((seq_len,), dtype=torch.long, device=device)

        idxs = np.zeros((seq_len,), dtype=np.int64)
        for i, b in enumerate(batch_vals):
            val = batch_onehot_map.get(b, None)
            if val is None:
                idxs[i] = 0
            else:
                # support either stored index (int) or one-hot (array/tensor)
                if isinstance(val, (int, np.integer)):
                    idxs[i] = int(val)
                else:
                    v = np.asarray(val)
                    if v.ndim == 1:
                        idxs[i] = int(np.argmax(v))
                    else:
                        idxs[i] = 0
        return torch.from_numpy(idxs).to(device, dtype=torch.long)

    # ---------- Inference over grouped, variable-length sequences ----------
    logger.info("Beginning grouped inference with variable-length sequences (padded=False).")
    total_cells = adata.n_obs
    preds_np = None  # will allocate once we see the first prediction's dimension

    # Prepare optional per-row batch values as strings
    batch_vals_all = None
    if batch_col is not None and batch_col in adata.obs:
        batch_vals_all = to_str_array(adata.obs[batch_col])

    processed = 0
    pbar = tqdm(total=total_cells, desc="Inferring", unit="cells")
    with torch.no_grad():
        for group_name, idxs in groups.items():
            # process windows of size <= cell_set_len
            for start in range(0, len(idxs), cell_set_len):
                win = idxs[start : start + cell_set_len]
                seq_len = len(win)

                X_batch_np = X_all[win]
                X_batch = torch.tensor(X_batch_np, dtype=torch.float32, device=device)

                pert_names_batch = pert_names_all[win]
                pert_batch = encode_pert_window(pert_names_batch)

                if batch_vals_all is not None:
                    batch_vals_batch = batch_vals_all[win]
                else:
                    batch_vals_batch = np.array(["0"] * seq_len)
                batch_indices = encode_batch_indices_window(batch_vals_batch, seq_len)

                # Build batch dict; variable-length => padded=False
                batch = {
                    "ctrl_cell_emb": X_batch,           # [S, E_in]
                    "pert_emb": pert_batch,             # [S, P]
                    "pert_name": list(pert_names_batch),
                    "batch": batch_indices,             # [S], long indices
                }

                # Model predict
                out = model.predict_step(batch, batch_idx=0, padded=False)

                # Choose decoded gene predictions if present, else latent
                if "pert_cell_counts_preds" in out and out["pert_cell_counts_preds"] is not None:
                    pred_tensor = out["pert_cell_counts_preds"]  # [S, Dg]
                else:
                    pred_tensor = out["preds"]                   # [S, De]

                pred_np = pred_tensor.detach().float().cpu().numpy()

                # Allocate global array lazily with correct width
                if preds_np is None:
                    preds_np = np.empty((total_cells, pred_np.shape[1]), dtype=np.float32)

                preds_np[win] = pred_np
                processed += seq_len
                pbar.update(seq_len)
    pbar.close()

    # ---------- Write results ----------
    if using_obsm:
        adata.obsm[args.embed_key] = preds_np
        logger.info(f"Predictions written to adata.obsm['{args.embed_key}'] with shape {preds_np.shape}")
    else:
        adata.X = preds_np
        logger.info(f"Predictions written to adata.X with shape {preds_np.shape}")

    output_path = args.output or args.adata.replace(".h5ad", "_with_preds.h5ad")
    adata.write_h5ad(output_path)
    logger.info(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on AnnData with a trained model checkpoint.")
    add_arguments_infer(parser)
    args = parser.parse_args()
    run_tx_infer(args)


if __name__ == "__main__":
    main()