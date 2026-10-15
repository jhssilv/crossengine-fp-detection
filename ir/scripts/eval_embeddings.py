"""
Evaluate trained IR embeddings for quality and engine-agnosticism.

Checks:
  1. Vocabulary coverage  — all 138 IR tokens present?
  2. Vector health        — no NaN/zero/collapsed vectors
  3. Nearest neighbours   — semantically similar tokens cluster together?
  4. Category coherence   — intra-category cosine sim > inter-category sim?
  5. PCA scatter plot     — 2D projection coloured by opcode category

Usage:
  python ir/eval_embeddings.py                         # evaluates both models
  python ir/eval_embeddings.py --model word2vec
  python ir/eval_embeddings.py --model fasttext --plot ir/embeddings/pca.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import fasttext
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent.parent.parent
EQDICT_PATH = ROOT / "ir/equivalence_dict.json"
EMBEDDINGS_DIR = ROOT / "ir/embeddings"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Opcode categories (manually grouped from opcode_map.md)
# Each IR token → category label for visualisation and coherence scoring.
# ---------------------------------------------------------------------------
CATEGORIES: dict[str, list[str]] = {
    "LOAD":      ["LOAD_INT", "LOAD_CONST", "LOAD_BOOL", "LOAD_NULL",
                  "LOAD_UNDEFINED", "LOAD_LOCAL", "LOAD_GLOBAL",
                  "LOAD_CLOSURE_VAR", "LOAD_THIS"],
    "STORE":     ["STORE_LOCAL", "STORE_GLOBAL", "STORE_CLOSURE_VAR"],
    "PROP_GET":  ["GET_NAMED_PROP", "GET_KEYED_PROP"],
    "PROP_SET":  ["SET_NAMED_PROP", "SET_KEYED_PROP", "DEFINE_OBJ_PROP"],
    "BINARY":    ["BINARY_ADD", "BINARY_SUB", "BINARY_MUL", "BINARY_DIV",
                  "BINARY_MOD", "BINARY_POW"],
    "BITWISE":   ["BIT_OR", "BIT_AND", "BIT_XOR",
                  "BIT_LSH", "BIT_RSH", "BIT_URSH"],
    "UNARY":     ["UNARY_NEG", "UNARY_BIT_NOT", "UNARY_NOT",
                  "UNARY_TYPEOF", "TYPEOF", "INC", "DEC"],
    "CMP":       ["CMP_EQ", "CMP_EQ_STRICT", "CMP_NE", "CMP_NE_STRICT",
                  "CMP_LT", "CMP_LE", "CMP_GT", "CMP_GE"],
    "JMP":       ["JMP_IF_TRUE", "JMP_IF_FALSE", "JUMP", "JUMP_TARGET",
                  "LOOP_HEAD", "LOGICAL_AND", "LOGICAL_OR", "NULLISH_COALESCE"],
    "CALL":      ["CALL_PROP", "CALL_FUNC", "CALL_NEW"],
    "CREATE":    ["CREATE_CLOSURE", "CREATE_OBJ", "CREATE_ARR",
                  "CREATE_FUNC_CTX"],
    "CONTROL":   ["RETURN", "POP", "DUP", "SWAP", "THROW", "RETHROW",
                  "TRY_CATCH", "EXCEPTION"],
    "COERCE":    ["TO_NUMBER", "TO_STRING", "TO_BOOLEAN", "INSTANCEOF",
                  "IN_OP", "TDZ_CHECK"],
    "OTHER":     ["GET_TEMPLATE", "GEN_YIELD", "GEN_RESUME", "GEN_NEXT",
                  "ITER_NEXT", "ITER_CLOSE", "FOR_IN_SETUP", "FOR_IN_NEXT",
                  "DELETE_PROP", "MODULE_LOAD", "MODULE_STORE",
                  "PRIVATE_GET", "PRIVATE_SET", "DEFINE_CLASS",
                  "SPREAD_CALL", "SPREAD_NEW"],
}

TOKEN_TO_CAT: dict[str, str] = {
    tok: cat for cat, toks in CATEGORIES.items() for tok in toks
}


def load_model(model_type: str) -> fasttext.FastText._FastText:
    path = EMBEDDINGS_DIR / f"{model_type}.bin"
    log.info("Loading %s from %s", model_type, path)
    model = fasttext.load_model(str(path))
    log.info("Loaded — vocab: %d  dim: %d", len(model.words), model.get_dimension())
    return model


def get_matrix(model: fasttext.FastText._FastText, tokens: list[str]) -> np.ndarray:
    """Return (N, dim) matrix of word vectors for the given tokens."""
    return np.array([model.get_word_vector(t) for t in tokens], dtype=np.float64)


def clean_tokens(
    model: fasttext.FastText._FastText, tokens: list[str]
) -> tuple[list[str], np.ndarray]:
    """Return (valid_tokens, matrix) with NaN/Inf/zero-norm rows removed."""
    mat = get_matrix(model, tokens)
    valid = np.isfinite(mat).all(axis=1) & (np.linalg.norm(mat, axis=1) > 1e-8)
    return [t for t, ok in zip(tokens, valid) if ok], mat[valid]


# ---------------------------------------------------------------------------
# Check 1 — vocabulary coverage
# ---------------------------------------------------------------------------
def check_coverage(model: fasttext.FastText._FastText, ir_tokens: list[str]) -> None:
    log.info("--- [1] Vocabulary coverage ---")
    in_vocab = [t for t in ir_tokens if t in model.words]
    out_of_vocab = [t for t in ir_tokens if t not in model.words]
    log.info(
        "%d/%d IR tokens in vocabulary (%.1f%%)",
        len(in_vocab), len(ir_tokens), 100 * len(in_vocab) / len(ir_tokens),
    )
    if out_of_vocab:
        log.warning("OOV tokens: %s", out_of_vocab)


# ---------------------------------------------------------------------------
# Check 2 — vector health
# ---------------------------------------------------------------------------
def check_vector_health(model: fasttext.FastText._FastText, tokens: list[str]) -> None:
    log.info("--- [2] Vector health ---")
    mat = get_matrix(model, tokens)
    nan_mask = ~np.isfinite(mat).all(axis=1)
    zero_mask = np.linalg.norm(mat, axis=1) < 1e-8
    nan_tokens = [t for t, bad in zip(tokens, nan_mask) if bad]
    log.info(
        "Vectors with NaN/Inf: %d | zero-norm: %d",
        int(nan_mask.sum()), int(zero_mask.sum()),
    )
    if nan_tokens:
        log.warning(
            "NaN/Inf tokens: %s — model likely trained without t=0.0 fix; retrain recommended.",
            nan_tokens,
        )

    # Remaining checks on clean vectors only
    valid_tokens, clean = clean_tokens(model, tokens)
    norms = np.linalg.norm(clean, axis=1)
    log.info(
        "Clean vectors: %d/%d | norm range: [%.4f, %.4f] mean %.4f",
        len(valid_tokens), len(tokens), norms.min(), norms.max(), norms.mean(),
    )
    normed = clean / (norms[:, None] + 1e-8)
    sim_matrix = normed @ normed.T
    np.fill_diagonal(sim_matrix, np.nan)
    mean_sim = float(np.nanmean(sim_matrix))
    log.info(
        "Mean pairwise cosine similarity: %.4f  (near 1.0 = collapsed, near 0 = well spread)",
        mean_sim,
    )


# ---------------------------------------------------------------------------
# Check 3 — nearest neighbours
# ---------------------------------------------------------------------------
PROBE_TOKENS = [
    "LOAD_LOCAL", "STORE_LOCAL",        # var access
    "GET_NAMED_PROP", "SET_NAMED_PROP", # property ops
    "BINARY_ADD", "BINARY_MUL",         # arithmetic
    "JMP_IF_TRUE", "JMP_IF_FALSE",      # control flow
    "CALL_PROP", "CALL_FUNC",           # calls
    "RETURN", "THROW",                  # control
    "CREATE_CLOSURE", "CREATE_OBJ",     # creation
]

def check_neighbours(model: fasttext.FastText._FastText, k: int = 5) -> None:
    log.info("--- [3] Nearest neighbours (k=%d) ---", k)
    all_tokens = [t for t in model.words if t != "<UNK>"]
    tokens, mat = clean_tokens(model, all_tokens)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normed = mat / (norms + 1e-8)

    tok_idx = {t: i for i, t in enumerate(tokens)}

    for probe in PROBE_TOKENS:
        if probe not in tok_idx:
            log.info("  %-22s → (not in clean vocab)", probe)
            continue
        sims = normed @ normed[tok_idx[probe]]
        top_idx = np.argsort(sims)[::-1][1:k + 1]
        neighbours = [(tokens[i], f"{sims[i]:.3f}") for i in top_idx]
        log.info("  %-22s → %s", probe, "  ".join(f"{n}({s})" for n, s in neighbours))


# ---------------------------------------------------------------------------
# Check 4 — category coherence
# ---------------------------------------------------------------------------
def check_category_coherence(model: fasttext.FastText._FastText) -> None:
    log.info("--- [4] Category coherence (intra vs inter cosine sim) ---")

    cat_vecs: dict[str, np.ndarray] = {}
    for cat, toks in CATEGORIES.items():
        valid_toks, mat = clean_tokens(model, toks)
        if len(valid_toks) < 2:
            continue
        cat_vecs[cat] = mat

    intra_sims, inter_sims = [], []
    cats = list(cat_vecs.keys())

    for i, cat in enumerate(cats):
        vecs = cat_vecs[cat]
        nv = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        sim = nv @ nv.T
        np.fill_diagonal(sim, np.nan)
        intra = float(np.nanmean(sim))
        intra_sims.append(intra)

        other_vecs = np.vstack([cat_vecs[c] for j, c in enumerate(cats) if j != i])
        no = other_vecs / (np.linalg.norm(other_vecs, axis=1, keepdims=True) + 1e-8)
        inter = float(np.mean(nv @ no.T))
        inter_sims.append(inter)

        log.info("  %-12s  intra=%.3f  inter=%.3f  gap=%.3f", cat, intra, inter, intra - inter)

    log.info(
        "  %-12s  intra=%.3f  inter=%.3f  gap=%.3f  ← mean",
        "OVERALL", np.mean(intra_sims), np.mean(inter_sims),
        np.mean(intra_sims) - np.mean(inter_sims),
    )


# ---------------------------------------------------------------------------
# Check 5 — PCA plot
# ---------------------------------------------------------------------------
CATEGORY_COLORS = {
    "LOAD": "#4e79a7", "STORE": "#f28e2b", "PROP_GET": "#e15759",
    "PROP_SET": "#76b7b2", "BINARY": "#59a14f", "BITWISE": "#edc948",
    "UNARY": "#b07aa1", "CMP": "#ff9da7", "JMP": "#9c755f",
    "CALL": "#bab0ac", "CREATE": "#d37295", "CONTROL": "#a0cbe8",
    "COERCE": "#ffbe7d", "OTHER": "#8cd17d",
}

def plot_pca(
    model: fasttext.FastText._FastText,
    model_name: str,
    out_path: Path,
) -> None:
    log.info("--- [5] PCA plot → %s ---", out_path)
    vocab_set = set(model.words)
    tokens, cats = [], []
    for cat, toks in CATEGORIES.items():
        valid_toks, _ = clean_tokens(model, toks)
        for t in valid_toks:
            tokens.append(t)
            cats.append(cat)

    _, mat = clean_tokens(model, tokens)
    coords = PCA(n_components=2).fit_transform(mat)

    fig, ax = plt.subplots(figsize=(12, 9))
    plotted_cats = set()
    for (x, y), tok, cat in zip(coords, tokens, cats):
        color = CATEGORY_COLORS.get(cat, "#aaaaaa")
        label = cat if cat not in plotted_cats else None
        ax.scatter(x, y, color=color, s=60, label=label, zorder=3)
        ax.annotate(tok, (x, y), fontsize=6.5, ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points")
        plotted_cats.add(cat)

    ax.legend(loc="upper right", fontsize=8, markerscale=1.2)
    ax.set_title(f"IR Embeddings PCA — {model_name}", fontsize=13)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Plot saved → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["word2vec", "fasttext", "both"], default="both")
    p.add_argument("--plot", type=Path, default=None,
                   help="Path to save PCA plot (e.g. ir/embeddings/pca_fasttext.png). "
                        "If omitted, plots are saved alongside the model.")
    p.add_argument("--neighbours", type=int, default=5, help="k for nearest-neighbour display")
    return p.parse_args()


def evaluate(model_type: str, plot_path: Path | None, k: int) -> None:
    log.info("========== Evaluating: %s ==========", model_type)
    with open(EQDICT_PATH) as f:
        eq = json.load(f)
    ir_tokens: list[str] = eq["ir_tokens"]

    model = load_model(model_type)

    check_coverage(model, ir_tokens)
    check_vector_health(model, [t for t in ir_tokens if t in model.words])
    check_neighbours(model, k)
    check_category_coherence(model)

    if plot_path is None:
        plot_path = EMBEDDINGS_DIR / f"pca_{model_type}.png"
    plot_pca(model, model_type, plot_path)


def main() -> None:
    args = parse_args()
    models = ["word2vec", "fasttext"] if args.model == "both" else [args.model]
    for m in models:
        evaluate(m, args.plot, args.neighbours)


if __name__ == "__main__":
    main()
