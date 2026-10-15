# Cross-engine browser fingerprinting detection from bytecodes

Undergraduate thesis (INF/UFRGS). The JavaScript engines V8 (Chromium) and SpiderMonkey (Firefox) are instrumented to capture the bytecodes of every function. Those functions are linked to fingerprinting API calls observed at runtime, and classifiers are trained on an intermediate representation (IR) shared by both engines. Extends ByteDefender (2025), which uses V8 only.

Thesis (in Portuguese, with English abstract): [LUME/UFRGS](https://lume.ufrgs.br/handle/10183/312164). LaTeX source in `thesis/TCC/` (`latexmk main.tex`).

Dataset and trained models: [doi:10.5281/zenodo.23394257](https://doi.org/10.5281/zenodo.23394257).

> **Em português.** Código do Trabalho de Conclusão de Curso *Detecção cross-engine de browser fingerprinting por representação intermediária de bytecodes JavaScript* (INF/UFRGS). Os motores V8 e SpiderMonkey foram instrumentados para registrar os bytecodes de cada função, e os opcodes dos dois motores são mapeados para uma representação intermediária comum de 145 tokens. Sobre ela, um LightGBM com inferência iterativa detecta funções de fingerprinting com F1 = 0,904, e um único modelo treinado nos dois motores atinge F1 = 0,903 no Chromium e 0,904 no Firefox. A monografia completa está no [LUME](https://lume.ufrgs.br/handle/10183/312164).

## Layout

| Path | Contents |
|---|---|
| `engines/` | V8 and SpiderMonkey patches, Chromium build scripts (SLURM) and instrumentation docs |
| `extension/` | Manifest V3 extension loaded in both browsers; hooks fingerprinting APIs and records the call stack |
| `crawlers/` | Firefox (Selenium) and Chromium (Playwright) crawlers, plus the Tranco list used (10,000 sites) |
| `pipeline/` | Bytecodes ↔ API calls join (`post_processor.py`) and dataset cleaning (`clean_dataset.py`) |
| `ir/` | IR equivalence dictionary, validation micro-scripts, dataset translation, embeddings and classifiers |
| `ir/*_model*/`, `ir/lgbm_crossengine*/` | `metrics.json` of each experiment reported in the thesis |
| `transformer/` | Transformer classifier and the SLURM scripts used on the cluster |
| `thesis/TCC/` | LaTeX source of the thesis |

Not versioned (generated locally): `navegadores/` (browser source trees and builds), `data/` (crawl output and labeled datasets), `ir/*.parquet`, `ir/corpus.txt`, `ir/script_*.json` and model binaries (`*.pkl`, `*.pt`, `*.bin`).

## Reproducing

Tested with Python 3.14. The crawl and intermediate datasets take about 150 GB.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install
```

The Firefox crawler and `ir/scripts/run_microscripts.py` also need [geckodriver](https://github.com/mozilla/geckodriver) (the micro-script runner expects it at `/usr/local/bin/geckodriver`).

Steps 1 to 3 can be skipped by downloading the datasets from [Zenodo](https://doi.org/10.5281/zenodo.23394257) into `data/` (`labeled_dataset_clean.parquet`) and `ir/` (`translated_dataset.parquet`). Query strings in their `script_url` column are pseudonymized, which does not change any result.

### 1. Instrumented browsers

See `engines/README.md`. The crawlers expect:

- `navegadores/v8-tcc/out/Release/chrome`
- `navegadores/spidermonkey-tcc/gecko-dev/obj-x86_64-pc-linux-gnu/dist/bin/firefox`

### 2. Crawl

```bash
python crawlers/firefox-crawler.py
python crawlers/chromium-crawler.py
```

Output goes to `data/`: `*_bytecodes.jsonl.gz` and `*_fp_calls.jsonl`. The crawlers resume where they stopped (`data/*_done.txt`).

### 3. Labeled dataset

```bash
python pipeline/post_processor.py      # → data/labeled_dataset.parquet
python pipeline/clean_dataset.py       # → data/labeled_dataset_clean.parquet
```

### 4. Intermediate representation

```bash
python ir/scripts/run_microscripts.py          # validates the mapping with 27 micro-scripts
python ir/scripts/build_equivalence_dict.py    # → ir/equivalence_dict.json
python ir/scripts/translate_dataset.py         # → ir/translated_dataset.parquet
python ir/scripts/train_embeddings.py          # → ir/embeddings/fasttext.{bin,vec}
python ir/scripts/eval_embeddings.py
```

### 5. Classifiers (binary detection)

```bash
python ir/scripts/train_rf.py --binary --out ir/rf_model_binary
python ir/scripts/train_lgbm.py --out ir/lgbm_model
python ir/scripts/train_lgbm.py --ngram-max 3 --out ir/lgbm_model_trigram
python ir/scripts/train_lgbm_meta.py
python ir/scripts/train_svm.py
python ir/scripts/eval_lgbm.py                 # inference latency and feature importance
```

BiLSTM and Transformer were trained on GPUs on the cluster; hyperparameters are in `ir/cluster/train_bilstm.sh` and `transformer/cluster/train*.sh`.

### 6. Script context and iterative inference

Each step uses the previous model's probabilities to recompute per-script statistics:

```bash
python ir/scripts/compute_script_density.py            # from ir/lgbm_model
python ir/scripts/train_lgbm_density.py                # → ir/lgbm_model_density   (F1 0.853)
python ir/scripts/compute_script_stats_predicted.py
python ir/scripts/train_lgbm_density_v2.py             # → ir/lgbm_model_density_v2 (F1 0.874)
python ir/scripts/compute_script_stats_iter.py --model-dir ir/lgbm_model_density_v2 \
    --prev-stats ir/script_stats_predicted.json --out ir/script_stats_iter2.json
python ir/scripts/train_lgbm_density_iter.py --stats ir/script_stats_iter2.json \
    --out ir/lgbm_model_density_v3b --neg-ratio 7       # iteration 2 (F1 0.901)
python ir/scripts/compute_script_stats_iter.py --model-dir ir/lgbm_model_density_v3b \
    --prev-stats ir/script_stats_iter2.json --out ir/script_stats_iter3.json
python ir/scripts/train_lgbm_density_iter.py --stats ir/script_stats_iter3.json \
    --out ir/lgbm_model_density_v4 --neg-ratio 7        # iteration 3 (F1 0.904)
```

Upper bound using ground-truth per-script labels (not deployable):

```bash
python ir/scripts/compute_script_stats.py
python ir/scripts/train_lgbm_sprint11.py               # → ir/lgbm_model_sprint11 (F1 0.928)
```

### 7. Cross-engine and multilabel

```bash
python ir/scripts/train_lgbm_crossengine.py     # zero-shot and joint model, bigrams
python ir/scripts/train_lgbm_crossengine_v2.py  # same, with iterative context
python ir/scripts/train_rf.py --out ir/rf_model
python ir/scripts/train_lgbm_multilabel.py
```

## Main results

| Model | F1 |
|---|---|
| RF, unigrams | 0.834 |
| LightGBM, bigrams | 0.843 |
| LightGBM, bigrams + script density | 0.853 |
| LightGBM + iterative inference (iter. 3) | **0.904** |
| LightGBM + ground-truth context (oracle) | 0.928 |
| Zero-shot V8 → SM / SM → V8 | 0.149 / 0.180 |
| Joint model with iterative context, Chromium / Firefox | 0.903 / 0.904 |

Full discussion in chapters 6 and 7 of the thesis.

## Citation

```bibtex
@thesis{silva2026crossengine,
  author      = {Silva, José Henrique de Souza},
  title       = {Detecção cross-engine de browser fingerprinting por representação intermediária de bytecodes JavaScript},
  type        = {Bachelor's thesis},
  institution = {Universidade Federal do Rio Grande do Sul},
  year        = {2026},
  url         = {https://lume.ufrgs.br/handle/10183/312164}
}

@dataset{silva2026crossengine_data,
  author    = {Silva, José Henrique de Souza},
  title     = {Cross-engine JavaScript bytecode dataset for browser fingerprinting detection (V8 + SpiderMonkey)},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23394257}
}
```

## License

MIT (see `LICENSE`). The ABNT/UFRGS LaTeX classes in `thesis/TCC/` keep their own licenses.
