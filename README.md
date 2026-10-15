# Detecção de browser fingerprinting por bytecodes cross-engine

Trabalho de Conclusão de Curso (INF/UFRGS). Instrumenta os motores JavaScript V8 (Chromium) e SpiderMonkey (Firefox) para capturar os bytecodes de cada função, liga essas funções a chamadas de APIs de fingerprinting observadas em tempo de execução e treina classificadores sobre uma representação intermediária (RI) comum aos dois motores. Estende o ByteDefender (2025), que usa só o V8.

A monografia está em `thesis/TCC/` (`latexmk main.tex`).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `engines/` | Patches do V8 e do SpiderMonkey, scripts de build do Chromium (SLURM) e documentação da instrumentação |
| `extension/` | Extensão (Manifest V3) carregada nos dois navegadores; intercepta APIs de fingerprinting e registra a pilha de chamadas |
| `crawlers/` | Crawlers Firefox (Selenium) e Chromium (Playwright) e a lista Tranco usada (10.000 sites) |
| `pipeline/` | Junção bytecodes ↔ chamadas de API (`post_processor.py`) e limpeza do dataset (`clean_dataset.py`) |
| `ir/` | Dicionário de equivalência da RI, micro-scripts de validação, tradução do dataset, embeddings e classificadores |
| `ir/*_model*/`, `ir/lgbm_crossengine*/` | `metrics.json` de cada experimento citado na monografia |
| `transformer/` | Classificador Transformer e scripts SLURM usados no cluster |
| `thesis/TCC/` | Fonte LaTeX da monografia |

Não versionados (gerados localmente): `navegadores/` (árvores de código e builds dos navegadores), `data/` (crawl e datasets rotulados), `ir/*.parquet`, `ir/corpus.txt`, `ir/script_*.json` e os binários dos modelos (`*.pkl`, `*.pt`, `*.bin`).

## Reprodução

Testado com Python 3.14. O crawl e os datasets intermediários ocupam cerca de 150 GB.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install
```

### 1. Navegadores instrumentados

Ver `engines/README.md`. Os crawlers esperam:

- `navegadores/v8-tcc/out/Release/chrome`
- `navegadores/spidermonkey-tcc/gecko-dev/obj-x86_64-pc-linux-gnu/dist/bin/firefox`

### 2. Crawl

```bash
python crawlers/firefox-crawler.py
python crawlers/chromium-crawler.py
```

Saída em `data/`: `*_bytecodes.jsonl.gz` e `*_fp_calls.jsonl`. Os crawlers retomam de onde pararam (`data/*_done.txt`).

### 3. Dataset rotulado

```bash
python pipeline/post_processor.py      # → data/labeled_dataset.parquet
python pipeline/clean_dataset.py       # → data/labeled_dataset_clean.parquet
```

### 4. Representação intermediária

```bash
python ir/scripts/run_microscripts.py          # valida o mapeamento com 27 micro-scripts
python ir/scripts/build_equivalence_dict.py    # → ir/equivalence_dict.json
python ir/scripts/translate_dataset.py         # → ir/translated_dataset.parquet
python ir/scripts/train_embeddings.py          # → ir/embeddings/fasttext.{bin,vec}
python ir/scripts/eval_embeddings.py
```

### 5. Classificadores (detecção binária)

```bash
python ir/scripts/train_rf.py --binary --out ir/rf_model_binary
python ir/scripts/train_lgbm.py --out ir/lgbm_model
python ir/scripts/train_lgbm.py --ngram-max 3 --out ir/lgbm_model_trigram
python ir/scripts/train_lgbm_meta.py
python ir/scripts/train_svm.py
python ir/scripts/eval_lgbm.py                 # latência de inferência e importância de features
```

BiLSTM e Transformer foram treinados em GPU no cluster; os hiperparâmetros estão em `ir/cluster/train_bilstm.sh` e `transformer/cluster/train*.sh`.

### 6. Contexto de script e inferência iterativa

Cada passo usa as probabilidades do modelo anterior para recalcular estatísticas por script:

```bash
python ir/scripts/compute_script_density.py            # a partir de ir/lgbm_model
python ir/scripts/train_lgbm_density.py                # → ir/lgbm_model_density   (F1 0,853)
python ir/scripts/compute_script_stats_predicted.py
python ir/scripts/train_lgbm_density_v2.py             # → ir/lgbm_model_density_v2 (F1 0,874)
python ir/scripts/compute_script_stats_iter.py --model-dir ir/lgbm_model_density_v2 \
    --prev-stats ir/script_stats_predicted.json --out ir/script_stats_iter2.json
python ir/scripts/train_lgbm_density_iter.py --stats ir/script_stats_iter2.json \
    --out ir/lgbm_model_density_v3b --neg-ratio 7       # iteração 2 (F1 0,901)
python ir/scripts/compute_script_stats_iter.py --model-dir ir/lgbm_model_density_v3b \
    --prev-stats ir/script_stats_iter2.json --out ir/script_stats_iter3.json
python ir/scripts/train_lgbm_density_iter.py --stats ir/script_stats_iter3.json \
    --out ir/lgbm_model_density_v4 --neg-ratio 7        # iteração 3 (F1 0,904)
```

Limite superior com rótulos reais por script (não implantável):

```bash
python ir/scripts/compute_script_stats.py
python ir/scripts/train_lgbm_sprint11.py               # → ir/lgbm_model_sprint11 (F1 0,928)
```

### 7. Cross-engine e multilabel

```bash
python ir/scripts/train_lgbm_crossengine.py     # zero-shot e modelo conjunto, bigramas
python ir/scripts/train_lgbm_crossengine_v2.py  # idem, com contexto iterativo
python ir/scripts/train_rf.py --out ir/rf_model
python ir/scripts/train_lgbm_multilabel.py
```

## Resultados principais

| Modelo | F1 |
|---|---|
| RF, unigramas | 0,834 |
| LightGBM, bigramas | 0,843 |
| LightGBM, bigramas + densidade de script | 0,853 |
| LightGBM + inferência iterativa (iter. 3) | **0,904** |
| LightGBM + contexto real (oracle) | 0,928 |
| Zero-shot V8 → SM / SM → V8 | 0,149 / 0,180 |
| Modelo conjunto com contexto iterativo, Chromium / Firefox | 0,903 / 0,904 |

Discussão completa nos capítulos 6 e 7 da monografia.
