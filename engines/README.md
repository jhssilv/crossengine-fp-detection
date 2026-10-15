# Instrumentação dos Motores JavaScript

Documentação técnica das modificações realizadas no V8 (Chromium) e no SpiderMonkey (Firefox) para extração de bytecodes e rastreamento de funções anônimas.

---

## Visão Geral

O objetivo da instrumentação é capturar, **no momento da compilação JIT**, a sequência de bytecodes de cada função JavaScript junto com metadados que permitam ligá-la a chamadas de APIs de fingerprinting observadas em tempo de execução.

Cada engine emite os dados em stderr com um prefixo distinto:

| Engine | Prefixo | Arquivo resultante |
|---|---|---|
| V8 (Chromium) | `[V8Log]` | `data/chromium_v8_bytecodes.jsonl.gz` |
| SpiderMonkey (Firefox) | `[SpiderMonkey Log]` | `data/firefox_sm_bytecodes.jsonl.gz` |

## Patches

| Patch | Arquivo alterado | Commit base |
|---|---|---|
| `v8-interpreter.patch` | `v8/src/interpreter/interpreter.cc` | V8 `15ab4d3626ed3b62c096f90073cf3d8a6e3c300a` |
| `spidermonkey-bytecode-emitter.patch` | `js/src/frontend/BytecodeEmitter.cpp` | gecko-dev `5836a062726f715fda621338a17b51aff30d0a8c` |

Os crawlers esperam as árvores em `navegadores/` (ignorado pelo git):

```bash
git -C navegadores/v8-tcc/v8 apply ../../../engines/v8-interpreter.patch
git -C navegadores/spidermonkey-tcc/gecko-dev apply ../../../engines/spidermonkey-bytecode-emitter.patch
```

---

## 1. V8 — `interpreter.cc`

### O que foi adicionado

A função `DoFinalizeJobImpl` é chamada pelo Ignition (interpretador de bytecodes do V8) ao finalizar a compilação de cada função JavaScript. A instrumentação fica dentro dessa função, após a geração do `BytecodeArray`.

#### 1.1 ID Estável por Função (`injected_id`)

Para cada função, calcula-se um identificador determinístico usando **FNV-1a 64 bits** sobre a semente:

```
"v8/" + script_url + "/" + start_pos + "/" + end_pos
```

`start_pos` e `end_pos` são offsets de caractere dentro do script-fonte. Funções anônimas recebem o mesmo ID sempre que o script é recarregado na mesma posição — isso é a base do rastreamento cross-sessão.

```cpp
const std::string seed =
    "v8/" + script_url_str +
    "/" + std::to_string(start_pos) +
    "/" + std::to_string(end_pos);
const std::string injected_id = "fn_" + std::to_string(FNV1a64(seed));
```

#### 1.2 Sequência de Bytecodes

Itera sobre o `BytecodeArray` via `BytecodeArrayIterator`, concatenando os nomes dos bytecodes separados por vírgula:

```cpp
interpreter::BytecodeArrayIterator it(bytecodes);
while (!it.done()) {
    bytecode_sequence += interpreter::Bytecodes::ToString(it.current_bytecode());
    bytecode_sequence += ',';
    it.Advance();
}
```

#### 1.3 Emissão para Stderr

Um mutex estático (`v8log_mutex`) garante que linhas de múltiplas threads de compilação não se misturem:

```cpp
std::cerr << "[V8Log] "
          << "{\"injected_id\":\"" << injected_id << "\""
          << ",\"script_url\":\""  << json_escape(script_url_str) << "\""
          << ",\"script_id\":"     << script_id
          << ",\"start_pos\":"     << start_pos
          << ",\"end_pos\":"       << end_pos
          << ",\"func_name\":\""   << json_escape(func_debug_name) << "\""
          << ",\"bytecodes\":\""   << json_escape(bytecode_sequence) << "\""
          << "}\n" << std::flush;
```

Funções sem `Script` associado (built-ins, extensões internas, WASM) são descartadas antes desse ponto via guarda `IsScript(raw_script)`.

### Formato de saída (V8)

Cada linha de stderr com prefixo `[V8Log]` contém um JSON:

```json
{
  "injected_id": "fn_12345678901234567",
  "script_url":  "https://example.com/bundle.js",
  "script_id":   42,
  "start_pos":   1024,
  "end_pos":     2048,
  "func_name":   "canvasFingerprint",
  "bytecodes":   "LdaConstant,Star0,CallProperty0,..."
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `injected_id` | string | Hash FNV-1a 64 bits, prefixado com `fn_`. Estável entre sessões. |
| `script_url` | string | URL do script (ou `"unknown"` para scripts inline). |
| `script_id` | int | ID interno do V8 para o script nessa sessão. |
| `start_pos` | int | Offset de início da função no script-fonte (bytes). |
| `end_pos` | int | Offset de fim da função no script-fonte (bytes). |
| `func_name` | string | Nome AST da função; `"(anonymous)"` para funções sem nome. |
| `bytecodes` | string | Sequência de opcodes separada por vírgula. |

---

## 2. SpiderMonkey — `gecko-dev`

### Localização do build

```
navegadores/spidermonkey-tcc/gecko-dev/obj-x86_64-pc-linux-gnu/dist/bin/firefox
```

### O que foi adicionado

No `BytecodeEmitter`, cada função anônima recebe o nome de exibição `anon_injected_<hash>`, onde o hash (`std::hash`) é calculado sobre URL do script e posição da função. Como o nome de exibição aparece em `Error.stack`, a extensão captura o mesmo identificador que o log de bytecodes, e a junção no Firefox é feita só por nome.

A instrumentação emite blocos de texto delimitados em stderr para cada função compilada. O parser do crawler (`parse_spidermonkey_logs()` no `firefox-crawler.py`) transforma esses blocos no mesmo JSONL do Chromium (sem `start_pos`/`end_pos`).

### Formato de saída (SpiderMonkey)

```
[SpiderMonkey Log]
Script URL: https://example.com/bundle.js
Internal Index: 7
Start: 1024
End: 2048
Function name: canvasFingerprint
Bytecode: [FunctionThis, SetLocal, Pop, GetGName, ...]
========================================
```

| Campo | Descrição |
|---|---|
| `Script URL` | URL do script. Scripts `resource://` e `chrome://` são filtrados pelo parser. |
| `Internal Index` | Índice sequencial da função dentro do script nessa sessão. |
| `Function name` | Nome da função; string vazia para anônimas. |
| `Bytecode` | Lista de opcodes SpiderMonkey entre colchetes, separados por vírgula e espaço. |

---

## 3. Captura de APIs de Fingerprinting (Extensão Compartilhada)

A extensão em `extension/` é carregada em ambos os browsers. O `content.js` roda no **MAIN world** (`document_start`, todos os frames), interceptando as APIs do paper ByteDefender:

Intercepta APIs de Canvas, Canvas Font, AudioContext, WebRTC, WebGL, `screen` e `navigator` (lista completa em `extension/content.js`). Também normaliza propriedades exclusivas do Firefox (`navigator.oscpu`, `buildID`, `mozInnerScreenX`, ...) para que bibliotecas de fingerprinting não pulem coletas ao detectar o navegador.

Cada chamada captura a stack trace (formatos V8 e Gecko suportados) e extrai:

```json
{
  "api":       "canvas.toDataURL",
  "funcName":  "canvasFingerprint",
  "scriptUrl": "https://example.com/bundle.js",
  "line":      42,
  "col":       9
}
```

Os dados são coletados pelo crawler via `driver.execute_script("return window.__fpCalls")` (Firefox) ou `page.evaluate(...)` (Chromium).

---

## 4. Como Recompilar

### 4.1 Chromium — PCAD/UFRGS

O binário instrumentado é gerado no cluster PCAD via SLURM.

Os scripts assumem a conta e os caminhos usados no PCAD; ajuste `CLUSTER_USER` e os caminhos antes de usar.

```bash
# 1. Copia o interpreter.cc modificado para o cluster
scp navegadores/v8-tcc/v8/src/interpreter/interpreter.cc \
    <usuario>@pcad.inf.ufrgs.br:~/tcc/v8/src/interpreter/interpreter.cc

# 2. Submete o job e monitora (faz download automático ao final)
bash engines/start-build.sh

# Retomar monitoramento de job existente:
bash engines/start-build.sh <JOB_ID>
```

O script SBATCH (`engines/build_chromium.slurm.sh`) copia o `interpreter.cc` para o source tree do Chromium, regenera os arquivos GN e compila com ninja. O binário final é extraído em:

```
navegadores/v8-tcc/out/Release/chrome
```

**Verificar instrumentação no binário:**
```bash
strings navegadores/v8-tcc/out/Release/chrome | grep -c '\[V8Log\]'
# Deve retornar > 0
```

### 4.2 Firefox — Build local

```bash
cd navegadores/spidermonkey-tcc/gecko-dev

# Compilar apenas o motor JS (rápido, ~10 min)
python3.12 mach build

# Ou compilar o browser completo (lento, ~60 min)
python3.12 mach build browser
```

**Atenção:** o build gera symlinks em `dist/bin/` com o caminho absoluto do source. Se a árvore for movida depois do build, crie um symlink no caminho antigo apontando para o novo (ou recompile).

---

## 5. Como Validar

### 5.1 V8

```bash
# Lança o Chrome e filtra linhas V8Log do stderr
navegadores/v8-tcc/out/Release/chrome \
  --no-sandbox \
  --js-flags='--single-threaded' \
  https://browserleaks.com/canvas \
  2>&1 | grep '^\[V8Log\]' | head -3
```

Saída esperada (uma linha JSON por função):
```
[V8Log] {"injected_id":"fn_...","script_url":"https://...","func_name":"...","bytecodes":"..."}
```

### 5.2 SpiderMonkey

```bash
navegadores/spidermonkey-tcc/gecko-dev/obj-x86_64-pc-linux-gnu/dist/bin/firefox \
  --headless --no-remote --profile /tmp/ff-test \
  https://browserleaks.com/canvas \
  2>&1 | grep -A4 '^\[SpiderMonkey Log\]' | head -20
```

---

## 6. Limitações Conhecidas

| Limitação | Detalhe |
|---|---|
| **Scripts lazy-loaded** | Bytecodes só são emitidos quando a função é compilada. Funções em lazy load só aparecem se a página as executar durante o crawl. |
| **Web Workers** | A extensão não injeta hooks em Workers (contexto isolado). APIs de fingerprinting chamadas de Workers não são capturadas. |
| `eval` / `new Function` | Funções geradas em runtime recebem `injected_id` baseado na posição do `eval`, não do código gerado. |
| **Compilação paralela (V8)** | O mutex `v8log_mutex` serializa as escritas, mas pode criar gargalo em compilações com muitos threads. Mitigado com `--single-threaded`. |
| **`script_url` vazio** | Scripts inline (sem `src=`) reportam a URL da página HTML. O `start_pos`/`end_pos` é relativo ao HTML completo. |
