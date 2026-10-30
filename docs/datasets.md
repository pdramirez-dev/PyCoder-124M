# Datasets de código Python en Hugging Face

Listado consultado el 2026-10-29 mediante la API de Hugging Face (`/api/datasets`) y el
dataset viewer (`datasets-server`). Se descargó una muestra de `Avelina/python-edu-cleaned`;
su revisión y los shards utilizados constan en `data/raw/python-edu.json`. Las cifras de filas y
tamaño provienen del viewer; en datasets grandes el viewer puede convertir solo una parte,
así que se deben tomar como orientativas.

Leyenda de acceso: **libre** = descarga directa sin aceptar condiciones; **gated** = hay que
aceptar los términos en la web e iniciar sesión (`hf auth login`); **solo IDs** = el dataset
trae `blob_id` y el contenido se descarga aparte desde Software Heritage (S3, requiere
credenciales AWS).

## Recomendados para PyCoder-124M

| Dataset | Acceso | Contenido | Filas | Parquet | Licencia declarada | Notas |
| --- | --- | --- | --- | --- | --- | --- |
| [`Avelina/python-edu-cleaned`](https://huggingface.co/datasets/Avelina/python-edu-cleaned) | libre | columna `text` | 7,7M | ~6 GB | odc-by | Python-Edu de SmolLM con el código ya incluido. Mejor punto de partida. |
| [`jon-tow/starcoderdata-python-edu`](https://huggingface.co/datasets/jon-tow/starcoderdata-python-edu) | libre | columna `content` | 12,9M | ~23 GB | cc-by-4.0 (tag) | Python de StarCoderData con puntuación educativa (`int_score`); filtrar por `int_score >= 3`. |
| [`HuggingFaceTB/stack-edu`](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) (config `Python`) | libre, **solo IDs** | `blob_id`, `detected_licenses` | 25,3M | 2,5 GB (metadatos) | — | Versión filtrada por calidad de The Stack v2; incluye licencia por archivo. |
| [`HuggingFaceTB/smollm-corpus`](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (config `python-edu`) | libre, **solo IDs** | `blob_id` | 7,7M | 0,6 GB (metadatos) | odc-by | Origen de `Avelina/python-edu-cleaned`. |

## Corpus grandes de BigCode (gated)

| Dataset | Acceso | Notas |
| --- | --- | --- |
| [`bigcode/the-stack-v2-dedup`](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) (config `Python`) | gated, **solo IDs** | Deduplicado; requiere descargar contenido de Software Heritage. |
| [`bigcode/the-stack-v2`](https://huggingface.co/datasets/bigcode/the-stack-v2) (config `Python`) | gated, **solo IDs** | Versión completa sin deduplicar. |
| [`bigcode/the-stack-dedup`](https://huggingface.co/datasets/bigcode/the-stack-dedup) | gated | The Stack v1 deduplicado, con contenido (`data/python`). |
| [`bigcode/starcoderdata`](https://huggingface.co/datasets/bigcode/starcoderdata) | gated | Corpus de StarCoder, con contenido (`python/`). |
| [`bigcode/the-stack-smol`](https://huggingface.co/datasets/bigcode/the-stack-smol) | gated | Muestra pequeña de The Stack; útil para probar el pipeline. |

## Otros (complementarios o con reservas)

| Dataset | Acceso | Filas | Licencia declarada | Notas |
| --- | --- | --- | --- | --- |
| [`codeparrot/codeparrot-clean`](https://huggingface.co/datasets/codeparrot/codeparrot-clean) | libre | — | — | Python de GitHub limpio y deduplicado; columna `license` por archivo, mezcla licencias (incluye copyleft). |
| [`codeparrot/github-code-clean`](https://huggingface.co/datasets/codeparrot/github-code-clean) | libre | — | apache-2.0 (tag) | Multilenguaje; filtrar por `language == "Python"`; licencias mixtas por archivo. |
| [`bigcode/python-stack-v1-functions-filtered`](https://huggingface.co/datasets/bigcode/python-stack-v1-functions-filtered) | libre | 574K | — | Funciones Python filtradas de The Stack v1. |
| [`Nan-Do/code-search-net-python`](https://huggingface.co/datasets/Nan-Do/code-search-net-python) | libre | 455K | apache-2.0 | Funciones con docstring (CodeSearchNet). |
| [`ajibawa-2023/Python-Code-Large`](https://huggingface.co/datasets/ajibawa-2023/Python-Code-Large) | libre | 645K | mit (tag) | Procedencia poco documentada; revisar antes de usar. |
| [`nampdn-ai/tiny-codes`](https://huggingface.co/datasets/nampdn-ai/tiny-codes) | gated | — | mit | Código sintético generado por LLM, multilenguaje. |

Descartados para preentrenamiento: datasets de instrucciones (`python_code_instructions_18k_alpaca`,
`python-codes-25k`) y los de licencia no comercial (`CodeExercise-Python-27k`, cc-by-nc-sa).

## Advertencias

- La licencia declarada en Hugging Face no sustituye a la licencia de cada archivo de origen.
  Conservar `repo_name`, `path` y la licencia por archivo cuando existan, y revisar los términos
  de cada dataset antes de entrenar.
- Varios de estos corpus se solapan (todos derivan de GitHub/The Stack). Si se combinan, aplicar
  deduplicación entre ellos.
- Para descontaminar la evaluación, excluir benchmarks como HumanEval y MBPP del corpus.

## Volumen necesario

100M tokens (objetivo de la primera fase) equivalen aproximadamente a 350–450 MB de código
Python. Una muestra de `Avelina/python-edu-cleaned` basta; no hace falta descargar el dataset
completo.
