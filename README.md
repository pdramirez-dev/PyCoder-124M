# PyCoder-124M

Modelo autoregresivo de código Python inicializado desde cero. La implementación usa un Transformer decoder-only con RoPE, RMSNorm, SwiGLU y embeddings compartidos, sin pesos preentrenados. La configuración de producción tiene **123 587 328 parámetros**: 12 bloques, dimensión 768, 12 cabezas, FFN 2048, vocabulario 50 304 y contexto 1024. `configs/model_tiny.yaml` es para pruebas, no para medir la calidad del modelo final.

## Requisitos e instalación

Python 3.12, [uv](https://docs.astral.sh/uv/) y macOS, Linux o Windows con una versión de PyTorch compatible. En Apple Silicon se selecciona MPS si está disponible; CUDA tiene prioridad y CPU es la alternativa. Instalar las dependencias y herramientas de desarrollo:

```sh
uv sync
```

## Demostración rápida

El conjunto de `data/demo` contiene ejemplos originales pequeños; no produce un modelo útil. Los comandos siguientes entrenan el tokenizer exclusivamente sobre los archivos de entrenamiento, separan validación antes de tokenizar y ejecutan solo tres pasos:

```sh
uv run python scripts/train_tokenizer.py --vocab-size 512
uv run python scripts/prepare_dataset.py
uv run python scripts/train.py --config configs/train_smoke.yaml
uv run python scripts/evaluate.py checkpoints/latest.pt
uv run python scripts/generate.py checkpoints/latest.pt --prompt 'def '
uv run pytest -q
uv run ruff check src scripts tests
```

Para un entrenamiento configurable, editar `configs/train.yaml` y ejecutar:

```sh
uv run python scripts/train.py --model configs/model_tiny.yaml --config configs/train.yaml
uv run tensorboard --logdir logs
```

Para 124M, preparar antes un corpus suficiente (ver "Corpus real") y un tokenizer con `--vocab-size 50304`. `configs/train_124m.yaml` entrena unos 100M tokens: 1 526 pasos de 65 536 tokens (batch 4 × acumulación 16 × contexto 1024), aproximadamente 12 h en un MacBook Air M5 en float32 antes de la limitación térmica. Cada checkpoint ocupa unos 1,5 GB.

```sh
uv run python scripts/train.py --model configs/model_124m.yaml --config configs/train_124m.yaml
``` Los parámetros de entrenamiento, tamaño de batch y acumulación son configurables. El corpus de demostración no alcanza para el contexto de producción. No se inicia automáticamente ningún entrenamiento prolongado.

## Corpus real

`docs/datasets.md` lista los datasets de Hugging Face evaluados. Para descargar una muestra de `Avelina/python-edu-cleaned` (por defecto 400 MB de texto, unos 100M tokens) en `data/raw/`, con su procedencia (revisión y shards) en `data/raw/python-edu.json`:

```sh
uv run python scripts/download_dataset.py --max-mb 400
uv run python scripts/train_tokenizer.py --source data/raw --vocab-size 50304 --validation-fraction 0.05
uv run python scripts/prepare_dataset.py --source data/raw --validation-fraction 0.05
```

`train_tokenizer.py` y `prepare_dataset.py` leen archivos `.py` y filas de archivos `.parquet` (columna `--text-column`, por defecto `text`). La muestra toma filas en orden de los shards, de forma determinista para una misma revisión.

Ambos scripts aplican el filtro básico de `configs/filters.yaml` (`--filters`) antes de deduplicar y separar, de modo que tokenizer y dataset ven los mismos archivos. Se descartan documentos no UTF-8, demasiado pequeños o grandes, con líneas muy largas, con poca proporción alfanumérica, marcados como autogenerados en sus primeras líneas o que no compilan con `ast.parse` (incluido Python 2). Se elimina un BOM inicial. `manifest.json` registra la configuración del filtro y los rechazos por motivo. Usar el mismo archivo de filtros en ambos scripts. No hay deduplicación aproximada, descontaminación ni filtrado de licencias por archivo.

## Checkpoints y reanudación

El último checkpoint se escribe atómicamente en `checkpoints/latest.pt`; incluye pesos, optimizador, scheduler, escalador, estados RNG y posición del flujo de entrenamiento. Ctrl+C solicita detenerse y guardar después de completar el paso actual. Reanudar con los mismos archivos, configuración y backend:

```sh
uv run python scripts/train.py --config configs/train.yaml --resume checkpoints/latest.pt
```

La configuración y el hash del archivo de entrenamiento se comprueban al reanudar. Conservar también el tokenizer y la partición originales; reemplazarlos puede invalidar la evaluación o la generación. Para una reproducción exacta usar el mismo hardware, versión de PyTorch y backend; algunas operaciones de aceleradores pueden ser no deterministas. Los checkpoints contienen objetos Python: cargar únicamente archivos de confianza.

## Datos y precisión

`prepare_dataset.py` acepta un directorio con archivos `.py`. Deduplica por hash de contenido exacto y separa archivos completos de forma determinista. No detecta duplicados aproximados ni código compartido entre archivos; un corpus real necesita filtrado adicional, análisis de contaminación, revisión de licencias y procedencia antes de entrenar. Si se cambia `--seed` o `--validation-fraction`, usar los mismos valores en `train_tokenizer.py` y `prepare_dataset.py`. El vocabulario BPE puede ser menor que el máximo solicitado si el corpus es pequeño; la generación restringe sus muestras al vocabulario efectivo.

El flujo concatena documentos delimitados por BOS/EOS y usa ventanas contiguas cíclicas. Atención entre documentos dentro de una ventana está permitida. La pérdida y perplejidad de validación reflejan este flujo y no constituyen benchmarks de corrección funcional de código. La integración de The Stack v2 y benchmarks funcionales quedan para una etapa posterior.

`precision: float32` es la opción predeterminada y compatible con CPU/MPS/CUDA. `float16` y `bfloat16` están habilitados solo en CUDA; se debe verificar el soporte del dispositivo. TensorBoard registra memoria asignada de MPS o CUDA cuando están disponibles. La memoria unificada total del sistema puede diferir de esa medida. No hay limitación térmica automática; ajustar pasos, batch y pausas para un MacBook Air sin ventilador.
