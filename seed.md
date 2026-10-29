# Proyecto: PyCoder-124M — Python Language Model from Scratch

Actúa como un Senior AI/ML Engineer especializado en PyTorch, arquitectura Transformer, entrenamiento de LLM desde cero y optimización para Apple Silicon.

Quiero construir **PyCoder-124M**, un modelo de lenguaje autoregresivo especializado en generación y autocompletado de código Python, entrenado completamente desde cero.

## 1. Objetivo

Crear un proyecto profesional, modular y extensible que permita:

- Implementar una arquitectura GPT decoder-only desde cero utilizando PyTorch.
- Inicializar todos los pesos aleatoriamente, sin modelos preentrenados.
- Entrenar un tokenizer BPE especializado en Python.
- Preentrenar el modelo mediante predicción del siguiente token.
- Generar y autocompletar código Python.
- Guardar checkpoints y reanudar entrenamientos interrumpidos.
- Ejecutar evaluaciones y benchmarks.
- Escalar posteriormente el modelo.

## 2. Hardware

Equipo inicial:

- Apple MacBook Air M5.
- 32 GB de memoria unificada.
- macOS con Apple Silicon.
- Backend PyTorch MPS (Metal Performance Shaders).

El proyecto debe ser compatible también con CUDA para entrenar posteriormente en GPUs NVIDIA.

## 3. Arquitectura del modelo

Nombre: `PyCoder-124M`

Arquitectura: GPT decoder-only Transformer.

Configuración inicial propuesta:

- 12 Transformer blocks.
- Hidden dimension: 768.
- 12 attention heads.
- Context length: 1024 tokens.
- Tokenizer BPE con vocabulario configurable.
- Causal self-attention.
- Rotary Position Embeddings (RoPE).
- RMSNorm.
- SwiGLU.
- Weight tying entre embeddings de entrada y salida.
- Inicialización de pesos desde cero.

Ajustar las dimensiones internas y el vocabulario para alcanzar aproximadamente 124 millones de parámetros.

Implementar el cálculo automático de parámetros y documentar el valor exacto obtenido.

Utilizar `torch.nn.functional.scaled_dot_product_attention` cuando sea compatible, con un fallback funcional.

## 4. Stack tecnológico

- Python 3.12.
- PyTorch.
- Hugging Face Tokenizers.
- NumPy.
- PyYAML.
- tqdm.
- TensorBoard.
- pytest.
- Ruff.
- uv para gestión de dependencias.

No utilizar `AutoModelForCausalLM` ni cargar modelos preentrenados.

## 5. Estructura del proyecto

```text
pycoder/
├── README.md
├── pyproject.toml
├── configs/
│   ├── model_124m.yaml
│   └── train.yaml
├── src/
│   └── pycoder/
│       ├── __init__.py
│       ├── config.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── attention.py
│       │   ├── transformer.py
│       │   ├── embeddings.py
│       │   └── gpt.py
│       ├── data/
│       │   ├── tokenizer.py
│       │   ├── dataset.py
│       │   └── preprocessing.py
│       ├── training/
│       │   ├── trainer.py
│       │   ├── checkpoint.py
│       │   └── optimizer.py
│       └── inference/
│           └── generate.py
├── scripts/
│   ├── train_tokenizer.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
├── tests/
│   ├── test_attention.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_training.py
├── data/
├── checkpoints/
└── logs/
```

## 6. Entrenamiento

Implementar un training loop con:

- AdamW optimizer.
- Learning rate warmup y cosine decay.
- Gradient accumulation.
- Gradient clipping.
- Evaluación periódica de validation loss.
- Mixed precision cuando el backend lo soporte de forma estable.
- Checkpointing periódico.
- Reanudación completa del entrenamiento, incluyendo optimizador, scheduler, RNG y posición del dataset.
- TensorBoard logging.
- Selección automática entre MPS, CUDA y CPU.

El entrenamiento debe poder detenerse mediante Ctrl+C y guardar un checkpoint de recuperación.

## 7. Dataset

En esta primera etapa, NO descargar datasets masivos.

Crear un pequeño dataset de demostración con funciones Python originales y ejemplos de código válidos para probar el pipeline.

El sistema debe permitir posteriormente integrar datasets como The Stack v2, filtrando únicamente código Python y respetando las licencias.

Separar entrenamiento y validación, evitando duplicados y contaminación entre conjuntos.

## 8. Validaciones

Implementar pruebas automatizadas que comprueben:

- Dimensiones de los tensores.
- Correcto funcionamiento de la máscara causal.
- Forward pass.
- Backpropagation.
- Reducción de loss en un dataset pequeño.
- Generación autoregresiva.
- Guardado y restauración de checkpoints.
- Funcionamiento en MPS, cuando esté disponible.

## 9. Optimización para MacBook Air M5

El MacBook Air no tiene ventilador, por lo que el sistema debe permitir entrenamientos controlados y reanudables.

Incluir:

- Configuración de batch size y gradient accumulation.
- Limitación configurable de pasos.
- Monitorización de memoria cuando sea posible.
- Un modelo diminuto para pruebas rápidas.
- Configuración de producción separada para 124M.

No iniciar entrenamientos prolongados automáticamente.

## 10. Primera entrega

En esta fase quiero exclusivamente:

1. Crear el scaffolding completo.
2. Implementar la arquitectura funcional.
3. Implementar un tokenizer y pipeline de datos mínimo.
4. Configurar el entrenamiento y checkpoints.
5. Crear los tests.
6. Documentar instalación y comandos.
7. Ejecutar un smoke test breve, si el entorno lo permite.

No realizar aún el pretraining de 100 millones de tokens.

Priorizar código limpio, tipado, modular, documentado y mantenible.

No crear implementaciones ficticias, funciones vacías ni placeholders que aparenten estar terminados.

Al finalizar, mostrar:

- Estructura de archivos generada.
- Número exacto de parámetros.
- Resultado de los tests.
- Resultado del smoke test.
- Consumo de memoria observado, si puede medirse.
- Comandos para ejecutar el proyecto.

**Comienza implementando el proyecto, no solamente describiendo cómo hacerlo.**