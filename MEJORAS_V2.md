# C.A. Dupin V2 - Mejoras del Sistema de Reconocimiento de Patrones

## 🚀 Novedades Principales

### 0. ⚡ Optimizaciones de Rendimiento (NUEVAS)

#### Automatic Mixed Precision (AMP)
- **Velocidad**: Acelera el entrenamiento 2-3x en GPUs con soporte Tensor Cores
- **Memoria**: Reduce el uso de memoria hasta un 50%
- **Cómo funciona**: Usa float16 para cálculos y float32 para mantener precisión
- **Activación**: Automática en GPU, configurable con `--use-amp`

```bash
# Entrenar con AMP activado (automático en GPU)
python dupin.py entrenar-patrones-v2 --epochs 30

# Desactivar AMP explícitamente
python dupin.py entrenar-patrones-v2 --use-amp False
```

#### torch.compile (PyTorch 2.0+)
- **Velocidad**: Optimiza el modelo compilándolo, ganando 10-30% más velocidad
- **Soporte**: Detecta automáticamente si PyTorch 2.0+ está disponible
- **Modos**: `reduce-overhead` para mejor rendimiento en entrenamiento
- **Activación**: Automática si PyTorch 2.0+ está instalado

```bash
# torch.compile se activa automáticamente con PyTorch 2.0+
python dupin.py entrenar-patrones-v2 --epochs 30

# Desactivar compilación
python dupin.py entrenar-patrones-v2 --use-compile False
```

#### DataLoader Paralelo con Prefetching
- **Mejora**: Carga de datos en paralelo con múltiples workers
- **Configuración automática**: 4 workers en CPU, 2 en GPU
- **Persistent workers**: Mantiene workers activos entre epochs
- **Prefetching**: Pre-carga batches para reducir tiempo de espera
- **Pin Memory**: Transferencia GPU optimizada

```bash
# Usar configuración automática (recomendado)
python dupin.py entrenar-patrones-v2

# Configurar workers manualmente
python dupin.py entrenar-patrones-v2 --num-workers 4
```

#### Channels Last Memory Format
- **Velocidad**: Mejora rendimiento en hardware moderno (10-20% en GPUs NVIDIA)
- **Formato**: NHWC (batch, height, width, channels) más eficiente que NCHW
- **Activación**: Automática en GPU con `--channels-last`

```bash
# Activado por defecto en GPU
python dupin.py entrenar-patrones-v2

# Desactivar si hay problemas de compatibilidad
python dupin.py entrenar-patrones-v2 --channels-last False
```

#### Gradient Checkpointing
- **Memoria**: Reduce uso de memoria 20-40% entrenando redes más profundas
- **Trade-off**: Un poco más lento pero permite batch sizes más grandes
- **Ideal**: Entrenamiento en GPU con memoria limitada
- **Activación**: `--use-gradient-checkpointing`

```bash
# Activar gradient checkpointing
python dupin.py entrenar-patrones-v2 --use-gradient-checkpointing
```

#### Image Caching
- **Velocidad**: Cache de imágenes pre-procesadas en memoria
- **Beneficio**: Elimina re-lectura de disco cada epoch
- **Automático**: Siempre activado en datasets de entrenamiento
- **Impacto**: 10-30% más rápido en datasets pequeños/medianos

#### Optimizaciones Adicionales
- **non_blocking=True**: Transferencias asíncronas GPU-CPU
- **Optimizador AdamW**: Mejor manejo de pesos y decaimiento
- **Betas optimizados**: (0.9, 0.999) para convergencia más rápida
- **Gradient Clipping**: Estabiliza entrenamiento con `max_norm=1.0`
- **Batch normalization mejorado**: Mejor estabilidad en entrenamiento

#### Resumen de Gananancias de Rendimiento

| Optimización | Ganancia Velocidad | Ahorro Memoria | Estado |
|--------------|-------------------|----------------|---------|
| AMP | 2-3x | 40-50% | ✓ Auto (GPU) |
| torch.compile | 10-30% | - | ✓ Auto (PyTorch 2+) |
| DataLoader Paralelo | 1.5-2x | - | ✓ Auto |
| Channels Last | 10-20% | - | ✓ Auto (GPU) |
| Image Cache | 10-30% | - | ✓ Siempre |
| Gradient Checkpointing | - | 20-40% | Opcional |

**Ganancia total combinada**: Hasta **5-8x más rápido** en GPUs modernas

### 1. Nuevas Técnicas de IA Implementadas

#### 🧠 Arquitectura Mejorada
- **SE Blocks (Squeeze-and-Excitation)**: Mecanismo de atención que permite a la red aprender qué características son más importantes
- **Bloques Residuales con SE**: Mejor flujo de gradientes y atención de canales
- **Arquitectura más profunda**: 4 capas con 64→128→256→512 canales (3 bloques cada una)

#### 📈 Optimizaciones de Entrenamiento
- **One-Cycle Learning Rate Policy**: Estrategia de learning rate que ajusta dinámicamente el LR durante el entrenamiento
- **Warmup**: Épocas de calentamiento para estabilizar el entrenamiento
- **Gradient Accumulation**: Permite batch sizes efectivos mayores
- **Early Stopping con warmup**: Detiene el entrenamiento cuando no hay mejora, ignorando épocas de warmup

#### 🎨 Data Augmentation Avanzado
- **RandAugment**: Auto-augmentation que aplica transformaciones aleatorias con magnitud controlada
- **Mixup**: Combina pares de imágenes para mejorar generalización
- **Configurable**: Cada técnica puede activarse/desactivarse

#### 🔍 Inferencia Mejorada
- **Multi-scale Inference**: Reconoce patrones en múltiples escalas (96x96, 128x128, 160x160)
- **Ensemble de predicciones**: Promedia resultados de múltiples escalas para mayor precisión

### 2. Nuevas Carpetas

El sistema V2 crea automáticamente dos carpetas:

```
fotos_entrenamiento/
├── por_patron/
│   ├── logo_empresa/      ← Coloca aquí fotos del logo de tu empresa
│   ├── producto_a/        ← Coloca aquí fotos del producto A
│   └── mi_patron/        ← Coloca aquí fotos de tu patrón
└── README.md

fotos_identificar/
├── foto1.jpg            ← Coloca aquí fotos para identificar
├── foto2.png
└── ...
```

### 3. Nuevo Flujo de Trabajo Simplificado

#### Paso 1: Crear Patrón
```bash
python dupin.py crear-patron-v2 "mi_logo" --descripcion "Logo de mi empresa"
```
- Crea automáticamente la carpeta `fotos_entrenamiento/por_patron/mi_logo/`
- Crea un README con instrucciones

#### Paso 2: Colocar Imágenes de Entrenamiento
- Copia o mueve las fotos a: `fotos_entrenamiento/por_patron/mi_logo/`
- Formatos soportados: JPG, JPEG, PNG, BMP, GIF, TIFF

#### Paso 3: Importar Imágenes
```bash
python dupin.py importar-entrenamiento
```
- Importa todas las imágenes de las carpetas de patrones
- Muestra un resumen de imágenes importadas por patrón

#### Paso 4: Entrenar Modelo
```bash
python dupin.py entrenar-patrones-v2 --epochs 50 --batch-size 16 --warmup 3
```

#### Paso 5: Identificar Imágenes
```bash
# Identificar todas las imágenes en fotos_identificar/
python dupin.py identificar-v2 --umbral 0.6

# Identificar una imagen específica
python dupin.py reconocer-v2 imagen.jpg --umbral 0.7 --multiscale
```

## 📋 Comandos V2 Disponibles

### crear-patron-v2
Crea un nuevo patrón con carpeta automática.

```bash
python dupin.py crear-patron-v2 "nombre_patron" --descripcion "Descripción opcional"
```

### importar-entrenamiento
Importa imágenes de todas las carpetas de entrenamiento.

```bash
python dupin.py importar-entrenamiento
```

### entrenar-patrones-v2
Entrena el modelo con técnicas de IA avanzadas.

```bash
python dupin.py entrenar-patrones-v2 \
    --epochs 50 \
    --batch-size 16 \
    --val-split 0.2 \
    --learning-rate 0.001 \
    --max-lr 0.01 \
    --warmup 3 \
    --early-stopping 10 \
    --dropout 0.4
```

Opciones avanzadas:
- `--focal-loss`: Usar Focal Loss para clases desbalanceadas
- `--label-smoothing 0.1`: Label smoothing para mejor generalización
- `--no-mixup`: Desactivar Mixup augmentation
- `--no-randaugment`: Desactivar RandAugment
- `--grad-accum 2`: Gradient accumulation (batch efectivo = batch_size * 2)

### reconocer-v2
Reconoce patrones en una imagen específica.

```bash
python dupin.py reconocer-v2 imagen.jpg --umbral 0.7 --multiscale
```

Opciones:
- `--umbral 0.5`: Umbral de confianza (0.0-1.0)
- `--multiscale`: Usar multi-scale inference para mayor precisión

### identificar-v2
Identifica patrones en todas las imágenes de `fotos_identificar/`.

```bash
# Identificación estándar (solo guarda detecciones por encima del umbral)
python dupin.py identificar-v2 --umbral 0.6 --output resultados.json

# Guardar alternativas (top-k) por imagen
python dupin.py identificar-v2 --umbral 0.6 --top-k 3 --output resultados.json

# Revisión humana interactiva (GUIA a la IA): aprueba/corrige y añade muestras para re-entrenar
python dupin.py identificar-v2 --umbral 0.6 --revisar --top-k 3 --incluir-todas

# En revisión: mover en vez de copiar al set de entrenamiento
python dupin.py identificar-v2 --revisar --mover

# En revisión: no agregar al set de entrenamiento (solo registrar feedback)
python dupin.py identificar-v2 --revisar --no-agregar
```

Genera:
- Archivo JSON con resultados
- Reporte legible en texto con estadísticas (TOP-1 por imagen)
- (Opcional con `--revisar`) `user_patterns/review_feedback_v2.json` con aprobaciones/correcciones
- (Opcional con `--revisar`) copia/mueve imágenes al patrón correcto en `fotos_entrenamiento/por_patron/` y las registra como nuevas muestras

### listar-patrones-v2
Lista todos los patrones con información detallada.

```bash
python dupin.py listar-patrones-v2
```

### info-v2
Muestra información detallada del sistema y modelo.

```bash
python dupin.py info-v2
```

### flujo-completo-v2
Flujo completo automatizado: crear patrón, importar y entrenar.

```bash
python dupin.py flujo-completo-v2 "mi_patron" --descripcion "Descripción" --epochs 30
```

## 🔬 Técnicas de IA Detalladas

### One-Cycle Learning Rate
- Aumenta el LR desde el valor inicial hasta `max_lr` en el 30% del entrenamiento
- Disminuye gradualmente hasta el valor inicial
- Permite convergencia más rápida y mejor

### RandAugment
- Aplica transformaciones aleatorias automáticamente
- Magnitud controlada (0-30)
- Transformaciones: brillo, contraste, saturación, rotación, flip, etc.

### Mixup
- Combina dos imágenes y sus etiquetas
- Crea muestras sintéticas interpoladas
- Mejora robustez y generalización

### SE Blocks
- Aprendizaje de pesos de atención por canal
- Permite a la red enfocarse en características importantes
- Reduce ruido de características irrelevantes

### Multi-scale Inference
- Analiza la imagen en múltiples tamaños
- Promedia predicciones
- Mayor precisión a costa de más tiempo de inferencia

## 📊 Métricas y Logging

El sistema V2 incluye:
- **Progress bars** con tqdm durante entrenamiento
- **Métricas detalladas**: loss, accuracy por epoch
- **Best model checkpoint**: Guarda automáticamente el mejor modelo
- **Training history**: Guarda historial de entrenamientos
- **Reportes detallados**: Para identificación de imágenes

## 🆚 Comparación: V1 vs V2

| Característica | V1 (pattern_learner.py) | V2 (pattern_learner_v2.py) |
|---------------|-------------------------|---------------------------|
| Arquitectura | ResNet básica | ResNet + SE Blocks |
| Data Augmentation | Fijo (flip, rotación, etc.) | RandAugment + Mixup |
| Learning Rate | Cosine Annealing | One-Cycle Policy |
| Warmup | ❌ | ✅ (configurable) |
| Gradient Accumulation | ❌ | ✅ |
| Multi-scale Inference | ❌ | ✅ |
| Carpetas automáticas | ❌ | ✅ |
| Importación por carpeta | ❌ | ✅ |
| Identificación en lote | Manual | Automática desde carpeta |

## 💡 Casos de Uso Recomendados

### Caso 1: Logo de Empresa
```bash
# Crear patrón
python dupin.py crear-patron-v2 "logo_empresa" --descripcion "Logo oficial de nuestra empresa"

# Colocar 20-50 imágenes del logo en fotos_entrenamiento/por_patron/logo_empresa/

# Importar
python dupin.py importar-entrenamiento

# Entrenar con técnicas avanzadas
python dupin.py entrenar-patrones-v2 --epochs 50 --warmup 5 --focal-loss

# Identificar imágenes
python dupin.py identificar-v2 --umbral 0.8
```

### Caso 2: Múltiples Productos
```bash
# Crear múltiples patrones
python dupin.py crear-patron-v2 "producto_a"
python dupin.py crear-patron-v2 "producto_b"
python dupin.py crear-patron-v2 "producto_c"

# Colocar imágenes en cada carpeta

# Importar todo de una vez
python dupin.py importar-entrenamiento

# Entrenar con dataset multiclase
python dupin.py entrenar-patrones-v2 --epochs 100 --batch-size 32 --label-smoothing 0.1
```

### Caso 3: Alta Precisión
```bash
# Entrenar con máximo de técnicas
python dupin.py entrenar-patrones-v2 \
    --epochs 100 \
    --batch-size 8 \
    --grad-accum 4 \
    --warmup 10 \
    --early-stopping 20 \
    --label-smoothing 0.1

# Usar multi-scale para reconocimiento
python dupin.py reconocer-v2 imagen.jpg --umbral 0.9 --multiscale
```

## 📁 Estructura Completa de Archivos

```
proyecto/
├── dupin.py                          ← Programa principal (con comandos V2)
├── core/
│   ├── pattern_learner.py              ← Sistema V1 (anterior)
│   └── pattern_learner_v2.py         ← Sistema V2 (nuevo)
├── fotos_entrenamiento/               ← CREADO AUTOMÁTICAMENTE
│   ├── por_patron/
│   │   ├── logo_empresa/
│   │   │   ├── logo1.jpg
│   │   │   ├── logo2.png
│   │   │   └── README.md
│   │   └── producto_a/
│   │       └── ...
│   └── README.md
├── fotos_identificar/                 ← CREADO AUTOMÁTICAMENTE
│   ├── imagen_a_identificar1.jpg
│   ├── imagen_a_identificar2.png
│   └── ...
├── user_patterns/
│   ├── patterns.json                 ← Metadatos de patrones
│   ├── patterns_model.pth            ← Modelo V1
│   ├── patterns_model_v2.pth         ← Modelo V2 (nuevo)
│   ├── pattern_0000/                ← Muestras internas V1
│   └── ...
└── resultados_identificacion_*.json    ← Resultados de identificación
└── resultados_identificacion_*_reporte.txt  ← Reportes legibles
```

## ⚙️ Optimizaciones de Rendimiento

### Entrenamiento Rápido
```bash
python dupin.py entrenar-patrones-v2 \
    --epochs 30 \
    --batch-size 32 \
    --warmup 2
```

### Máxima Precisión
```bash
python dupin.py entrenar-patrones-v2 \
    --epochs 100 \
    --batch-size 8 \
    --grad-accum 4 \
    --warmup 10
```

### Inferencia Rápida
```bash
python dupin.py reconocer-v2 imagen.jpg --umbral 0.7
# Sin --multiscale (más rápido)
```

### Inferencia Precisa
```bash
python dupin.py reconocer-v2 imagen.jpg --umbral 0.5 --multiscale
# Con --multiscale (más preciso, más lento)
```

## 🔧 Solución de Problemas

### Problema: "No hay modelo entrenado"
**Solución:**
```bash
python dupin.py entrenar-patrones-v2 --epochs 30
```

### Problema: "No hay muestras de entrenamiento"
**Solución:**
```bash
# 1. Crear patrón
python dupin.py crear-patron-v2 "mi_patron"

# 2. Colocar imágenes en fotos_entrenamiento/por_patron/mi_patron/

# 3. Importar
python dupin.py importar-entrenamiento
```

### Problema: Baja precisión
**Soluciones:**
1. Añadir más imágenes de entrenamiento (mínimo 20 por patrón)
2. Usar más épocas de entrenamiento
3. Activar Focal Loss para clases desbalanceadas
4. Usar Label Smoothing para mejor generalización
5. Reducir el umbral de confianza

### Problema: Overfitting
**Soluciones:**
1. Aumentar el dropout (`--dropout 0.5`)
2. Usar Label Smoothing (`--label-smoothing 0.1`)
3. Reducir épocas de entrenamiento
4. Aumentar el data augmentation (activar RandAugment y Mixup)

## 📚 Referencias

- **One-Cycle Policy**: https://arxiv.org/abs/1708.07120
- **RandAugment**: https://arxiv.org/abs/1909.13719
- **Mixup**: https://arxiv.org/abs/1710.09412
- **SE Blocks**: https://arxiv.org/abs/1709.01507

## 🎯 Próximas Mejoras Planeadas

- [ ] Knowledge Distillation para modelos más compactos
- [ ] AutoML para búsqueda automática de hiperparámetros
- [ ] Soporte para video en tiempo real con V2
- [ ] Exportación a ONNX para despliegue
- [ ] API REST para integración
