# Implementación de Human-in-the-Loop Training para C.A. Dupin

## Resumen de Cambios

Esta implementación añade capacidades de entrenamiento local con retroalimentación humana para C.A. Dupin, permitiendo que los usuarios entrenen modelos con sus propios datos y mejoren el sistema mediante correcciones humanas.

## Cambios Principales

### 1. `core/model_trainer.py`

#### Nuevas Clases:
- **`ImagePairDataset`**: Dataset PyTorch que carga imágenes organizadas por clase en subdirectorios
- **Métodos añadidos a `ModelTrainer`**:
  - `_create_pairs()`: Genera pares de imágenes positivos (misma clase) y negativos (distinta clase)
  - `fine_tune_with_feedback()`: Ajusta el modelo usando correcciones humanas
  - `save_feedback()` / `load_feedback()`: Persistencia de feedback humano en JSON

#### Mejoras:
- Reemplazó el entrenamiento simulado con entrenamiento real usando DataLoader
- Implementó transformaciones de imágenes (resize, to_tensor)
- Ajustó arquitectura de la red siamesa para input de 100x100
- Reducción de dimensiones en capas fully connected para efficiency

### 2. `core/human_feedback.py`

#### Nuevas Funcionalidades:
- **`_load_feedback()`**: Carga feedback previo de `feedback.json` al iniciar
- **`_save_feedback()`**: Guarda feedback automáticamente en cada interacción
- **`_exportar_feedback()`**: Nueva opción en menú para exportar feedback
- **`get_feedback_data()`**: Método público para acceder al feedback

#### Mejoras:
- Almacenamiento de rutas completas de imágenes (no solo nombres)
- Feedback persistente entre sesiones
- Nueva opción de menú (4) para exportar feedback

### 3. `dupin.py`

#### Nueva Función:
- **`ajustar_modelo()`**: Integra el ajuste del modelo con feedback humano
  - Carga modelo existente
  - Carga archivo JSON de feedback
  - Ejecuta fine-tuning
  - Guarda modelo mejorado

#### Nueva CLI Command:
- `ajustar`: Comando para ajustar modelos con feedback
  - `--modelo`: Ruta al modelo pre-entrenado (.pth)
  - `--feedback`: Ruta al JSON de feedback
  - `--output`: Ruta de salida del modelo ajustado (default: modelo_ajustado.pth)

### 4. `README.md`

#### Actualizaciones:
- Documentación sobre estructura de directorios para entrenamiento
- Instrucciones para el modo interactivo
- Nueva sección: "Ajustar modelo con feedback humano"
- Explicación del flujo de trabajo human-in-the-loop

### 5. `.gitignore`

#### Nueva Entrada:
- `feedback.json`: Excluye archivos de feedback humano del control de versiones (puede contener rutas locales)

## Flujo de Trabajo del Usuario

### 1. Entrenamiento Inicial
```bash
python dupin.py entrenar ./mis_datos --epochs 20 --output modelo_inicial.pth
```

Requiere estructura de directorios:
```
mis_datos/
├── persona_1/
│   ├── foto1.jpg
│   └── foto2.jpg
└── persona_2/
    ├── foto3.jpg
    └── foto4.jpg
```

### 2. Retroalimentación Humana
```bash
python dupin.py interactivo ./imagenes_a_revisar
```

Durante el modo interactivo:
- Comparar pares de imágenes
- Validar o corregir similitudes calculadas por IA
- Feedback se guarda automáticamente en `feedback.json`
- Exportar feedback para reentrenamiento (opción 4)

### 3. Ajuste del Modelo
```bash
python dupin.py ajustar --modelo modelo_inicial.pth \
                        --feedback ./imagenes_a_revisar/feedback.json \
                        --output modelo_mejorado.pth
```

El modelo ajustado incorpora las correcciones humanas.

## Detalles Técnicos

### Arquitectura de la Red Siamésa
- Input: 100x100x3 (RGB)
- 4 capas convolucionales con ReLU y MaxPool
- 512-dimensional embedding
- Contrastive loss con margin=2.0

### Generación de Pares para Entrenamiento
- Pares positivos: 2 imágenes de la misma clase
- Pares negativos: 2 imágenes de clases diferentes
- Balance: ~50% positivos, ~50% negativos
- Máximo: min(N*2, 1000) pares por dataset

### Fine-Tuning con Feedback
- Learning rate reducido (0.1x del original)
- Usa pares de imágenes del feedback humano
- Etiquetas derivadas de similitudes humanas
- Útil para correcciones específicas

## Ventajas de la Implementación

1. **Localidad Completa**: Todo funciona offline sin servicios en la nube
2. **Privacidad**: Los datos nunca salen de la máquina del usuario
3. **Human-in-the-Loop**: La decisión final siempre es humana
4. **Persistencia**: El feedback se guarda y acumula entre sesiones
5. **Incremental**: Los modelos mejoran progresivamente con más feedback
6. **Transparencia**: Los usuarios ven y pueden corregir todas las decisiones de IA

## Limitaciones y Consideraciones

1. Requiere datos organizados por clase en subdirectorios para entrenamiento
2. El tamaño de batch por defecto es 16 (ajustable según memoria disponible)
3. El número máximo de pares está limitado a 1000 por dataset
4. Fine-tuning requiere imágenes originales disponibles en las rutas del feedback
5. No hay sistema de validación cruzada implementado aún

## Próximas Mejoras Posibles

1. Sistema de validación automática durante entrenamiento
2. Data augmentation para mejorar generalización
3. Métricas de evaluación más detalladas (precision, recall, F1)
4. Interfaz gráfica para el modo interactivo
5. Soporte para streaming de video en tiempo real
6. Exportación de modelos a formatos estándar (ONNX, TensorFlow)
7. Sistema de versionado de modelos
8. Migración continua del modelo sin interrupciones
