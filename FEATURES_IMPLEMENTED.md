# Características Implementadas

Este documento describe todas las características implementadas para C.A. Dupin.

## ✅ Características Principales

### 1. Comparación de Imágenes y Regiones Visuales
- **Comparación de imágenes completas**: Usa métodos ORB, SIFT, histograma y SSIM
- **Comparación de regiones específicas (ROI)**: Permite comparar solo regiones de interés en las imágenes
- **Comparación de múltiples ROIs**: Compara varias regiones simultáneamente
- **Comparación con detalles**: Retorna información detallada sobre la comparación

**Comandos**:
```bash
python dupin.py comparar imagen1.jpg imagen2.jpg --umbral 0.8
python dupin.py comparar-prob imagen1.jpg imagen2.jpg --metodo orb
python dupin.py comparar-prob img1.jpg img2.jpg --roi1 10 10 100 100 --roi2 20 20 100 100
```

### 2. Aprendizaje de Patrones Definidos por el Usuario
- **Definición de patrones personalizados**: Los usuarios pueden definir sus propios patrones visuales
- **Entrenamiento de patrones**: Entrena un modelo con los patrones definidos
- **Reconocimiento de patrones**: Detecta patrones definidos en nuevas imágenes
- **Gestión de muestras**: Permite añadir múltiples muestras por patrón

**Comandos**:
```bash
python dupin.py definir-patron "mi_logo" --descripcion "Logotipo de mi marca" --imagen logo.jpg
python dupin.py entrenar-patrones --epochs 15
python dupin.py reconocer-patron foto.jpg --umbral 0.7
python dupin.py reconocer-patron foto.jpg --roi 50 50 200 200 --umbral 0.8
python dupin.py listar-patrones
```

### 3. Visualización de Probabilidades y Similitudes
- **Probabilidades detalladas**: Muestra:
  - Probabilidad de similitud
  - Probabilidad de ser idénticos
  - Probabilidad de ser diferentes
  - Nivel de confianza (muy alta, alta, media, baja, muy baja)
- **Detalles técnicos**: Muestra keypoints, matches, distancia promedio
- **Visualización clara**: Formato fácil de entender

**Salida de ejemplo**:
```
📊 Resultados:
  Similitud: 87.34%

📈 Probabilidades:
  Similares:      87.34%
  Idénticos:      76.28%
  Diferentes:     12.66%

🔍 Nivel de confianza: ALTA
```

### 4. Marcado de Regiones Específicas o Imágenes Completas
- **Selección interactiva de ROI**: Interfaz visual para seleccionar regiones
- **Selección múltiple**: Puede seleccionar múltiples regiones por imagen
- **Detección automática**: Detecta ROIs automáticamente usando contornos, bordes o color
- **Gestión de ROIs**: Guarda, carga y visualiza ROIs

**Comandos**:
```bash
python dupin.py roi --imagen foto.jpg
```

### 5. Aprendizaje desde Aprobaciones y Correcciones Humanas
- **Feedback de aprobación**: Aprueba patrones detectados correctamente
- **Feedback de corrección**: Corrige patrones incorrectamente identificados
- **Feedback específico de ROI**: Feedback a nivel de región de interés
- **Estadísticas de feedback**: Muestra tasa de aprobación, correcciones, etc.
- **Exportación de datos de aprendizaje**: Exporta todo el feedback para reentrenamiento

**Comandos**:
```bash
python dupin.py aprobar foto.jpg --tipo "logo_empresa"
python dupin.py corregir foto.jpg "Este es otro logotipo" --tipo "logo"
python dupin.py aprobar foto.jpg --roi 50 50 200 200 --tipo "logo"
python dupin.py corregir foto.jpg "corrección" --roi 50 50 200 200 --tipo "logo"
```

### 6. Funciona Offline
- **Sin dependencias en la nube**: Todo el procesamiento es local
- **Sin API keys**: No requiere claves de servicios externos
- **Privacidad total**: Las imágenes nunca salen del sistema local
- **Entrenamiento local**: Los modelos se entrenan en tu propia máquina

### 7. Extensible y Modular
- **Sistema de módulos**: Arquitectura modular para añadir nuevos reconocedores
- **Módulos preconfigurados**: Rostros, estrellas, billetes, humanos, animales, plantas
- **Módulos personalizables**: Puedes crear tus propios módulos
- **Gestor de módulos**: Activa/desactiva módulos según necesidades

**Comandos**:
```bash
python dupin.py modulos
python dupin.py entrenar-modulos ./datos --modules faces animals
```

## 📋 Módulos Core

### ImageMatcher (core/image_matcher.py)
Comparación de imágenes usando múltiples métodos:
- ORB (Oriented FAST and Rotated BRIEF)
- SIFT (Scale-Invariant Feature Transform)
- Histograma de color
- SSIM (Structural Similarity Index)

**Funciones clave**:
- `compare()`: Comparación básica
- `compare_with_details()`: Comparación con detalles
- `compare_multiple_rois()`: Comparación de múltiples regiones
- `find_matches()`: Búsqueda en base de datos
- `_calculate_probability()`: Cálculo de probabilidades

### PatternLearner (core/pattern_learner.py)
Sistema de aprendizaje de patrones personalizados:
- Definición de patrones por el usuario
- Entrenamiento con redes neuronales CNN
- Reconocimiento de patrones entrenados
- Tracking de aprobaciones y correcciones

**Funciones clave**:
- `define_pattern()`: Define nuevo patrón
- `add_pattern_sample()`: Añade muestras de entrenamiento
- `train_patterns()`: Entrena el modelo
- `recognize_pattern()`: Reconoce patrones en imágenes
- `record_feedback()`: Registra feedback humano
- `list_patterns()`: Lista todos los patrones
- `delete_pattern()`: Elimina un patrón

### HumanFeedbackLoop (core/human_feedback.py)
Gestión de retroalimentación humana mejorada:
- Feedback a nivel de imagen
- Feedback específico de ROI
- Aprobación de patrones
- Corrección de patrones
- Estadísticas de feedback

**Funciones nuevas**:
- `add_roi_feedback()`: Feedback específico de ROI
- `approve_pattern()`: Aprobar patrón
- `correct_pattern()`: Corregir patrón
- `get_roi_statistics()`: Estadísticas de ROI
- `export_learning_data()`: Exportar datos de aprendizaje
- `batch_approve_corrections()`: Aprobación en lote

### ROIManager (core/roi_manager.py)
Gestión de regiones de interés:
- Selección interactiva
- Detección automática
- Visualización de ROIs
- Gestión de archivos de ROI

**Funciones existentes** (mejoradas):
- `select_roi_interactive()`: Selección visual
- `auto_detect_roi()`: Detección automática
- `extract_roi_regions()`: Extracción de regiones
- `visualize_rois()`: Visualización
- `save_rois()` / `load_rois()`: Persistencia

### VisualInterface (core/visual_interface.py)
Visualización de resultados:
- Muestra detecciones con bounding boxes
- Visualización de probabilidades
- Mapas de calor
- Comparación con ground truth
- Exportación de reportes

## 🔧 Comandos CLI Disponibles

### Comandos Existentes
- `comparar`: Comparar dos imágenes
- `entrenar`: Entrenar modelo general
- `ajustar`: Ajustar modelo con feedback
- `interactivo`: Modo interactivo
- `roi`: Selección de ROI
- `camara`: Cámara en vivo
- `visual`: Análisis visual
- `modulos`: Listar módulos
- `entrenar-modulos`: Entrenar módulos específicos

### Nuevos Comandos
- `definir-patron`: Definir patrón visual personalizado
- `entrenar-patrones`: Entrenar modelo con patrones definidos
- `reconocer-patron`: Reconocer patrones en imagen
- `listar-patrones`: Listar patrones definidos
- `comparar-prob`: Comparar con probabilidades detalladas
- `aprobar`: Aprobar patrón detectado
- `corregir`: Corregir patrón detectado

## 📊 Ejemplos de Uso

### Flujo Completo de Trabajo

#### 1. Definir un patrón personalizado
```bash
python dupin.py definir-patron "logo_empresa" \
  --descripcion "Logotipo corporativo azul" \
  --imagen logo.jpg \
  --roi 100 100 200 100
```

#### 2. Añadir más muestras
```bash
python dupin.py definir-patron "logo_empresa" \
  --imagen logo2.jpg
```

#### 3. Entrenar el modelo
```bash
python dupin.py entrenar-patrones --epochs 20
```

#### 4. Reconocer en nuevas imágenes
```bash
python dupin.py reconocer-patron nueva_imagen.jpg --umbral 0.8
```

#### 5. Dar feedback humano
```bash
# Si la detección fue correcta
python dupin.py aprobar nueva_imagen.jpg --tipo "logo_empresa"

# Si fue incorrecta
python dupin.py corregir nueva_imagen.jpg "Es otro logo" --tipo "logo"
```

#### 6. Comparar con probabilidades
```bash
python dupin.py comparar-prob img1.jpg img2.jpg \
  --roi1 50 50 100 100 \
  --roi2 30 30 100 100 \
  --metodo orb
```

## 🏗️ Arquitectura

### Flujo de Datos

```
Imagen → ROI Manager → Image Matcher → Probabilidades
                              ↓
                      Pattern Learner → Reconocimiento
                              ↓
                    Human Feedback Loop → Aprendizaje
                              ↓
                      Visual Interface → Visualización
```

### Componentes Interconectados

1. **ImageMatcher**: Compara imágenes con soporte de ROI
2. **PatternLearner**: Aprende patrones personalizados
3. **HumanFeedbackLoop**: Recibe feedback humano
4. **ROIManager**: Gestiona regiones de interés
5. **VisualInterface**: Muestra resultados visuales

## 💾 Archivos Generados

- `user_patterns/`: Directorio de patrones de usuario
  - `patterns.json`: Metadatos de patrones
  - `pattern_XXXX/`: Muestras de cada patrón
  - `patterns_model.pth`: Modelo entrenado de patrones

- `feedback.json`: Feedback de imágenes completas
- `roi_feedback.json`: Feedback específico de ROIs
- `learning_data.json`: Datos exportados para aprendizaje
- `rois_seleccionadas.json`: ROIs guardadas

## 🎯 Casos de Uso

### Detección de Logos
```bash
# Definir logo de marca
python dupin.py definir-patron "logo_nike" --imagen nike.jpg

# Buscar en imágenes
python dupin.py reconocer-patron foto.jpg

# Feedback de correcciones
python dupin.py corregir foto.jpg "Es logo_adidas" --tipo "logo_nike"
```

### Comparación de Documentos
```bash
# Comparar firmas en regiones específicas
python dupin.py comparar-prob firma1.jpg firma2.jpg \
  --roi1 100 200 300 100 \
  --roi2 80 180 300 100
```

### Control de Calidad
```bash
# Definir patrón de producto correcto
python dupin.py definir-patron "producto_ok" --imagen producto_ok.jpg

# Verificar producción
python dupin.py reconocer-patron producto_línea.jpg

# Marcar defectos
python dupin.py corregir producto_línea.jpg "producto_defecto" --tipo "producto_ok"
```

## 🔒 Seguridad y Privacidad

- ✅ Todo el procesamiento es local
- ✅ No se envían datos a servicios externos
- ✅ No requiere conexión a internet
- ✅ Los modelos son propiedad del usuario
- ✅ Feedback guardado localmente

## 📈 Métricas de Rendimiento

El sistema mantiene estadísticas de:
- Precisión de patrones (aprobaciones / total)
- Número de muestras por patrón
- Tasa de aprobación de feedback
- Tiempos de entrenamiento y reconocimiento

## 🚀 Rendimiento

- **Comparación**: < 1 segundo para imágenes estándar
- **Entrenamiento de patrones**: ~2-5 segundos por época
- **Reconocimiento**: < 0.5 segundos por imagen
- **Selección de ROI**: Interactiva en tiempo real

## 📝 Notas

- Todos los modelos usan PyTorch
- Imágenes preprocesadas a 100x100 para CNN
- Soporta formatos: JPG, PNG, BMP, TIFF, GIF
- Múltiples idiomas en la interfaz (es, en, fr)
- Compatible con Python 3.7+

## 🛠️ Extensibilidad

El sistema está diseñado para ser extensible:

### Añadir nuevo módulo de reconocimiento:
1. Heredar de `BaseRecognitionModule`
2. Implementar `train()`, `predict()`, `evaluate()`
3. Registrar en `ModuleManager`

### Añadir nuevo método de comparación:
1. Añadir método a `ImageMatcher`
2. Implementar lógica de comparación
3. Actualizar argumentos CLI

### Añadir nueva visualización:
1. Extender `VisualInterface`
2. Añadir método de renderizado
3. Integrar con CLI
