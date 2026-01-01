# 🔍 Carpeta de Identificación - C.A. Dupin V2

Esta carpeta es donde colocas las imágenes que quieres identificar usando los patrones que ya has entrenado.

## 📂 Cómo Usar

### Paso 1: Entrenar Primero
Antes de identificar, debes tener un modelo entrenado:

```bash
# Si no has entrenado, sigue los pasos de fotos_entrenamiento/
# O usa un flujo completo:
python dupin.py flujo-completo-v2 "tu_patron" --epochs 30
```

### Paso 2: Colocar Imágenes a Identificar
Copia o mueve las imágenes que quieres identificar a esta carpeta:

```bash
# Ejemplo
cp ~/mis_fotos_a_identificar/* fotos_identificar/
```

**Formatos soportados:** JPG, JPEG, PNG, BMP, GIF, TIFF

### Paso 3: Identificar Patrones
Usa el comando para identificar todas las imágenes:

```bash
python dupin.py identificar-v2 --umbral 0.6
```

Esto:
- Analiza todas las imágenes en `fotos_identificar/`
- Detecta patrones previamente entrenados
- Genera un archivo JSON con resultados
- Genera un reporte legible en texto

## 📊 Resultados

Los resultados se guardan en archivos con timestamp:

- **Resultados JSON:** `resultados_identificacion_YYYYMMDD_HHMMSS.json`
- **Reporte TXT:** `resultados_identificacion_YYYYMMDD_HHMMSS_reporte.txt`

### Formato del JSON
```json
{
  "fotos_identificar/imagen1.jpg": [
    {
      "pattern_id": "pattern_0000",
      "pattern_name": "logo_empresa",
      "probability": 0.95
    }
  ],
  "fotos_identificar/imagen2.jpg": [
    {
      "pattern_id": "pattern_0001",
      "pattern_name": "producto_a",
      "probability": 0.87
    }
  ]
}
```

### Formato del Reporte TXT
```
═════════════════════════════════════════════════════════
  REPORTE DE IDENTIFICACIÓN DE PATRONES
  Generado: 2025-01-01 12:34:56
═════════════════════════════════════════════════════════

📊 RESUMEN POR PATRÓN:
  • logo_empresa: 15 detecciones (conf. promedio: 92.35%)
  • producto_a: 8 detecciones (conf. promedio: 85.12%)

📁 Total de imágenes analizadas: 23
🎯 Total de patrones detectados: 2

═════════════════════════════════════════════════════════
DETALLES POR IMAGEN:
═════════════════════════════════════════════════════════

🖼️  imagen1.jpg
   └─ logo_empresa (95.23%)

🖼️  imagen2.jpg
   └─ producto_a (87.45%)
```

## 🎯 Ajustes de Umbral

El umbral de confianza determina qué tan estricto es el sistema:

```bash
# Umbral bajo (más detecciones, posibles falsos positivos)
python dupin.py identificar-v2 --umbral 0.3

# Umbral medio (balanceado)
python dupin.py identificar-v2 --umbral 0.5

# Umbral alto (menos detecciones, más preciso)
python dupin.py identificar-v2 --umbral 0.7

# Umbral muy alto (solo detecciones muy confiables)
python dupin.py identificar-v2 --umbral 0.9
```

### Recomendaciones de Umbral
- **0.9+:** Para aplicaciones críticas donde falsos positivos no son aceptables
- **0.7-0.8:** Para aplicaciones con buen balance entre precisión y recall
- **0.5-0.6:** Para explorar y encontrar patrones incluso en casos difíciles
- **0.3-0.4:** Para descubrir patrones que podrían ser difíciles de detectar

## 🔬 Identificar una Imagen Específica

Si quieres identificar una sola imagen:

```bash
python dupin.py reconocer-v2 imagen.jpg --umbral 0.7
```

### Multi-scale Inference
Para mayor precisión (más lento):

```bash
python dupin.py reconocer-v2 imagen.jpg --umbral 0.7 --multiscale
```

El sistema analizará la imagen en múltiples escalas:
- 96x96
- 128x128  
- 160x160

Y promediará las predicciones para mayor precisión.

## 💡 Casos de Uso

### Caso 1: Verificar Logo en Fotos
```bash
# Colocar fotos en fotos_identificar/
cp ~/mis_fotos/*.jpg fotos_identificar/

# Identificar con umbral alto para precisión
python dupin.py identificar-v2 --umbral 0.8

# Revisar el reporte
cat resultados_identificacion_*_reporte.txt
```

### Caso 2: Buscar Productos en Lotes
```bash
# Copiar todo un lote de fotos
cp ~/lote_fotos/* fotos_identificar/

# Identificar con umbral medio para capturar más
python dupin.py identificar-v2 --umbral 0.5 --output lote_enero.json

# El archivo lote_enero.json contendrá todos los resultados
```

### Caso 3: Análisis Detallado de Una Foto
```bash
# Identificar una foto específica con máxima precisión
python dupin.py reconocer-v2 foto_especial.jpg --umbral 0.5 --multiscale

# El sistema mostrará en consola:
# - Patrón detectado
# - Probabilidad
# - Nivel de confianza (Muy alta, Alta, Media, Baja)
```

## 📈 Procesamiento en Lote

El sistema `identificar-v2` puede procesar cientos de imágenes de forma eficiente:

```bash
# Colocar muchas imágenes
cp ~/gran_coleccion/* fotos_identificar/

# Identificar todas
python dupin.py identificar-v2 --umbral 0.6

# El sistema procesará todas y generará un reporte completo
```

### Ventajas del Procesamiento en Lote
- Procesa automáticamente todas las imágenes de la carpeta
- Genera estadísticas agregadas
- Crea reportes estructurados
- Exporta en JSON para integración con otros sistemas

## ❓ Problemas Comunes

### "No hay modelo entrenado"
**Solución:**
```bash
# Primero entrena un modelo
python dupin.py entrenar-patrones-v2 --epochs 30

# O usa el flujo completo
python dupin.py flujo-completo-v2 "patron" --epochs 30
```

### "No se encontraron patrones"
**Causas posibles:**
1. El umbral es demasiado alto
2. Las imágenes no contienen los patrones entrenados
3. El modelo necesita más entrenamiento

**Soluciones:**
- Bajar el umbral: `--umbral 0.3`
- Verificar que las imágenes son correctas
- Entrenar con más datos: `--epochs 100`

### "Probabilidades muy bajas"
**Soluciones:**
1. Reentrenar el modelo con más datos
2. Usar imágenes más variadas en el entrenamiento
3. Verificar que las imágenes a identificar son similares a las de entrenamiento
4. Usar multi-scale inference: `--multiscale`

## 🔄 Flujo de Trabajo Completo

```
1. Crear Patrones en fotos_entrenamiento/
   ↓
2. Colocar imágenes de entrenamiento
   ↓
3. Importar: python dupin.py importar-entrenamiento
   ↓
4. Entrenar: python dupin.py entrenar-patrones-v2
   ↓
5. Colocar imágenes a identificar en fotos_identificar/
   ↓
6. Identificar: python dupin.py identificar-v2
   ↓
7. Revisar resultados (JSON + reporte TXT)
```

## 📚 Comandos Relacionados

```bash
# Ver información del sistema
python dupin.py info-v2

# Listar patrones entrenados
python dupin.py listar-patrones-v2

# Ver información de un patrón específico
python dupin.py info-v2 | grep "nombre_patron"

# Identificar una sola imagen
python dupin.py reconocer-v2 imagen.jpg --umbral 0.7 --multiscale
```

## 🌟 Consejos para Mejorar Detecciones

1. **Imágenes Claras:** Usa fotos nítidas y bien iluminadas
2. **Angulos Similares:** El patrón debe verse similar al entrenamiento
3. **Tamaño Adecuado:** Imágenes muy pequeñas o muy grandes pueden afectar
4. **Sin Oclusiones:** El patrón debe estar completamente visible
5. **Fondo Neutral:** Fondos complejos pueden confundir al modelo

---

**C.A. Dupin - Sistema de Reconocimiento Visual Inteligente**
