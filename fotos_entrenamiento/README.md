# 📚 Carpeta de Entrenamiento - C.A. Dupin V2

Esta carpeta contiene las imágenes de entrenamiento para el sistema de reconocimiento de patrones V2.

## 📂 Estructura

```
fotos_entrenamiento/
├── por_patron/              ← Carpetas organizadas por patrón
│   ├── nombre_patron1/      ← Coloca aquí las fotos de este patrón
│   ├── nombre_patron2/      ← Coloca aquí las fotos de este patrón
│   └── ...
└── README.md                ← Este archivo
```

## 🎯 Cómo Usar

### Paso 1: Crear un Patrón
Usa el comando para crear un nuevo patrón:

```bash
python dupin.py crear-patron-v2 "nombre_del_patron" --descripcion "Descripción opcional"
```

Esto creará automáticamente una carpeta en `por_patron/nombre_del_patron/`

### Paso 2: Colocar las Imágenes
Copia o mueve las imágenes de entrenamiento a la carpeta del patrón:

```bash
# Ejemplo: Si creaste el patrón "logo_empresa"
cp ~/mis_logos/* fotos_entrenamiento/por_patron/logo_empresa/
```

**Formatos soportados:** JPG, JPEG, PNG, BMP, GIF, TIFF

**Cantidad recomendada:** Mínimo 20-30 imágenes por patrón para buenos resultados

### Paso 3: Importar las Imágenes
Importa todas las imágenes desde las carpetas:

```bash
python dupin.py importar-entrenamiento
```

Este comando:
- Busca imágenes en todas las carpetas de `por_patron/`
- Las importa al sistema interno
- Muestra un resumen de imágenes importadas

### Paso 4: Entrenar el Modelo
Entrena el modelo con las imágenes importadas:

```bash
python dupin.py entrenar-patrones-v2 --epochs 50 --warmup 3
```

## 💡 Consejos para Mejores Resultados

1. **Variabilidad:** Usa fotos en diferentes condiciones:
   - Diferentes ángulos
   - Diferentes iluminaciones
   - Diferentes fondos
   - Diferentes tamaños

2. **Cantidad:** Mínimo 20-30 imágenes por patrón
   - 50-100 imágenes = Buenos resultados
   - 100+ imágenes = Excelentes resultados

3. **Calidad:** Las imágenes deben ser claras y nítidas
   - Evita fotos borrosas
   - El patrón debe estar visible
   - Buena iluminación

4. **Balance:** Si tienes múltiples patrones, intenta mantener un número similar de imágenes por patrón

5. **Diversidad:** No uses la misma imagen varias veces
   - Cada imagen debe ser única
   - Más diversidad = mejor generalización

## 📊 Ejemplos de Uso

### Ejemplo 1: Logo de Empresa
```bash
# Crear patrón
python dupin.py crear-patron-v2 "logo_empresa" --descripcion "Logo oficial de nuestra empresa"

# Copiar 50 fotos del logo a diferentes ángulos
cp logos/*.jpg fotos_entrenamiento/por_patron/logo_empresa/

# Importar
python dupin.py importar-entrenamiento

# Entrenar
python dupin.py entrenar-patrones-v2 --epochs 50 --batch-size 16 --warmup 3
```

### Ejemplo 2: Múltiples Productos
```bash
# Crear patrones
python dupin.py crear-patron-v2 "producto_a"
python dupin.py crear-patron-v2 "producto_b"
python dupin.py crear-patron-v2 "producto_c"

# Colocar 30 fotos de cada producto
cp fotos_producto_a/* fotos_entrenamiento/por_patron/producto_a/
cp fotos_producto_b/* fotos_entrenamiento/por_patron/producto_b/
cp fotos_producto_c/* fotos_entrenamiento/por_patron/producto_c/

# Importar y entrenar
python dupin.py importar-entrenamiento
python dupin.py entrenar-patrones-v2 --epochs 100 --batch-size 32 --label-smoothing 0.1
```

## 🔄 Proceso Completo

```
1. Crear Patrón
   ↓
2. Colocar Imágenes en fotos_entrenamiento/por_patron/<nombre>/
   ↓
3. Importar: python dupin.py importar-entrenamiento
   ↓
4. Entrenar: python dupin.py entrenar-patrones-v2
   ↓
5. Identificar: python dupin.py identificar-v2
```

## ❓ Problemas Comunes

### "No hay imágenes para el patrón"
- Verifica que hay imágenes en la carpeta del patrón
- Verifica que los formatos son correctos (JPG, PNG, etc.)

### "Baja precisión"
- Añade más imágenes de entrenamiento
- Aumenta las épocas de entrenamiento
- Asegúrate de tener variedad en las imágenes

### "Overfitting"
- Usa más imágenes de entrenamiento
- Aumenta el dropout: `--dropout 0.5`
- Usa label smoothing: `--label-smoothing 0.1`

## 📚 Más Información

Para más detalles sobre las técnicas de IA implementadas en V2, consulta:
- `MEJORAS_V2.md` - Documentación completa del sistema V2
- `python dupin.py info-v2` - Información del sistema actual
- `python dupin.py listar-patrones-v2` - Listar patrones definidos

---

**C.A. Dupin - Sistema de Reconocimiento Visual Inteligente**
