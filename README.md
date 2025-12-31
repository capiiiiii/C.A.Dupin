# C.A. Dupin

**C.A. Dupin** (Chevalier Auguste Dupin) es un sistema abierto de análisis visual diseñado para encontrar, comparar y aprender patrones visuales a partir de imágenes y video, con el ser humano como guía central del proceso. El sistema combina modelos de visión por computadora con interacción directa del usuario, permitiendo construir conocimiento visual de forma progresiva, transparente y controlada.

## 🌟 Características Principales

- **Aprendizaje Guiado por Humanos**: El sistema observa, compara y aprende junto al usuario. Las decisiones finales siempre son humanas.
- **Definición de Patrones Personalizados**: Define tus propios patrones visuales mediante imágenes de ejemplo y regiones de interés (ROI).
- **Análisis Multi-módulo**: Arquitectura modular que incluye reconocimiento de rostros, cuerpos, animales, plantas, objetos, billetes, estrellas y más.
- **Razonamiento Visual**: El sistema no solo muestra resultados, sino que expone su proceso interno mediante mapas de calor y visualización de coincidencias.
- **Entrenamiento Incremental Local**: Entrena y refina modelos localmente sin depender de servicios en la nube, garantizando total privacidad.
- **Soporte Multi-idioma**: Interfaz disponible en español, inglés y francés.

## 🚀 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt
```

## 🛠️ Uso del Sistema

### Análisis Integral (Nuevo)

Realiza un análisis profundo que combina todos los módulos y patrones aprendidos, mostrando el razonamiento visual:

```bash
python dupin.py analizar imagen.jpg --umbral 0.6
```

### Aprendizaje de Patrones de Usuario

1. **Definir un patrón**:
   ```bash
   python dupin.py definir-patron "mi_logo" --descripcion "Logo corporativo" --imagen logo.jpg
   ```

2. **Entrenar el conocimiento**:
   ```bash
   python dupin.py entrenar-patrones --epochs 15
   ```

3. **Reconocer en nuevas imágenes (o directorios)**:
   ```bash
   python dupin.py reconocer-patron ./mis_fotos --umbral 0.7
   ```

### Retroalimentación Humana

Aprueba o corrige las detecciones del sistema para mejorar su precisión:

```bash
# Aprobar detección correcta
python dupin.py aprobar foto.jpg --tipo "mi_logo"

# Corregir detección errónea
python dupin.py corregir foto.jpg "otro_objeto" --tipo "mi_logo"
```

### Comparación con Probabilidades Detalladas

Compara dos imágenes o regiones específicas viendo el razonamiento técnico:

```bash
python dupin.py comparar-prob img1.jpg img2.jpg --metodo sift --razonamiento
```

### Cámara en Vivo

Análisis multimodular en tiempo real:

```bash
python dupin.py camara
```

## 📂 Estructura del Proyecto

- `dupin.py`: Punto de entrada CLI principal.
- `core/`: Módulos nucleares del sistema.
  - `image_matcher.py`: Motores de comparación visual.
  - `pattern_learner.py`: Sistema de aprendizaje de patrones CNN.
  - `human_feedback.py`: Gestión del loop de retroalimentación.
  - `roi_manager.py`: Selección y gestión de regiones de interés.
  - `module_manager.py`: Orquestador de módulos de reconocimiento.
  - `visual_interface.py`: Generación de visualizaciones y razonamiento.

## 📚 Documentación Detallada

- [**FEATURES_IMPLEMENTED.md**](FEATURES_IMPLEMENTED.md): Listado completo de capacidades.
- [**DESCRIPCION_SISTEMA.md**](DESCRIPCION_SISTEMA.md): Filosofía y visión del proyecto.
- [**IMPLEMENTACION.md**](IMPLEMENTACION.md): Detalles técnicos y arquitectura.

## ⚖️ Filosofía

C.A. Dupin se basa en la idea de que la inteligencia artificial debe ser una extensión de la capacidad humana, no un reemplazo. El sistema expone su "pensamiento" para que el usuario pueda comprender por qué se tomó una decisión y corregirla si es necesario, fomentando una relación de aprendizaje mutuo.

---
Desarrollado como software de código abierto para la comunidad.
