# C.A.Dupin

Chevalier Auguste Dupin es una herramienta abierta de coincidencias visuales asistidas por humanos. Permite entrenar modelos, comparar imágenes y corregir resultados en tiempo real, mostrando patrones y similitudes sin imponer juicios. La decisión final siempre es humana.

Cualquiera que quiera contribuir con codigo es bienvenido

## Características

- **Comparación de imágenes**: Compara dos imágenes y obtén un score de similitud
- **Búsqueda de coincidencias**: Encuentra imágenes similares en una base de datos
- **Entrenamiento de modelos**: Entrena modelos personalizados con tus propios datos
- **Loop de retroalimentación humana**: Modo interactivo para revisar y corregir resultados en tiempo real

## 📚 Documentación

Para una descripción completa del sistema, su arquitectura y filosofía, consulta:
- **[DESCRIPCION_SISTEMA.md](DESCRIPCION_SISTEMA.md)** - Descripción general del sistema
- **[FEATURES_IMPLEMENTED.md](FEATURES_IMPLEMENTED.md)** - Características implementadas en detalle
- **[IMPLEMENTACION.md](IMPLEMENTACION.md)** - Detalles técnicos de implementación

## Instalación

```bash
# Clonar el repositorio
git clone <repository-url>
cd C.A.Dupin

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Comparar dos imágenes

```bash
python dupin.py comparar imagen1.jpg imagen2.jpg
```

Con umbral personalizado:

```bash
python dupin.py comparar imagen1.jpg imagen2.jpg --umbral 0.9
```

### Entrenar un modelo

```bash
python dupin.py entrenar ./directorio_datos --epochs 20 --output mi_modelo.pth
```

**Nota**: El directorio de entrenamiento debe tener imágenes organizadas en subdirectorios por clase:
```
directorio_datos/
├── clase_a/
│   ├── imagen1.jpg
│   └── imagen2.jpg
└── clase_b/
    ├── imagen3.jpg
    └── imagen4.jpg
```

### Modo interactivo (Human-in-the-Loop)

```bash
python dupin.py interactivo ./directorio_imagenes
```

El modo interactivo permite:
- Comparar imágenes manualmente
- Buscar imágenes similares
- Proporcionar feedback sobre las coincidencias
- Corregir resultados del modelo en tiempo real
- Exportar feedback para mejorar el modelo

Durante el modo interactivo, el feedback se guarda automáticamente en `feedback.json`.

### Ajustar modelo con feedback humano

Una vez que hayas recopilado feedback humano, puedes usarlo para ajustar el modelo:

```bash
python dupin.py ajustar --modelo mi_modelo.pth --feedback feedback.json --output modelo_ajustado.pth
```

Esto mejora el modelo basándose en las correcciones que los humanos han proporcionado.

## Estructura del proyecto

```
C.A.Dupin/
├── dupin.py                    # Programa principal
├── core/
│   ├── __init__.py
│   ├── image_matcher.py        # Comparación de imágenes
│   ├── model_trainer.py        # Entrenamiento de modelos
│   └── human_feedback.py       # Loop de retroalimentación humana
├── requirements.txt            # Dependencias
└── README.md
```

## Métodos de comparación

El sistema soporta varios métodos de comparación:

- **ORB** (Oriented FAST and Rotated BRIEF): Detección de características rápida
- **SIFT** (Scale-Invariant Feature Transform): Detección robusta de características
- **Histogram**: Comparación basada en histogramas de color
- **SSIM**: Índice de similitud estructural

## Filosofía

C.A.Dupin está diseñado bajo el principio de que **la decisión final siempre es humana**. El sistema:

- Muestra patrones y similitudes
- No impone juicios automáticos
- Permite corrección en tiempo real
- Aprende de la retroalimentación humana

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/amazing-feature`)
3. Commit tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## Licencia

Este proyecto es de código abierto y está disponible para todos los que quieran contribuir.
