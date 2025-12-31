#!/usr/bin/env python3
"""
C.A.Dupin - Herramienta de coincidencias visuales asistidas por humanos

Este programa permite entrenar modelos, comparar imágenes y corregir resultados
en tiempo real con soporte para ROI, cámara en vivo y módulos multiidioma.

Características principales:
- 📷 Soporte para imágenes y cámara en vivo
- 🖼️ Marcado de regiones de interés (ROI)
- 🔁 Corrección en tiempo real (aprobar / rechazar / corregir)
- 🧠 Entrenamiento incremental sin límite artificial de ejemplos
- 📊 Visualización clara de lo que el modelo está identificando
- 🌐 Interfaz disponible en múltiples idiomas
- 🧩 Módulos de reconocimiento preconfigurados y entrenables
"""

import sys
import argparse
from pathlib import Path
import cv2

# Importar módulos principales
from core.image_matcher import ImageMatcher
from core.model_trainer import ModelTrainer
from core.human_feedback import HumanFeedbackLoop
from core.roi_manager import ROIManager
from core.camera_manager import CameraManager
from core.visual_interface import VisualInterface
from core.language_manager import LanguageManager
from core.module_manager import ModuleManager
from core.pattern_learner import PatternLearner


def configurar_idioma(language_manager, idioma):
    """Configura el idioma de la interfaz."""
    if idioma and language_manager.set_language(idioma):
        return True
    return False


def comparar_imagenes(imagen1_path, imagen2_path, umbral=0.85, language_manager=None):
    """Compara dos imágenes y muestra la similitud."""
    matcher = ImageMatcher()
    
    # Obtener textos traducidos
    texts = {}
    if language_manager:
        texts = {
            'titulo': language_manager.get_text('comparing_images', 'interface'),
            'imagen1': language_manager.get_text('image_1', 'interface'),
            'imagen2': language_manager.get_text('image_2', 'interface'),
            'similitud': language_manager.get_text('similarity', 'interface'),
            'similares': language_manager.get_text('similar', 'interface'),
            'no_similares': language_manager.get_text('not_similar', 'interface'),
            'umbral': f">= {umbral:.0%}"
        }
    
    try:
        similitud = matcher.compare(imagen1_path, imagen2_path)
        
        print(f"\n{texts.get('titulo', 'Comparación de imágenes:')}")
        print(f"  {texts.get('imagen1', 'Imagen 1')}: {imagen1_path}")
        print(f"  {texts.get('imagen2', 'Imagen 2')}: {imagen2_path}")
        print(f"  {texts.get('similitud', 'Similitud')}: {similitud:.2%}")
        
        if similitud >= umbral:
            print(f"  ✓ {texts.get('similares', 'Las imágenes son similares')} ({texts.get('umbral', f'>= {umbral:.0%}')})")
        else:
            print(f"  ✗ {texts.get('no_similares', 'Las imágenes no son similares')} (< {umbral:.0%})")
        
        return similitud
    except Exception as e:
        print(f"Error comparando imágenes: {e}")
        return 0.0


def entrenar_modelo(directorio_datos, epochs=10, output_path="modelo.pth", 
                   language_manager=None):
    """Entrena un modelo con los datos proporcionados."""
    trainer = ModelTrainer()
    
    # Obtener textos traducidos
    texts = {}
    if language_manager:
        texts = {
            'titulo': language_manager.get_text('training_model', 'training'),
            'directorio': language_manager.get_text('data_directory', 'training'),
            'epocas': language_manager.get_text('epochs', 'training'),
            'completado': language_manager.get_text('completed', 'training'),
            'guardado': language_manager.get_text('saved_model', 'training')
        }
    
    print(f"\n{texts.get('titulo', 'Entrenando modelo con datos de:')} {directorio_datos}")
    print(f"{texts.get('epocas', 'Épocas')}: {epochs}")
    
    try:
        modelo = trainer.train(directorio_datos, epochs=epochs)
        trainer.save_model(modelo, output_path)
        
        print(f"\n✓ {texts.get('completado', 'Modelo guardado en:')} {output_path}")
        return modelo
    except Exception as e:
        print(f"Error entrenando modelo: {e}")
        return None


def modo_roi(interactivo=True, imagen_path=None, language_manager=None):
    """Modo de selección de regiones de interés."""
    roi_manager = ROIManager()
    
    # Obtener textos traducidos
    texts = {}
    if language_manager:
        texts = {
            'titulo': language_manager.get_text('title', 'roi'),
            'instrucciones': language_manager.get_text('instructions', 'roi'),
            'seleccionando': language_manager.get_text('selecting', 'roi'),
            'no_imagen': language_manager.get_text('no_image', 'roi')
        }
    
    print(f"\n=== {texts.get('titulo', 'Modo ROI - Regiones de Interés')} ===")
    
    if imagen_path:
        print(f"{texts.get('seleccionando', 'Seleccionando ROI en')}: {imagen_path}")
        rois = roi_manager.select_roi_interactive(imagen_path)
        
        if rois:
            print(f"\n✓ {len(rois)} ROI(s) seleccionada(s)")
            
            # Guardar ROI
            roi_file = "rois_seleccionadas.json"
            roi_manager.save_rois(rois, roi_file)
            
            # Visualizar ROI
            print(f"\n{texts.get('visualization', 'Visualizando ROI...')}")
            roi_manager.visualize_rois(imagen_path, rois)
            
            return rois
    else:
        print(texts.get('no_imagen', 'Especifica una imagen con --imagen'))
        return []
    
    return []


def modo_camara(language_manager=None):
    """Modo de cámara en vivo."""
    camera_manager = CameraManager()
    
    # Obtener textos traducidos
    texts = {}
    if language_manager:
        texts = {
            'titulo': language_manager.get_text('title', 'camera'),
            'iniciando': language_manager.get_text('initializing', 'camera'),
            'camaras_disponibles': language_manager.get_text('available_cameras', 'camera')
        }
    
    print(f"\n=== {texts.get('titulo', 'Modo Cámara en Vivo')} ===")
    
    # Mostrar cámaras disponibles
    cameras = camera_manager.list_cameras()
    print(f"{texts.get('camaras_disponibles', 'Cámaras disponibles')}: {len(cameras)}")
    for cam in cameras:
        print(f"  ID {cam['id']}: {cam['resolution']} @ {cam['fps']} FPS")
    
    if not cameras:
        print("No se encontraron cámaras disponibles")
        return
    
    # Inicializar cámara
    if camera_manager.initialize_camera():
        print(f"\n{texts.get('iniciando', 'Iniciando captura...')}")
        
        def procesar_frame(frame):
            """Procesar cada frame capturado."""
            # Aquí se puede integrar análisis con módulos
            return frame
        
        camera_manager.start_capture(callback=procesar_frame)
        
        print("Controles:")
        print("- ESC: Salir")
        print("- ESPACIO: Tomar foto")
        print("- R: Iniciar/detener grabación")
        
        try:
            input("Presiona ENTER para detener la cámara...")
        except KeyboardInterrupt:
            pass
        finally:
            camera_manager.stop_capture()
    else:
        print("Error inicializando la cámara")


def modo_visual(imagen_path, language_manager=None):
    """Modo de visualización con análisis."""
    visual_interface = VisualInterface()
    module_manager = ModuleManager()
    
    # Obtener textos traducidos
    texts = {}
    if language_manager:
        texts = {
            'titulo': language_manager.get_text('title', 'detections'),
            'cargando': language_manager.get_text('loading', 'interface'),
            'analizando': language_manager.get_text('analyzing', 'detections'),
            'detecciones': language_manager.get_text('detections_found', 'detections')
        }
    
    print(f"\n=== {texts.get('titulo', 'Modo Visual - Análisis Inteligente')} ===")
    
    if not visual_interface.load_image(imagen_path):
        return
    
    print(f"{texts.get('analizando', 'Analizando imagen...')}")
    
    # Realizar análisis con todos los módulos
    predictions = module_manager.predict(imagen_path, active_only=False)
    
    # Mostrar resultados
    total_detections = 0
    for module_id, module_predictions in predictions.items():
        if module_predictions:
            total_detections += len(module_predictions)
            
            # Añadir detecciones a la interfaz visual
            for pred in module_predictions:
                visual_interface.add_detection(
                    class_name=pred['class'],
                    confidence=pred['confidence'],
                    bounding_box=pred['bbox']
                )
    
    print(f"\n{texts.get('detecciones', 'Total de detecciones')}: {total_detections}")
    
    # Mostrar visualización
    if total_detections > 0:
        visual_interface.show_detections()
        
        # Guardar visualización
        output_path = f"analisis_{Path(imagen_path).stem}.jpg"
        visual_interface.save_visualization(output_path)
        
        # Exportar reporte
        report_path = f"reporte_{Path(imagen_path).stem}.json"
        visual_interface.export_analysis_report(report_path)
    else:
        print("No se encontraron detecciones para mostrar")


def modo_interactivo(directorio_imagenes, language_manager=None):
    """Inicia el modo interactivo mejorado con retroalimentación humana."""
    
    # Obtener textos traducidos
    texts = {}
    if language_manager:
        texts = {
            'titulo': language_manager.get_text('title', 'feedback'),
            'instrucciones': language_manager.get_text('interactive_mode', 'interface')
        }
    
    print(f"\n=== C.A. Dupin - {texts.get('titulo', 'Modo Interactivo Avanzado')} ===")
    print(texts.get('instrucciones', 'Revisando patrones y similitudes con análisis multi-módulo.\n'))
    
    # Inicializar componentes
    feedback_loop = HumanFeedbackLoop(directorio_imagenes)
    module_manager = ModuleManager()
    visual_interface = VisualInterface()
    
    # Activar todos los módulos por defecto
    for module_info in module_manager.get_available_modules():
        module_manager.activate_module(module_info['module_id'])
    
    feedback_loop.start()


def ajustar_modelo(modelo_path, feedback_path, output_path="modelo_ajustado.pth", language_manager=None):
    """Ajusta un modelo usando retroalimentación humana."""
    
    # Obtener textos traducidos
    texts = {}
    if language_manager:
        texts = {
            'titulo': language_manager.get_text('fine_tuning', 'training'),
            'cargando_modelo': language_manager.get_text('loading_model', 'training'),
            'cargando_feedback': language_manager.get_text('loading_feedback', 'feedback'),
            'sin_feedback': language_manager.get_text('no_feedback', 'feedback'),
            'ajustando': language_manager.get_text('adjusting', 'training'),
            'completado': language_manager.get_text('completed', 'training'),
            'guardado': language_manager.get_text('saved_model', 'training')
        }
    
    print(f"\n=== {texts.get('titulo', 'Ajustando Modelo con Feedback Humano')} ===")
    
    trainer = ModelTrainer()
    
    print(f"{texts.get('cargando_modelo', 'Cargando modelo desde')}: {modelo_path}")
    modelo = trainer.load_model(modelo_path)
    
    print(f"{texts.get('cargando_feedback', 'Cargando feedback desde')}: {feedback_path}")
    feedback_data = trainer.load_feedback(feedback_path)
    
    if not feedback_data:
        print(texts.get('sin_feedback', 'No se encontró feedback para ajustar el modelo.'))
        return
    
    modelo_ajustado = trainer.fine_tune_with_feedback(modelo, feedback_data)
    trainer.save_model(modelo_ajustado, output_path)
    
    print(f"\n✓ {texts.get('completado', 'Modelo ajustado guardado en')}: {output_path}")
    return modelo_ajustado


def definir_patron(nombre, descripcion="", imagen_path=None, roi=None, language_manager=None):
    """Define un nuevo patrón visual para aprendizaje."""
    pattern_learner = PatternLearner()
    
    print(f"\n=== Definir Patrón Visual ===")
    print(f"Nombre: {nombre}")
    if descripcion:
        print(f"Descripción: {descripcion}")
    if imagen_path:
        print(f"Imagen de ejemplo: {imagen_path}")
    if roi:
        print(f"ROI: {roi}")
    
    pattern_id = pattern_learner.define_pattern(
        name=nombre,
        description=descripcion,
        image_path=imagen_path,
        roi=roi
    )
    
    print(f"\n✓ Patrón definido con ID: {pattern_id}")
    return pattern_id


def entrenar_patrones(epochs=10, language_manager=None):
    """Entrena el modelo con los patrones definidos por el usuario."""
    pattern_learner = PatternLearner()
    
    print(f"\n=== Entrenando Patrones Definidos por Usuario ===")
    
    patterns = pattern_learner.list_patterns()
    if not patterns:
        print("No hay patrones definidos. Usa 'definir-patron' primero.")
        return None
    
    print(f"Patrones a entrenar: {len(patterns)}")
    for pattern in patterns:
        print(f"  - {pattern['name']} (muestras: {pattern['samples']})")
    
    success = pattern_learner.train_patterns(epochs=epochs)
    
    if success:
        print("\n✓ Entrenamiento de patrones completado")
        return pattern_learner
    else:
        print("\n✗ Error en el entrenamiento de patrones")
        return None


def reconocer_patron(imagen_path, roi=None, threshold=0.5, language_manager=None):
    """Reconoce patrones en una imagen."""
    pattern_learner = PatternLearner()
    
    print(f"\n=== Reconocimiento de Patrones ===")
    print(f"Imagen: {imagen_path}")
    if roi:
        print(f"ROI: {roi}")
    print(f"Umbral de confianza: {threshold:.0%}")
    
    detections = pattern_learner.recognize_pattern(
        image_path=imagen_path,
        roi=roi,
        threshold=threshold
    )
    
    if detections:
        print(f"\n✓ Encontrados {len(detections)} patrones:")
        for detection in detections:
            print(f"\n  Patrón: {detection['pattern_name']}")
            print(f"  Probabilidad: {detection['probability']:.2%}")
            print(f"  Confianza: {detection['probability']:.2%}")
    else:
        print("\nNo se encontraron patrones")
    
    return detections


def listar_patrones(language_manager=None):
    """Lista todos los patrones definidos por el usuario."""
    pattern_learner = PatternLearner()
    
    patterns = pattern_learner.list_patterns()
    
    if not patterns:
        print("\nNo hay patrones definidos.")
        return
    
    print(f"\n=== Patrones Definidos ({len(patterns)}) ===\n")
    
    for pattern in patterns:
        print(f"ID: {pattern['id']}")
        print(f"Nombre: {pattern['name']}")
        if pattern['description']:
            print(f"Descripción: {pattern['description']}")
        print(f"Muestras: {pattern['samples']}")
        print(f"Aprobaciones: {pattern['approved']}")
        print(f"Correcciones: {pattern['corrected']}")
        if pattern['samples'] > 0:
            print(f"Precisión: {pattern['accuracy']:.2%}")
        print(f"Creado: {pattern['created_at'][:10]}")
        print()


def comparar_con_probabilidades(imagen1_path, imagen2_path, roi1=None, roi2=None, 
                               metodo='orb', language_manager=None):
    """Compara dos imágenes mostrando probabilidades detalladas."""
    matcher = ImageMatcher(metodo=metodo)
    
    print(f"\n=== Comparación con Probabilidades Detalladas ===")
    print(f"Imagen 1: {imagen1_path}")
    if roi1:
        print(f"  ROI 1: {roi1}")
    print(f"Imagen 2: {imagen2_path}")
    if roi2:
        print(f"  ROI 2: {roi2}")
    print(f"Método: {metodo.upper()}")
    
    result = matcher.compare_with_details(
        imagen1_path, imagen2_path,
        roi1=roi1, roi2=roi2
    )
    
    print(f"\n📊 Resultados:")
    print(f"  Similitud: {result['similarity']:.2%}")
    print(f"\n📈 Probabilidades:")
    
    prob = result['probability']
    print(f"  Similares:      {prob['similar']:.2%}")
    print(f"  Idénticos:      {prob['identical']:.2%}")
    print(f"  Diferentes:     {prob['different']:.2%}")
    print(f"\n🔍 Nivel de confianza: {prob['confidence_level'].upper()}")
    
    if result['details']:
        print(f"\n📋 Detalles técnicos:")
        for key, value in result['details'].items():
            print(f"  {key}: {value}")
    
    return result


def aprobar_patron(imagen_path, roi=None, pattern_type='general', language_manager=None):
    """Aprueba un patrón detectado para aprendizaje."""
    directorio = Path(imagen_path).parent
    feedback_loop = HumanFeedbackLoop(str(directorio))
    
    feedback_loop.approve_pattern(
        image_path=imagen_path,
        roi=roi,
        pattern_type=pattern_type
    )
    
    print(f"\n✓ Patrón aprobado para aprendizaje")
    return True


def corregir_patron(imagen_path, correction, roi=None, pattern_type='general', language_manager=None):
    """Corrige un patrón detectado para aprendizaje."""
    directorio = Path(imagen_path).parent
    feedback_loop = HumanFeedbackLoop(str(directorio))
    
    feedback_loop.correct_pattern(
        image_path=imagen_path,
        roi=roi,
        correction=correction,
        pattern_type=pattern_type
    )
    
    print(f"\n✓ Patrón corregido: {correction}")
    return True


def listar_modulos(language_manager=None):
    """Lista todos los módulos disponibles."""
    module_manager = ModuleManager()
    
    print("\n=== Módulos de Reconocimiento Disponibles ===")
    
    modules = module_manager.get_available_modules()
    
    for module in modules:
        status = "✓ ACTIVO" if module['is_active'] else "○ INACTIVO"
        trained = "✓" if module['is_trained'] else "○"
        
        print(f"\n{status} {module['name']} ({module['module_id']})")
        print(f"  Descripción: {module['description']}")
        print(f"  Entrenado: {trained}")
        if module['is_trained']:
            print(f"  Precisión: {module['accuracy']:.2%}")
        print(f"  Configuración: {len(module['config'])} parámetros")


def entrenar_modulos(directorio_datos, modules=None, language_manager=None):
    """Entrena módulos específicos."""
    module_manager = ModuleManager()
    
    if modules is None:
        # Entrenar todos los módulos activos
        modules = module_manager.get_active_modules()
    else:
        # Activar módulos específicos
        module_manager.active_modules.clear()
        for module_id in modules:
            module_manager.activate_module(module_id)
    
    print(f"\n=== Entrenando Módulos ===")
    print(f"Módulos a entrenar: {modules}")
    
    # Preparar rutas de datos (usar el mismo directorio para todos los módulos)
    data_paths = {module_id: directorio_datos for module_id in modules}
    
    # Entrenar todos los módulos
    results = module_manager.train_all_modules(data_paths)
    
    print(f"\n=== Resultados del Entrenamiento ===")
    for module_id, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {module_id}: {'Éxito' if success else 'Falló'}")


def main():
    parser = argparse.ArgumentParser(
        description='C.A. Dupin - Herramienta de coincidencias visuales asistidas por humanos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Comparar dos imágenes
  python dupin.py comparar imagen1.jpg imagen2.jpg --umbral 0.8

  # Entrenar modelo con datos locales
  python dupin.py entrenar ./datos_entrenamiento --epochs 20

  # Modo interactivo con feedback humano
  python dupin.py interactivo ./imagenes_revisar

  # Seleccionar regiones de interés en una imagen
  python dupin.py roi --imagen foto.jpg

  # Cámara en vivo para análisis en tiempo real
  python dupin.py camara

  # Análisis visual con todos los módulos
  python dupin.py visual --imagen foto.jpg

  # Ajustar modelo con feedback humano
  python dupin.py ajustar --modelo modelo.pth --feedback feedback.json

  # Listar módulos disponibles
  python dupin.py modulos

  # Entrenar módulos específicos
  python dupin.py entrenar-modulos ./datos --modules faces animals

  # Cambiar idioma de la interfaz
  python dupin.py comparar img1.jpg img2.jpg --idioma en
        """
    )
    
    # Argumento global para idioma
    parser.add_argument('--idioma', '--language', choices=['es', 'en', 'fr'],
                       default='es', help='Idioma de la interfaz')
    
    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponibles')
    
    # Comando: comparar
    comparar_parser = subparsers.add_parser('comparar', help='Comparar dos imágenes')
    comparar_parser.add_argument('imagen1', help='Ruta a la primera imagen')
    comparar_parser.add_argument('imagen2', help='Ruta a la segunda imagen')
    comparar_parser.add_argument('--umbral', type=float, default=0.85,
                                help='Umbral de similitud (0.0-1.0)')
    
    # Comando: entrenar
    entrenar_parser = subparsers.add_parser('entrenar', help='Entrenar modelo')
    entrenar_parser.add_argument('directorio', help='Directorio con imágenes de entrenamiento')
    entrenar_parser.add_argument('--epochs', type=int, default=10,
                                help='Número de épocas de entrenamiento')
    entrenar_parser.add_argument('--output', default='modelo.pth',
                                help='Ruta para guardar el modelo')
    
    # Comando: ajustar
    ajustar_parser = subparsers.add_parser('ajustar', help='Ajustar modelo con feedback humano')
    ajustar_parser.add_argument('--modelo', required=True,
                               help='Ruta al modelo entrenado')
    ajustar_parser.add_argument('--feedback', required=True,
                               help='Ruta al archivo JSON con feedback')
    ajustar_parser.add_argument('--output', default='modelo_ajustado.pth',
                               help='Ruta para guardar el modelo ajustado')
    
    # Comando: interactivo
    interactivo_parser = subparsers.add_parser('interactivo',
                                               help='Modo interactivo con retroalimentación humana')
    interactivo_parser.add_argument('directorio', help='Directorio con imágenes a revisar')
    
    # Comando: ROI
    roi_parser = subparsers.add_parser('roi', help='Selección de regiones de interés')
    roi_parser.add_argument('--imagen', required=True, help='Ruta a la imagen para ROI')
    
    # Comando: cámara
    camara_parser = subparsers.add_parser('camara', help='Cámara en vivo')
    
    # Comando: visual
    visual_parser = subparsers.add_parser('visual', help='Análisis visual con módulos')
    visual_parser.add_argument('--imagen', required=True, help='Ruta a la imagen a analizar')
    
    # Comando: módulos
    modulos_parser = subparsers.add_parser('modulos', help='Listar módulos disponibles')
    
    # Comando: entrenar-módulos
    entrenar_modulos_parser = subparsers.add_parser('entrenar-modulos',
                                                    help='Entrenar módulos específicos')
    entrenar_modulos_parser.add_argument('directorio', help='Directorio con datos de entrenamiento')
    entrenar_modulos_parser.add_argument('--modules', nargs='+',
                                        help='IDs de módulos a entrenar (faces, stars, etc.)')

    # Comando: definir-patron
    definir_patron_parser = subparsers.add_parser('definir-patron',
                                               help='Definir un nuevo patrón visual')
    definir_patron_parser.add_argument('nombre', help='Nombre del patrón')
    definir_patron_parser.add_argument('--descripcion', default='', help='Descripción del patrón')
    definir_patron_parser.add_argument('--imagen', help='Imagen de ejemplo del patrón')
    definir_patron_parser.add_argument('--roi', nargs=4, type=int,
                                   metavar=('X', 'Y', 'W', 'H'),
                                   help='Región de interés (x y w h)')

    # Comando: entrenar-patrones
    entrenar_patrones_parser = subparsers.add_parser('entrenar-patrones',
                                                 help='Entrenar modelo con patrones definidos')
    entrenar_patrones_parser.add_argument('--epochs', type=int, default=10,
                                     help='Número de épocas')

    # Comando: reconocer-patron
    reconocer_patron_parser = subparsers.add_parser('reconocer-patron',
                                                  help='Reconocer patrones en una imagen')
    reconocer_patron_parser.add_argument('imagen', help='Ruta a la imagen')
    reconocer_patron_parser.add_argument('--roi', nargs=4, type=int,
                                      metavar=('X', 'Y', 'W', 'H'),
                                      help='Región de interés (x y w h)')
    reconocer_patron_parser.add_argument('--umbral', type=float, default=0.5,
                                      help='Umbral de confianza (0.0-1.0)')

    # Comando: listar-patrones
    listar_patrones_parser = subparsers.add_parser('listar-patrones',
                                               help='Listar patrones definidos')

    # Comando: comparar-prob
    comparar_prob_parser = subparsers.add_parser('comparar-prob',
                                               help='Comparar imágenes con probabilidades detalladas')
    comparar_prob_parser.add_argument('imagen1', help='Ruta a la primera imagen')
    comparar_prob_parser.add_argument('imagen2', help='Ruta a la segunda imagen')
    comparar_prob_parser.add_argument('--roi1', nargs=4, type=int,
                                    metavar=('X', 'Y', 'W', 'H'),
                                    help='ROI de imagen 1 (x y w h)')
    comparar_prob_parser.add_argument('--roi2', nargs=4, type=int,
                                    metavar=('X', 'Y', 'W', 'H'),
                                    help='ROI de imagen 2 (x y w h)')
    comparar_prob_parser.add_argument('--metodo', default='orb',
                                    choices=['orb', 'sift', 'histogram', 'ssim'],
                                    help='Método de comparación')

    # Comando: aprobar
    aprobar_parser = subparsers.add_parser('aprobar',
                                         help='Aprobar un patrón detectado')
    aprobar_parser.add_argument('imagen', help='Ruta a la imagen')
    aprobar_parser.add_argument('--roi', nargs=4, type=int,
                             metavar=('X', 'Y', 'W', 'H'),
                             help='Región de interés (x y w h)')
    aprobar_parser.add_argument('--tipo', default='general',
                             help='Tipo de patrón')

    # Comando: corregir
    corregir_parser = subparsers.add_parser('corregir',
                                          help='Corregir un patrón detectado')
    corregir_parser.add_argument('imagen', help='Ruta a la imagen')
    corregir_parser.add_argument('correccion', help='Texto de corrección')
    corregir_parser.add_argument('--roi', nargs=4, type=int,
                               metavar=('X', 'Y', 'W', 'H'),
                               help='Región de interés (x y w h)')
    corregir_parser.add_argument('--tipo', default='general',
                               help='Tipo de patrón')

    args = parser.parse_args()
    
    # Inicializar gestor de idiomas
    language_manager = LanguageManager(args.idioma)
    
    if args.comando == 'comparar':
        comparar_imagenes(args.imagen1, args.imagen2, args.umbral, language_manager)
    elif args.comando == 'entrenar':
        entrenar_modelo(args.directorio, args.epochs, args.output, language_manager)
    elif args.comando == 'ajustar':
        ajustar_modelo(args.modelo, args.feedback, args.output, language_manager)
    elif args.comando == 'interactivo':
        modo_interactivo(args.directorio, language_manager)
    elif args.comando == 'roi':
        modo_roi(imagen_path=args.imagen, language_manager=language_manager)
    elif args.comando == 'camara':
        modo_camara(language_manager)
    elif args.comando == 'visual':
        modo_visual(args.imagen, language_manager)
    elif args.comando == 'modulos':
        listar_modulos(language_manager)
    elif args.comando == 'entrenar-modulos':
        entrenar_modulos(args.directorio, args.modules, language_manager)
    elif args.comando == 'definir-patron':
        definir_patron(
            nombre=args.nombre,
            descripcion=args.descripcion,
            imagen_path=args.imagen,
            roi=tuple(args.roi) if args.roi else None,
            language_manager=language_manager
        )
    elif args.comando == 'entrenar-patrones':
        entrenar_patrones(epochs=args.epochs, language_manager=language_manager)
    elif args.comando == 'reconocer-patron':
        reconocer_patron(
            imagen_path=args.imagen,
            roi=tuple(args.roi) if args.roi else None,
            threshold=args.umbral,
            language_manager=language_manager
        )
    elif args.comando == 'listar-patrones':
        listar_patrones(language_manager)
    elif args.comando == 'comparar-prob':
        comparar_con_probabilidades(
            imagen1_path=args.imagen1,
            imagen2_path=args.imagen2,
            roi1=tuple(args.roi1) if args.roi1 else None,
            roi2=tuple(args.roi2) if args.roi2 else None,
            metodo=args.metodo,
            language_manager=language_manager
        )
    elif args.comando == 'aprobar':
        aprobar_patron(
            imagen_path=args.imagen,
            roi=tuple(args.roi) if args.roi else None,
            pattern_type=args.tipo,
            language_manager=language_manager
        )
    elif args.comando == 'corregir':
        corregir_patron(
            imagen_path=args.imagen,
            correction=args.correccion,
            roi=tuple(args.roi) if args.roi else None,
            pattern_type=args.tipo,
            language_manager=language_manager
        )
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())