#!/usr/bin/env python3
"""
C.A.Dupin - Herramienta de coincidencias visuales asistidas por humanos

Este programa permite entrenar modelos, comparar imágenes y corregir resultados
en tiempo real con soporte para ROI, cámara en vivo y módulos multiidioma.

Características principales:
- 📷 Soporte para imágenes y cámara en vivo
- 🖼️ Marcado de regiones de interés (ROI)
- 🔁 Corrección en tiempo real (aprobar / rechazar / corregir)
- 🧠 Entrenamiento mejorado con técnicas avanzadas sin límite de ejemplos
- 📊 Visualización clara de lo que el modelo está identificando
- 🌐 Interfaz disponible en múltiples idiomas
- 🧩 Módulos de reconocimiento preconfigurados y entrenables

Mejoras de IA implementadas (SIN APIs externas):
- 🎨 Data Augmentation (rotación, flip, jitter, blur, perspectiva)
- 📈 Learning Rate Scheduling (CosineAnnealingWarmRestarts)
- 🛑 Early Stopping con restauración de mejores pesos
- 🔀 Test Time Augmentation (TTA) para inferencia robusta
- 🎯 Focal Loss para clases desbalanceadas
- ✨ Label Smoothing para mejor generalización
- 🏗️ Residual Blocks en la arquitectura de la red
- 📏 Gradient Clipping para estabilidad
- ⚖️ Batch Normalization en todas las capas
- 📊 Métricas detalladas (Precision, Recall, F1, Matriz de confusión)
"""

import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

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
from core.pattern_learner_v2 import ImprovedPatternLearnerV2
from core.video_pattern_recognition import VideoStreamPatternRecognizer, RealTimePatternAnalyzer


def configurar_idioma(language_manager, idioma):
    """Configura el idioma de la interfaz."""
    if idioma and language_manager.set_language(idioma):
        return True
    return False


def comparar_imagenes(imagen1_path, imagen2_path, umbral=0.85, metodo='orb', 
                      modelo_path='modelo.pth', language_manager=None):
    """Compara dos imágenes y muestra la similitud."""
    matcher = ImageMatcher(metodo=metodo, model_path=modelo_path)
    
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
    """Modo de cámara en vivo con análisis en tiempo real."""
    camera_manager = CameraManager()
    module_manager = ModuleManager()
    visual_interface = VisualInterface()
    
    # Activar todos los módulos por defecto para la cámara
    for module_info in module_manager.get_available_modules():
        module_manager.activate_module(module_info['module_id'])
    
    # Obtener textos traducidos
    texts = {}
    if language_manager:
        texts = {
            'titulo': language_manager.get_text('title', 'camera'),
            'iniciando': language_manager.get_text('initializing', 'camera'),
            'camaras_disponibles': language_manager.get_text('available_cameras', 'camera')
        }
    
    print(f"\n=== {texts.get('titulo', 'Modo Cámara en Vivo - Análisis Multimodular')} ===")
    
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
        print(f"\n{texts.get('iniciando', 'Iniciando captura y análisis...')}")
        
        def procesar_frame(frame):
            """Procesar cada frame capturado con los módulos activos."""
            # Limpiar detecciones previas
            visual_interface.clear_detections()
            visual_interface.current_image = frame
            
            # Realizar análisis con módulos (usamos umbral bajo para cámara)
            predictions = module_manager.predict(frame, active_only=True)
            
            # Añadir detecciones
            for module_id, module_predictions in predictions.items():
                for pred in module_predictions:
                    visual_interface.add_detection(
                        class_name=f"{module_id}:{pred['class']}",
                        confidence=pred['confidence'],
                        bounding_box=pred['bbox']
                    )
            
            # Visualizar resultados en el frame
            display_frame = visual_interface.visualize_detections()
            
            # Añadir info de cámara
            stats = camera_manager.get_stats()
            cv2.putText(display_frame, f"FPS: {stats['frames_per_second']:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return display_frame
        
        camera_manager.start_capture(callback=procesar_frame)
        
        print("\nControles:")
        print("- ESC: Salir")
        print("- ESPACIO: Tomar foto")
        print("- R: Iniciar/detener grabación")
        print("\nPresiona CTRL+C en esta terminal para detener el proceso si la ventana no responde.")
        
        try:
            # Mantener el programa en ejecución mientras la cámara captura
            while camera_manager.is_capturing:
                cv2.waitKey(100)
        except KeyboardInterrupt:
            pass
        finally:
            camera_manager.stop_capture()
            cv2.destroyAllWindows()
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


def modo_analisis(imagen_path, threshold=0.5, language_manager=None):
    """Realiza un análisis profundo y explicativo de una imagen."""
    visual_interface = VisualInterface()
    module_manager = ModuleManager()
    pattern_learner = PatternLearner()
    
    print(f"\n=== C.A. Dupin - Análisis Integral de Imagen ===")
    print(f"Imagen: {imagen_path}")
    print(f"Umbral de confianza: {threshold:.0%}")
    
    if not visual_interface.load_image(imagen_path):
        return
    
    print(f"\n1. 🧩 Consultando módulos de reconocimiento base...")
    # Activar todos los módulos
    for module_info in module_manager.get_available_modules():
        module_manager.activate_module(module_info['module_id'])
    
    module_predictions = module_manager.predict(imagen_path, active_only=True)
    
    # Añadir detecciones de módulos
    for module_id, preds in module_predictions.items():
        for pred in preds:
            visual_interface.add_detection(
                class_name=f"{module_id}:{pred['class']}",
                confidence=pred['confidence'],
                bounding_box=pred['bbox']
            )
    
    print(f"\n2. 🧠 Buscando patrones definidos por el usuario...")
    pattern_detections = pattern_learner.recognize_pattern(
        image_path=imagen_path,
        threshold=threshold,
        include_reasoning=True
    )
    
    # Añadir detecciones de patrones
    for det in pattern_detections:
        visual_interface.add_detection(
            class_name=f"Patrón:{det['pattern_name']}",
            confidence=det['probability'],
            bounding_box=det['bbox']
        )
    
    summary = visual_interface.get_detection_summary()
    print(f"\n📊 Resumen del Análisis:")
    print(f"  Total detecciones: {summary['total']}")
    if 'classes' in summary:
        for cls, count in summary['classes'].items():
            print(f"  - {cls}: {count}")
            
    print(f"\n3. 🎨 Generando representaciones visuales del razonamiento...")
    
    # Mostrar detecciones generales
    print("\nMostrando detecciones encontradas...")
    visual_interface.show_detections()
    
    # Mostrar razonamiento para cada patrón encontrado
    for det in pattern_detections:
        if 'heatmap' in det:
            print(f"\nExponiendo razonamiento interno para: {det['pattern_name']}")
            print(f"Las áreas más brillantes influyeron más en la identificación.")
            
            # Superponer heatmap
            img = cv2.imread(imagen_path)
            img = cv2.resize(img, (500, 500))
            heatmap = cv2.resize(det['heatmap'], (500, 500))
            heatmap = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            combined = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
            
            cv2.putText(combined, f"Razonamiento: {det['pattern_name']}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Zonas de influencia del modelo", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f"Razonamiento Interno - {det['pattern_name']}", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    print("\n✓ Análisis completo finalizado.")
    return summary


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


def entrenar_patrones(epochs=30, batch_size=16, val_split=0.2, 
                      learning_rate=0.001, use_focal_loss=False, 
                      label_smoothing=0.0, early_stopping_patience=10,
                      dropout_rate=0.4, language_manager=None):
    """Entrena el modelo con los patrones definidos por el usuario usando técnicas avanzadas."""
    pattern_learner = PatternLearner()
    
    print(f"\n=== Entrenando Patrones Definidos por Usuario (Modo Mejorado) ===")
    
    patterns = pattern_learner.list_patterns()
    if not patterns:
        print("No hay patrones definidos. Usa 'definir-patron' primero.")
        return None
    
    print(f"Patrones a entrenar: {len(patterns)}")
    for pattern in patterns:
        print(f"  - {pattern['name']} (muestras: {pattern['samples']})")
    
    print(f"\n🔧 Configuración del entrenamiento:")
    print(f"  - Épocas: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Validación split: {val_split*100:.0f}%")
    print(f"  - Focal Loss: {use_focal_loss}")
    print(f"  - Label Smoothing: {label_smoothing}")
    print(f"  - Early Stopping: {early_stopping_patience} épocas (0 = desactivado)")
    print(f"  - Dropout rate: {dropout_rate}")
    
    history = pattern_learner.train_patterns(
        epochs=epochs,
        batch_size=batch_size,
        val_split=val_split,
        learning_rate=learning_rate,
        use_focal_loss=use_focal_loss,
        label_smoothing=label_smoothing,
        early_stopping_patience=early_stopping_patience,
        dropout_rate=dropout_rate
    )
    
    if history:
        print("\n✓ Entrenamiento de patrones completado")
        return pattern_learner
    else:
        print("\n✗ Error en el entrenamiento de patrones")
        return None


def reconocer_patron(imagen_path, roi=None, threshold=0.5, 
                    mostrar_razonamiento=False, use_tta=False, 
                    tta_transforms=5, language_manager=None):
    """Reconoce patrones en una imagen o directorio con opción de mostrar razonamiento visual y TTA."""
    pattern_learner = PatternLearner()
    path = Path(imagen_path)
    
    if path.is_dir():
        extensiones = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        imagenes = [f for f in path.iterdir() if f.suffix.lower() in extensiones]
        
        print(f"\n=== Reconocimiento de Patrones en Directorio ===")
        print(f"Directorio: {imagen_path}")
        print(f"Imágenes encontradas: {len(imagenes)}")
        print(f"Umbral de confianza: {threshold:.0%}")
        
        all_detections = {}
        for img_path in imagenes:
            print(f"\nAnalizando {img_path.name}...")
            detections = pattern_learner.recognize_pattern(
                image_path=str(img_path),
                roi=roi,
                threshold=threshold,
                include_reasoning=False,  # No mostramos razonamiento en lote por defecto
                use_tta=use_tta
            )
            if detections:
                all_detections[str(img_path)] = detections
                print(f"  ✓ Encontrados {len(detections)} patrones")
                for det in detections:
                    print(f"    - {det['pattern_name']} ({det['probability']:.2%})")
                    if 'consistency' in det:
                        print(f"      Consistencia TTA: {det['consistency']:.2%}")
            else:
                print("  No se encontraron patrones")
        
        return all_detections

    print(f"\n=== Reconocimiento de Patrones ===")
    print(f"Imagen: {imagen_path}")
    if roi:
        print(f"ROI: {roi}")
    print(f"Umbral de confianza: {threshold:.0%}")
    if use_tta:
        print(f"Test Time Augmentation: Activo ({tta_transforms} transformaciones)")
    
    detections = pattern_learner.recognize_pattern(
        image_path=imagen_path,
        roi=roi,
        threshold=threshold,
        include_reasoning=mostrar_razonamiento,
        use_tta=use_tta
    )
    
    if detections:
        print(f"\n✓ Encontrados {len(detections)} patrones:")
        for detection in detections:
            print(f"\n  Patrón: {detection['pattern_name']}")
            print(f"  Probabilidad: {detection['probability']:.2%}")
            
            # Mostrar información TTA si está disponible
            if 'consistency' in detection:
                print(f"  Consistencia TTA: {detection['consistency']:.2%}")
                print(f"  Intervalo de confianza 95%: "
                      f"[{detection['confidence_interval_lower']:.2%}, "
                      f"{detection['confidence_interval_upper']:.2%}]")
            
            if mostrar_razonamiento and 'heatmap' in detection:
                print(f"  🎨 Mostrando mapa de calor de razonamiento para {detection['pattern_name']}...")
                
                # Cargar imagen original para superponer
                img = cv2.imread(imagen_path)
                if roi:
                    x, y, w, h = roi
                    img = img[y:y+h, x:x+w]
                
                if img is not None:
                    img = cv2.resize(img, (400, 400))
                    heatmap = cv2.resize(detection['heatmap'], (400, 400))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    
                    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
                    
                    cv2.imshow(f"Razonamiento: {detection['pattern_name']}", superimposed_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
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
                               metodo='orb', modelo_path='modelo.pth', 
                               mostrar_razonamiento=False, language_manager=None):
    """Compara dos imágenes mostrando probabilidades detalladas y razonamiento visual."""
    matcher = ImageMatcher(metodo=metodo, model_path=modelo_path)
    visual_interface = VisualInterface()
    
    print(f"\n=== Comparación con Probabilidades Detalladas ===")
    print(f"Imagen 1: {imagen1_path}")
    if roi1:
        print(f"  ROI 1: {roi1}")
    print(f"Imagen 2: {imagen2_path}")
    if roi2:
        print(f"  ROI 2: {roi2}")
    print(f"Método: {metodo.upper()}")
    
    # Si queremos mostrar razonamiento visual y usamos métodos de características
    raw_data = None
    if mostrar_razonamiento and metodo in ['orb', 'sift']:
        img1 = matcher.cargar_imagen(imagen1_path)
        img2 = matcher.cargar_imagen(imagen2_path)
        
        if roi1:
            x, y, w, h = roi1
            img1 = img1[y:y+h, x:x+w]
        if roi2:
            x, y, w, h = roi2
            img2 = img2[y:y+h, x:x+w]
            
        similarity, details, raw_data = matcher._compare_features_with_details(img1, img2, return_raw=True)
        result = {
            'similarity': similarity,
            'method': metodo,
            'details': details,
            'probability': matcher._calculate_probability(similarity)
        }
    else:
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
            
    # Mostrar razonamiento visual si se solicita
    if mostrar_razonamiento and raw_data:
        print("\n🎨 Generando visualización de razonamiento...")
        match_img = visual_interface.visualize_matches(
            img1, img2, 
            raw_data['kp1'], raw_data['kp2'], 
            raw_data['good_matches']
        )
        cv2.imshow("C.A. Dupin - Razonamiento de Coincidencia", match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


def aprobar_patron(imagen_path, roi=None, pattern_type='general', language_manager=None):
    """Aprueba un patrón detectado y lo añade como muestra para aprendizaje."""
    directorio = Path(imagen_path).parent
    feedback_loop = HumanFeedbackLoop(str(directorio))
    pattern_learner = PatternLearner()
    
    # Registrar en historial de feedback
    feedback_loop.approve_pattern(
        image_path=imagen_path,
        roi=roi,
        pattern_type=pattern_type
    )
    
    # Si es un patrón definido por el usuario, añadir como muestra
    # Buscamos si el pattern_type coincide con algún nombre de patrón
    patterns = pattern_learner.list_patterns()
    pattern_id = None
    for p in patterns:
        if p['name'] == pattern_type or p['id'] == pattern_type:
            pattern_id = p['id']
            break
    
    if pattern_id:
        pattern_learner.add_pattern_sample(pattern_id, imagen_path, roi)
        pattern_learner.record_feedback(pattern_id, is_correct=True)
        print(f"✓ Patrón '{pattern_type}' incorporado al conocimiento del sistema")
    else:
        print(f"○ Patrón '{pattern_type}' registrado en feedback (no vinculado a patrón de usuario)")
    
    return True


def corregir_patron(imagen_path, correction, roi=None, pattern_type='general', language_manager=None):
    """Corrige un patrón detectado y lo añade como muestra del patrón correcto."""
    directorio = Path(imagen_path).parent
    feedback_loop = HumanFeedbackLoop(str(directorio))
    pattern_learner = PatternLearner()
    
    # Registrar en historial de feedback
    feedback_loop.correct_pattern(
        image_path=imagen_path,
        roi=roi,
        correction=correction,
        pattern_type=pattern_type
    )
    
    # Si la corrección coincide con un patrón conocido, añadir como muestra
    patterns = pattern_learner.list_patterns()
    pattern_id = None
    for p in patterns:
        if p['name'] == correction or p['id'] == correction:
            pattern_id = p['id']
            break
    
    if pattern_id:
        pattern_learner.add_pattern_sample(pattern_id, imagen_path, roi)
        # También registramos que el patrón original fue incorrecto si era un patrón de usuario
        orig_pattern_id = None
        for p in patterns:
            if p['name'] == pattern_type or p['id'] == pattern_type:
                orig_pattern_id = p['id']
                break
        if orig_pattern_id:
            pattern_learner.record_feedback(orig_pattern_id, is_correct=False)
            
        print(f"✓ Corrección '{correction}' incorporada como nuevo ejemplo")
    else:
        print(f"○ Corrección registrada: {correction}")
    
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


def reconocer_video_patrones(video_path=None, camera_index=None, roi=None, 
                           threshold=0.5, frame_skip=1, use_tta=False,
                           save_output=None, language_manager=None):
    """
    Reconoce patrones en video (archivo o cámara en vivo).
    
    Args:
        video_path: Ruta al archivo de video (opcional)
        camera_index: Índice de cámara para video en vivo (opcional)
        roi: Región de interés (x, y, w, h)
        threshold: Umbral de confianza mínimo
        frame_skip: Procesar cada N frames (optimización)
        use_tta: Usar Test Time Augmentation (lento)
        save_output: Ruta para guardar video con detecciones
        language_manager: Gestor de idiomas
    """
    print("\n" + "="*60)
    print("🎥 RECONOCIMIENTO DE PATRONES EN VIDEO")
    print("="*60)
    
    try:
        # Crear reconocedor de video
        video_recognizer = VideoStreamPatternRecognizer()
        
        # Configurar parámetros de optimización
        video_recognizer.set_optimization_params(
            frame_skip=frame_skip,
            confidence_threshold=threshold
        )
        
        # Procesar video
        if video_path:
            print(f"📁 Video archivo: {video_path}")
            detection_history = video_recognizer.recognize_from_video_file(
                video_path=video_path,
                output_path=save_output,
                roi=roi,
                threshold=threshold,
                use_tta=use_tta,
                display=True,
                save_detections=True
            )
            
        elif camera_index is not None:
            print(f"📹 Cámara en vivo: Índice {camera_index}")
            print("\n⚡ Iniciando reconocimiento en tiempo real...")
            detection_history = video_recognizer.recognize_from_camera(
                camera_index=camera_index,
                roi=roi,
                threshold=threshold,
                use_tta=use_tta,
                save_video=save_output is not None,
                output_path=save_output
            )
        
        # Mostrar estadísticas finales
        stats = video_recognizer.get_performance_stats()
        if stats:
            print(f"\n📊 ESTADÍSTICAS DE RENDIMIENTO:")
            print(f"  - Frames procesados: {stats['total_frames']}")
            print(f"  - FPS promedio: {stats['frames_per_second']:.1f}")
            print(f"  - Tiempo total: {stats['elapsed_time']:.1f}s")
            print(f"  - Frames con detecciones: {stats['detection_history_frames']}")
        
        if detection_history:
            print(f"\n✓ Reconocimiento completado: {len(detection_history)} frames con detecciones")
            print("\n📄 Se ha guardado el historial de detecciones en formato JSON")
        else:
            print("\n⚠️ No se detectaron patrones en el video")
        
        return detection_history
        
    except Exception as e:
        print(f"\n❌ Error en reconocimiento de video: {e}")
        import traceback
        traceback.print_exc()
        return None


def generar_reporte_temporal(detection_file=None, output_path=None):
    """
    Genera un reporte temporal de detecciones desde archivo JSON.
    
    Args:
        detection_file: Ruta al archivo de detecciones
        output_path: Ruta para guardar reporte
    """
    if not detection_file or not Path(detection_file).exists():
        print(f"⚠️ Archivo de detecciones no encontrado: {detection_file}")
        return
    
    # Cargar detecciones
    try:
        with open(detection_file, 'r', encoding='utf-8') as f:
            detection_data = json.load(f)
    except Exception as e:
        print(f"❌ Error cargando detecciones: {e}")
        return
    
    print(f"\n📊 Generando reporte temporal...")
    
    # Analizar frecuencia
    frequency = RealTimePatternAnalyzer.analyze_detection_frequency(
        detection_data.get('detection_history', [])
    )
    
    # Generar reporte
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reporte_temporal_{timestamp}.json"
    
    RealTimePatternAnalyzer.generate_temporal_report(
        detection_data.get('detection_history', []),
        output_path
    )
    
    print(f"✓ Reporte guardado: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════
# FUNCIONES V2 - Sistema mejorado de reconocimiento de patrones
# ═══════════════════════════════════════════════════════════════

def crear_patron_v2(nombre, descripcion="", language_manager=None):
    """
    Crea un nuevo patrón con sistema V2.
    Crea automáticamente la carpeta en fotos_entrenamiento/por_patron/.
    """
    pattern_learner = ImprovedPatternLearnerV2()
    
    print(f"\n=== C.A. Dupin V2 - Crear Patrón ===")
    print(f"🎯 Creando patrón: {nombre}")
    if descripcion:
        print(f"📝 Descripción: {descripcion}")
    
    pattern_id = pattern_learner.define_pattern_from_folder(
        name=nombre,
        description=descripcion
    )
    
    pattern_folder = pattern_learner.patterns_training_dir / nombre
    print(f"\n📁 Siguientes pasos:")
    print(f"   1. Coloca fotos de '{nombre}' en:")
    print(f"      {pattern_folder}")
    print(f"   2. Ejecuta: python dupin.py importar-entrenamiento")
    print(f"   3. Entrena: python dupin.py entrenar-patrones-v2")
    
    return pattern_id


def importar_entrenamiento(language_manager=None):
    """
    Importa todas las imágenes de las carpetas de entrenamiento.
    """
    pattern_learner = ImprovedPatternLearnerV2()
    
    print(f"\n=== C.A. Dupin V2 - Importar Entrenamiento ===")
    print(f"📂 Buscando imágenes en fotos_entrenamiento/por_patron/...")
    
    results = pattern_learner.import_all_patterns_from_folders()
    
    if not results:
        print(f"\n⚠️  No se encontraron imágenes.")
        print(f"   Crea patrones con: python dupin.py crear-patron-v2 <nombre>")
        print(f"   Y coloca fotos en las carpetas creadas.")
        return {}
    
    total_images = sum(results.values())
    print(f"\n✅ Importación completada")
    print(f"   Total de imágenes importadas: {total_images}")
    
    return results


def entrenar_patrones_v2(epochs=30, batch_size=16, val_split=0.2,
                        learning_rate=0.001, max_lr=0.01,
                        use_focal_loss=False, label_smoothing=0.0,
                        early_stopping_patience=10, warmup_epochs=3,
                        dropout_rate=0.4, use_mixup=True,
                        use_randaugment=True, gradient_accumulation=1,
                        use_amp=None, use_compile=None,
                        use_gradient_checkpointing=False,
                        num_workers=None, channels_last=True,
                        language_manager=None):
    """
    Entrena el modelo V2 con todas las técnicas avanzadas y optimizaciones de rendimiento.
    """
    pattern_learner = ImprovedPatternLearnerV2()

    print(f"\n=== C.A. Dupin V2 - Entrenamiento Mejorado ===")

    patterns = pattern_learner.list_patterns()
    if not patterns:
        print("❌ No hay patrones definidos.")
        print("   Crea patrones con: python dupin.py crear-patron-v2 <nombre>")
        return None

    print(f"📊 Patrones encontrados: {len(patterns)}")
    for p in patterns:
        print(f"   • {p['name']}: {p['samples']} muestras (carpeta: {p['folder_images']} imágenes)")

    history = pattern_learner.train_patterns_v2(
        epochs=epochs,
        batch_size=batch_size,
        val_split=val_split,
        learning_rate=learning_rate,
        max_lr=max_lr,
        use_focal_loss=use_focal_loss,
        label_smoothing=label_smoothing,
        early_stopping_patience=early_stopping_patience,
        warmup_epochs=warmup_epochs,
        dropout_rate=dropout_rate,
        use_mixup=use_mixup,
        use_randaugment=use_randaugment,
        gradient_accumulation_steps=gradient_accumulation,
        save_checkpoints=True,
        use_amp=use_amp,
        use_compile=use_compile,
        use_gradient_checkpointing=use_gradient_checkpointing,
        num_workers=num_workers,
        channels_last=channels_last
    )

    if history:
        print(f"\n✅ Entrenamiento completado exitosamente")
        print(f"💾 Modelo guardado en: user_patterns/patterns_model_v2.pth")
        return pattern_learner
    else:
        print(f"\n❌ Error en el entrenamiento")
        return None


def identificar_carpetas_v2(threshold=0.5, output_file=None, language_manager=None):
    """
    Identifica patrones en todas las imágenes de fotos_identificar/.
    """
    pattern_learner = ImprovedPatternLearnerV2()
    
    print(f"\n=== C.A. Dupin V2 - Identificar desde Carpeta ===")
    print(f"📁 Buscando imágenes en fotos_identificar/...")
    
    # Verificar que el modelo existe
    model_info = pattern_learner.get_model_info()
    if not model_info['model_exists']:
        print(f"\n❌ No hay modelo entrenado.")
        print(f"   Primero entrena con: python dupin.py entrenar-patrones-v2")
        return {}
    
    print(f"📊 Modelo: {model_info['model_size_mb']:.2f} MB")
    print(f"🎯 Patrones entrenados: {model_info['num_patterns']}")
    print(f"   {', '.join(model_info['pattern_names'])}")
    
    results = pattern_learner.identify_from_folder(output_file=output_file)
    
    return results


def listar_patrones_v2(language_manager=None):
    """
    Lista todos los patrones con información detallada V2.
    """
    pattern_learner = ImprovedPatternLearnerV2()
    
    patterns = pattern_learner.list_patterns()
    model_info = pattern_learner.get_model_info()
    
    print(f"\n=== C.A. Dupin V2 - Patrones Definidos ===")
    print(f"📊 Total de patrones: {len(patterns)}")
    
    if model_info['model_exists']:
        print(f"🤖 Modelo entrenado: ✓ ({model_info['model_size_mb']:.2f} MB)")
        print(f"📅 Última modificación: {model_info['model_modified']}")
        print(f"📚 Entrenamientos previos: {model_info['training_history_count']}")
    else:
        print(f"🤖 Modelo entrenado: ✗ (entrena con: python dupin.py entrenar-patrones-v2)")
    
    if patterns:
        print(f"\n📋 Lista de patrones:\n")
        for p in patterns:
            print(f"  🎯 {p['name']}")
            print(f"     ID: {p['id']}")
            if p['description']:
                print(f"     Descripción: {p['description']}")
            print(f"     Muestras importadas: {p['samples']}")
            print(f"     Imágenes en carpeta: {p['folder_images']}")
            if p['approved'] + p['corrected'] > 0:
                print(f"     Feedback: {p['approved']}✓ / {p['corrected']}✗")
                print(f"     Precisión: {p['accuracy']:.2%}")
            print(f"     Creado: {p['created_at'][:10]}")
            print()
    else:
        print(f"\n⚠️  No hay patrones definidos.")
        print(f"   Crea un patrón con: python dupin.py crear-patron-v2 <nombre>")
    
    return patterns


def info_modelo_v2(language_manager=None):
    """
    Muestra información detallada sobre el modelo V2.
    """
    pattern_learner = ImprovedPatternLearnerV2()
    
    print(f"\n=== C.A. Dupin V2 - Información del Modelo ===")
    
    patterns = pattern_learner.list_patterns()
    model_info = pattern_learner.get_model_info()
    
    print(f"\n📁 Directorios:")
    print(f"   fotos_entrenamiento/       : Para fotos de entrenamiento")
    print(f"   fotos_entrenamiento/por_patron/ : Organizado por patrón")
    print(f"   fotos_identificar/        : Para fotos a identificar")
    print(f"   user_patterns/             : Datos y modelo del sistema")
    
    print(f"\n🤖 Estado del modelo:")
    if model_info['model_exists']:
        print(f"   ✅ Modelo entrenado disponible")
        print(f"   📏 Tamaño: {model_info['model_size_mb']:.2f} MB")
        print(f"   📅 Modificado: {model_info['model_modified']}")
        print(f"   📍 Ruta: {model_info['model_path']}")
    else:
        print(f"   ❌ No hay modelo entrenado")
        print(f"   💡 Entrena con: python dupin.py entrenar-patrones-v2")
    
    print(f"\n🎯 Patrones definidos: {len(patterns)}")
    if patterns:
        for p in patterns:
            folder_status = "✓" if p['folder_images'] > 0 else "○"
            print(f"   {folder_status} {p['name']}: {p['samples']} muestras, "
                  f"{p['folder_images']} imágenes en carpeta")
    
    if not patterns:
        print(f"   💡 Crea un patrón con: python dupin.py crear-patron-v2 <nombre>")
    
    print(f"\n📚 Historial de entrenamientos: {model_info['training_history_count']}")
    
    return model_info


def reconocer_patron_v2(imagen_path, threshold=0.5, multiscale=False, 
                       language_manager=None):
    """
    Reconoce patrones en una imagen usando el sistema V2.
    """
    pattern_learner = ImprovedPatternLearnerV2()
    
    print(f"\n=== C.A. Dupin V2 - Reconocimiento ===")
    print(f"📷 Imagen: {imagen_path}")
    print(f"🎯 Umbral: {threshold:.0%}")
    
    # Verificar modelo
    model_info = pattern_learner.get_model_info()
    if not model_info['model_exists']:
        print(f"\n❌ No hay modelo entrenado.")
        print(f"   Entrena con: python dupin.py entrenar-patrones-v2")
        return []
    
    if multiscale:
        print(f"🔍 Multi-scale inference: Activo")
        detections = pattern_learner.recognize_pattern_multiscale(
            image_path=imagen_path,
            scales=[96, 128, 160],
            threshold=threshold
        )
    else:
        detections = pattern_learner.recognize_pattern(
            image_path=imagen_path,
            threshold=threshold
        )
    
    if detections:
        print(f"\n✅ Patrones encontrados: {len(detections)}")
        for i, det in enumerate(detections, 1):
            print(f"\n  {i}. {det['pattern_name']}")
            print(f"     Probabilidad: {det['probability']:.2%}")
            print(f"     Confianza: ", end="")
            conf = det['probability']
            if conf >= 0.9:
                print("🔥 Muy alta")
            elif conf >= 0.75:
                print("✨ Alta")
            elif conf >= 0.6:
                print("👍 Media")
            else:
                print("⚠️ Baja")
    else:
        print(f"\n❌ No se encontraron patrones con el umbral actual.")
        print(f"   Intenta un umbral más bajo: --umbral 0.3")
    
    return detections


def flujo_completo_v2(nombre_patron, descripcion="", epochs=30, 
                      language_manager=None):
    """
    Flujo completo V2: crear patrón, importar imágenes y entrenar.
    """
    print(f"\n{'='*60}")
    print(f"  C.A. DUPIN V2 - FLUJO COMPLETO DE ENTRENAMIENTO")
    print(f"{'='*60}")
    
    # Paso 1: Crear patrón
    print(f"\n📌 PASO 1: Crear patrón")
    pattern_learner = ImprovedPatternLearnerV2()
    pattern_id = pattern_learner.define_pattern_from_folder(nombre_patron, descripcion)
    
    pattern_folder = pattern_learner.patterns_training_dir / nombre_patron
    
    print(f"\n💡 Ahora tienes 3 opciones:")
    print(f"\n   OPCIÓN A - Flujo interactivo:")
    print(f"   1. Coloca tus fotos en: {pattern_folder}")
    print(f"   2. Presiona ENTER cuando estés listo")
    
    input(f"\n   [ENTER para continuar] ")
    
    # Paso 2: Importar imágenes
    print(f"\n📌 PASO 2: Importar imágenes de entrenamiento")
    results = pattern_learner.import_all_patterns_from_folders()
    
    if nombre_patron not in results:
        print(f"\n⚠️  No se encontraron imágenes para '{nombre_patron}'")
        print(f"   Coloca imágenes en: {pattern_folder}")
        print(f"   Y ejecuta: python dupin.py importar-entrenamiento")
        return pattern_learner
    
    print(f"\n✓ Importadas {results[nombre_patron]} imágenes")
    
    # Paso 3: Entrenar
    print(f"\n📌 PASO 3: Entrenar modelo")
    history = pattern_learner.train_patterns_v2(
        epochs=epochs,
        batch_size=16,
        val_split=0.2,
        use_randaugment=True,
        use_mixup=True
    )
    
    if history:
        print(f"\n{'='*60}")
        print(f"  ✅ ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print(f"{'='*60}")
        print(f"\n📊 Resultados finales:")
        print(f"   - Mejor val_loss: {min(history['val_loss']):.4f}")
        print(f"   - Mejor val_acc: {max(history['val_acc']):.4f}")
        print(f"   - Épocas entrenadas: {len(history['train_loss'])}")
        
        print(f"\n🎯 Próximos pasos:")
        print(f"   1. Identificar: python dupin.py identificar-v2")
        print(f"   2. Reconocer una imagen: python dupin.py reconocer-v2 <imagen>")
        print(f"   3. Ver info: python dupin.py info-v2")
    
    return pattern_learner


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

  # Definir y entrenar patrones personalizados (modo mejorado)
  python dupin.py definir-patron "mi_logo" --imagen logo.jpg
  python dupin.py definir-patron "mi_logo" --imagen logo2.jpg
  python dupin.py entrenar-patrones --epochs 30 --batch-size 16 --val-split 0.2
  python dupin.py entrenar-patrones --epochs 50 --focal-loss --early-stopping 15

  # Reconocer patrones con TTA (Test Time Augmentation) para mejor precisión
  python dupin.py reconocer-patron imagen.jpg --umbral 0.7 --tta
  python dupin.py reconocer-patron imagen.jpg --umbral 0.6 --tta --tta-transforms 7

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
    comparar_parser.add_argument('--metodo', default='orb',
                                choices=['orb', 'sift', 'histogram', 'ssim', 'siamese'],
                                help='Método de comparación')
    comparar_parser.add_argument('--modelo', default='modelo.pth',
                                help='Ruta al modelo siamés (si se usa metodo=siamese)')
    
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
                                                  help='Entrenar modelo con patrones definidos (modo mejorado)')
    entrenar_patrones_parser.add_argument('--epochs', type=int, default=30,
                                      help='Número de épocas (default: 30)')
    entrenar_patrones_parser.add_argument('--batch-size', type=int, default=16,
                                      help='Tamaño del batch (default: 16)')
    entrenar_patrones_parser.add_argument('--val-split', type=float, default=0.2,
                                      help='Proporción de validación (default: 0.2)')
    entrenar_patrones_parser.add_argument('--learning-rate', type=float, default=0.001,
                                      help='Learning rate inicial (default: 0.001)')
    entrenar_patrones_parser.add_argument('--focal-loss', action='store_true',
                                      help='Usar Focal Loss para clases desbalanceadas')
    entrenar_patrones_parser.add_argument('--label-smoothing', type=float, default=0.0,
                                      help='Factor de label smoothing (default: 0.0)')
    entrenar_patrones_parser.add_argument('--early-stopping', type=int, default=10,
                                      help='Paciencia para early stopping (0 = desactivado, default: 10)')
    entrenar_patrones_parser.add_argument('--dropout', type=float, default=0.4,
                                      help='Tasa de dropout (default: 0.4)')

    # Comando: reconocer-patron
    reconocer_patron_parser = subparsers.add_parser('reconocer-patron',
                                                  help='Reconocer patrones en una imagen')
    reconocer_patron_parser.add_argument('imagen', help='Ruta a la imagen')
    reconocer_patron_parser.add_argument('--roi', nargs=4, type=int,
                                      metavar=('X', 'Y', 'W', 'H'),
                                      help='Región de interés (x y w h)')
    reconocer_patron_parser.add_argument('--umbral', type=float, default=0.5,
                                      help='Umbral de confianza (0.0-1.0)')
    reconocer_patron_parser.add_argument('--razonamiento', action='store_true',
                                      help='Mostrar razonamiento visual del reconocimiento')
    reconocer_patron_parser.add_argument('--tta', action='store_true',
                                      help='Usar Test Time Augmentation para mejor precisión')
    reconocer_patron_parser.add_argument('--tta-transforms', type=int, default=5,
                                      help='Número de transformaciones TTA (default: 5)')

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
                                    choices=['orb', 'sift', 'histogram', 'ssim', 'siamese'],
                                    help='Método de comparación')
    comparar_prob_parser.add_argument('--modelo', default='modelo.pth',
                                    help='Ruta al modelo siamés (si se usa metodo=siamese)')
    comparar_prob_parser.add_argument('--razonamiento', action='store_true',
                                    help='Mostrar razonamiento visual de la coincidencia')

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

    # Comando: análisis
    analisis_parser = subparsers.add_parser('analizar', help='Análisis integral (módulos + patrones)')
    analisis_parser.add_argument('imagen', help='Ruta a la imagen')
    analisis_parser.add_argument('--umbral', type=float, default=0.5,
                               help='Umbral de confianza')

    # Comando: video-patron (análisis de video)
    video_patron_parser = subparsers.add_parser('video-patron', 
                                               help='Reconocer patrones en video')
    video_patron_parser.add_argument('--video', help='Ruta al archivo de video')
    video_patron_parser.add_argument('--camara', type=int, default=None,
                                    help='Índice de cámara para video en vivo')
    video_patron_parser.add_argument('--roi', nargs=4, type=int,
                                    metavar=('X', 'Y', 'W', 'H'),
                                    help='Región de interés (x y w h)')
    video_patron_parser.add_argument('--umbral', type=float, default=0.5,
                                    help='Umbral de confianza (0.0-1.0)')
    video_patron_parser.add_argument('--frame-skip', type=int, default=1,
                                    help='Procesar cada N frames (optimización, default: 1)')
    video_patron_parser.add_argument('--tta', action='store_true',
                                    help='Usar Test Time Augmentation (más preciso pero lento)')
    video_patron_parser.add_argument('--guardar', help='Ruta para guardar video con detecciones')

    # Comando: camara-patron (reconocimiento en tiempo real desde cámara)
    camara_patron_parser = subparsers.add_parser('camara-patron',
                                               help='Reconocer patrones en tiempo real desde cámara')
    camara_patron_parser.add_argument('--camara', type=int, default=0,
                                     help='Índice de cámara (default: 0)')
    camara_patron_parser.add_argument('--roi', nargs=4, type=int,
                                     metavar=('X', 'Y', 'W', 'H'),
                                     help='Región de interés (x y w h)')
    camara_patron_parser.add_argument('--umbral', type=float, default=0.5,
                                     help='Umbral de confianza (0.0-1.0)')
    camara_patron_parser.add_argument('--optimizacion', type=int, default=1,
                                     help='Nivel de optimización: 1=todos, 2=cada 2 frames, 3=cada 3 frames')
    camara_patron_parser.add_argument('--guardar', help='Ruta para guardar video con detecciones')

    # Comando: reporte-temporal
    reporte_temporal_parser = subparsers.add_parser('reporte-temporal',
                                                   help='Generar reporte temporal de detecciones')
    reporte_temporal_parser.add_argument('detecciones', help='Ruta al archivo JSON de detecciones')
    reporte_temporal_parser.add_argument('--output', help='Ruta para guardar reporte (auto-generado si no se especifica)')

    # ═══════════════════════════════════════════════════════════════
    # COMANDOS V2 - Sistema mejorado de reconocimiento de patrones
    # ═══════════════════════════════════════════════════════════════
    
    # Comando: crear-patron-v2
    crear_patron_v2_parser = subparsers.add_parser('crear-patron-v2',
                                                  help='V2: Crear un nuevo patrón (crea carpeta automáticamente)')
    crear_patron_v2_parser.add_argument('nombre', help='Nombre del patrón')
    crear_patron_v2_parser.add_argument('--descripcion', default='', help='Descripción del patrón')
    
    # Comando: importar-entrenamiento
    importar_entrenamiento_parser = subparsers.add_parser('importar-entrenamiento',
                                                         help='V2: Importar imágenes de las carpetas de entrenamiento')
    
    # Comando: entrenar-patrones-v2
    entrenar_patrones_v2_parser = subparsers.add_parser('entrenar-patrones-v2',
                                                      help='V2: Entrenar modelo con técnicas de IA avanzadas y optimizaciones')
    entrenar_patrones_v2_parser.add_argument('--epochs', type=int, default=30,
                                          help='Número de épocas (default: 30)')
    entrenar_patrones_v2_parser.add_argument('--batch-size', type=int, default=16,
                                          help='Tamaño del batch (default: 16)')
    entrenar_patrones_v2_parser.add_argument('--val-split', type=float, default=0.2,
                                          help='Proporción de validación (default: 0.2)')
    entrenar_patrones_v2_parser.add_argument('--learning-rate', type=float, default=0.001,
                                          help='Learning rate inicial (default: 0.001)')
    entrenar_patrones_v2_parser.add_argument('--max-lr', type=float, default=0.01,
                                          help='Learning rate máximo para One-Cycle (default: 0.01)')
    entrenar_patrones_v2_parser.add_argument('--focal-loss', action='store_true',
                                          help='Usar Focal Loss para clases desbalanceadas')
    entrenar_patrones_v2_parser.add_argument('--label-smoothing', type=float, default=0.0,
                                          help='Factor de label smoothing (default: 0.0)')
    entrenar_patrones_v2_parser.add_argument('--early-stopping', type=int, default=10,
                                          help='Paciencia para early stopping (default: 10)')
    entrenar_patrones_v2_parser.add_argument('--warmup', type=int, default=3,
                                          help='Épocas de warmup (default: 3)')
    entrenar_patrones_v2_parser.add_argument('--dropout', type=float, default=0.4,
                                          help='Tasa de dropout (default: 0.4)')
    entrenar_patrones_v2_parser.add_argument('--no-mixup', action='store_true',
                                          help='Desactivar Mixup augmentation')
    entrenar_patrones_v2_parser.add_argument('--no-randaugment', action='store_true',
                                          help='Desactivar RandAugment')
    entrenar_patrones_v2_parser.add_argument('--grad-accum', type=int, default=1,
                                          help='Gradient accumulation steps (default: 1)')

    # Optimizaciones de rendimiento
    entrenar_patrones_v2_parser.add_argument('--use-amp', action='store_true', default=None,
                                          help='Usar Automatic Mixed Precision (auto: True en GPU)')
    entrenar_patrones_v2_parser.add_argument('--no-amp', action='store_true',
                                          help='Desactivar Automatic Mixed Precision')
    entrenar_patrones_v2_parser.add_argument('--use-compile', action='store_true', default=None,
                                          help='Usar torch.compile (auto: True si PyTorch 2.0+)')
    entrenar_patrones_v2_parser.add_argument('--no-compile', action='store_true',
                                          help='Desactivar torch.compile')
    entrenar_patrones_v2_parser.add_argument('--use-gradient-checkpointing', action='store_true',
                                          help='Usar gradient checkpointing para ahorrar memoria')
    entrenar_patrones_v2_parser.add_argument('--num-workers', type=int, default=None,
                                          help='Workers para DataLoader (auto: 4 si CPU, 2 si GPU)')
    entrenar_patrones_v2_parser.add_argument('--no-channels-last', action='store_true',
                                          help='Desactivar channels_last memory format')
    
    # Comando: reconocer-v2
    reconocer_v2_parser = subparsers.add_parser('reconocer-v2',
                                              help='V2: Reconocer patrones en una imagen')
    reconocer_v2_parser.add_argument('imagen', help='Ruta a la imagen')
    reconocer_v2_parser.add_argument('--umbral', type=float, default=0.5,
                                    help='Umbral de confianza (0.0-1.0)')
    reconocer_v2_parser.add_argument('--multiscale', action='store_true',
                                    help='Usar multi-scale inference para mayor precisión')
    
    # Comando: identificar-v2
    identificar_v2_parser = subparsers.add_parser('identificar-v2',
                                                help='V2: Identificar patrones en todas las imágenes de fotos_identificar/')
    identificar_v2_parser.add_argument('--umbral', type=float, default=0.5,
                                     help='Umbral de confianza (0.0-1.0)')
    identificar_v2_parser.add_argument('--output', help='Ruta del archivo JSON de salida')
    
    # Comando: listar-patrones-v2
    listar_patrones_v2_parser = subparsers.add_parser('listar-patrones-v2',
                                                   help='V2: Listar patrones con información detallada')
    
    # Comando: info-v2
    info_v2_parser = subparsers.add_parser('info-v2',
                                          help='V2: Mostrar información del sistema y modelo')
    
    # Comando: flujo-completo-v2
    flujo_completo_v2_parser = subparsers.add_parser('flujo-completo-v2',
                                                   help='V2: Flujo completo: crear patrón, importar y entrenar')
    flujo_completo_v2_parser.add_argument('nombre', help='Nombre del patrón')
    flujo_completo_v2_parser.add_argument('--descripcion', default='', help='Descripción del patrón')
    flujo_completo_v2_parser.add_argument('--epochs', type=int, default=30,
                                          help='Número de épocas (default: 30)')

    args = parser.parse_args()
    
    # Inicializar gestor de idiomas
    language_manager = LanguageManager(args.idioma)
    
    if args.comando == 'comparar':
        comparar_imagenes(args.imagen1, args.imagen2, args.umbral, args.metodo, args.modelo, language_manager)
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
        entrenar_patrones(
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            learning_rate=args.learning_rate,
            use_focal_loss=args.focal_loss,
            label_smoothing=args.label_smoothing,
            early_stopping_patience=args.early_stopping,
            dropout_rate=args.dropout,
            language_manager=language_manager
        )
    elif args.comando == 'reconocer-patron':
        reconocer_patron(
            imagen_path=args.imagen,
            roi=tuple(args.roi) if args.roi else None,
            threshold=args.umbral,
            mostrar_razonamiento=args.razonamiento,
            use_tta=args.tta,
            tta_transforms=args.tta_transforms,
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
            modelo_path=args.modelo,
            mostrar_razonamiento=args.razonamiento,
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
    elif args.comando == 'analizar':
        modo_analisis(
            imagen_path=args.imagen,
            threshold=args.umbral,
            language_manager=language_manager
        )
    elif args.comando == 'video-patron':
        reconocer_video_patrones(
            video_path=args.video,
            camera_index=args.camara,
            roi=tuple(args.roi) if args.roi else None,
            threshold=args.umbral,
            frame_skip=args.frame_skip,
            use_tta=args.tta,
            save_output=args.guardar,
            language_manager=language_manager
        )
    elif args.comando == 'camara-patron':
        reconocer_video_patrones(
            video_path=None,
            camera_index=args.camara,
            roi=tuple(args.roi) if args.roi else None,
            threshold=args.umbral,
            frame_skip=args.optimizacion,
            use_tta=False,  # No usar TTA en tiempo real (demasiado lento)
            save_output=args.guardar,
            language_manager=language_manager
        )
    elif args.comando == 'reporte-temporal':
        generar_reporte_temporal(
            detection_file=args.detecciones,
            output_path=args.output
        )
    # ═══════════════════════════════════════════════════════════════
    # COMANDOS V2 - Sistema mejorado de reconocimiento de patrones
    # ═══════════════════════════════════════════════════════════════
    elif args.comando == 'crear-patron-v2':
        crear_patron_v2(
            nombre=args.nombre,
            descripcion=args.descripcion,
            language_manager=language_manager
        )
    elif args.comando == 'importar-entrenamiento':
        importar_entrenamiento(language_manager=language_manager)
    elif args.comando == 'entrenar-patrones-v2':
        # Manejar argumentos booleanos para optimizaciones
        use_amp = args.use_amp if args.use_amp is not None else (not args.no_amp if args.no_amp else None)
        use_compile = args.use_compile if args.use_compile is not None else (not args.no_compile if args.no_compile else None)

        entrenar_patrones_v2(
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            learning_rate=args.learning_rate,
            max_lr=args.max_lr,
            use_focal_loss=args.focal_loss,
            label_smoothing=args.label_smoothing,
            early_stopping_patience=args.early_stopping,
            warmup_epochs=args.warmup,
            dropout_rate=args.dropout,
            use_mixup=not args.no_mixup,
            use_randaugment=not args.no_randaugment,
            gradient_accumulation=args.grad_accum,
            use_amp=use_amp,
            use_compile=use_compile,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            num_workers=args.num_workers,
            channels_last=not args.no_channels_last,
            language_manager=language_manager
        )
    elif args.comando == 'reconocer-v2':
        reconocer_patron_v2(
            imagen_path=args.imagen,
            threshold=args.umbral,
            multiscale=args.multiscale,
            language_manager=language_manager
        )
    elif args.comando == 'identificar-v2':
        identificar_carpetas_v2(
            threshold=args.umbral,
            output_file=args.output,
            language_manager=language_manager
        )
    elif args.comando == 'listar-patrones-v2':
        listar_patrones_v2(language_manager=language_manager)
    elif args.comando == 'info-v2':
        info_modelo_v2(language_manager=language_manager)
    elif args.comando == 'flujo-completo-v2':
        flujo_completo_v2(
            nombre_patron=args.nombre,
            descripcion=args.descripcion,
            epochs=args.epochs,
            language_manager=language_manager
        )
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())