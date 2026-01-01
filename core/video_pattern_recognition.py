"""
Módulo para reconocimiento de patrones en video en vivo.
Extiende la funcionalidad de pattern_learner.py al procesamiento de video en tiempo real.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import time
import json
from datetime import datetime

from .pattern_learner import ImprovedPatternLearner


class VideoStreamPatternRecognizer:
    """
    Reconocedor de patrones en flujo de video en tiempo real.
    Utiliza el modelo entrenado de ImprovedPatternLearner.
    """
    
    def __init__(self, model_path: str = "patterns_model.pth", 
                 patterns_file: str = "user_patterns/patterns.json"):
        """
        Inicializa el reconocedor de patrones para video.
        
        Args:
            model_path: Ruta al modelo entrenado
            patterns_file: Ruta al archivo de patrones definidos
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pattern_learner = ImprovedPatternLearner(model_path)
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        self.detection_history = []
        
        # Parámetros de optimización
        self.frame_skip = 1  # Procesar cada frame
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4  # Non-maximum suppression
        self.detection_cooldown = 0  # Frames entre detecciones del mismo patrón
        
        # Tracking de detecciones
        self.active_detections = {}
        self.detection_id_counter = 0
        
        print(f"✓ VideoStreamPatternRecognizer inicializado (Device: {self.device})")
    
    def process_frame(self, frame: np.ndarray, roi: Tuple[int, int, int, int] = None,
                     threshold: float = 0.5, use_tta: bool = False) -> List[Dict]:
        """
        Procesa un frame individual y detecta patrones.
        
        Args:
            frame: Frame de video en formato BGR
            roi: Región de interés opcional (x, y, w, h)
            threshold: Umbral de confianza mínimo
            use_tta: Usar Test Time Augmentation
            
        Returns:
            Lista de detecciones en el frame
        """
        # Incrementar frame counter
        self.frame_count += 1
        
        # Calcular FPS
        if self.frame_count == 1:
            self.start_time = time.time()
        elif self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
        
        # Extraer ROI si se especifica
        detection_frame = frame
        detection_roi = roi
        
        if roi:
            x, y, w, h = roi
            h_full, w_full = frame.shape[:2]
            x = max(0, min(x, w_full - 1))
            y = max(0, min(y, h_full - 1))
            w = min(w, w_full - x)
            h = min(h, h_full - y)
            if w > 0 and h > 0:
                detection_frame = frame[y:y+h, x:x+w]
                detection_roi = (x, y, w, h)
        else:
            detection_roi = (0, 0, frame.shape[1], frame.shape[0])
        
        # Guardar frame temporalmente para el reconocedor
        temp_frame_path = Path("temp_frame.jpg")
        cv2.imwrite(str(temp_frame_path), frame)
        
        try:
            # Realizar detección
            if use_tta and hasattr(self.pattern_learner, 'recognize_pattern_tta'):
                detections = self.pattern_learner.recognize_pattern_tta(
                    str(temp_frame_path), roi=detection_roi, 
                    threshold=threshold, include_reasoning=False
                )
            else:
                detections = self.pattern_learner.recognize_pattern(
                    str(temp_frame_path), roi=detection_roi,
                    threshold=threshold, include_reasoning=False
                )
            
            # Actualizar tracking
            self._update_detections_tracking(detections, self.frame_count)
            
            # Añadir metadatos
            for detection in detections:
                detection['frame'] = self.frame_count
                detection['fps'] = self.fps
                
            return detections
            
        finally:
            # Limpiar archivo temporal
            if temp_frame_path.exists():
                temp_frame_path.unlink()
    
    def _update_detections_tracking(self, detections: List[Dict], frame_num: int):
        """
        Actualiza el tracking de detecciones entre frames.
        
        Args:
            detections: Detecciones actuales
            frame_num: Número de frame actual
        """
        # Limpiar detecciones antiguas
        current_time = frame_num
        to_remove = []
        
        for det_id, detection in self.active_detections.items():
            if current_time - detection['last_seen'] > self.detection_cooldown + 10:
                to_remove.append(det_id)
        
        for det_id in to_remove:
            del self.active_detections[det_id]
        
        # Actualizar o añadir nuevas detecciones
        for detection in detections:
            pattern_name = detection['pattern_name']
            
            # Buscar detección existente del mismo patrón
            found_match = False
            for det_id, active_det in self.active_detections.items():
                if active_det['pattern_name'] == pattern_name:
                    # Actualizar detección existente
                    active_det['last_seen'] = frame_num
                    active_det['confidence'] = detection['probability']
                    if 'confidence' in detection:
                        active_det['confidence_history'].append(detection['probability'])
                    found_match = True
                    break
            
            if not found_match:
                # Crear nueva detección track
                det_id = f"det_{self.detection_id_counter}"
                self.detection_id_counter += 1
                
                self.active_detections[det_id] = {
                    'id': det_id,
                    'pattern_name': pattern_name,
                    'pattern_id': detection.get('pattern_id', ''),
                    'first_seen': frame_num,
                    'last_seen': frame_num,
                    'confidence': detection['probability'],
                    'confidence_history': [detection['probability']]
                }
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict],
                       font_scale: float = 0.6, thickness: int = 2) -> np.ndarray:
        """
        Dibuja las detecciones sobre el frame.
        
        Args:
            frame: Frame original
            detections: Lista de detecciones
            font_scale: Escala de fuente para texto
            thickness: Grosor de líneas
            
        Returns:
            Frame con detecciones dibujadas
        """
        output_frame = frame.copy()
        
        for detection in detections:
            # Obtener bounding box
            bbox = detection.get('bbox', (0, 0, frame.shape[1], frame.shape[0]))
            x, y, w, h = bbox
            
            # Calificar color según confianza
            confidence = detection['probability']
            if confidence >= 0.8:
                color = (0, 255, 0)  # Verde - alta confianza
            elif confidence >= 0.6:
                color = (0, 255, 255)  # Amarillo - media confianza
            else:
                color = (0, 165, 255)  # Naranja - baja confianza
            
            # Dibujar rectángulo
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Preparar texto
            pattern_name = detection['pattern_name']
            conf_text = f"{confidence:.2f}"
            
            # Dibujar fondo del texto
            label = f"{pattern_name}: {conf_text}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            cv2.rectangle(
                output_frame, 
                (x, y - text_height - 10), 
                (x + text_width, y), 
                color, 
                -1
            )
            
            # Dibujar texto
            cv2.putText(
                output_frame, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 0, 0) if confidence >= 0.8 else (255, 255, 255), 
                1
            )
            
            # Añadir información adicional (TTA si está disponible)
            if 'consistency' in detection:
                consistency = detection['consistency']
                consistency_text = f"Consistencia: {consistency:.2f}"
                cv2.putText(
                    output_frame,
                    consistency_text,
                    (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.8,
                    (200, 200, 200),
                    1
                )
        
        # Añadir información de FPS
        if self.fps > 0:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(
                output_frame, 
                fps_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
        
        # Añadir número de detecciones
        det_text = f"Dets: {len(detections)}"
        cv2.putText(
            output_frame, 
            det_text, 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        return output_frame
    
    def recognize_from_video_file(self, video_path: str, output_path: str = None,
                                 roi: Tuple[int, int, int, int] = None,
                                 threshold: float = 0.5, use_tta: bool = False,
                                 display: bool = True,
                                 save_detections: bool = True) -> List[Dict]:
        """
        Reconoce patrones en un archivo de video.
        
        Args:
            video_path: Ruta al archivo de video
            output_path: Ruta opcional para guardar video con detecciones
            roi: Región de interés
            threshold: Umbral de confianza
            use_tta: Usar TTA (lento para video)
            display: Mostrar video en tiempo real
            save_detections: Guardar detecciones en archivo JSON
            
        Returns:
            Historia de detecciones
        """
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Obtener propiedades del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Preparar writer de salida
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.detection_history = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Procesar cada N frames (optimización)
                if frame_count % self.frame_skip == 0:
                    detections = self.process_frame(frame, roi, threshold, use_tta)
                    
                    # Guardar detecciones en historia
                    if detections:
                        frame_detections = {
                            'frame': frame_count,
                            'timestamp': frame_count / fps if fps > 0 else 0,
                            'detections': detections
                        }
                        self.detection_history.append(frame_detections)
                else:
                    # Usar detecciones previas si no procesamos este frame
                    detections = []
                
                # Dibujar detecciones
                output_frame = self.draw_detections(frame, detections)
                
                # Mostrar
                if display:
                    cv2.imshow('Video Recognition', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Guardar
                if writer:
                    writer.write(output_frame)
                
                # Mostrar progreso
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"Procesando... {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Guardar detecciones
        if save_detections:
            self._save_detection_history(video_path)
        
        print(f"✓ Video procesado: {frame_count} frames, {len(self.detection_history)} frames con detecciones")
        return self.detection_history
    
    def recognize_from_camera(self, camera_index: int = 0, roi: Tuple[int, int, int, int] = None,
                             threshold: float = 0.5, use_tta: bool = False,
                             save_video: bool = False, output_path: str = None,
                             max_duration: int = None) -> List[Dict]:
        """
        Reconoce patrones desde cámara en tiempo real.
        
        Args:
            camera_index: Índice de la cámara (0 default)
            roi: Región de interés
            threshold: Umbral de confianza
            use_tta: Usar TTA (muy lento para tiempo real)
            save_video: Guardar video con detecciones
            output_path: Ruta para guardar video
            max_duration: Duración máxima en segundos (None = infinito)
            
        Returns:
            Historia de detecciones
        """
        # Abrir cámara
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir la cámara {camera_index}")
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Obtener propiedades
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ Cámara iniciada: {width}x{height} @ {fps:.1f} FPS")
        
        # Preparar writer
        writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.detection_history = []
        frame_count = 0
        start_time = time.time()
        
        print("\n⌨️ Controles:")
        print("  - 'q': Salir")
        print("  - 's': Guardar captura de pantalla")
        print("  - 'p': Pausar/reanudar")
        print("  - 'c': Limpiar detecciones activas")
        print("\n🎥 Iniciando reconocimiento en tiempo real...\n")
        
        paused = False
        
        try:
            while True:
                # Verificar tiempo máximo
                if max_duration and (time.time() - start_time) > max_duration:
                    print(f"⏱️ Tiempo máximo alcanzado: {max_duration}s")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ No se pudo capturar frame")
                    break
                
                frame_count += 1
                
                if not paused:
                    # Procesar frame
                    if frame_count % self.frame_skip == 0:
                        detections = self.process_frame(frame, roi, threshold, use_tta)
                        
                        if detections:
                            frame_detections = {
                                'frame': frame_count,
                                'timestamp': time.time() - start_time,
                                'detections': detections
                            }
                            self.detection_history.append(frame_detections)
                    else:
                        detections = []
                    
                    # Dibujar detecciones
                    output_frame = self.draw_detections(frame, detections)
                else:
                    # Mostrar frame sin procesar (pausado)
                    output_frame = self.draw_detections(
                        frame, [], 
                        font_scale=0.7, thickness=2
                    )
                    cv2.putText(
                        output_frame, 
                        "PAUSADO", 
                        (width // 2 - 50, height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255), 
                        2
                    )
                
                # Mostrar tiempo
                elapsed = time.time() - start_time
                time_text = f"Tiempo: {elapsed:.1f}s"
                cv2.putText(output_frame, time_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mostrar frame
                cv2.imshow('Video Recognition en Vivo', output_frame)
                
                # Guardar frame si es necesario
                if writer:
                    writer.write(output_frame)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Saliendo...")
                    break
                elif key == ord('s'):
                    # Guardar screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, output_frame)
                    print(f"📸 Captura guardada: {screenshot_path}")
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'⏸️' if paused else '▶️'} {'Pausado' if paused else 'Reanudado'}")
                elif key == ord('c'):
                    self.active_detections.clear()
                    print("🗑️ Detecciones activas limpiadas")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\n✓ Stream finalizado:")
        print(f"  - Frames procesados: {frame_count}")
        print(f"  - Duración: {elapsed:.1f}s")
        print(f"  - FPS promedio: {frame_count/elapsed:.1f}")
        print(f"  - Frames con detecciones: {len(self.detection_history)}")
        
        if save_video:
            self._save_detection_history(f"camera_{camera_index}")
        
        return self.detection_history
    
    def _save_detection_history(self, source: str):
        """
        Guarda el historial de detecciones en archivo JSON.
        
        Args:
            source: Origen del video (ruta o ID de cámara)
        """
        if not self.detection_history:
            return
        
        # Calcular estadísticas
        all_patterns = {}
        total_detections = 0
        
        for frame_data in self.detection_history:
            for detection in frame_data['detections']:
                pattern_name = detection['pattern_name']
                if pattern_name not in all_patterns:
                    all_patterns[pattern_name] = {
                        'count': 0,
                        'avg_confidence': 0,
                        'max_confidence': 0,
                        'frames': []
                    }
                all_patterns[pattern_name]['count'] += 1
                all_patterns[pattern_name]['avg_confidence'] += detection['probability']
                all_patterns[pattern_name]['max_confidence'] = max(
                    all_patterns[pattern_name]['max_confidence'],
                    detection['probability']
                )
                all_patterns[pattern_name]['frames'].append(frame_data['frame'])
                total_detections += 1
        
        # Calcular promedios
        for pattern in all_patterns.values():
            pattern['avg_confidence'] /= pattern['count']
        
        # Preparar datos de salida
        output_data = {
            'source': str(source),
            'timestamp': datetime.now().isoformat(),
            'total_frames_processed': self.frame_count,
            'frames_with_detections': len(self.detection_history),
            'fps': self.fps,
            'summary': {
                'total_detections': total_detections,
                'unique_patterns': len(all_patterns),
                'patterns': all_patterns
            },
            'detection_history': self.detection_history
        }
        
        # Guardar archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_source = str(source).replace('/', '_').replace('\\', '_')
        output_path = f"detections_{safe_source}_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"📄 Historial de detecciones guardado: {output_path}")
        except Exception as e:
            print(f"⚠️ Error guardando historial: {e}")
    
    def set_optimization_params(self, frame_skip: int = 1, confidence_threshold: float = 0.5,
                               nms_threshold: float = 0.4, detection_cooldown: int = 0):
        """
        Configura parámetros de optimización para tiempo real.
        
        Args:
            frame_skip: Procesar cada N frames (1 = todos)
            confidence_threshold: Umbral mínimo de confianza
            nms_threshold: Umbral para Non-Maximum Suppression
            detection_cooldown: Frames entre detecciones del mismo patrón
        """
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.detection_cooldown = detection_cooldown
        
        print(f"⚙️ Parámetros optimizados:")
        print(f"  - Frame skip: {frame_skip}")
        print(f"  - Confidence threshold: {confidence_threshold}")
        print(f"  - NMS threshold: {nms_threshold}")
        print(f"  - Detection cooldown: {detection_cooldown} frames")
    
    def get_performance_stats(self) -> Dict:
        """
        Obtiene estadísticas de rendimiento.
        
        Returns:
            Diccionario con estadísticas
        """
        if self.frame_count == 0:
            return {}
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_frames': self.frame_count,
            'fps': self.fps,
            'elapsed_time': elapsed,
            'frames_per_second': self.frame_count / elapsed if elapsed > 0 else 0,
            'active_detections': len(self.active_detections),
            'detection_history_frames': len(self.detection_history)
        }


class RealTimePatternAnalyzer:
    """
    Analizador de patrones en tiempo real con análisis adicional.
    """
    
    @staticmethod
    def analyze_detection_frequency(detection_history: List[Dict], time_window: float = 1.0):
        """
        Analiza la frecuencia de detecciones por patrón.
        
        Args:
            detection_history: Historial de detecciones
            time_window: Ventana de tiempo en segundos
            
        Returns:
            Diccionario con frecuencias
        """
        pattern_frequency = {}
        
        if not detection_history:
            return pattern_frequency
        
        # Ordenar por timestamp
        sorted_history = sorted(detection_history, key=lambda x: x.get('timestamp', 0))
        
        for frame_data in sorted_history:
            timestamp = frame_data.get('timestamp', 0)
            for detection in frame_data.get('detections', []):
                pattern_name = detection['pattern_name']
                
                if pattern_name not in pattern_frequency:
                    pattern_frequency[pattern_name] = {
                        'timestamps': [],
                        'frequency_per_second': 0
                    }
                
                pattern_frequency[pattern_name]['timestamps'].append(timestamp)
        
        # Calcular frecuencia
        for pattern, data in pattern_frequency.items():
            timestamps = data['timestamps']
            if len(timestamps) >= 2:
                duration = timestamps[-1] - timestamps[0]
                if duration > 0:
                    data['frequency_per_second'] = len(timestamps) / duration
        
        return pattern_frequency
    
    @staticmethod
    def generate_temporal_report(detection_history: List[Dict], output_path: str):
        """
        Genera un reporte temporal de detecciones.
        
        Args:
            detection_history: Historial de detecciones
            output_path: Ruta para guardar reporte
        """
        if not detection_history:
            print("No hay detecciones para generar reporte")
            return
        
        # Analizar frecuencia
        frequency = RealTimePatternAnalyzer.analyze_detection_frequency(detection_history)
        
        # Calcular estadísticas
        total_frames = len(detection_history)
        total_detections = sum(len(f['detections']) for f in detection_history)
        
        # Patrones únicos
        unique_patterns = set()
        for frame_data in detection_history:
            for detection in frame_data['detections']:
                unique_patterns.add(detection['pattern_name'])
        
        # Preparar reporte
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_frames': total_frames,
                'total_detections': total_detections,
                'unique_patterns': len(unique_patterns),
                'pattern_names': list(unique_patterns)
            },
            'frequency_analysis': frequency,
            'raw_history': detection_history
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📊 Reporte temporal guardado: {output_path}")
        except Exception as e:
            print(f"⚠️ Error guardando reporte: {e}")


def create_video_recognizer(model_path: str = None) -> VideoStreamPatternRecognizer:
    """
    Crea una instancia del reconocedor de video.
    
    Args:
        model_path: Ruta al modelo (opcional)
        
    Returns:
        Instancia de VideoStreamPatternRecognizer
    """
    if model_path and Path(model_path).exists():
        return VideoStreamPatternRecognizer(model_path)
    else:
        return VideoStreamPatternRecognizer()
