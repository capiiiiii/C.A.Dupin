"""
Módulo para gestión de cámara y captura en tiempo real
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List
import json
from datetime import datetime


class CameraManager:
    """Gestor de cámara para captura y procesamiento en tiempo real."""
    
    def __init__(self, camera_id: int = 0):
        """
        Inicializa el gestor de cámara.
        
        Args:
            camera_id: ID de la cámara (0 para cámara por defecto)
        """
        self.camera_id = camera_id
        self.cap = None
        self.is_capturing = False
        self.current_frame = None
        self.frame_callback = None
        self.capture_thread = None
        self.fps = 30
        self.resolution = (640, 480)
        self.frame_count = 0
        self.start_time = None
        
        # Configuración de grabación
        self.recording = False
        self.video_writer = None
        self.record_path = None
        
        # Estadísticas
        self.stats = {
            'frames_captured': 0,
            'frames_per_second': 0,
            'capture_time': 0,
            'errors': 0
        }
    
    def initialize_camera(self, camera_id: int = None) -> bool:
        """Inicializa la cámara con la configuración especificada."""
        if camera_id is not None:
            self.camera_id = camera_id
        
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"Error: No se pudo abrir la cámara {self.camera_id}")
                return False
            
            # Configurar resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            print(f"✓ Cámara inicializada (ID: {self.camera_id})")
            print(f"  Resolución: {self.resolution[0]}x{self.resolution[1]}")
            print(f"  FPS: {self.fps}")
            
            return True
            
        except Exception as e:
            print(f"Error inicializando cámara: {e}")
            return False
    
    def start_capture(self, callback: Callable = None) -> bool:
        """
        Inicia la captura continua de frames.
        
        Args:
            callback: Función a llamar con cada frame capturado
            
        Returns:
            bool: True si se inició correctamente
        """
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                return False
        
        self.frame_callback = callback
        self.is_capturing = True
        self.frame_count = 0
        self.start_time = time.time()
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        print("✓ Captura de cámara iniciada")
        return True
    
    def stop_capture(self):
        """Detiene la captura de frames."""
        self.is_capturing = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.recording:
            self.stop_recording()
        
        print("✓ Captura de cámara detenida")
    
    def _capture_loop(self):
        """Loop principal de captura de frames."""
        while self.is_capturing:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.stats['errors'] += 1
                    print(f"Error leyendo frame de la cámara")
                    time.sleep(0.1)
                    continue
                
                self.current_frame = frame.copy()
                self.frame_count += 1
                self.stats['frames_captured'] += 1
                
                # Calcular FPS en tiempo real
                if self.frame_count % 30 == 0:  # Actualizar cada 30 frames
                    elapsed = time.time() - self.start_time
                    self.stats['frames_per_second'] = self.frame_count / elapsed if elapsed > 0 else 0
                
                # Procesar frame con callback si existe
                if self.frame_callback:
                    try:
                        processed_frame = self.frame_callback(frame)
                        if processed_frame is not None:
                            frame = processed_frame
                    except Exception as e:
                        print(f"Error en callback de frame: {e}")
                
                # Mostrar frame en ventana
                self.display_frame(frame)
                
                # Manejar grabación
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)
                
                # Controlar FPS
                target_delay = 1.0 / self.fps
                actual_delay = time.time() - (self.start_time + self.frame_count * target_delay)
                if actual_delay < target_delay:
                    time.sleep(target_delay - actual_delay)
                
            except Exception as e:
                self.stats['errors'] += 1
                print(f"Error en captura: {e}")
                time.sleep(0.1)
    
    def display_frame(self, frame: np.ndarray, show_info: bool = True):
        """
        Muestra el frame en una ventana.
        
        Args:
            frame: Frame a mostrar
            show_info: Si se debe mostrar la información de estado (FPS, etc.)
        """
        display_frame = frame.copy()
        
        if show_info:
            # Mostrar FPS
            fps_text = f"FPS: {self.stats['frames_per_second']:.1f}"
            cv2.putText(display_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar número de frame
            frame_text = f"Frame: {self.frame_count}"
            cv2.putText(display_frame, frame_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar estado de grabación
            if self.recording:
                cv2.putText(display_frame, "● REC", (display_frame.shape[1] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mostrar controles
            cv2.putText(display_frame, "ESC: salir | SPC: foto | R: grabar", 
                       (10, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("C.A. Dupin - Camera Live", display_frame)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self.is_capturing = False
        elif key == 32:  # Espacio - tomar foto
            self.take_photo()
        elif key == ord('r') or key == ord('R'):
            if not self.recording:
                self.start_recording()
            else:
                self.stop_recording()
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Obtiene el frame actual capturado."""
        return self.current_frame.copy() if self.current_frame is not None else None
    
    def take_photo(self, output_path: str = None) -> bool:
        """
        Toma una foto del frame actual.
        
        Args:
            output_path: Ruta donde guardar la foto (opcional)
            
        Returns:
            bool: True si se guardó correctamente
        """
        if self.current_frame is None:
            print("Error: No hay frame disponible para capturar")
            return False
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"captura_{timestamp}.jpg"
        
        try:
            cv2.imwrite(output_path, self.current_frame)
            print(f"✓ Foto guardada: {output_path}")
            return True
        except Exception as e:
            print(f"Error guardando foto: {e}")
            return False
    
    def start_recording(self, output_path: str = None) -> bool:
        """
        Inicia la grabación de video.
        
        Args:
            output_path: Ruta donde guardar el video (opcional)
            
        Returns:
            bool: True si se inició correctamente
        """
        if self.recording:
            print("Ya se está grabando")
            return False
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"grabacion_{timestamp}.avi"
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps, 
                (self.resolution[0], self.resolution[1])
            )
            
            if not self.video_writer.isOpened():
                print("Error: No se pudo crear el archivo de video")
                return False
            
            self.record_path = output_path
            self.recording = True
            print(f"✓ Grabación iniciada: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error iniciando grabación: {e}")
            return False
    
    def stop_recording(self) -> bool:
        """Detiene la grabación de video."""
        if not self.recording:
            return False
        
        try:
            self.recording = False
            
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            print(f"✓ Grabación guardada: {self.record_path}")
            return True
            
        except Exception as e:
            print(f"Error deteniendo grabación: {e}")
            return False
    
    def set_resolution(self, width: int, height: int):
        """Cambia la resolución de la cámara."""
        self.resolution = (width, height)
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"✓ Resolución cambiada a {width}x{height}")
    
    def set_fps(self, fps: int):
        """Cambia los FPS de captura."""
        self.fps = fps
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            print(f"✓ FPS cambiado a {fps}")
    
    def list_cameras(self) -> List[dict]:
        """Lista las cámaras disponibles en el sistema."""
        cameras = []
        
        for i in range(10):  # Verificar hasta 10 cámaras
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Obtener propiedades de la cámara
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    cameras.append({
                        'id': i,
                        'resolution': f"{width}x{height}",
                        'fps': fps,
                        'available': True
                    })
                    
                    cap.release()
                else:
                    cap.release()
            except Exception:
                continue
        
        return cameras
    
    def detect_motion(self, threshold: float = 25.0) -> List[dict]:
        """
        Detecta movimiento en el frame actual.
        
        Args:
            threshold: Umbral de detección de movimiento
            
        Returns:
            Lista de regiones con movimiento detectado
        """
        if self.current_frame is None:
            return []
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Inicializar frame de referencia si no existe
        if not hasattr(self, 'reference_frame'):
            self.reference_frame = gray
            return []
        
        # Calcular diferencia
        frame_delta = cv2.absdiff(self.reference_frame, gray)
        thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilatar para llenar huecos
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filtrar áreas pequeñas
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'area': area,
                    'confidence': min(area / 1000, 1.0)
                })
        
        # Actualizar frame de referencia
        self.reference_frame = gray
        
        return motion_regions
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas de captura."""
        stats = self.stats.copy()
        
        if self.start_time:
            stats['capture_duration'] = time.time() - self.start_time
        
        return stats
    
    def reset_stats(self):
        """Reinicia las estadísticas de captura."""
        self.stats = {
            'frames_captured': 0,
            'frames_per_second': 0,
            'capture_time': 0,
            'errors': 0
        }
        self.frame_count = 0
        self.start_time = time.time()
    
    def save_session_info(self, output_path: str = "camera_session.json"):
        """Guarda información de la sesión de cámara."""
        session_info = {
            'camera_id': self.camera_id,
            'resolution': self.resolution,
            'fps': self.fps,
            'session_start': datetime.now().isoformat() if self.start_time else None,
            'stats': self.get_stats(),
            'recording_active': self.recording,
            'recorded_file': self.record_path if self.recording else None
        }
        
        with open(output_path, 'w') as f:
            json.dump(session_info, f, indent=2)
        
        print(f"Información de sesión guardada: {output_path}")
    
    def __del__(self):
        """Limpieza al destruir el objeto."""
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.stop_capture()
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()