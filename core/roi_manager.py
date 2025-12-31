"""
Módulo para gestión de regiones de interés (ROI) en imágenes
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Optional


class ROIManager:
    """Gestor de regiones de interés para análisis selectivo de imágenes."""
    
    def __init__(self):
        self.rois = []  # Lista de regiones definidas
        self.current_image = None
        self.current_window = "C.A. Dupin - ROI Selector"
    
    def set_image(self, image_path: str) -> bool:
        """Establece la imagen actual para seleccionar ROI."""
        if not Path(image_path).exists():
            print(f"Error: Imagen no encontrada: {image_path}")
            return False
        
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            print(f"Error: No se pudo cargar la imagen: {image_path}")
            return False
        
        return True
    
    def select_roi_interactive(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Permite al usuario seleccionar múltiples ROI de forma interactiva.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Lista de tuplas (x, y, width, height) para cada ROI
        """
        if not self.set_image(image_path):
            return []
        
        print("\n=== Selección de Regiones de Interés (ROI) ===")
        print("Instrucciones:")
        print("- Arrastra el mouse para seleccionar una región")
        print("- Presiona 'n' para siguiente región")
        print("- Presiona 'c' para continuar sin más regiones")
        print("- Presiona 'r' para reiniciar selección")
        print("- Presiona 'ESC' para cancelar\n")
        
        self.rois = []
        self._mouse_rect = None
        self._drawing = False
        self._show_instructions = True
        
        # Crear ventana y configurar mouse callback
        cv2.namedWindow(self.current_window)
        cv2.setMouseCallback(self.current_window, self._mouse_callback)
        
        while True:
            display_image = self.current_image.copy()
            
            # Mostrar ROI existentes
            for i, (x, y, w, h) in enumerate(self.rois):
                cv2.rectangle(display_image, (x, y), (x + w, h + h), (0, 255, 0), 2)
                cv2.putText(display_image, f"ROI {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Mostrar rectángulo actual mientras se arrastra
            if self._mouse_rect and self._drawing:
                x, y, w, h = self._mouse_rect
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(display_image, "Nueva ROI", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Mostrar instrucciones
            if self._show_instructions:
                cv2.putText(display_image, "n: siguiente ROI | c: continuar | r: reiniciar | ESC: cancelar", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_image, f"ROI seleccionadas: {len(self.rois)}", 
                           (10, display_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(self.current_window, display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n') or key == ord('N'):
                if self._mouse_rect and self._drawing:
                    self.rois.append(self._mouse_rect)
                    self._mouse_rect = None
                    self._drawing = False
                    print(f"✓ ROI agregada: {self.rois[-1]}")
            
            elif key == ord('c') or key == ord('C'):
                if self._mouse_rect and self._drawing:
                    self.rois.append(self._mouse_rect)
                    self._mouse_rect = None
                    self._drawing = False
                break
            
            elif key == ord('r') or key == ord('R'):
                self.rois = []
                self._mouse_rect = None
                self._drawing = False
                print("✓ Selección reiniciada")
            
            elif key == 27:  # ESC
                self.rois = []
                break
        
        cv2.destroyAllWindows()
        
        if self.rois:
            print(f"\n✓ {len(self.rois)} ROI(s) seleccionada(s)")
            for i, roi in enumerate(self.rois):
                print(f"  ROI {i+1}: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        else:
            print("\nNo se seleccionaron ROI")
        
        return self.rois
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._mouse_rect = (x, y, 0, 0)
            self._show_instructions = False
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drawing:
                self._mouse_rect = (self._mouse_rect[0], self._mouse_rect[1], 
                                  x - self._mouse_rect[0], y - self._mouse_rect[1])
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self._drawing:
                self._drawing = False
                if abs(self._mouse_rect[2]) > 5 and abs(self._mouse_rect[3]) > 5:
                    # Normalizar coordenadas (asegurarse de que width y height sean positivos)
                    x1, y1 = self._mouse_rect[0], self._mouse_rect[1]
                    x2, y2 = x, y
                    
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    
                    self._mouse_rect = (x, y, w, h)
                else:
                    self._mouse_rect = None
    
    def extract_roi_regions(self, image_path: str, rois: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Extrae las regiones de interés de una imagen.
        
        Args:
            image_path: Ruta a la imagen
            rois: Lista de ROI (x, y, width, height)
            
        Returns:
            Lista de imágenes recortadas
        """
        if not Path(image_path).exists():
            print(f"Error: Imagen no encontrada: {image_path}")
            return []
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: No se pudo cargar la imagen: {image_path}")
            return []
        
        roi_images = []
        for i, (x, y, w, h) in enumerate(rois):
            # Validar coordenadas
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w > 0 and h > 0:
                roi_image = image[y:y+h, x:x+w]
                roi_images.append(roi_image)
            else:
                print(f"Advertencia: ROI {i+1} fuera de los límites de la imagen")
        
        return roi_images
    
    def save_rois(self, rois: List[Tuple[int, int, int, int]], output_path: str):
        """Guarda las ROI en un archivo JSON."""
        roi_data = {
            'rois': [{'x': roi[0], 'y': roi[1], 'width': roi[2], 'height': roi[3]} 
                    for roi in rois],
            'created_at': str(np.datetime64('now'))
        }
        
        with open(output_path, 'w') as f:
            json.dump(roi_data, f, indent=2)
        
        print(f"ROI guardadas en: {output_path}")
    
    def load_rois(self, roi_file_path: str) -> List[Tuple[int, int, int, int]]:
        """Carga ROI desde un archivo JSON."""
        if not Path(roi_file_path).exists():
            print(f"Error: Archivo de ROI no encontrado: {roi_file_path}")
            return []
        
        try:
            with open(roi_file_path, 'r') as f:
                roi_data = json.load(f)
            
            rois = [(roi['x'], roi['y'], roi['width'], roi['height']) 
                   for roi in roi_data.get('rois', [])]
            
            print(f"Cargadas {len(rois)} ROI desde: {roi_file_path}")
            return rois
        except Exception as e:
            print(f"Error cargando ROI: {e}")
            return []
    
    def visualize_rois(self, image_path: str, rois: List[Tuple[int, int, int, int]] = None):
        """Visualiza las ROI en la imagen."""
        if rois is None:
            rois = self.rois
        
        if not rois:
            print("No hay ROI para visualizar")
            return
        
        if not self.set_image(image_path):
            return
        
        display_image = self.current_image.copy()
        
        for i, (x, y, w, h) in enumerate(rois):
            # Color diferente para cada ROI
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 3)
            
            # Etiqueta con número de ROI
            cv2.putText(display_image, f"ROI {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Información de la ROI
            info = f"({x},{y}) {w}x{h}"
            cv2.putText(display_image, info, (x, y+h+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(display_image, f"Total ROI: {len(rois)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("C.A. Dupin - ROI Visualization", display_image)
        print("Presiona cualquier tecla para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def auto_detect_roi(self, image_path: str, method: str = 'contour') -> List[Tuple[int, int, int, int]]:
        """
        Detecta automáticamente regiones de interés usando diferentes métodos.
        
        Args:
            image_path: Ruta a la imagen
            method: Método de detección ('contour', 'edge', 'color')
            
        Returns:
            Lista de ROI detectadas automáticamente
        """
        if not self.set_image(image_path):
            return []
        
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        rois = []
        
        if method == 'contour':
            # Detectar contornos para encontrar objetos
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filtrar objetos pequeños
                    x, y, w, h = cv2.boundingRect(contour)
                    rois.append((x, y, w, h))
        
        elif method == 'edge':
            # Detectar bordes para encontrar regiones de interés
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filtrar bordes pequeños
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 5.0:  # Filtrar formas muy irregulares
                        rois.append((x, y, w, h))
        
        elif method == 'color':
            # Detectar regiones con colores dominantes
            hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
            
            # Detectar regiones de color uniforme
            for hue in range(0, 180, 30):
                lower = np.array([hue, 30, 30])
                upper = np.array([hue + 30, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 2000:  # Solo regiones grandes
                        x, y, w, h = cv2.boundingRect(contour)
                        rois.append((x, y, w, h))
        
        # Eliminar ROI superpuestas o muy pequeñas
        filtered_rois = self._filter_rois(rois)
        
        print(f"Detectadas {len(filtered_rois)} ROI automáticamente con método '{method}'")
        return filtered_rois
    
    def _filter_rois(self, rois: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Filtra ROI eliminando duplicados y regiones inválidas."""
        if not rois:
            return []
        
        filtered = []
        
        for roi in rois:
            x, y, w, h = roi
            
            # Filtrar ROI muy pequeñas
            if w < 20 or h < 20:
                continue
            
            # Verificar superposición con ROI existentes
            is_overlap = False
            for existing_roi in filtered:
                ex, ey, ew, eh = existing_roi
                
                # Calcular superposición
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                
                # Si la superposición es mayor al 50% de la ROI más pequeña, la descartamos
                roi_area = w * h
                existing_area = ew * eh
                min_area = min(roi_area, existing_area)
                
                if overlap_area > 0.5 * min_area:
                    is_overlap = True
                    break
            
            if not is_overlap:
                filtered.append(roi)
        
        return filtered
    
    def get_roi_info(self, rois: List[Tuple[int, int, int, int]]) -> dict:
        """Obtiene información estadística sobre las ROI."""
        if not rois:
            return {'count': 0, 'total_area': 0, 'avg_area': 0, 'avg_aspect_ratio': 0}
        
        areas = [roi[2] * roi[3] for roi in rois]
        aspect_ratios = [roi[2] / roi[3] for roi in rois]
        
        return {
            'count': len(rois),
            'total_area': sum(areas),
            'avg_area': np.mean(areas),
            'min_area': min(areas),
            'max_area': max(areas),
            'avg_aspect_ratio': np.mean(aspect_ratios),
            'min_aspect_ratio': min(aspect_ratios),
            'max_aspect_ratio': max(aspect_ratios)
        }