"""
Módulo de interfaz visual para mostrar identificaciones del modelo
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import colorsys


class VisualInterface:
    """Interfaz visual para mostrar identificaciones y resultados del modelo."""
    
    def __init__(self):
        self.current_image = None
        self.detections = []
        self.roi_overlay = None
        self.confidence_threshold = 0.5
        self.colors = self._generate_colors(20)  # Colores para diferentes clases
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.window_name = "C.A. Dupin - Análisis Visual"
        
    def _generate_colors(self, n_colors: int) -> List[Tuple[int, int, int]]:
        """Genera una paleta de colores distintos."""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            color = tuple(int(c * 255) for c in rgb)
            colors.append(color)
        return colors
    
    def load_image(self, image_path: str) -> bool:
        """Carga una imagen para análisis visual."""
        if not Path(image_path).exists():
            print(f"Error: Imagen no encontrada: {image_path}")
            return False
        
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            print(f"Error: No se pudo cargar la imagen: {image_path}")
            return False
        
        return True
    
    def add_detection(self, class_name: str, confidence: float, 
                     bounding_box: Tuple[int, int, int, int],
                     roi_id: Optional[int] = None):
        """Añade una detección al análisis visual."""
        detection = {
            'class_name': class_name,
            'confidence': confidence,
            'bbox': bounding_box,
            'roi_id': roi_id,
            'timestamp': datetime.now().isoformat(),
            'color': self.colors[len(self.detections) % len(self.colors)]
        }
        
        self.detections.append(detection)
    
    def clear_detections(self):
        """Limpia todas las detecciones."""
        self.detections = []
    
    def visualize_detections(self, show_confidence: bool = True, 
                           show_class_names: bool = True) -> np.ndarray:
        """
        Crea una visualización de las detecciones en la imagen.
        
        Args:
            show_confidence: Si mostrar los scores de confianza
            show_class_names: Si mostrar los nombres de las clases
            
        Returns:
            Imagen con visualizaciones superpuestas
        """
        if self.current_image is None:
            print("Error: No hay imagen cargada")
            return None
        
        # Crear copia de la imagen
        display_image = self.current_image.copy()
        
        # Ordenar detecciones por confianza (mostrar las más altas primero)
        sorted_detections = sorted(self.detections, key=lambda x: x['confidence'], reverse=True)
        
        for i, detection in enumerate(sorted_detections):
            x, y, w, h = detection['bbox']
            color = detection['color']
            
            # Dibujar bounding box
            cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 3)
            
            # Preparar texto para mostrar
            texts = []
            if show_class_names:
                texts.append(detection['class_name'])
            
            if show_confidence:
                conf_text = f"{detection['confidence']:.2%}"
                texts.append(conf_text)
            
            # Añadir información de ROI si existe
            if detection['roi_id'] is not None:
                texts.append(f"ROI {detection['roi_id']}")
            
            # Mostrar texto
            self._draw_text_box(display_image, (x, y - 10), texts, color)
            
            # Añadir número de detección
            cv2.circle(display_image, (x + w, y), 20, color, -1)
            cv2.putText(display_image, str(i + 1), (x + w - 10, y + 7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Añadir información general
        self._add_info_overlay(display_image)
        
        return display_image
    
    def _draw_text_box(self, image: np.ndarray, position: Tuple[int, int], 
                      texts: List[str], color: Tuple[int, int, int]):
        """Dibuja un cuadro de texto con fondo semitransparente."""
        x, y = position
        
        # Calcular tamaño del texto
        font_scale = 0.6
        font_thickness = 2
        line_height = 25
        
        max_width = 0
        text_heights = []
        
        for text in texts:
            (text_width, text_height), _ = cv2.getTextSize(text, self.font, font_scale, font_thickness)
            max_width = max(max_width, text_width)
            text_heights.append(text_height)
        
        # Dibujar fondo semitransparente
        overlay = image.copy()
        cv2.rectangle(overlay, (x - 5, y - line_height * len(texts) - 5), 
                     (x + max_width + 10, y + 10), color, -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Dibujar texto
        for i, text in enumerate(texts):
            text_y = y - (len(texts) - i - 1) * line_height
            cv2.putText(image, text, (x, text_y), self.font, font_scale, 
                       (255, 255, 255), font_thickness)
    
    def _add_info_overlay(self, image: np.ndarray):
        """Añade información general como overlay."""
        height, width = image.shape[:2]
        
        # Fondo semitransparente para información
        overlay = image.copy()
        cv2.rectangle(overlay, (10, height - 80), (350, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Información general
        info_texts = [
            f"C.A. Dupin - Análisis Visual",
            f"Detections: {len(self.detections)}",
            f"Confidence Threshold: {self.confidence_threshold:.2%}"
        ]
        
        for i, text in enumerate(info_texts):
            y = height - 60 + i * 20
            cv2.putText(image, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (255, 255, 255), 1)
    
    def show_detections(self, delay: int = 0) -> bool:
        """
        Muestra las detecciones en una ventana.
        
        Args:
            delay: Delay en milisegundos (0 = espera infinita)
            
        Returns:
            bool: True si se mostró correctamente
        """
        display_image = self.visualize_detections()
        
        if display_image is None:
            return False
        
        cv2.imshow(self.window_name, display_image)
        
        if delay > 0:
            cv2.waitKey(delay)
            cv2.destroyAllWindows()
        else:
            print("Presiona cualquier tecla para cerrar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return True
    
    def save_visualization(self, output_path: str, show_confidence: bool = True, 
                          show_class_names: bool = True) -> bool:
        """
        Guarda la visualización en un archivo.
        
        Args:
            output_path: Ruta donde guardar la imagen
            show_confidence: Si mostrar scores de confianza
            show_class_names: Si mostrar nombres de clases
            
        Returns:
            bool: True si se guardó correctamente
        """
        display_image = self.visualize_detections(show_confidence, show_class_names)
        
        if display_image is None:
            return False
        
        try:
            cv2.imwrite(output_path, display_image)
            print(f"✓ Visualización guardada: {output_path}")
            return True
        except Exception as e:
            print(f"Error guardando visualización: {e}")
            return False
    
    def create_heatmap(self, alpha: float = 0.4) -> np.ndarray:
        """Crea un mapa de calor de las detecciones."""
        if self.current_image is None:
            return None
        
        heatmap = np.zeros((self.current_image.shape[0], self.current_image.shape[1]), dtype=np.float32)
        
        for detection in self.detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            
            # Crear máscara para la región
            mask = np.zeros((self.current_image.shape[0], self.current_image.shape[1]), dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
            # Aplicar confianza a la máscara
            mask = mask.astype(np.float32) * confidence
            heatmap += mask
        
        # Normalizar heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Convertir a colores
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        
        # Superponer a la imagen original
        result = cv2.addWeighted(self.current_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return result
    
    def get_detection_summary(self) -> Dict:
        """Obtiene un resumen de las detecciones."""
        if not self.detections:
            return {'total': 0}
        
        classes = [d['class_name'] for d in self.detections]
        confidences = [d['confidence'] for d in self.detections]
        
        # Contar por clase
        class_counts = {}
        for cls in classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Estadísticas de confianza
        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        
        return {
            'total': len(self.detections),
            'classes': class_counts,
            'confidence_stats': {
                'average': avg_confidence,
                'minimum': min_confidence,
                'maximum': max_confidence
            },
            'high_confidence_detections': len([c for c in confidences if c > self.confidence_threshold])
        }
    
    def filter_detections(self, min_confidence: float = None, class_name: str = None) -> List[Dict]:
        """Filtra las detecciones según criterios."""
        filtered = self.detections.copy()
        
        if min_confidence is not None:
            filtered = [d for d in filtered if d['confidence'] >= min_confidence]
        
        if class_name is not None:
            filtered = [d for d in filtered if d['class_name'] == class_name]
        
        return filtered
    
    def compare_with_ground_truth(self, ground_truth: List[Dict]) -> Dict:
        """
        Compara las detecciones con una verdad de campo.
        
        Args:
            ground_truth: Lista de detecciones de verdad de campo con formato:
                         [{'class_name': str, 'bbox': (x,y,w,h), 'confidence': float}]
        
        Returns:
            Diccionario con métricas de evaluación
        """
        if not self.detections:
            return {'precision': 0, 'recall': 0, 'f1_score': 0}
        
        # Calcular IoU (Intersection over Union) para cada par
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        
        iou_threshold = 0.5
        
        # Marcar detecciones de verdad de campo como no evaluadas
        gt_evaluated = [False] * len(ground_truth)
        
        for detection in self.detections:
            best_iou = 0
            best_gt_idx = -1
            
            # Buscar la mejor coincidencia en la verdad de campo
            for i, gt in enumerate(ground_truth):
                if gt_evaluated[i]:
                    continue
                
                iou = self._calculate_iou(detection['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Evaluar si es un true positive o false positive
            if best_iou >= iou_threshold:
                tp += 1
                gt_evaluated[best_gt_idx] = True
            else:
                fp += 1
        
        # Contar false negatives
        fn = sum(1 for evaluated in gt_evaluated if not evaluated)
        
        # Calcular métricas
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'iou_threshold': iou_threshold
        }
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calcula la Intersección sobre Unión (IoU) entre dos bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Coordenadas de intersección
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        # Calcular áreas
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        # Calcular IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def export_analysis_report(self, output_path: str) -> bool:
        """Exporta un reporte completo del análisis."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'image_info': {
                'width': self.current_image.shape[1] if self.current_image is not None else 0,
                'height': self.current_image.shape[0] if self.current_image is not None else 0
            },
            'detections': self.detections,
            'summary': self.get_detection_summary(),
            'settings': {
                'confidence_threshold': self.confidence_threshold
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✓ Reporte de análisis guardado: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error guardando reporte: {e}")
            return False
    
    def load_analysis_from_file(self, file_path: str) -> bool:
        """Carga un análisis desde un archivo JSON."""
        if not Path(file_path).exists():
            print(f"Error: Archivo no encontrado: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.detections = data.get('detections', [])
            
            if 'settings' in data:
                self.confidence_threshold = data['settings'].get('confidence_threshold', 0.5)
            
            print(f"✓ Análisis cargado desde: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error cargando análisis: {e}")
            return False