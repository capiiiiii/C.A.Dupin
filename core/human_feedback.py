"""
Módulo para gestionar el loop de retroalimentación humana
"""

import os
import json
from pathlib import Path
from datetime import datetime
from .image_matcher import ImageMatcher


class HumanFeedbackLoop:
    """Gestiona el ciclo de retroalimentación humana para corrección de resultados."""
    
    def __init__(self, directorio_imagenes):
        self.directorio = Path(directorio_imagenes)
        self.matcher = ImageMatcher()
        self.feedback_history = []
        self.roi_feedback = {}  # ROI-specific feedback
        
        if not self.directorio.exists():
            raise ValueError(f"Directorio no encontrado: {directorio_imagenes}")
        
        self._load_feedback()
        self._load_roi_feedback()
    
    def _load_feedback(self):
        """Carga el feedback previo si existe."""
        feedback_file = self.directorio / "feedback.json"
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    self.feedback_history = json.load(f)
                print(f"Cargados {len(self.feedback_history)} registros de feedback previo")
            except Exception as e:
                print(f"Error cargando feedback: {e}")
                self.feedback_history = []
    
    def _load_roi_feedback(self):
        """Carga el feedback específico de ROIs si existe."""
        roi_feedback_file = self.directorio / "roi_feedback.json"
        if roi_feedback_file.exists():
            try:
                with open(roi_feedback_file, 'r') as f:
                    self.roi_feedback = json.load(f)
                print(f"Cargados {len(self.roi_feedback)} registros de feedback de ROI")
            except Exception as e:
                print(f"Error cargando feedback de ROI: {e}")
                self.roi_feedback = {}
    
    def _save_feedback(self):
        """Guarda el feedback en un archivo JSON."""
        feedback_file = self.directorio / "feedback.json"
        try:
            with open(feedback_file, 'w') as f:
                json.dump(self.feedback_history, f, indent=2, default=str)
        except Exception as e:
            print(f"Error guardando feedback: {e}")
    
    def _save_roi_feedback(self):
        """Guarda el feedback específico de ROIs en un archivo JSON."""
        roi_feedback_file = self.directorio / "roi_feedback.json"
        try:
            with open(roi_feedback_file, 'w') as f:
                json.dump(self.roi_feedback, f, indent=2, default=str)
        except Exception as e:
            print(f"Error guardando feedback de ROI: {e}")
    
    def start(self):
        """Inicia el loop de retroalimentación interactivo."""
        print("=== Loop de Retroalimentación Humana ===\n")
        print("Este módulo permite revisar y corregir coincidencias en tiempo real.")
        print("La decisión final siempre es humana.\n")
        
        imagenes = self._listar_imagenes()
        
        if len(imagenes) < 2:
            print("Se necesitan al menos 2 imágenes para comparar.")
            return
        
        print(f"Encontradas {len(imagenes)} imágenes.\n")
        
        while True:
            print("\nOpciones:")
            print("1. Comparar dos imágenes")
            print("2. Buscar similares")
            print("3. Ver historial de feedback")
            print("4. Exportar feedback para entrenamiento")
            print("5. Salir")
            
            try:
                opcion = input("\nSeleccione una opción (1-5): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nFinalizando sesión...")
                break
            
            if opcion == '1':
                self._comparar_interactivo(imagenes)
            elif opcion == '2':
                self._buscar_similares(imagenes)
            elif opcion == '3':
                self._mostrar_historial()
            elif opcion == '4':
                self._exportar_feedback()
            elif opcion == '5':
                print("Finalizando sesión...")
                break
            else:
                print("Opción no válida.")
    
    def _listar_imagenes(self):
        """Lista todas las imágenes en el directorio."""
        extensiones = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        imagenes = []
        
        for archivo in self.directorio.rglob('*'):
            if archivo.suffix.lower() in extensiones:
                imagenes.append(archivo)
        
        return sorted(imagenes)
    
    def _comparar_interactivo(self, imagenes):
        """Compara dos imágenes seleccionadas por el usuario."""
        print("\nImágenes disponibles:")
        for i, img in enumerate(imagenes):
            print(f"{i + 1}. {img.name}")
        
        try:
            idx1 = int(input("\nSeleccione primera imagen (número): ")) - 1
            idx2 = int(input("Seleccione segunda imagen (número): ")) - 1
            
            if 0 <= idx1 < len(imagenes) and 0 <= idx2 < len(imagenes) and idx1 != idx2:
                img1 = imagenes[idx1]
                img2 = imagenes[idx2]
                
                print(f"\nComparando:")
                print(f"  {img1.name}")
                print(f"  {img2.name}")
                
                similitud = self.matcher.compare(str(img1), str(img2))
                print(f"\nSimilitud calculada: {similitud:.2%}")
                
                feedback = input("\n¿Es correcta esta evaluación? (s/n): ").strip().lower()
                
                if feedback in ['s', 'si', 'sí', 'y', 'yes']:
                    print("✓ Feedback registrado: CORRECTO")
                    self.feedback_history.append({
                        'img1': str(img1),
                        'img2': str(img2),
                        'similitud': similitud,
                        'correcto': True
                    })
                    self._save_feedback()
                elif feedback in ['n', 'no']:
                    similitud_real = input("¿Cuál sería la similitud correcta? (0.0-1.0): ").strip()
                    try:
                        similitud_real = float(similitud_real)
                        print("✓ Feedback registrado: CORREGIDO")
                        self.feedback_history.append({
                            'img1': str(img1),
                            'img2': str(img2),
                            'similitud_calculada': similitud,
                            'similitud_real': similitud_real,
                            'correcto': False
                        })
                        self._save_feedback()
                    except ValueError:
                        print("Valor no válido.")
            else:
                print("Selección no válida.")
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nOperación cancelada.")
    
    def _buscar_similares(self, imagenes):
        """Busca imágenes similares a una seleccionada."""
        print("\nImágenes disponibles:")
        for i, img in enumerate(imagenes):
            print(f"{i + 1}. {img.name}")
        
        try:
            idx = int(input("\nSeleccione imagen de referencia (número): ")) - 1
            
            if 0 <= idx < len(imagenes):
                query_img = imagenes[idx]
                otras_imagenes = [img for i, img in enumerate(imagenes) if i != idx]
                
                print(f"\nBuscando imágenes similares a: {query_img.name}")
                print("Procesando...")
                
                resultados = self.matcher.find_matches(
                    str(query_img),
                    [str(img) for img in otras_imagenes],
                    top_n=5
                )
                
                print("\nTop 5 coincidencias:")
                for i, (img_path, score) in enumerate(resultados, 1):
                    print(f"{i}. {Path(img_path).name} - Similitud: {score:.2%}")
            else:
                print("Selección no válida.")
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nOperación cancelada.")
    
    def _mostrar_historial(self):
        """Muestra el historial de feedback."""
        if not self.feedback_history:
            print("\nNo hay feedback registrado aún.")
            return
        
        print(f"\n=== Historial de Feedback ({len(self.feedback_history)} entradas) ===\n")
        
        for i, entry in enumerate(self.feedback_history, 1):
            print(f"{i}. {entry['img1']} vs {entry['img2']}")
            if entry['correcto']:
                print(f"   Similitud: {entry['similitud']:.2%} - ✓ CORRECTO")
            else:
                print(f"   Calculada: {entry['similitud_calculada']:.2%}")
                print(f"   Correcta: {entry['similitud_real']:.2%} - ✗ CORREGIDO")
            print()
    
    def _exportar_feedback(self):
        """Exporta el feedback para ser usado en entrenamiento de modelo."""
        if not self.feedback_history:
            print("\nNo hay feedback para exportar.")
            return
        
        feedback_file = self.directorio / "feedback.json"
        print(f"\nFeedback guardado en: {feedback_file}")
        print(f"Total de registros: {len(self.feedback_history)}")
        print("\nPuedes usar este archivo para ajustar el modelo con:")
        print("python dupin.py ajustar --feedback feedback.json --modelo modelo.pth")
    
    def get_feedback_data(self):
        """Retorna los datos de feedback para uso externo."""
        return self.feedback_history
    
    def add_roi_feedback(self, image_path, roi, comparison_result, is_correct, correction=None):
        """
        Añade feedback específico para una comparación de ROI.
        
        Args:
            image_path: Ruta a la imagen
            roi: Región de interés (x, y, w, h)
            comparison_result: Resultado de la comparación
            is_correct: Si el resultado fue correcto
            correction: Corrección si no fue correcto
        """
        feedback_key = f"{Path(image_path).name}_{roi[0]}_{roi[1]}_{roi[2]}_{roi[3]}"
        
        feedback_entry = {
            'image_path': str(image_path),
            'roi': roi,
            'comparison_result': comparison_result,
            'is_correct': is_correct,
            'correction': correction,
            'timestamp': datetime.now().isoformat()
        }
        
        self.roi_feedback[feedback_key] = feedback_entry
        self._save_roi_feedback()
        print(f"✓ Feedback de ROI registrado para {feedback_key}")
    
    def approve_pattern(self, image_path, roi=None, pattern_type='general'):
        """
        Aprueba un patrón detectado para aprendizaje futuro.
        
        Args:
            image_path: Ruta a la imagen
            roi: Región de interés (opcional)
            pattern_type: Tipo de patrón
        """
        feedback_entry = {
            'type': 'approval',
            'image_path': str(image_path),
            'roi': roi,
            'pattern_type': pattern_type,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_history.append(feedback_entry)
        self._save_feedback()
        print(f"✓ Patrón aprobado: {pattern_type}")
    
    def correct_pattern(self, image_path, roi=None, correction='', pattern_type='general'):
        """
        Corrige un patrón detectado para aprendizaje futuro.
        
        Args:
            image_path: Ruta a la imagen
            roi: Región de interés (opcional)
            correction: Texto de corrección
            pattern_type: Tipo de patrón
        """
        feedback_entry = {
            'type': 'correction',
            'image_path': str(image_path),
            'roi': roi,
            'pattern_type': pattern_type,
            'correction': correction,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_history.append(feedback_entry)
        self._save_feedback()
        print(f"✓ Patrón corregido: {pattern_type} -> {correction}")
    
    def get_roi_statistics(self):
        """
        Obtiene estadísticas del feedback de ROIs.
        
        Returns:
            dict: Estadísticas de feedback de ROIs
        """
        total = len(self.roi_feedback)
        approved = sum(1 for f in self.roi_feedback.values() if f.get('is_correct', False))
        corrected = total - approved
        
        return {
            'total_feedback': total,
            'approved': approved,
            'corrected': corrected,
            'approval_rate': approved / total if total > 0 else 0.0
        }
    
    def export_learning_data(self, output_path='learning_data.json'):
        """
        Exporta todos los datos de feedback para aprendizaje de modelos.
        
        Args:
            output_path: Ruta del archivo de salida
        """
        learning_data = {
            'image_feedback': self.feedback_history,
            'roi_feedback': self.roi_feedback,
            'statistics': {
                'total_image_feedback': len(self.feedback_history),
                'total_roi_feedback': len(self.roi_feedback),
                'roi_stats': self.get_roi_statistics()
            },
            'exported_at': datetime.now().isoformat()
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(learning_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"✓ Datos de aprendizaje exportados a: {output_path}")
        except Exception as e:
            print(f"Error exportando datos de aprendizaje: {e}")
    
    def batch_approve_corrections(self, image_paths, corrections):
        """
        Aprueba múltiples correcciones en lote.
        
        Args:
            image_paths: Lista de rutas de imágenes
            corrections: Lista de correcciones correspondientes
        """
        for img_path, correction in zip(image_paths, corrections):
            self.correct_pattern(img_path, correction=correction)
        print(f"✓ {len(image_paths)} correcciones aprobadas en lote")