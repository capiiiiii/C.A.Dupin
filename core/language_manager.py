"""
Módulo para gestión multiidioma de C.A. Dupin
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


class LanguageManager:
    """Gestor de idiomas para soporte multiidioma en C.A. Dupin."""
    
    def __init__(self, language: str = 'es'):
        """
        Inicializa el gestor de idiomas.
        
        Args:
            language: Código del idioma ('es', 'en', 'fr', etc.)
        """
        self.current_language = language
        self.translations = {}
        self.available_languages = {}
        self.translation_files = {}
        self._load_translations()
    
    def _load_translations(self):
        """Carga todas las traducciones disponibles."""
        # Rutas donde buscar archivos de traducción
        base_path = Path(__file__).parent.parent
        translations_path = base_path / 'translations'
        
        # Crear directorio si no existe
        translations_path.mkdir(exist_ok=True)
        
        # Cargar traducciones para todos los idiomas disponibles
        self._load_language_translations(translations_path)
        
        # Si no hay traducciones, crear archivo base
        if not self.translations:
            self._create_default_translations(translations_path)
    
    def _load_language_translations(self, translations_path: Path):
        """Carga las traducciones para todos los idiomas."""
        if not translations_path.exists():
            return
        
        # Buscar archivos JSON de traducción
        for lang_file in translations_path.glob('*.json'):
            lang_code = lang_file.stem
            
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                
                self.translations[lang_code] = translations
                self.available_languages[lang_code] = translations.get('meta', {}).get('name', lang_code.upper())
                self.translation_files[lang_code] = str(lang_file)
                
            except Exception as e:
                print(f"Error cargando traducciones para {lang_code}: {e}")
    
    def _create_default_translations(self, translations_path: Path):
        """Crea archivos de traducción por defecto."""
        # Traducciones por defecto en español
        default_translations = {
            'meta': {
                'name': 'Español',
                'code': 'es',
                'description': 'Español - C.A. Dupin'
            },
            'interface': {
                'main_title': 'C.A. Dupin - Análisis Visual Inteligente',
                'loading': 'Cargando...',
                'processing': 'Procesando...',
                'complete': 'Completado',
                'error': 'Error',
                'cancel': 'Cancelar',
                'save': 'Guardar',
                'load': 'Cargar',
                'settings': 'Configuración',
                'help': 'Ayuda',
                'exit': 'Salir'
            },
            'roi': {
                'title': 'Selección de Regiones de Interés',
                'instructions': 'Instrucciones',
                'drag_mouse': 'Arrastra el mouse para seleccionar una región',
                'press_n': "Presiona 'n' para siguiente región",
                'press_c': "Presiona 'c' para continuar sin más regiones",
                'press_r': "Presiona 'r' para reiniciar selección",
                'press_esc': "Presiona 'ESC' para cancelar",
                'no_rois': 'No se seleccionaron ROI',
                'rois_selected': 'ROI seleccionada(s)',
                'visualization': 'Visualización de ROI'
            },
            'camera': {
                'title': 'Cámara en Vivo',
                'initializing': 'Inicializando cámara...',
                'capture_started': 'Captura iniciada',
                'capture_stopped': 'Captura detenida',
                'photo_saved': 'Foto guardada',
                'recording_started': 'Grabación iniciada',
                'recording_stopped': 'Grabación detenida',
                'motion_detected': 'Movimiento detectado',
                'no_cameras': 'No se encontraron cámaras'
            },
            'detections': {
                'analyzing': 'Analizando imagen...',
                'detections_found': 'Detecciones encontradas',
                'confidence': 'Confianza',
                'class_name': 'Clase',
                'bounding_box': 'Bounding Box',
                'high_confidence': 'Alta confianza',
                'low_confidence': 'Baja confianza',
                'no_detections': 'No se encontraron detecciones'
            },
            'feedback': {
                'title': 'Retroalimentación Humana',
                'correct': '¿Es correcta esta evaluación?',
                'similarity': 'Similitud calculada',
                'correct_similarity': '¿Cuál sería la similitud correcta?',
                'feedback_saved': 'Feedback guardado',
                'history': 'Historial de feedback',
                'export_feedback': 'Exportar feedback'
            },
            'training': {
                'title': 'Entrenamiento de Modelo',
                'initializing': 'Inicializando entrenamiento...',
                'epoch': 'Época',
                'loss': 'Pérdida',
                'completed': 'Entrenamiento completado',
                'saving_model': 'Guardando modelo...',
                'loading_model': 'Cargando modelo...',
                'fine_tuning': 'Ajuste fino con feedback'
            },
            'modules': {
                'title': 'Módulos de Reconocimiento',
                'faces': 'Rostros',
                'stars': 'Estrellas y Cuerpos Celestes',
                'currency': 'Billetes y Patrones Monetarios',
                'humans': 'Cuerpos y Siluetas Humanas',
                'animals': 'Animales',
                'plants': 'Plantas',
                'custom': 'Objetos Personalizados',
                'not_trained': 'No entrenado',
                'trained': 'Entrenado',
                'accuracy': 'Precisión'
            },
            'errors': {
                'image_not_found': 'Imagen no encontrada',
                'model_not_loaded': 'Modelo no cargado',
                'invalid_roi': 'ROI inválida',
                'camera_error': 'Error de cámara',
                'processing_error': 'Error de procesamiento',
                'file_error': 'Error de archivo'
            },
            'controls': {
                'start': 'Iniciar',
                'stop': 'Detener',
                'pause': 'Pausar',
                'reset': 'Reiniciar',
                'next': 'Siguiente',
                'previous': 'Anterior',
                'zoom_in': 'Acercar',
                'zoom_out': 'Alejar',
                'fullscreen': 'Pantalla completa'
            }
        }
        
        # Guardar traducción por defecto (español)
        es_file = translations_path / 'es.json'
        with open(es_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, indent=2, ensure_ascii=False)
        
        # Crear traducción en inglés como ejemplo
        en_translations = {
            'meta': {
                'name': 'English',
                'code': 'en',
                'description': 'English - C.A. Dupin'
            },
            'interface': {
                'main_title': 'C.A. Dupin - Intelligent Visual Analysis',
                'loading': 'Loading...',
                'processing': 'Processing...',
                'complete': 'Complete',
                'error': 'Error',
                'cancel': 'Cancel',
                'save': 'Save',
                'load': 'Load',
                'settings': 'Settings',
                'help': 'Help',
                'exit': 'Exit'
            },
            'roi': {
                'title': 'Region of Interest Selection',
                'instructions': 'Instructions',
                'drag_mouse': 'Drag mouse to select a region',
                'press_n': "Press 'n' for next region",
                'press_c': "Press 'c' to continue without more regions",
                'press_r': "Press 'r' to reset selection",
                'press_esc': "Press 'ESC' to cancel",
                'no_rois': 'No ROIs selected',
                'rois_selected': 'ROI(s) selected',
                'visualization': 'ROI Visualization'
            },
            'camera': {
                'title': 'Live Camera',
                'initializing': 'Initializing camera...',
                'capture_started': 'Capture started',
                'capture_stopped': 'Capture stopped',
                'photo_saved': 'Photo saved',
                'recording_started': 'Recording started',
                'recording_stopped': 'Recording stopped',
                'motion_detected': 'Motion detected',
                'no_cameras': 'No cameras found'
            },
            'detections': {
                'analyzing': 'Analyzing image...',
                'detections_found': 'Detections found',
                'confidence': 'Confidence',
                'class_name': 'Class',
                'bounding_box': 'Bounding Box',
                'high_confidence': 'High confidence',
                'low_confidence': 'Low confidence',
                'no_detections': 'No detections found'
            },
            'feedback': {
                'title': 'Human Feedback',
                'correct': 'Is this evaluation correct?',
                'similarity': 'Calculated similarity',
                'correct_similarity': 'What would be the correct similarity?',
                'feedback_saved': 'Feedback saved',
                'history': 'Feedback history',
                'export_feedback': 'Export feedback'
            },
            'training': {
                'title': 'Model Training',
                'initializing': 'Initializing training...',
                'epoch': 'Epoch',
                'loss': 'Loss',
                'completed': 'Training completed',
                'saving_model': 'Saving model...',
                'loading_model': 'Loading model...',
                'fine_tuning': 'Fine-tuning with feedback'
            },
            'modules': {
                'title': 'Recognition Modules',
                'faces': 'Faces',
                'stars': 'Stars and Celestial Bodies',
                'currency': 'Banknotes and Monetary Patterns',
                'humans': 'Human Bodies and Silhouettes',
                'animals': 'Animals',
                'plants': 'Plants',
                'custom': 'Custom Objects',
                'not_trained': 'Not trained',
                'trained': 'Trained',
                'accuracy': 'Accuracy'
            },
            'errors': {
                'image_not_found': 'Image not found',
                'model_not_loaded': 'Model not loaded',
                'invalid_roi': 'Invalid ROI',
                'camera_error': 'Camera error',
                'processing_error': 'Processing error',
                'file_error': 'File error'
            },
            'controls': {
                'start': 'Start',
                'stop': 'Stop',
                'pause': 'Pause',
                'reset': 'Reset',
                'next': 'Next',
                'previous': 'Previous',
                'zoom_in': 'Zoom in',
                'zoom_out': 'Zoom out',
                'fullscreen': 'Fullscreen'
            }
        }
        
        en_file = translations_path / 'en.json'
        with open(en_file, 'w', encoding='utf-8') as f:
            json.dump(en_translations, f, indent=2, ensure_ascii=False)
        
        # Cargar las traducciones recién creadas
        self.translations = {'es': default_translations, 'en': en_translations}
        self.available_languages = {'es': 'Español', 'en': 'English'}
        self.translation_files = {'es': str(es_file), 'en': str(en_file)}
        
        print(f"✓ Archivos de traducción creados en: {translations_path}")
    
    def set_language(self, language_code: str) -> bool:
        """
        Cambia el idioma actual.
        
        Args:
            language_code: Código del idioma ('es', 'en', 'fr', etc.)
            
        Returns:
            bool: True si el idioma se cambió correctamente
        """
        if language_code not in self.translations:
            print(f"Idioma no disponible: {language_code}")
            print(f"Idiomas disponibles: {list(self.available_languages.keys())}")
            return False
        
        self.current_language = language_code
        print(f"✓ Idioma cambiado a: {self.available_languages.get(language_code, language_code)}")
        return True
    
    def get_text(self, key: str, category: str = 'interface') -> str:
        """
        Obtiene el texto traducido.
        
        Args:
            key: Clave del texto
            category: Categoría del texto ('interface', 'roi', 'camera', etc.)
            
        Returns:
            Texto traducido o la clave si no se encuentra
        """
        if self.current_language not in self.translations:
            return key
        
        category_translations = self.translations[self.current_language].get(category, {})
        return category_translations.get(key, key)
    
    def get_available_languages(self) -> Dict[str, str]:
        """Obtiene los idiomas disponibles."""
        return self.available_languages.copy()
    
    def add_translation(self, language_code: str, category: str, key: str, value: str) -> bool:
        """
        Añade una nueva traducción.
        
        Args:
            language_code: Código del idioma
            category: Categoría del texto
            key: Clave del texto
            value: Valor traducido
            
        Returns:
            bool: True si se añadió correctamente
        """
        if language_code not in self.translations:
            self.translations[language_code] = {}
        
        if category not in self.translations[language_code]:
            self.translations[language_code][category] = {}
        
        self.translations[language_code][category][key] = value
        return True
    
    def remove_translation(self, language_code: str, category: str, key: str) -> bool:
        """
        Elimina una traducción.
        
        Args:
            language_code: Código del idioma
            category: Categoría del texto
            key: Clave del texto
            
        Returns:
            bool: True si se eliminó correctamente
        """
        if (language_code in self.translations and 
            category in self.translations[language_code] and
            key in self.translations[language_code][category]):
            
            del self.translations[language_code][category][key]
            return True
        
        return False
    
    def save_translations(self, language_code: str = None) -> bool:
        """
        Guarda las traducciones en archivos.
        
        Args:
            language_code: Idioma específico a guardar (None para todos)
            
        Returns:
            bool: True si se guardaron correctamente
        """
        base_path = Path(__file__).parent.parent
        translations_path = base_path / 'translations'
        translations_path.mkdir(exist_ok=True)
        
        try:
            if language_code:
                # Guardar idioma específico
                if language_code in self.translations:
                    file_path = translations_path / f'{language_code}.json'
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.translations[language_code], f, indent=2, ensure_ascii=False)
                    print(f"✓ Traducciones guardadas: {file_path}")
            else:
                # Guardar todos los idiomas
                for lang_code, translations in self.translations.items():
                    file_path = translations_path / f'{lang_code}.json'
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(translations, f, indent=2, ensure_ascii=False)
                print(f"✓ Todas las traducciones guardadas en: {translations_path}")
            
            return True
            
        except Exception as e:
            print(f"Error guardando traducciones: {e}")
            return False
    
    def export_translation_template(self, output_path: str) -> bool:
        """
        Exporta una plantilla de traducción vacía.
        
        Args:
            output_path: Ruta donde guardar la plantilla
            
        Returns:
            bool: True si se exportó correctamente
        """
        template = {
            'meta': {
                'name': 'Nombre del Idioma',
                'code': 'código_idioma',
                'description': 'Descripción del idioma'
            },
            'interface': {},
            'roi': {},
            'camera': {},
            'detections': {},
            'feedback': {},
            'training': {},
            'modules': {},
            'errors': {},
            'controls': {}
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Plantilla de traducción exportada: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exportando plantilla: {e}")
            return False
    
    def import_translation(self, file_path: str, language_code: str = None) -> bool:
        """
        Importa traducciones desde un archivo.
        
        Args:
            file_path: Ruta del archivo de traducciones
            language_code: Código del idioma (obtenido del archivo si es None)
            
        Returns:
            bool: True si se importó correctamente
        """
        if not Path(file_path).exists():
            print(f"Error: Archivo no encontrado: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            # Obtener código del idioma
            if language_code is None:
                language_code = translations.get('meta', {}).get('code')
            
            if not language_code:
                print("Error: No se pudo determinar el código del idioma")
                return False
            
            self.translations[language_code] = translations
            self.available_languages[language_code] = translations.get('meta', {}).get('name', language_code.upper())
            self.translation_files[language_code] = file_path
            
            print(f"✓ Traducciones importadas para: {language_code}")
            return True
            
        except Exception as e:
            print(f"Error importando traducciones: {e}")
            return False
    
    def get_translation_stats(self) -> Dict:
        """Obtiene estadísticas de las traducciones."""
        stats = {
            'current_language': self.current_language,
            'available_languages': len(self.available_languages),
            'language_details': {}
        }
        
        for lang_code, translations in self.translations.items():
            category_counts = {}
            total_keys = 0
            
            for category, category_translations in translations.items():
                if isinstance(category_translations, dict):
                    count = len(category_translations)
                    category_counts[category] = count
                    total_keys += count
            
            stats['language_details'][lang_code] = {
                'name': self.available_languages.get(lang_code, lang_code),
                'total_keys': total_keys,
                'categories': category_counts
            }
        
        return stats
    
    def __str__(self) -> str:
        return f"LanguageManager(current: {self.current_language}, available: {list(self.available_languages.keys())})"
    
    def __repr__(self) -> str:
        return self.__str__()