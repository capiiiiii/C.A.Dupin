"""
Módulo de gestión y coordinación de módulos de reconocimiento
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from .modules import (
    BaseRecognitionModule, 
    FaceRecognitionModule,
    StarRecognitionModule, 
    CurrencyRecognitionModule,
    HumanBodyModule,
    AnimalRecognitionModule,
    PlantRecognitionModule,
    CustomObjectModule
)


class ModuleManager:
    """Gestor principal de módulos de reconocimiento."""
    
    def __init__(self, modules_config_path: str = None):
        """
        Inicializa el gestor de módulos.
        
        Args:
            modules_config_path: Ruta al archivo de configuración de módulos
        """
        self.modules = {}
        self.active_modules = set()
        self.config_path = modules_config_path
        self.global_config = {
            'confidence_threshold': 0.7,
            'enable_roi_processing': True,
            'enable_feedback_integration': True,
            'batch_processing': False,
            'max_detections_per_image': 20
        }
        
        # Inicializar módulos base
        self._initialize_default_modules()
        
        # Cargar configuración si existe
        if modules_config_path and Path(modules_config_path).exists():
            self.load_configuration(modules_config_path)
    
    def _initialize_default_modules(self):
        """Inicializa los módulos de reconocimiento base."""
        # Crear instancias de todos los módulos base
        base_modules = [
            FaceRecognitionModule(),
            StarRecognitionModule(),
            CurrencyRecognitionModule(),
            HumanBodyModule(),
            AnimalRecognitionModule(),
            PlantRecognitionModule(),
            CustomObjectModule()
        ]
        
        for module in base_modules:
            self.register_module(module)
        
        print(f"✓ {len(base_modules)} módulos base inicializados")
    
    def register_module(self, module: BaseRecognitionModule) -> bool:
        """
        Registra un nuevo módulo de reconocimiento.
        
        Args:
            module: Instancia del módulo a registrar
            
        Returns:
            bool: True si se registró correctamente
        """
        if module.module_id in self.modules:
            print(f"⚠️  Módulo '{module.module_id}' ya está registrado")
            return False
        
        self.modules[module.module_id] = module
        print(f"✓ Módulo registrado: {module.name} ({module.module_id})")
        return True
    
    def unregister_module(self, module_id: str) -> bool:
        """
        Desregistra un módulo.
        
        Args:
            module_id: ID del módulo a desregistrar
            
        Returns:
            bool: True si se desregistró correctamente
        """
        if module_id not in self.modules:
            print(f"⚠️  Módulo '{module_id}' no encontrado")
            return False
        
        # Desactivar módulo si está activo
        if module_id in self.active_modules:
            self.deactivate_module(module_id)
        
        del self.modules[module_id]
        print(f"✓ Módulo desregistrado: {module_id}")
        return True
    
    def activate_module(self, module_id: str) -> bool:
        """
        Activa un módulo para procesamiento.
        
        Args:
            module_id: ID del módulo a activar
            
        Returns:
            bool: True si se activó correctamente
        """
        if module_id not in self.modules:
            print(f"⚠️  Módulo '{module_id}' no encontrado")
            return False
        
        self.active_modules.add(module_id)
        print(f"✓ Módulo activado: {module_id}")
        return True
    
    def deactivate_module(self, module_id: str) -> bool:
        """
        Desactiva un módulo.
        
        Args:
            module_id: ID del módulo a desactivar
            
        Returns:
            bool: True si se desactivó correctamente
        """
        if module_id not in self.modules:
            print(f"⚠️  Módulo '{module_id}' no encontrado")
            return False
        
        self.active_modules.discard(module_id)
        print(f"✓ Módulo desactivado: {module_id}")
        return True
    
    def get_active_modules(self) -> List[str]:
        """Obtiene la lista de módulos activos."""
        return list(self.active_modules)
    
    def get_available_modules(self) -> List[Dict]:
        """Obtiene información de todos los módulos disponibles."""
        modules_info = []
        
        for module_id, module in self.modules.items():
            info = module.get_info()
            info['is_active'] = module_id in self.active_modules
            modules_info.append(info)
        
        return modules_info
    
    def train_module(self, module_id: str, data_path: str, **kwargs) -> bool:
        """
        Entrena un módulo específico.
        
        Args:
            module_id: ID del módulo a entrenar
            data_path: Ruta a los datos de entrenamiento
            **kwargs: Argumentos adicionales
            
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        if module_id not in self.modules:
            print(f"⚠️  Módulo '{module_id}' no encontrado")
            return False
        
        module = self.modules[module_id]
        success = module.train(data_path, **kwargs)
        
        if success:
            # Guardar modelo entrenado
            model_path = f"{module_id}_model.json"
            module.save_model(model_path)
        
        return success
    
    def train_all_modules(self, data_paths: Dict[str, str], **kwargs) -> Dict[str, bool]:
        """
        Entrena todos los módulos activos.
        
        Args:
            data_paths: Diccionario con module_id -> data_path
            **kwargs: Argumentos adicionales
            
        Returns:
            Diccionario con resultados de entrenamiento
        """
        results = {}
        
        print(f"\n🚀 Iniciando entrenamiento de {len(self.active_modules)} módulos...")
        
        for module_id in self.active_modules:
            if module_id in data_paths:
                print(f"\n--- Entrenando {module_id} ---")
                results[module_id] = self.train_module(module_id, data_paths[module_id], **kwargs)
            else:
                print(f"⚠️  No hay datos de entrenamiento para {module_id}")
                results[module_id] = False
        
        return results
    
    def predict(self, image_input, active_only: bool = True, **kwargs) -> Dict[str, List[Dict]]:
        """
        Realiza predicciones con los módulos activos.
        
        Args:
            image_input: Imagen o path de imagen
            active_only: Si usar solo módulos activos
            **kwargs: Argumentos adicionales
            
        Returns:
            Diccionario con module_id -> lista de predicciones
        """
        all_predictions = {}
        
        # Determinar qué módulos usar
        modules_to_use = self.active_modules if active_only else self.modules.keys()
        
        if not modules_to_use:
            print("⚠️  No hay módulos activos para realizar predicciones")
            return all_predictions
        
        print(f"\n🔍 Realizando análisis con {len(modules_to_use)} módulos...")
        
        for module_id in modules_to_use:
            if module_id not in self.modules:
                continue
            
            try:
                module = self.modules[module_id]
                predictions = module.predict(image_input, **kwargs)
                
                # Filtrar por umbral de confianza
                if predictions:
                    min_confidence = self.global_config.get('confidence_threshold', 0.7)
                    filtered_predictions = [
                        pred for pred in predictions 
                        if pred.get('confidence', 0) >= min_confidence
                    ]
                    all_predictions[module_id] = filtered_predictions
                    
                    print(f"  {module.name}: {len(filtered_predictions)} detecciones")
                else:
                    all_predictions[module_id] = []
                    print(f"  {module.name}: Sin detecciones")
                    
            except Exception as e:
                print(f"  ❌ Error en {module_id}: {e}")
                all_predictions[module_id] = []
        
        return all_predictions
    
    def batch_predict(self, image_inputs: List, **kwargs) -> List[Dict[str, List[Dict]]]:
        """
        Realiza predicciones en lote.
        
        Args:
            image_inputs: Lista de imágenes o paths
            **kwargs: Argumentos adicionales
            
        Returns:
            Lista de diccionarios con predicciones para cada imagen
        """
        results = []
        
        print(f"\n📦 Procesando {len(image_inputs)} imágenes en lote...")
        
        for i, image_input in enumerate(image_inputs):
            print(f"\n--- Imagen {i+1}/{len(image_inputs)} ---")
            predictions = self.predict(image_input, **kwargs)
            results.append(predictions)
        
        return results
    
    def evaluate_module(self, module_id: str, test_data_path: str) -> float:
        """
        Evalúa un módulo específico.
        
        Args:
            module_id: ID del módulo a evaluar
            test_data_path: Ruta a los datos de prueba
            
        Returns:
            Precisión del módulo
        """
        if module_id not in self.modules:
            print(f"⚠️  Módulo '{module_id}' no encontrado")
            return 0.0
        
        module = self.modules[module_id]
        accuracy = module.evaluate(test_data_path)
        
        print(f"📊 Precisión de {module.name}: {accuracy:.2%}")
        return accuracy
    
    def evaluate_all_modules(self, test_data_paths: Dict[str, str]) -> Dict[str, float]:
        """
        Evalúa todos los módulos con datos de prueba.
        
        Args:
            test_data_paths: Diccionario con module_id -> test_data_path
            
        Returns:
            Diccionario con precisiones por módulo
        """
        results = {}
        
        print(f"\n📈 Evaluando {len(self.modules)} módulos...")
        
        for module_id, module in self.modules.items():
            if module_id in test_data_paths:
                print(f"\n--- Evaluando {module_id} ---")
                accuracy = self.evaluate_module(module_id, test_data_paths[module_id])
                results[module_id] = accuracy
            else:
                print(f"⚠️  No hay datos de prueba para {module_id}")
                results[module_id] = 0.0
        
        return results
    
    def create_custom_module(self, module_id: str, name: str, description: str = "",
                           base_class: type = CustomObjectModule) -> bool:
        """
        Crea un nuevo módulo personalizado.
        
        Args:
            module_id: ID único del módulo
            name: Nombre del módulo
            description: Descripción del módulo
            base_class: Clase base para el módulo
            
        Returns:
            bool: True si se creó correctamente
        """
        if module_id in self.modules:
            print(f"⚠️  Módulo '{module_id}' ya existe")
            return False
        
        try:
            # Crear instancia del módulo personalizado
            custom_module = base_class()
            custom_module.module_id = module_id
            custom_module.name = name
            custom_module.description = description
            
            self.register_module(custom_module)
            print(f"✓ Módulo personalizado creado: {name} ({module_id})")
            return True
            
        except Exception as e:
            print(f"❌ Error creando módulo personalizado: {e}")
            return False
    
    def save_configuration(self, output_path: str = None) -> bool:
        """
        Guarda la configuración actual de módulos.
        
        Args:
            output_path: Ruta donde guardar la configuración
            
        Returns:
            bool: True si se guardó correctamente
        """
        if output_path is None:
            output_path = "modules_config.json"
        
        config = {
            'active_modules': list(self.active_modules),
            'global_config': self.global_config,
            'modules': {}
        }
        
        for module_id, module in self.modules.items():
            config['modules'][module_id] = module.get_info()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            self.config_path = output_path
            print(f"✓ Configuración guardada: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error guardando configuración: {e}")
            return False
    
    def load_configuration(self, config_path: str) -> bool:
        """
        Carga configuración desde archivo.
        
        Args:
            config_path: Ruta del archivo de configuración
            
        Returns:
            bool: True si se cargó correctamente
        """
        if not Path(config_path).exists():
            print(f"⚠️  Archivo de configuración no encontrado: {config_path}")
            return False
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Cargar configuración global
            self.global_config.update(config.get('global_config', {}))
            
            # Cargar módulos activos
            self.active_modules = set(config.get('active_modules', []))
            
            # Cargar información de módulos
            modules_info = config.get('modules', {})
            for module_id, info in modules_info.items():
                if module_id in self.modules:
                    self.modules[module_id].config.update(info.get('config', {}))
            
            self.config_path = config_path
            print(f"✓ Configuración cargada: {config_path}")
            print(f"  Módulos activos: {len(self.active_modules)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando configuración: {e}")
            return False
    
    def get_module_statistics(self) -> Dict:
        """Obtiene estadísticas de todos los módulos."""
        stats = {
            'total_modules': len(self.modules),
            'active_modules': len(self.active_modules),
            'trained_modules': 0,
            'avg_accuracy': 0.0,
            'modules_by_type': {},
            'module_details': {}
        }
        
        total_accuracy = 0.0
        trained_count = 0
        
        for module_id, module in self.modules.items():
            info = module.get_info()
            stats['module_details'][module_id] = info
            
            if module.is_trained:
                stats['trained_modules'] += 1
                total_accuracy += module.accuracy
                trained_count += 1
            
            # Categorizar por tipo
            module_type = module.__class__.__name__.replace('Module', '').lower()
            if module_type not in stats['modules_by_type']:
                stats['modules_by_type'][module_type] = 0
            stats['modules_by_type'][module_type] += 1
        
        stats['avg_accuracy'] = total_accuracy / trained_count if trained_count > 0 else 0.0
        
        return stats
    
    def export_module_report(self, output_path: str = "modules_report.json") -> bool:
        """
        Exporta un reporte completo de módulos.
        
        Args:
            output_path: Ruta donde guardar el reporte
            
        Returns:
            bool: True si se exportó correctamente
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_module_statistics(),
            'modules': self.get_available_modules(),
            'active_modules': list(self.active_modules),
            'global_config': self.global_config
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✓ Reporte de módulos exportado: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error exportando reporte: {e}")
            return False
    
    def reset_all_modules(self):
        """Reinicia todos los módulos (desentrena)."""
        for module in self.modules.values():
            module.is_trained = False
            module.accuracy = 0.0
            module.last_trained = None
        
        self.active_modules.clear()
        print("✓ Todos los módulos reiniciados")
    
    def get_best_module_for_image(self, image_input) -> Tuple[str, List[Dict]]:
        """
        Encuentra el mejor módulo para analizar una imagen específica.
        
        Args:
            image_input: Imagen o path de imagen
            
        Returns:
            Tupla (module_id, predictions) del mejor módulo
        """
        best_module = None
        best_score = 0.0
        best_predictions = []
        
        for module_id in self.active_modules:
            if module_id not in self.modules:
                continue
            
            module = self.modules[module_id]
            predictions = module.predict(image_input)
            
            # Calcular score basado en número y confianza de predicciones
            if predictions:
                avg_confidence = np.mean([p.get('confidence', 0) for p in predictions])
                num_predictions = len(predictions)
                score = avg_confidence * num_predictions
                
                if score > best_score:
                    best_score = score
                    best_module = module_id
                    best_predictions = predictions
        
        return best_module, best_predictions
    
    def __str__(self) -> str:
        return f"ModuleManager({len(self.modules)} módulos, {len(self.active_modules)} activos)"
    
    def __repr__(self) -> str:
        return self.__str__()