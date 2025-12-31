"""
Módulo para aprendizaje de patrones visuales definidos por el usuario
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class PatternDataset(Dataset):
    """Dataset para entrenamiento de patrones personalizados."""
    
    def __init__(self, patterns_data: List[Dict], transform=None):
        """
        Inicializa el dataset de patrones.
        
        Args:
            patterns_data: Lista de diccionarios con patrones
            transform: Transformaciones a aplicar
        """
        self.patterns = patterns_data
        self.transform = transform
        
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        image_path = pattern['image_path']
        roi = pattern.get('roi', None)
        
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo cargar imagen: {image_path}")
        
        # Extraer ROI si existe
        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        # Convertir a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        # Obtener etiqueta
        label = pattern.get('pattern_id', 0)
        
        return image, label


class PatternNetwork(nn.Module):
    """Red neuronal para clasificación de patrones personalizados."""
    
    def __init__(self, num_classes=10):
        super(PatternNetwork, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 12 * 12, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class PatternLearner:
    """Sistema para aprender patrones visuales definidos por el usuario."""
    
    def __init__(self, model_path: str = "patterns_model.pth"):
        """
        Inicializa el sistema de aprendizaje de patrones.
        
        Args:
            model_path: Ruta para guardar/cargar el modelo
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.patterns = {}  # pattern_id -> pattern_data
        self.model = None
        self.pattern_counter = 0
        
        # Crear directorio para patrones si no existe
        self.patterns_dir = Path("user_patterns")
        self.patterns_dir.mkdir(exist_ok=True)
        
        # Cargar modelo existente si está disponible
        self._load_patterns()
    
    def _load_patterns(self):
        """Carga los patrones definidos por el usuario."""
        patterns_file = self.patterns_dir / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.patterns = data.get('patterns', {})
                    self.pattern_counter = data.get('counter', 0)
                    self.model = PatternNetwork(num_classes=max(10, len(self.patterns) + 1))
                    self.model.to(self.device)
                    print(f"✓ Cargados {len(self.patterns)} patrones definidos por el usuario")
            except Exception as e:
                print(f"Error cargando patrones: {e}")
                self.model = PatternNetwork()
                self.model.to(self.device)
        else:
            self.model = PatternNetwork()
            self.model.to(self.device)
    
    def _save_patterns(self):
        """Guarda los patrones definidos por el usuario."""
        data = {
            'patterns': self.patterns,
            'counter': self.pattern_counter,
            'last_updated': datetime.now().isoformat()
        }
        
        patterns_file = self.patterns_dir / "patterns.json"
        try:
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando patrones: {e}")
    
    def define_pattern(self, name: str, description: str = "", 
                      image_path: str = None, roi: Tuple[int, int, int, int] = None) -> str:
        """
        Define un nuevo patrón visual.
        
        Args:
            name: Nombre del patrón
            description: Descripción del patrón
            image_path: Ruta a una imagen de ejemplo
            roi: Región de interés (x, y, w, h) en la imagen
            
        Returns:
            ID del patrón creado
        """
        pattern_id = f"pattern_{self.pattern_counter:04d}"
        self.pattern_counter += 1
        
        self.patterns[pattern_id] = {
            'id': pattern_id,
            'name': name,
            'description': description,
            'image_path': image_path,
            'roi': roi,
            'created_at': datetime.now().isoformat(),
            'samples': 0,
            'approved': 0,
            'corrected': 0
        }
        
        self._save_patterns()
        print(f"✓ Patrón definido: {name} (ID: {pattern_id})")
        
        return pattern_id
    
    def add_pattern_sample(self, pattern_id: str, image_path: str, 
                          roi: Tuple[int, int, int, int] = None) -> bool:
        """
        Añade una muestra de entrenamiento para un patrón.
        
        Args:
            pattern_id: ID del patrón
            image_path: Ruta a la imagen de muestra
            roi: Región de interés (x, y, w, h)
            
        Returns:
            True si se añadió correctamente
        """
        if pattern_id not in self.patterns:
            print(f"Patrón {pattern_id} no encontrado")
            return False
        
        if not Path(image_path).exists():
            print(f"Imagen no encontrada: {image_path}")
            return False
        
        # Guardar muestra
        pattern_dir = self.patterns_dir / pattern_id
        pattern_dir.mkdir(exist_ok=True)
        
        sample_idx = self.patterns[pattern_id]['samples'] + 1
        sample_path = pattern_dir / f"sample_{sample_idx:04d}.json"
        
        sample_data = {
            'image_path': image_path,
            'roi': roi,
            'added_at': datetime.now().isoformat()
        }
        
        try:
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2)
            
            self.patterns[pattern_id]['samples'] += 1
            self._save_patterns()
            print(f"✓ Muestra añadida a {self.patterns[pattern_id]['name']}")
            return True
        except Exception as e:
            print(f"Error añadiendo muestra: {e}")
            return False
    
    def train_patterns(self, epochs: int = 10, batch_size: int = 8) -> bool:
        """
        Entrena el modelo con los patrones definidos.
        
        Args:
            epochs: Número de épocas
            batch_size: Tamaño del batch
            
        Returns:
            True si el entrenamiento fue exitoso
        """
        if not self.patterns:
            print("No hay patrones definidos para entrenar")
            return False
        
        # Preparar datos de entrenamiento
        training_data = []
        
        for pattern_id, pattern in self.patterns.items():
            pattern_dir = self.patterns_dir / pattern_id
            if not pattern_dir.exists():
                continue
            
            # Cargar muestras del patrón
            for sample_file in pattern_dir.glob("sample_*.json"):
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        sample = json.load(f)
                    training_data.append({
                        'pattern_id': list(self.patterns.keys()).index(pattern_id),
                        'image_path': sample['image_path'],
                        'roi': sample.get('roi')
                    })
                except Exception as e:
                    print(f"Error cargando muestra {sample_file}: {e}")
                    continue
        
        if len(training_data) == 0:
            print("No hay muestras de entrenamiento disponibles")
            return False
        
        print(f"🏃 Entrenando modelo con {len(training_data)} muestras...")
        
        # Preparar dataset
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
        
        dataset = PatternDataset(training_data, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Actualizar número de clases del modelo
        num_classes = len(self.patterns)
        self.model = PatternNetwork(num_classes=num_classes)
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Entrenar
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Época {epoch+1}/{epochs} - Pérdida: {avg_loss:.4f}")
        
        # Guardar modelo
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': num_classes,
            'patterns': self.patterns
        }, self.model_path)
        
        print(f"✓ Entrenamiento completado. Modelo guardado en: {self.model_path}")
        return True
    
    def recognize_pattern(self, image_path: str, roi: Tuple[int, int, int, int] = None,
                        threshold: float = 0.5) -> List[Dict]:
        """
        Reconoce patrones en una imagen.
        
        Args:
            image_path: Ruta a la imagen
            roi: Región de interés (x, y, w, h)
            threshold: Umbral de confianza mínimo
            
        Returns:
            Lista de detecciones encontradas
        """
        if not self.patterns:
            print("No hay patrones entrenados")
            return []
        
        # Cargar modelo
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"No se pudo cargar imagen: {image_path}")
            return []
        
        # Extraer ROI si existe
        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        # Preprocesar
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = outputs[0].cpu().numpy()
        
        # Filtrar por umbral
        pattern_ids = list(self.patterns.keys())
        detections = []
        
        for i, prob in enumerate(probabilities):
            if prob >= threshold and i < len(pattern_ids):
                pattern_id = pattern_ids[i]
                pattern = self.patterns[pattern_id]
                detections.append({
                    'pattern_id': pattern_id,
                    'pattern_name': pattern['name'],
                    'probability': float(prob),
                    'bbox': roi if roi else (0, 0, image.shape[1], image.shape[0])
                })
        
        # Ordenar por probabilidad
        detections.sort(key=lambda x: x['probability'], reverse=True)
        
        return detections
    
    def record_feedback(self, pattern_id: str, is_correct: bool, 
                      correction: str = None) -> bool:
        """
        Registra feedback humano sobre una detección.
        
        Args:
            pattern_id: ID del patrón
            is_correct: Si la detección fue correcta
            correction: Corrección si no fue correcta
            
        Returns:
            True si se registró correctamente
        """
        if pattern_id not in self.patterns:
            return False
        
        if is_correct:
            self.patterns[pattern_id]['approved'] += 1
        else:
            self.patterns[pattern_id]['corrected'] += 1
        
        self._save_patterns()
        return True
    
    def get_pattern_info(self, pattern_id: str) -> Optional[Dict]:
        """Obtiene información de un patrón específico."""
        return self.patterns.get(pattern_id)
    
    def list_patterns(self) -> List[Dict]:
        """Lista todos los patrones definidos."""
        patterns_list = []
        
        for pattern_id, pattern in self.patterns.items():
            info = {
                'id': pattern_id,
                'name': pattern['name'],
                'description': pattern['description'],
                'samples': pattern['samples'],
                'approved': pattern['approved'],
                'corrected': pattern['corrected'],
                'accuracy': pattern['approved'] / (pattern['approved'] + pattern['corrected']) 
                           if (pattern['approved'] + pattern['corrected']) > 0 else 0.0,
                'created_at': pattern['created_at']
            }
            patterns_list.append(info)
        
        return patterns_list
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """Elimina un patrón."""
        if pattern_id not in self.patterns:
            print(f"Patrón {pattern_id} no encontrado")
            return False
        
        del self.patterns[pattern_id]
        self._save_patterns()
        
        # Eliminar directorio del patrón
        pattern_dir = self.patterns_dir / pattern_id
        if pattern_dir.exists():
            import shutil
            shutil.rmtree(pattern_dir)
        
        print(f"✓ Patrón eliminado: {pattern_id}")
        return True
