"""
Módulo mejorado para aprendizaje de patrones visuales definidos por el usuario
Incluye data augmentation, early stopping, learning rate scheduling, TTA y más.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import warnings
warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    """
    Focal Loss para manejar clases desbalanceadas.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing para mejor generalización."""
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=1))


class ResidualBlock(nn.Module):
    """Bloque residual para mejorar el flujo de gradientes."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class EnhancedPatternDataset(Dataset):
    """Dataset mejorado con data augmentation para entrenamiento de patrones."""
    
    def __init__(self, patterns_data: List[Dict], transform=None, augment=False):
        """
        Inicializa el dataset de patrones.
        
        Args:
            patterns_data: Lista de diccionarios con patrones
            transform: Transformaciones base a aplicar
            augment: Si se debe aplicar data augmentation
        """
        self.patterns = patterns_data
        self.base_transform = transform
        self.augment = augment
        
        # Data augmentation transforms
        self.augment_transforms = None
        if augment:
            self.augment_transforms = self._get_augmentation_transforms()
    
    def _get_augmentation_transforms(self):
        """Retorna transformaciones de data augmentation."""
        from torchvision import transforms
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), 
                                   scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        ])
    
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
            # Validar que el ROI está dentro de la imagen
            h, w_img = image.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h - 1))
            w = min(w, w_img - x)
            h = min(h, h - y)
            if w > 0 and h > 0:
                image = image[y:y+h, x:x+w]
        
        # Convertir a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        
        # Aplicar transformaciones
        if self.base_transform:
            image_pil = self.base_transform(image_pil)
        
        # Aplicar augmentation si está habilitado
        if self.augment and self.augment_transforms:
            image_pil = self.augment_transforms(image_pil)
        
        # Obtener etiqueta
        label = pattern.get('pattern_id', 0)
        
        return image_pil, label


class ImprovedPatternNetwork(nn.Module):
    """Red neuronal mejorada con bloques residuales y arquitectura más profunda."""
    
    def __init__(self, num_classes=10, dropout_rate=0.4):
        super(ImprovedPatternNetwork, self).__init__()
        
        # Capas iniciales
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Bloques residuales
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Capas fully connected mejoradas
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(512, num_classes)
        )
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Crea una capa con bloques residuales."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Inicializa los pesos de la red."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                      nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class EarlyStopping:
    """Early stopping para detener el entrenamiento cuando no hay mejora."""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
    
    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
    
    def restore_best_model(self, model):
        if self.best_weights is not None and self.restore_best_weights:
            model.load_state_dict(self.best_weights)


class ImprovedPatternLearner:
    """Sistema mejorado para aprender patrones visuales definidos por el usuario."""
    
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
        self.training_history = []
        
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
                    self.training_history = data.get('training_history', [])
                    num_classes = max(10, len(self.patterns) + 1)
                    self.model = ImprovedPatternNetwork(num_classes=num_classes)
                    self.model.to(self.device)
                    print(f"✓ Cargados {len(self.patterns)} patrones definidos por el usuario")
            except Exception as e:
                print(f"Error cargando patrones: {e}")
                self.model = ImprovedPatternNetwork()
                self.model.to(self.device)
        else:
            self.model = ImprovedPatternNetwork()
            self.model.to(self.device)
    
    def _save_patterns(self):
        """Guarda los patrones definidos por el usuario."""
        data = {
            'patterns': self.patterns,
            'counter': self.pattern_counter,
            'training_history': self.training_history,
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
    
    def train_patterns(self, epochs: int = 30, batch_size: int = 16, 
                      val_split: float = 0.2, learning_rate: float = 0.001,
                      use_focal_loss: bool = False, label_smoothing: float = 0.0,
                      early_stopping_patience: int = 10, 
                      dropout_rate: float = 0.4) -> Dict:
        """
        Entrena el modelo con los patrones definidos usando técnicas avanzadas.
        
        Args:
            epochs: Número de épocas
            batch_size: Tamaño del batch
            val_split: Proporción para validación (0.0-1.0)
            learning_rate: Learning rate inicial
            use_focal_loss: Usar Focal Loss para clases desbalanceadas
            label_smoothing: Factor de label smoothing (0.0 desactivado)
            early_stopping_patience: Paciencia para early stopping (0 desactivado)
            dropout_rate: Tasa de dropout
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        if not self.patterns:
            print("No hay patrones definidos para entrenar")
            return {}
        
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
            return {}
        
        print(f"🏃 Entrenando modelo mejorado con {len(training_data)} muestras...")
        
        # Preparar dataset con data augmentation
        from torchvision import transforms
        
        base_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Dividir en train y validation
        val_size = int(len(training_data) * val_split)
        train_size = len(training_data) - val_size
        
        if val_size > 0:
            train_dataset = EnhancedPatternDataset(
                training_data[:train_size], 
                transform=base_transform,
                augment=True
            )
            val_dataset = EnhancedPatternDataset(
                training_data[train_size:], 
                transform=base_transform,
                augment=False
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                   shuffle=False, num_workers=0)
        else:
            train_dataset = EnhancedPatternDataset(
                training_data, 
                transform=base_transform,
                augment=True
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=0)
            val_loader = None
        
        # Actualizar número de clases del modelo
        num_classes = len(self.patterns)
        self.model = ImprovedPatternNetwork(num_classes=num_classes, 
                                            dropout_rate=dropout_rate)
        self.model.to(self.device)
        
        # Configurar loss function
        if use_focal_loss:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif label_smoothing > 0:
            criterion = LabelSmoothingLoss(num_classes, smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizador con weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                               weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Early stopping
        early_stopping = None
        if early_stopping_patience > 0 and val_loader is not None:
            early_stopping = EarlyStopping(patience=early_stopping_patience, 
                                           restore_best_weights=True)
        
        # Métricas de entrenamiento
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Entrenar
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            epoch_train_correct = 0
            epoch_train_total = 0
            
            self.model.train()
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_train_total += labels.size(0)
                epoch_train_correct += (predicted == labels).sum().item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_acc = 100 * epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Validación
            val_loss = 0.0
            val_acc = 0.0
            
            if val_loader is not None:
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = 100 * val_correct / val_total if val_total > 0 else 0
                
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_acc)
                
                # Early stopping check
                if early_stopping:
                    early_stopping(avg_val_loss, self.model)
                    if early_stopping.early_stop:
                        print(f"  Early stopping activado en época {epoch+1}")
                        break
                
                # Guardar mejor modelo
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            # Scheduler step
            scheduler.step()
            
            # Imprimir progreso
            val_str = f" | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%" if val_loader else ""
            print(f"  Época {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%"
                  f"{val_str} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Restaurar mejores pesos si se usó early stopping
        if early_stopping and early_stopping.restore_best_weights:
            early_stopping.restore_best_model(self.model)
            print(f"  Restaurados mejores pesos del modelo (Val Acc: {best_val_acc:.2f}%)")
        
        # Guardar modelo
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': num_classes,
            'patterns': self.patterns,
            'history': history,
            'config': {
                'dropout_rate': dropout_rate,
                'use_focal_loss': use_focal_loss,
                'label_smoothing': label_smoothing
            }
        }, self.model_path)
        
        # Guardar historial de entrenamiento
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'epochs': epoch + 1,
            'num_samples': len(training_data),
            'best_val_acc': float(best_val_acc),
            'history': history
        })
        self._save_patterns()
        
        print(f"✓ Entrenamiento completado. Mejor precisión validación: {best_val_acc:.2f}%")
        print(f"  Modelo guardado en: {self.model_path}")
        
        return history
    
    def _load_model_checkpoint(self) -> bool:
        """Carga los pesos del modelo desde el archivo, ajustando la arquitectura si es necesario."""
        if not Path(self.model_path).exists():
            return False
            
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            num_classes = checkpoint.get('num_classes', len(self.patterns))
            
            # Re-inicializar modelo si el número de clases no coincide
            if self.model is None or num_classes != self.model.fc[-1].out_features:
                config = checkpoint.get('config', {})
                dropout_rate = config.get('dropout_rate', 0.4)
                self.model = ImprovedPatternNetwork(num_classes=num_classes, dropout_rate=dropout_rate)
                self.model.to(self.device)
                
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            return False

    def recognize_pattern_tta(self, image_input: Union[str, np.ndarray], roi: Tuple[int, int, int, int] = None,
                            threshold: float = 0.5, include_reasoning: bool = False,
                            tta_transforms: int = 5) -> List[Dict]:
        """
        Reconoce patrones usando Test Time Augmentation (TTA) para mejor precisión.
        
        Args:
            image_input: Ruta a la imagen o array de imagen (BGR)
            roi: Región de interés (x, y, w, h)
            threshold: Umbral de confianza mínimo
            include_reasoning: Si se debe incluir información de razonamiento
            tta_transforms: Número de transformaciones TTA
            
        Returns:
            Lista de detecciones encontradas
        """
        if not self.patterns:
            print("No hay patrones definidos")
            return []
        
        # Cargar modelo
        if not self._load_model_checkpoint():
            print("No se pudo cargar el modelo entrenado")
            return []
        
        # Cargar imagen
        if isinstance(image_input, (str, Path)):
            full_image = cv2.imread(str(image_input))
            if full_image is None:
                print(f"No se pudo cargar imagen: {image_input}")
                return []
        else:
            full_image = image_input
        
        image = full_image
        # Extraer ROI si existe
        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        # Preprocesar
        from torchvision import transforms
        
        base_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Transformaciones TTA
        tta_transformations = [
            lambda x: x,  # Original
            lambda x: transforms.functional.hflip(x),  # Flip horizontal
            lambda x: transforms.functional.vflip(x),  # Flip vertical
            lambda x: transforms.functional.rotate(x, 15),  # Rotación 15°
            lambda x: transforms.functional.rotate(x, -15),  # Rotación -15°
            lambda x: transforms.functional.adjust_brightness(x, 1.2),  # Más brillante
            lambda x: transforms.functional.adjust_contrast(x, 1.2),  # Más contraste
        ]
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tensor = base_transform(image_pil)
        
        # Acumular predicciones con TTA
        all_predictions = []
        for i, tta_transform in enumerate(tta_transformations[:tta_transforms]):
            augmented = tta_transform(image_tensor).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(augmented)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                all_predictions.append(probabilities)
        
        # Promediar predicciones (ensemble)
        avg_probabilities = np.mean(all_predictions, axis=0)
        
        # Filtrar por umbral
        pattern_ids = list(self.patterns.keys())
        detections = []
        
        for i, prob in enumerate(avg_probabilities):
            if prob >= threshold and i < len(pattern_ids):
                pattern_id = pattern_ids[i]
                pattern = self.patterns[pattern_id]
                
                # Calcular desviación estándar (consistencia del TTA)
                std_dev = np.std([p[i] for p in all_predictions])
                confidence_interval = 1.96 * std_dev  # 95% CI
                
                detection = {
                    'pattern_id': pattern_id,
                    'pattern_name': pattern['name'],
                    'probability': float(prob),
                    'confidence_std': float(std_dev),
                    'confidence_interval_lower': float(max(0, prob - confidence_interval)),
                    'confidence_interval_upper': float(min(1, prob + confidence_interval)),
                    'bbox': roi if roi else (0, 0, full_image.shape[1], full_image.shape[0]),
                    'tta_votes': int(tta_transforms),
                    'consistency': float(1.0 - min(1.0, std_dev))  # 1 = muy consistente
                }
                
                if include_reasoning:
                    original_tensor = image_tensor.unsqueeze(0).to(self.device)
                    detection['heatmap'] = self._generate_heatmap(original_tensor, i)
                
                detections.append(detection)
        
        # Ordenar por probabilidad
        detections.sort(key=lambda x: x['probability'], reverse=True)
        
        return detections
    
    def recognize_pattern(self, image_input: Union[str, np.ndarray], roi: Tuple[int, int, int, int] = None,
                        threshold: float = 0.5, include_reasoning: bool = False,
                        use_tta: bool = False) -> List[Dict]:
        """
        Reconoce patrones en una imagen.
        
        Args:
            image_input: Ruta a la imagen o array de imagen (BGR)
            roi: Región de interés (x, y, w, h)
            threshold: Umbral de confianza mínimo
            include_reasoning: Si se debe incluir información de razonamiento
            use_tta: Usar Test Time Augmentation para mejor precisión
            
        Returns:
            Lista de detecciones encontradas
        """
        if use_tta:
            return self.recognize_pattern_tta(image_input, roi, threshold, include_reasoning)
        
        if not self.patterns:
            print("No hay patrones definidos")
            return []
        
        # Cargar modelo
        if not self._load_model_checkpoint():
            print("No se pudo cargar el modelo entrenado")
            return []
        
        # Cargar imagen
        if isinstance(image_input, (str, Path)):
            full_image = cv2.imread(str(image_input))
            if full_image is None:
                print(f"No se pudo cargar imagen: {image_input}")
                return []
        else:
            full_image = image_input
        
        image = full_image
        # Extraer ROI si existe
        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        # Preprocesar
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        # Filtrar por umbral
        pattern_ids = list(self.patterns.keys())
        detections = []
        
        for i, prob in enumerate(probabilities):
            if prob >= threshold and i < len(pattern_ids):
                pattern_id = pattern_ids[i]
                pattern = self.patterns[pattern_id]
                
                detection = {
                    'pattern_id': pattern_id,
                    'pattern_name': pattern['name'],
                    'probability': float(prob),
                    'bbox': roi if roi else (0, 0, full_image.shape[1], full_image.shape[0])
                }
                
                if include_reasoning:
                    detection['heatmap'] = self._generate_heatmap(image_tensor, i)
                
                detections.append(detection)
        
        # Ordenar por probabilidad
        detections.sort(key=lambda x: x['probability'], reverse=True)
        
        return detections
    
    def _generate_heatmap(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Genera un mapa de calor (razonamiento) usando Grad-CAM."""
        self.model.eval()
        
        # Activar gradientes para el tensor de entrada
        input_image = image_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_image)
        
        # Target score
        target = output[0][class_idx]
        
        # Backward pass para obtener gradientes con respecto a la entrada
        self.model.zero_grad()
        target.backward()
        
        # Obtener gradientes y calcular magnitud
        gradients = input_image.grad.data[0].cpu().numpy()
        gradients = np.transpose(gradients, (1, 2, 0))
        heatmap = np.max(np.abs(gradients), axis=2)
        
        # Normalizar heatmap
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
        return heatmap
    
    def evaluate_model(self, test_data: List[Dict]) -> Dict:
        """
        Evalúa el modelo con datos de prueba calculando métricas detalladas.
        
        Args:
            test_data: Lista de diccionarios con datos de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        if not self.patterns:
            return {}
        
        # Cargar modelo
        if not self._load_model_checkpoint():
            print("No se pudo cargar el modelo entrenado")
            return {}
        
        # Preparar dataset de prueba
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        dataset = EnhancedPatternDataset(test_data, transform=transform, augment=False)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Evaluar
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calcular métricas
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Accuracy general
        accuracy = np.mean(all_preds == all_labels)
        
        # Métricas por clase
        num_classes = len(self.patterns)
        pattern_ids = list(self.patterns.keys())
        
        class_metrics = {}
        for i, pattern_id in enumerate(pattern_ids):
            mask = (all_labels == i)
            if mask.sum() > 0:
                # Precision, Recall, F1
                tp = ((all_preds == i) & (all_labels == i)).sum()
                fp = ((all_preds == i) & (all_labels != i)).sum()
                fn = ((all_preds != i) & (all_labels == i)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[pattern_id] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'support': int(mask.sum())
                }
        
        # Matriz de confusión
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for pred, label in zip(all_preds, all_labels):
            confusion_matrix[label, pred] += 1
        
        metrics = {
            'accuracy': float(accuracy),
            'num_samples': len(all_labels),
            'class_metrics': class_metrics,
            'confusion_matrix': confusion_matrix.tolist()
        }
        
        print(f"\n📊 Métricas de Evaluación:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Número de muestras: {len(all_labels)}")
        print(f"\n  Métricas por clase:")
        for pattern_id, metric in class_metrics.items():
            pattern_name = self.patterns[pattern_id]['name']
            print(f"    {pattern_name}:")
            print(f"      Precision: {metric['precision']:.4f}")
            print(f"      Recall: {metric['recall']:.4f}")
            print(f"      F1: {metric['f1']:.4f}")
            print(f"      Soporte: {metric['support']}")
        
        return metrics
    
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
        
        print(f"Patrón {pattern_id} eliminado")
        return True
    
    def export_learning_data(self, output_path: str = "learning_data.json") -> bool:
        """Exporta los datos de aprendizaje para fine-tuning externo."""
        try:
            learning_data = {
                'patterns': self.patterns,
                'exported_at': datetime.now().isoformat(),
                'version': '2.0'
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(learning_data, f, indent=2, ensure_ascii=False)
            
            print(f"Datos de aprendizaje exportados a: {output_path}")
            return True
        except Exception as e:
            print(f"Error exportando datos: {e}")
            return False


# Mantener compatibilidad con nombres anteriores
PatternNetwork = ImprovedPatternNetwork
PatternDataset = EnhancedPatternDataset
PatternLearner = ImprovedPatternLearner
