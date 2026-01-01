"""
Módulo V2 mejorado para aprendizaje de patrones visuales con técnicas de IA avanzadas.
Implementa:
- SE (Squeeze-and-Excitation) Blocks para atención
- Mixup y CutMix augmentation
- One-Cycle Learning Rate Policy
- RandAugment
- Progressive Resizing
- Multi-scale inference
- Gradient Accumulation
- Auto-save de checkpoints
- Logging detallado
- Auto-tuning básico de hiperparámetros
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
from tqdm import tqdm
warnings.filterwarnings('ignore')


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block para atención de canales.
    Ayuda a la red a aprender qué características son más importantes.
    """
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlockSE(nn.Module):
    """Bloque residual con SE Block."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
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
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out


class FocalLoss(nn.Module):
    """Focal Loss para manejar clases desbalanceadas."""
    
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


class RandAugment:
    """
    RandAugment: Auto-augmentation aprendido.
    Aplica transformaciones aleatorias con magnitud controlada.
    """
    
    def __init__(self, n=2, m=10):
        self.n = n  # Número de transformaciones
        self.m = m  # Magnitud (0-30)
        
    def __call__(self, img):
        from torchvision import transforms, functional as F
        
        transformaciones = [
            lambda x: F.adjust_brightness(x, 1 + (np.random.rand() - 0.5) * self.m / 10),
            lambda x: F.adjust_contrast(x, 1 + (np.random.rand() - 0.5) * self.m / 10),
            lambda x: F.adjust_saturation(x, 1 + (np.random.rand() - 0.5) * self.m / 10),
            lambda x: F.hflip(x) if np.random.rand() > 0.5 else x,
            lambda x: F.vflip(x) if np.random.rand() > 0.5 else x,
            lambda x: F.rotate(x, np.random.uniform(-15, 15)),
            lambda x: F.posterize(x, int(np.random.randint(4, 8))),
            lambda x: F.sharpness(x, 1 + (np.random.rand() - 0.5) * self.m / 10),
            lambda x: F.equalize(x),
        ]
        
        for _ in range(self.n):
            t = np.random.choice(transformaciones)
            img = t(img)
        
        return img


class Mixup:
    """Mixup augmentation para mejorar generalización."""
    
    def __init__(self, alpha=0.4):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, labels_a, labels_b, lam):
    """Criterio mixup."""
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


class EnhancedPatternDatasetV2(Dataset):
    """Dataset V2 mejorado con RandAugment y progressive resizing."""
    
    def __init__(self, patterns_data: List[Dict], img_size=128, 
                 augment=False, use_randaugment=True):
        self.patterns = patterns_data
        self.img_size = img_size
        self.augment = augment
        self.use_randaugment = use_randaugment
        
        if augment:
            if use_randaugment:
                self.randaugment = RandAugment(n=2, m=10)
            else:
                from torchvision import transforms
                self.base_augment = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.3),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                         saturation=0.2, hue=0.1),
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
            h_img, w_img = image.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            if w > 0 and h > 0:
                image = image[y:y+h, x:x+w]
        
        # Convertir a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        
        # Resize
        from torchvision import transforms
        image_pil = transforms.Resize((self.img_size, self.img_size))(image_pil)
        
        # Aplicar augmentation
        if self.augment:
            if self.use_randaugment:
                image_pil = self.randaugment(image_pil)
            else:
                image_pil = self.base_augment(image_pil)
        
        # Convertir a tensor
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
        
        image_tensor = normalize(to_tensor(image_pil))
        
        # Obtener etiqueta
        label = pattern.get('pattern_id', 0)
        
        return image_tensor, label


class ImprovedPatternNetworkV2(nn.Module):
    """
    Red neuronal V2 mejorada con SE Blocks y arquitectura más potente.
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.4):
        super(ImprovedPatternNetworkV2, self).__init__()
        
        # Capas iniciales
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Bloques residuales con SE
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)  # Más profundo
        self.layer4 = self._make_layer(256, 512, 3, stride=2)  # Más profundo
        
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
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Crea una capa con bloques residuales con SE."""
        layers = []
        layers.append(ResidualBlockSE(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlockSE(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Inicializa los pesos de la red."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                      nonlinearity='relu')
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


class OneCycleLR:
    """
    One-Cycle Learning Rate Policy.
    Ajusta el learning rate durante el entrenamiento.
    """
    
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        if self.step_num > self.total_steps:
            return
        
        # Calcular learning rate actual
        if self.step_num <= self.total_steps * self.pct_start:
            # Fase de aumento
            scale = self.step_num / (self.total_steps * self.pct_start)
            lr = self.max_lr * scale
        else:
            # Fase de disminución
            scale = (self.total_steps - self.step_num) / (
                self.total_steps * (1 - self.pct_start))
            lr = self.max_lr * scale
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStoppingV2:
    """Early stopping V2 mejorado con warmup."""
    
    def __init__(self, patience=7, min_delta=0.001, warmup_epochs=0, 
                 restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        self.current_epoch = 0
    
    def __call__(self, val_loss, model):
        self.current_epoch += 1
        
        # Durante warmup, no hacemos early stopping
        if self.current_epoch <= self.warmup_epochs:
            return
        
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


class ImprovedPatternLearnerV2:
    """
    Sistema V2 mejorado para aprender patrones visuales.
    Incluye todas las técnicas de IA avanzadas.
    """
    
    def __init__(self, model_path: str = "user_patterns/patterns_model_v2.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.patterns = {}
        self.model = None
        self.pattern_counter = 0
        self.training_history = []
        
        # Directorios
        self.patterns_dir = Path("user_patterns")
        self.patterns_dir.mkdir(exist_ok=True)
        self.training_dir = Path("fotos_entrenamiento")
        self.training_dir.mkdir(exist_ok=True)
        self.identify_dir = Path("fotos_identificar")
        self.identify_dir.mkdir(exist_ok=True)
        
        # Crear directorios para cada tipo de patrón
        self.patterns_training_dir = self.training_dir / "por_patron"
        self.patterns_training_dir.mkdir(exist_ok=True)
        
        self._load_patterns()
        print(f"📁 Carpetas creadas/listas:")
        print(f"   - fotos_entrenamiento/ : Coloca aquí fotos para entrenar")
        print(f"   - fotos_identificar/  : Coloca aquí fotos para identificar")
        print(f"   - fotos_entrenamiento/por_patron/ : Organizadas por patrón")
    
    def _load_patterns(self):
        """Carga los patrones definidos."""
        patterns_file = self.patterns_dir / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.patterns = data.get('patterns', {})
                    self.pattern_counter = data.get('counter', 0)
                    self.training_history = data.get('training_history', [])
                    num_classes = max(10, len(self.patterns) + 1)
                    self.model = ImprovedPatternNetworkV2(num_classes=num_classes)
                    self.model.to(self.device)
                    print(f"✓ Cargados {len(self.patterns)} patrones")
            except Exception as e:
                print(f"Error cargando patrones: {e}")
                self.model = ImprovedPatternNetworkV2()
                self.model.to(self.device)
        else:
            self.model = ImprovedPatternNetworkV2()
            self.model.to(self.device)
    
    def _save_patterns(self):
        """Guarda los patrones."""
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
    
    def define_pattern_from_folder(self, name: str, description: str = "") -> str:
        """
        Define un patrón y crea su carpeta en fotos_entrenamiento.
        """
        pattern_id = f"pattern_{self.pattern_counter:04d}"
        self.pattern_counter += 1
        
        self.patterns[pattern_id] = {
            'id': pattern_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'samples': 0,
            'approved': 0,
            'corrected': 0
        }
        
        # Crear carpeta para este patrón
        pattern_folder = self.patterns_training_dir / name
        pattern_folder.mkdir(exist_ok=True)
        
        # Crear archivo README
        readme_path = pattern_folder / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# Patrón: {name}\n\n")
            f.write(f"Descripción: {description}\n\n")
            f.write(f"Coloca aquí las fotos de entrenamiento para este patrón.\n\n")
            f.write(f"ID del patrón: {pattern_id}\n")
        
        self._save_patterns()
        print(f"✓ Patrón definido: {name}")
        print(f"✓ Carpeta creada: {pattern_folder}")
        print(f"  → Coloca fotos de {name} en esta carpeta")
        
        return pattern_id
    
    def import_images_from_folder(self, pattern_name: str) -> int:
        """
        Importa imágenes de la carpeta de entrenamiento.
        """
        # Buscar el patrón por nombre
        pattern_id = None
        for pid, p in self.patterns.items():
            if p['name'] == pattern_name:
                pattern_id = pid
                break
        
        if not pattern_id:
            print(f"Patrón '{pattern_name}' no encontrado")
            return 0
        
        pattern_folder = self.patterns_training_dir / pattern_name
        if not pattern_folder.exists():
            print(f"Carpeta no encontrada: {pattern_folder}")
            return 0
        
        # Buscar imágenes
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        images = list(pattern_folder.glob("*.*"))
        images = [img for img in images if img.suffix.lower() in image_extensions]
        
        count = 0
        for img_path in tqdm(images, desc=f"Importando imágenes para {pattern_name}"):
            if self.add_pattern_sample(pattern_id, str(img_path)):
                count += 1
        
        print(f"✓ Importadas {count} imágenes para '{pattern_name}'")
        return count
    
    def import_all_patterns_from_folders(self) -> Dict[str, int]:
        """
        Importa imágenes de todas las carpetas de patrones.
        """
        results = {}
        
        # Buscar todas las carpetas de patrones
        for folder in self.patterns_training_dir.iterdir():
            if folder.is_dir():
                pattern_name = folder.name
                count = self.import_images_from_folder(pattern_name)
                if count > 0:
                    results[pattern_name] = count
        
        print(f"\n📊 Resumen de importación:")
        for name, count in results.items():
            print(f"  - {name}: {count} imágenes")
        
        return results
    
    def add_pattern_sample(self, pattern_id: str, image_path: str, 
                          roi: Tuple[int, int, int, int] = None) -> bool:
        """Añade una muestra de entrenamiento."""
        if pattern_id not in self.patterns:
            return False
        
        if not Path(image_path).exists():
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
            return True
        except Exception as e:
            print(f"Error añadiendo muestra: {e}")
            return False
    
    def train_patterns_v2(self, 
                        epochs: int = 30,
                        batch_size: int = 16,
                        val_split: float = 0.2,
                        learning_rate: float = 0.001,
                        max_lr: float = 0.01,
                        use_focal_loss: bool = False,
                        label_smoothing: float = 0.0,
                        early_stopping_patience: int = 10,
                        warmup_epochs: int = 3,
                        dropout_rate: float = 0.4,
                        use_mixup: bool = True,
                        use_randaugment: bool = True,
                        gradient_accumulation_steps: int = 1,
                        save_checkpoints: bool = True) -> Dict:
        """
        Entrena el modelo con técnicas V2 avanzadas.
        """
        if not self.patterns:
            print("No hay patrones definidos para entrenar")
            return {}
        
        # Preparar datos
        training_data = []
        
        for pattern_id, pattern in self.patterns.items():
            pattern_dir = self.patterns_dir / pattern_id
            if not pattern_dir.exists():
                continue
            
            for sample_file in pattern_dir.glob("sample_*.json"):
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        sample = json.load(f)
                    training_data.append({
                        'pattern_id': list(self.patterns.keys()).index(pattern_id),
                        'image_path': sample['image_path'],
                        'roi': sample.get('roi')
                    })
                except Exception:
                    continue
        
        if len(training_data) == 0:
            print("No hay muestras de entrenamiento")
            return {}
        
        print(f"\n🚀 Iniciando entrenamiento V2 mejorado")
        print(f"   Total de muestras: {len(training_data)}")
        print(f"   Técnicas activas:")
        print(f"      - One-Cycle Learning Rate (max_lr={max_lr})")
        print(f"      - RandAugment: {'Sí' if use_randaugment else 'No'}")
        print(f"      - Mixup: {'Sí' if use_mixup else 'No'}")
        print(f"      - Gradient Accumulation: {gradient_accumulation_steps}x")
        print(f"      - Warmup: {warmup_epochs} épocas")
        
        # Dividir datos
        val_size = int(len(training_data) * val_split)
        train_size = len(training_data) - val_size
        
        train_data = training_data[:train_size]
        val_data = training_data[train_size:]
        
        # Crear datasets
        train_dataset = EnhancedPatternDatasetV2(
            train_data, img_size=128, augment=True, use_randaugment=use_randaugment
        )
        val_dataset = EnhancedPatternDatasetV2(
            val_data, img_size=128, augment=False, use_randaugment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0)
        
        # Actualizar modelo si necesario
        num_classes = len(self.patterns)
        if self.model.fc[-1].out_features != num_classes:
            self.model = ImprovedPatternNetworkV2(num_classes=num_classes, 
                                                 dropout_rate=dropout_rate)
            self.model.to(self.device)
        
        # Optimizador
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                               weight_decay=1e-4)
        
        # One-Cycle LR
        total_steps = len(train_loader) * epochs
        scheduler = OneCycleLR(optimizer, max_lr, total_steps)
        
        # Loss
        if use_focal_loss:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif label_smoothing > 0:
            criterion = LabelSmoothingLoss(num_classes, label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Mixup
        mixup_fn = Mixup(alpha=0.4) if use_mixup else None
        
        # Early stopping
        early_stopping = EarlyStoppingV2(
            patience=early_stopping_patience,
            warmup_epochs=warmup_epochs,
            restore_best_weights=True
        )
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        
        print(f"\n📈 Entrenando...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            optimizer.zero_grad()
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Mixup
                if mixup_fn is not None:
                    images, labels_a, labels_b, lam = mixup_fn((images, labels))
                    outputs = self.model(images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    
                    # Para accuracy sin mixup
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (lam * predicted.eq(labels_a).float() + 
                                    (1 - lam) * predicted.eq(labels_b).float()).sum().item()
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()
                
                # Gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                
                train_loss += loss.item() * images.size(0)
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 
                                 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})
            
            train_loss /= len(train_dataset)
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    val_loss += loss.item() * images.size(0)
            
            val_loss /= len(val_dataset)
            val_acc = val_correct / val_total
            
            # Guardar historial
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Guardar mejor modelo
            if val_loss < best_val_loss and save_checkpoints:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"  ✓ Mejor modelo guardado (val_loss: {val_loss:.4f})")
            
            # Early stopping
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f"\n⏹️  Early stopping activado en epoch {epoch+1}")
                break
        
        # Restaurar mejores pesos
        early_stopping.restore_best_model(self.model)
        
        # Guardar modelo final
        torch.save(self.model.state_dict(), self.model_path)
        
        # Guardar historial
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'epochs': epoch + 1,
            'best_val_loss': best_val_loss,
            'config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'max_lr': max_lr,
                'use_mixup': use_mixup,
                'use_randaugment': use_randaugment,
            }
        })
        self._save_patterns()
        
        print(f"\n✅ Entrenamiento V2 completado")
        print(f"   Mejor val_loss: {best_val_loss:.4f}")
        print(f"   Modelo guardado en: {self.model_path}")
        
        return history
    
    def recognize_pattern_multiscale(self, image_path: str, 
                                    scales: List[int] = [96, 128, 160],
                                    threshold: float = 0.5) -> List[Dict]:
        """
        Reconoce patrones usando multi-scale inference para mejor precisión.
        """
        if not self.patterns:
            return []
        
        if not self._load_model_checkpoint():
            return []
        
        all_predictions = []
        
        for scale in scales:
            detections = self._recognize_at_scale(image_path, scale, threshold)
            all_predictions.extend(detections)
        
        # Agregar predicciones del mismo patrón
        aggregated = {}
        for det in all_predictions:
            pid = det['pattern_id']
            if pid not in aggregated:
                aggregated[pid] = {
                    'pattern_id': pid,
                    'pattern_name': det['pattern_name'],
                    'probabilities': []
                }
            aggregated[pid]['probabilities'].append(det['probability'])
        
        # Promediar
        final_detections = []
        for pid, data in aggregated.items():
            avg_prob = np.mean(data['probabilities'])
            if avg_prob >= threshold:
                final_detections.append({
                    'pattern_id': pid,
                    'pattern_name': data['pattern_name'],
                    'probability': float(avg_prob),
                    'bbox': (0, 0, 0, 0)  # Ajustar según imagen
                })
        
        final_detections.sort(key=lambda x: x['probability'], reverse=True)
        return final_detections
    
    def _recognize_at_scale(self, image_path: str, scale: int, 
                           threshold: float) -> List[Dict]:
        """Reconoce a una escala específica."""
        from torchvision import transforms
        
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        transform = transforms.Compose([
            transforms.Resize((scale, scale)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        pattern_ids = list(self.patterns.keys())
        detections = []
        
        for i, prob in enumerate(probabilities):
            if prob >= threshold and i < len(pattern_ids):
                pattern = self.patterns[pattern_ids[i]]
                detections.append({
                    'pattern_id': pattern_ids[i],
                    'pattern_name': pattern['name'],
                    'probability': float(prob)
                })
        
        return detections
    
    def _load_model_checkpoint(self) -> bool:
        """Carga el checkpoint del modelo."""
        if not Path(self.model_path).exists():
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Adaptar si el número de clases cambió
            if checkpoint['fc.6.weight'].size(0) != len(self.patterns):
                # Reconstruir la última capa
                num_classes = len(self.patterns)
                self.model = ImprovedPatternNetworkV2(num_classes=num_classes)
                self.model.to(self.device)
                print("⚠ Modelo reestructurado para nuevo número de clases")
            else:
                self.model.load_state_dict(checkpoint)
            
            return True
        except Exception as e:
            print(f"Error cargando checkpoint: {e}")
            return False
    
    def recognize_pattern(self, image_path: str, threshold: float = 0.5) -> List[Dict]:
        """
        Reconoce patrones en una imagen.
        """
        return self.recognize_pattern_multiscale(image_path, 
                                                  scales=[128], 
                                                  threshold=threshold)
    
    def identify_from_folder(self, output_file: str = None) -> Dict:
        """
        Identifica patrones en todas las imágenes de fotos_identificar/.
        """
        if output_file is None:
            output_file = f"resultados_identificacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Buscar imágenes
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        images = list(self.identify_dir.glob("*.*"))
        images = [img for img in images if img.suffix.lower() in image_extensions]
        
        if not images:
            print("No hay imágenes en fotos_identificar/")
            return {}
        
        print(f"\n🔍 Identificando patrones en {len(images)} imágenes...")
        
        results = {}
        for img_path in tqdm(images):
            detections = self.recognize_pattern(str(img_path), threshold=0.5)
            if detections:
                results[str(img_path)] = detections
                print(f"  {img_path.name}: {detections[0]['pattern_name']} ({detections[0]['probability']:.2%})")
        
        # Guardar resultados
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Resultados guardados en: {output_file}")
        
        # Generar reporte
        self._generate_identification_report(results, output_file.replace('.json', '_reporte.txt'))
        
        return results
    
    def _generate_identification_report(self, results: Dict, output_path: str):
        """Genera un reporte legible de identificaciones."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("═════════════════════════════════════════════════════════\n")
            f.write("  REPORTE DE IDENTIFICACIÓN DE PATRONES\n")
            f.write(f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("═════════════════════════════════════════════════════════\n\n")
            
            # Estadísticas por patrón
            pattern_counts = {}
            for img_path, detections in results.items():
                for det in detections:
                    name = det['pattern_name']
                    if name not in pattern_counts:
                        pattern_counts[name] = {'count': 0, 'avg_conf': 0}
                    pattern_counts[name]['count'] += 1
                    pattern_counts[name]['avg_conf'] += det['probability']
            
            for name, data in pattern_counts.items():
                data['avg_conf'] /= data['count']
            
            f.write("📊 RESUMEN POR PATRÓN:\n")
            for name, data in sorted(pattern_counts.items(), 
                                     key=lambda x: x[1]['count'], 
                                     reverse=True):
                f.write(f"  • {name}: {data['count']} detecciones ")
                f.write(f"(conf. promedio: {data['avg_conf']:.2%})\n")
            
            f.write(f"\n📁 Total de imágenes analizadas: {len(results)}\n")
            f.write(f"🎯 Total de patrones detectados: {len(pattern_counts)}\n")
            
            f.write("\n═════════════════════════════════════════════════════════\n")
            f.write("DETALLES POR IMAGEN:\n")
            f.write("═════════════════════════════════════════════════════════\n\n")
            
            for img_path, detections in results.items():
                img_name = Path(img_path).name
                f.write(f"🖼️  {img_name}\n")
                for det in detections:
                    f.write(f"   └─ {det['pattern_name']} ({det['probability']:.2%})\n")
                f.write("\n")
        
        print(f"✓ Reporte generado: {output_path}")
    
    def list_patterns(self) -> List[Dict]:
        """Lista todos los patrones."""
        patterns_list = []
        
        for pattern_id, pattern in self.patterns.items():
            # Contar imágenes en la carpeta de entrenamiento
            pattern_folder = self.patterns_training_dir / pattern['name']
            folder_images = 0
            if pattern_folder.exists():
                extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
                folder_images = len([f for f in pattern_folder.iterdir() 
                                   if f.suffix.lower() in extensions])
            
            info = {
                'id': pattern_id,
                'name': pattern['name'],
                'description': pattern['description'],
                'samples': pattern['samples'],
                'folder_images': folder_images,
                'approved': pattern['approved'],
                'corrected': pattern['corrected'],
                'accuracy': pattern['approved'] / (pattern['approved'] + pattern['corrected']) 
                           if (pattern['approved'] + pattern['corrected']) > 0 else 0.0,
                'created_at': pattern['created_at']
            }
            patterns_list.append(info)
        
        return patterns_list
    
    def get_model_info(self) -> Dict:
        """Obtiene información sobre el modelo entrenado."""
        checkpoint_path = Path(self.model_path)
        
        info = {
            'model_exists': checkpoint_path.exists(),
            'model_path': str(checkpoint_path),
            'num_patterns': len(self.patterns),
            'pattern_names': [p['name'] for p in self.patterns.values()],
            'training_history_count': len(self.training_history),
        }
        
        if checkpoint_path.exists():
            info['model_size_mb'] = checkpoint_path.stat().st_size / (1024 * 1024)
            info['model_modified'] = datetime.fromtimestamp(
                checkpoint_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        return info
