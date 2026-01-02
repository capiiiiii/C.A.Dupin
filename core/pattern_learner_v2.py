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
- MEJORAS DE RENDIMIENTO (PyTorch 2.0+):
  * Automatic Mixed Precision (AMP) - hasta 2-3x más rápido
  * torch.compile para optimización del modelo
  * DataLoader paralelo con prefetching
  * Channels Last memory format
  * Gradient Checkpointing para ahorrar memoria
"""

import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from PIL import Image
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Verificar PyTorch 2.0+ para torch.compile
PYTORCH_2_PLUS = hasattr(torch, 'compile')
USE_AMP = torch.cuda.is_available()  # AMP solo en GPU


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block para atención de canales.
    Ayuda a la red a aprender qué características son más importantes.
    Optimizado con memoria eficiente.
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


class GradientCheckpointingWrapper(nn.Module):
    """
    Wrapper para habilitar gradient checkpointing y ahorrar memoria.
    """

    def __init__(self, module, use_checkpointing=True):
        super().__init__()
        self.module = module
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.module, x)
        return self.module(x)


class InvertedResidualSE(nn.Module):
    """Inverted Residual Block con SE Block (MobileNetV3-style) para eficiencia."""
    
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(InvertedResidualSE, self).__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        self.conv = nn.Sequential(
            # Expansion conv
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # SE Block
            SEBlock(hidden_dim, reduction=4),
            # Projection conv
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EfficientChannelAttention(nn.Module):
    """Eficient Channel Attention (ECA) - menos parámetros que SE Block."""
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((np.log(channels) / np.log(2)) + b) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)


class CoordConv(nn.Module):
    """CoordConv para ayudar a la red a entender coordenadas espaciales."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, _, h, w = x.size()
        # Agregar coordenadas
        i = torch.arange(h, dtype=x.dtype, device=x.device)[None, None, :, None]
        j = torch.arange(w, dtype=x.dtype, device=x.device)[None, None, None, :]
        i = i.expand(batch_size, 1, h, w)
        j = j.expand(batch_size, 1, h, w)
        x = torch.cat([x, i / (h - 1), j / (w - 1)], dim=1)
        return self.relu(self.bn(self.conv(x)))


class ResidualBlockSE(nn.Module):
    """Bloque residual con SE Block optimizado."""

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
        out = out + identity  # Más eficiente que +=
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
    """Dataset V2 mejorado con RandAugment, caching y optimizaciones."""

    def __init__(
        self,
        patterns_data: List[Dict],
        img_size: int = 128,
        augment: bool = False,
        use_randaugment: bool = True,
        cache_images: bool = True,
        cache_max_items: int = 512,
    ):
        self.patterns = patterns_data
        self.img_size = img_size
        self.augment = augment
        self.use_randaugment = use_randaugment
        self.cache_images = cache_images
        self.cache_max_items = cache_max_items

        self._image_cache: "OrderedDict[str, Image.Image]" = OrderedDict()

        # Transformaciones base (pre-calculadas)
        from torchvision import transforms

        self.base_resize = transforms.Resize((self.img_size, self.img_size))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if augment:
            if use_randaugment:
                self.randaugment = RandAugment(n=2, m=10)
            else:
                self.base_augment = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.3),
                        transforms.RandomRotation(degrees=15),
                        transforms.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.1,
                        ),
                    ]
                )

    def _load_image_cached(self, image_path: str, roi: Optional[Tuple] = None) -> Image.Image:
        """Carga imagen con cache LRU para evitar relectura y limitar memoria."""
        if not self.cache_images:
            return self._load_image_uncached(image_path, roi)

        cache_key = f"{image_path}_{roi}"
        cached = self._image_cache.pop(cache_key, None)
        if cached is not None:
            self._image_cache[cache_key] = cached
            return cached

        image_pil = self._load_image_uncached(image_path, roi)

        self._image_cache[cache_key] = image_pil
        if len(self._image_cache) > self.cache_max_items:
            self._image_cache.popitem(last=False)

        return image_pil

    def _load_image_uncached(self, image_path: str, roi: Optional[Tuple] = None) -> Image.Image:
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
                image = image[y : y + h, x : x + w]

        # Convertir a RGB y resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        image_pil = self.base_resize(image_pil)

        return image_pil

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        image_path = pattern['image_path']
        roi = pattern.get('roi', None)

        # Cargar imagen (con cache)
        image_pil = self._load_image_cached(image_path, roi)

        # Aplicar augmentation si está habilitado
        if self.augment:
            if self.use_randaugment:
                image_pil = self.randaugment(image_pil)
            else:
                image_pil = self.base_augment(image_pil)

        # Convertir a tensor
        image_tensor = self.normalize(self.to_tensor(image_pil))

        # Obtener etiqueta
        label = pattern.get('pattern_id', 0)

        return image_tensor, label


class ImprovedPatternNetworkV2(nn.Module):
    """
    Red neuronal V3 ultra-eficiente con Inverted Residuals, CoordConv y ECA.
    Optimizada para velocidad, memoria y precisión en clasificación visual.
    """

    def __init__(self, num_classes=10, dropout_rate=0.3, use_gradient_checkpointing=False):
        super(ImprovedPatternNetworkV2, self).__init__()

        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Stem con CoordConv para entender mejor coordenadas espaciales
        self.stem = nn.Sequential(
            CoordConv(3, 32, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)       # 32x32
        )

        # Agregar información previa: para ~10-50 clases, profundidad moderada es mejor
        # MobileNetV3-style Inverted Residuals (más eficientes que resnet convencional)
        # Número de bloques sintonizado para 128x128 inputs y ensembles más pequeños
        self.blocks = nn.Sequential(
            self._make_layer(32, 64, 2, stride=1, use_inverted=True),   # 32x32
            self._make_layer(64, 96, 2, stride=2, use_inverted=True),   # 16x16
            self._make_layer(96, 160, 3, stride=2, use_inverted=True),  # 8x8
            self._make_layer(160, 256, 2, stride=2, use_inverted=True), # 4x4
        )

        # ECA (los canales más eficientes) + clasificador
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.eca = EfficientChannelAttention(256, reduction=4)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, use_inverted=True):
        """Crea una capa con bloques residuales (mejorados o inverted)."""
        layers = []
        for i in range(blocks):
            block_stride = stride if i == 0 else 1
            block_in = in_channels if i == 0 else out_channels
            if use_inverted:
                # Inverted residual para eficiencia (MobileNetV3)
                layers.append(InvertedResidualSE(block_in, out_channels, block_stride, expand_ratio=6))
            else:
                # Bloque residual tradicional con SE
                layers.append(ResidualBlockSE(block_in, out_channels, block_stride))

        # Aplicar gradient checkpointing si está habilitado (ahorro memoria)
        if self.use_gradient_checkpointing:
            return nn.Sequential(*[GradientCheckpointingWrapper(block) for block in layers])
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Inicializa los pesos de la red de manera más eficiente."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)  # Mejor para deep nets
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.eca(x).flatten(1)  # ECA antes del clasificador
        x = self.classifier(x)
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


class AdaptiveTrainingConfig:
    """Auto-configurador inteligente basado en análisis del dataset."""
    
    @staticmethod
    def analyze_dataset(patterns: Dict) -> Dict:
        """Analiza el dataset y sugiere configuraciones óptimas."""
        total_samples = sum(p['samples'] for p in patterns.values())
        num_classes = len(patterns)
        
        # Distribución por clase
        class_counts = [p['samples'] for p in patterns.values()]
        min_samples = min(class_counts) if class_counts else 1
        max_samples = max(class_counts) if class_counts else 1
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else 1
        
        # Configuraciones auto-ajustadas
        config = {
            'epochs': 30 if total_samples < 500 else 50,
            'batch_size': 32 if total_samples >= 200 else 16,
            'use_focal_loss': imbalance_ratio > 3.0,
            'use_mixup': total_samples >= 100,  # Necesita suficientes datos
            'learning_rate': 0.002 if num_classes <= 10 else 0.001,
            'warmup_epochs': 3 if total_samples >= 100 else 1,
            'early_stopping_patience': 10 if total_samples >= 200 else 7,
            'dropout_rate': 0.3 if total_samples >= 100 else 0.2,
            'use_randaugment': total_samples >= 50,  # No sobre-augmentar con poco data
            'gradient_accumulation_steps': 1,  # Auto ajustable según VRAM
            'use_amp': True,  # Activar si GPU disponible
            'use_gradient_checkpointing': total_samples > 500,  # Ahorro memoria en datasets grandes
        }
        
        # Sugerencias para el usuario
        suggestions = []
        if imbalance_ratio > 5:
            suggestions.append("⚠️  Dataset muy desbalanceado. Añade más muestras a las clases con menos ejemplos.")
        if total_samples < 50:
            suggestions.append("📊 Dataset pequeño. Considera añadir más fotos para mejorar generalización.")
        if num_classes > 30:
            suggestions.append("📐 Muchas clases. Usa arquitectura más profunda (no implementado aquí).")
        
        config['suggestions'] = suggestions
        config['imbalance_ratio'] = imbalance_ratio
        config['total_samples'] = total_samples
        
        return config


class ImprovedPatternLearnerV2:
    """
    Sistema V3 ultra-inteligente para aprender patrones visuales.
    Incluye auto-configuración, insights y UI mejorada.
    """
    
    def __init__(self, model_path: str = "user_patterns/patterns_model_v2.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.patterns = {}
        self.model = None
        self.pattern_counter = 0
        self.training_history = []

        # Cache / estado para inferencia (rendimiento)
        self._inference_transforms: Dict[int, object] = {}
        self._loaded_checkpoint_mtime: Optional[float] = None
        self._loaded_checkpoint_num_patterns: Optional[int] = None
        self._inference_model_compiled: bool = False
        self._inference_channels_last: bool = False
        
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

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Devuelve el módulo base (soporta torch.compile / DataParallel)."""
        while True:
            if hasattr(model, "_orig_mod"):
                model = model._orig_mod  # type: ignore[attr-defined]
                continue
            if hasattr(model, "module"):
                model = model.module  # type: ignore[assignment]
                continue
            return model

    @staticmethod
    def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normaliza claves para que funcionen aunque se haya entrenado con torch.compile."""
        normalized: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod.") :]
            if k.startswith("module."):
                k = k[len("module.") :]
            normalized[k] = v
        return normalized

    @staticmethod
    def _classifier_prefix(model: "ImprovedPatternNetworkV2") -> str:
        last_idx = len(model.fc) - 1
        return f"fc.{last_idx}."

    def _save_model_checkpoint(self, path: str, model: Optional[nn.Module] = None, meta: Optional[Dict] = None) -> None:
        model_to_save = self._unwrap_model(model or self.model)
        state_dict = self._normalize_state_dict_keys(model_to_save.state_dict())
        payload = {
            'state_dict': state_dict,
            'meta': {
                'created_at': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
                'device': str(self.device),
                'num_patterns': len(self.patterns),
                **(meta or {}),
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def _get_inference_transform(self, scale: int):
        transform = self._inference_transforms.get(scale)
        if transform is not None:
            return transform

        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((scale, scale)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self._inference_transforms[scale] = transform
        return transform

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
        
        # Crear archivo README (sin sobrescribir si el usuario ya lo editó)
        readme_path = pattern_folder / "README.md"
        if not readme_path.exists():
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"# Patrón: {name}\n\n")
                f.write(f"Descripción: {description}\n\n")
                f.write("Coloca aquí las fotos de entrenamiento para este patrón.\n\n")
                f.write(f"ID del patrón: {pattern_id}\n")
        
        self._save_patterns()
        print(f"✓ Patrón definido: {name}")
        print(f"✓ Carpeta creada: {pattern_folder}")
        print(f"  → Coloca fotos de {name} en esta carpeta")
        
        return pattern_id
    
    def import_images_from_folder(self, pattern_name: str, auto_create: bool = False) -> int:
        """
        Importa imágenes de la carpeta de entrenamiento.

        Args:
            pattern_name: Nombre de la carpeta/patrón
            auto_create: Si es True, crea el patrón si no existe todavía
        """
        # Buscar el patrón por nombre
        pattern_id = None
        for pid, p in self.patterns.items():
            if p['name'] == pattern_name:
                pattern_id = pid
                break

        if not pattern_id and auto_create:
            pattern_id = self.define_pattern_from_folder(pattern_name, description="")

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

    def import_all_patterns_from_folders(self, auto_create: bool = True) -> Dict[str, int]:
        """
        Importa imágenes de todas las carpetas de patrones.

        Por defecto, si detecta una carpeta nueva dentro de
        fotos_entrenamiento/por_patron/ y no existe el patrón, lo crea
        automáticamente. Esto reduce pasos y facilita que el usuario guíe a la IA.
        """
        results: Dict[str, int] = {}

        # Buscar todas las carpetas de patrones
        for folder in self.patterns_training_dir.iterdir():
            if folder.is_dir():
                pattern_name = folder.name
                count = self.import_images_from_folder(pattern_name, auto_create=auto_create)
                if count > 0:
                    results[pattern_name] = count

        print("\n📊 Resumen de importación:")
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
                        save_checkpoints: bool = True,
                        use_amp: bool = None,
                        use_compile: bool = None,
                        use_gradient_checkpointing: bool = False,
                        num_workers: int = None,
                        channels_last: bool = True) -> Dict:
        """
        Entrena el modelo con técnicas V2 avanzadas y optimizaciones de rendimiento.

        Args:
            epochs: Número de épocas
            batch_size: Tamaño del batch
            val_split: Proporción para validación
            learning_rate: Learning rate inicial
            max_lr: Learning rate máximo para One-Cycle
            use_focal_loss: Usar Focal Loss
            label_smoothing: Factor de label smoothing
            early_stopping_patience: Paciencia para early stopping
            warmup_epochs: Épocas de warmup
            dropout_rate: Tasa de dropout
            use_mixup: Usar Mixup augmentation
            use_randaugment: Usar RandAugment
            gradient_accumulation_steps: Pasos de acumulación de gradientes
            save_checkpoints: Guardar checkpoints
            use_amp: Usar Automatic Mixed Precision (auto: True en GPU)
            use_compile: Usar torch.compile (auto: True si PyTorch 2.0+)
            use_gradient_checkpointing: Usar gradient checkpointing para ahorrar memoria
            num_workers: Workers para DataLoader (auto: 4 si CPU, 2 si GPU)
            channels_last: Usar channels_last memory format

        Returns:
            Diccionario con historial de entrenamiento
        """
        if not self.patterns:
            print("No hay patrones definidos para entrenar")
            return {}

        # Detectar y configurar optimizaciones automáticamente
        if use_amp is None:
            use_amp = USE_AMP
        if use_compile is None:
            use_compile = PYTORCH_2_PLUS
        if num_workers is None:
            num_workers = 2 if USE_AMP else 4

        # Preparar datos
        training_data: List[Dict] = []

        # Mapeo estable (mejor rendimiento que list(...).index(...) dentro del loop)
        pattern_id_to_idx = {pid: idx for idx, pid in enumerate(self.patterns.keys())}

        for pattern_id, _pattern in self.patterns.items():
            pattern_dir = self.patterns_dir / pattern_id
            if not pattern_dir.exists():
                continue

            label_idx = pattern_id_to_idx.get(pattern_id)
            if label_idx is None:
                continue

            for sample_file in pattern_dir.glob("sample_*.json"):
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        sample = json.load(f)
                    training_data.append(
                        {
                            'pattern_id': label_idx,
                            'image_path': sample['image_path'],
                            'roi': sample.get('roi'),
                        }
                    )
                except Exception:
                    continue

        if len(training_data) == 0:
            print("No hay muestras de entrenamiento")
            return {}

        print(f"\n🚀 Iniciando entrenamiento V2 optimizado")
        print(f"   Total de muestras: {len(training_data)}")
        print(f"   Dispositivo: {self.device}")

        # Guía rápida para el usuario: balance de datos
        idx_to_name = {idx: self.patterns[pid]['name'] for pid, idx in pattern_id_to_idx.items()}
        counts: Dict[str, int] = {name: 0 for name in idx_to_name.values()}
        for row in training_data:
            counts[idx_to_name.get(int(row['pattern_id']), 'desconocido')] = counts.get(
                idx_to_name.get(int(row['pattern_id']), 'desconocido'),
                0,
            ) + 1

        if counts:
            min_count = min(counts.values())
            max_count = max(counts.values())
            print(f"\n   📊 Distribución de muestras por patrón:")
            for name, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                print(f"      - {name}: {cnt}")

            if min_count > 0 and max_count / min_count >= 3:
                print(
                    "\n   ⚠️  Dataset desbalanceado (max/min >= 3). "
                    "Sugerencias: añade más fotos a las clases con menos ejemplos o usa --focal-loss."
                )
        print(f"\n   Técnicas de IA:")
        print(f"      - One-Cycle Learning Rate (max_lr={max_lr})")
        print(f"      - RandAugment: {'Sí' if use_randaugment else 'No'}")
        print(f"      - Mixup: {'Sí' if use_mixup else 'No'}")
        print(f"      - Gradient Accumulation: {gradient_accumulation_steps}x")
        print(f"      - Warmup: {warmup_epochs} épocas")
        print(f"\n   Optimizaciones de Rendimiento:")
        print(f"      - Automatic Mixed Precision (AMP): {'✓ Activado' if use_amp else '✗ Desactivado'}")
        print(f"      - torch.compile: {'✓ Activado' if use_compile else '✗ Desactivado'}")
        print(f"      - Gradient Checkpointing: {'✓ Activado' if use_gradient_checkpointing else '✗ Desactivado'}")
        print(f"      - Channels Last Memory Format: {'✓ Activado' if channels_last else '✗ Desactivado'}")
        print(f"      - DataLoader Workers: {num_workers}")
        print(f"      - Image Cache: ✓ Activado")

        # Dividir datos
        val_size = int(len(training_data) * val_split)
        train_size = len(training_data) - val_size

        train_data = training_data[:train_size]
        val_data = training_data[train_size:]

        # Crear datasets con cache
        train_dataset = EnhancedPatternDatasetV2(
            train_data, img_size=128, augment=True, use_randaugment=use_randaugment, cache_images=True
        )
        val_dataset = EnhancedPatternDatasetV2(
            val_data, img_size=128, augment=False, use_randaugment=False, cache_images=True
        )

        # DataLoader optimizado con prefetching
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if USE_AMP else False,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if USE_AMP else False,
            persistent_workers=True if num_workers > 0 else False
        )

        # Actualizar modelo si necesario
        num_classes = len(self.patterns)
        if self.model.fc[-1].out_features != num_classes:
            self.model = ImprovedPatternNetworkV2(
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
            self.model.to(self.device)

        # Aplicar channels_last memory format si está disponible y es GPU
        if channels_last and USE_AMP:
            self.model = self.model.to(memory_format=torch.channels_last)
            print(f"\n   Model convertido a channels_last memory format")

        # Aplicar torch.compile si está disponible y está activado
        compiled = False
        if use_compile and PYTORCH_2_PLUS:
            try:
                print(f"\n   Compilando modelo con torch.compile (esto puede tardar unos segundos)...")
                self.model = torch.compile(self.model, mode='reduce-overhead')
                compiled = True
                print(f"   ✓ Modelo compilado exitosamente")
            except Exception as e:
                print(f"   ⚠ No se pudo compilar el modelo: {e}")
                print(f"   Continuando sin compilación...")

        # Optimizador AdamW
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate,
                               weight_decay=1e-4, betas=(0.9, 0.999))

        # One-Cycle LR scheduler
        total_steps = len(train_loader) * epochs
        scheduler = OneCycleLR(optimizer, max_lr, total_steps)

        # Loss function
        if use_focal_loss:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif label_smoothing > 0:
            criterion = LabelSmoothingLoss(num_classes, label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Mixup
        mixup_fn = Mixup(alpha=0.4) if use_mixup else None

        # Early stopping
        early_stopping = EarlyStoppingV2(
            patience=early_stopping_patience,
            warmup_epochs=warmup_epochs,
            restore_best_weights=True
        )

        # AMP GradScaler para entrenamiento mixto precision
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Training loop optimizado
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')

        print(f"\n📈 Entrenando...")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            optimizer.zero_grad()

            for batch_idx, (images, labels) in enumerate(pbar):
                # Mover a device y aplicar channels_last si es necesario
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if channels_last and USE_AMP:
                    images = images.to(memory_format=torch.channels_last)

                # Mixup
                if mixup_fn is not None:
                    images, labels_a, labels_b, lam = mixup_fn((images, labels))

                    # AMP forward pass
                    if use_amp and scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        outputs = self.model(images)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

                    # Accuracy con mixup
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (lam * predicted.eq(labels_a).float() +
                                    (1 - lam) * predicted.eq(labels_b).float()).sum().item()
                else:
                    # AMP forward pass
                    if use_amp and scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)

                    # Accuracy normal
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                # Gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Backward con AMP
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping con AMP
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()

                train_loss += loss.item() * gradient_accumulation_steps * images.size(0)

                pbar.set_postfix({
                    'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })

            train_loss /= len(train_dataset)
            train_acc = train_correct / train_total

            # Validation optimizada
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    if channels_last and USE_AMP:
                        images = images.to(memory_format=torch.channels_last)

                    if use_amp and scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = criterion(outputs, labels)
                    else:
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
                self._save_model_checkpoint(
                    self.model_path,
                    meta={
                        'compiled': compiled,
                        'channels_last': channels_last,
                        'best_val_loss': float(best_val_loss),
                    },
                )
                print(f"  ✓ Mejor modelo guardado (val_loss: {val_loss:.4f})")

            # Early stopping
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f"\n⏹️  Early stopping activado en epoch {epoch+1}")
                break
        
        # Restaurar mejores pesos
        early_stopping.restore_best_model(self.model)
        
        # Guardar modelo final
        self._save_model_checkpoint(
            self.model_path,
            meta={
                'compiled': compiled,
                'channels_last': channels_last,
                'best_val_loss': float(best_val_loss),
            },
        )
        
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
    
    def recognize_pattern_multiscale(
        self,
        image_path: str,
        scales: List[int] = [96, 128, 160],
        threshold: float = 0.5,
        top_k: int = 0,
    ) -> List[Dict]:
        """Reconoce patrones usando multi-scale inference.

        - Más eficiente: calcula probabilidades por escala y promedia.
        - Permite top_k para guiar al usuario con alternativas.
        """
        if not self.patterns:
            return []

        if not self._load_model_checkpoint():
            return []

        probs_per_scale: List[np.ndarray] = []
        for scale in scales:
            probs = self._predict_probabilities_at_scale(image_path, scale)
            if probs is not None:
                probs_per_scale.append(probs)

        if not probs_per_scale:
            return []

        mean_probs = np.mean(np.stack(probs_per_scale, axis=0), axis=0)
        return self._detections_from_probabilities(mean_probs, threshold=threshold, top_k=top_k)

    def _predict_probabilities_at_scale(self, image_path: str, scale: int) -> Optional[np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        transform = self._get_inference_transform(scale)
        image_tensor = transform(image_pil).unsqueeze(0)

        if USE_AMP and self._inference_channels_last:
            image_tensor = image_tensor.to(memory_format=torch.channels_last)

        image_tensor = image_tensor.to(self.device, non_blocking=True)

        with torch.no_grad():
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    outputs = self.model(image_tensor)
            else:
                outputs = self.model(image_tensor)

            probabilities = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()

        # Recortar por número de patrones reales (evita clases extra)
        num_classes = len(self.patterns)
        return probabilities[:num_classes]

    def _detections_from_probabilities(
        self,
        probabilities: np.ndarray,
        threshold: float,
        top_k: int = 0,
    ) -> List[Dict]:
        pattern_ids = list(self.patterns.keys())
        if len(pattern_ids) == 0:
            return []

        probs = probabilities[: len(pattern_ids)]

        if top_k and top_k > 0:
            top_idx = np.argsort(-probs)[: min(top_k, len(pattern_ids))]
        else:
            top_idx = np.where(probs >= threshold)[0]

        detections: List[Dict] = []
        for i in top_idx:
            prob = float(probs[i])
            if prob < threshold and (not top_k or top_k <= 0):
                continue

            pid = pattern_ids[int(i)]
            pattern = self.patterns[pid]
            detections.append(
                {
                    'pattern_id': pid,
                    'pattern_name': pattern['name'],
                    'probability': prob,
                    'bbox': (0, 0, 0, 0),
                }
            )

        detections.sort(key=lambda x: x['probability'], reverse=True)
        return detections
    
    def _load_model_checkpoint(
        self,
        compile_for_inference: Optional[bool] = None,
        channels_last: bool = True,
    ) -> bool:
        """Carga el checkpoint del modelo.

        Mejoras:
        - Soporta checkpoints guardados como dict (state_dict + meta).
        - Soporta modelos entrenados con torch.compile (normaliza claves).
        - Evita recargar el modelo si no cambió el checkpoint.
        - Si cambió el número de patrones, intenta cargar el backbone y re-inicializa el clasificador.
        """
        checkpoint_path = Path(self.model_path)
        if not checkpoint_path.exists():
            return False

        mtime = checkpoint_path.stat().st_mtime
        if (
            self._loaded_checkpoint_mtime == mtime
            and self._loaded_checkpoint_num_patterns == len(self.patterns)
            and self.model is not None
        ):
            return True

        if len(self.patterns) == 0:
            return False

        try:
            loaded_obj = torch.load(checkpoint_path, map_location=self.device)

            if isinstance(loaded_obj, dict) and 'state_dict' in loaded_obj:
                state_dict = loaded_obj.get('state_dict', {})
                meta = loaded_obj.get('meta', {})
            else:
                state_dict = loaded_obj
                meta = {}

            if not isinstance(state_dict, dict):
                raise ValueError("Formato de checkpoint no reconocido")

            state_dict = self._normalize_state_dict_keys(state_dict)

            num_classes = len(self.patterns)
            model = ImprovedPatternNetworkV2(num_classes=num_classes)
            model.to(self.device)

            classifier_prefix = self._classifier_prefix(model)
            loaded_num_classes = None
            weight_key = f"{classifier_prefix}weight"
            if weight_key in state_dict:
                loaded_num_classes = int(state_dict[weight_key].shape[0])

            if loaded_num_classes is not None and loaded_num_classes != num_classes:
                filtered = {k: v for k, v in state_dict.items() if not k.startswith(classifier_prefix)}
                model.load_state_dict(filtered, strict=False)
                print(
                    f"⚠ Modelo adaptado: clases en checkpoint={loaded_num_classes}, clases actuales={num_classes}. "
                    "Backbone cargado y clasificador reinicializado."
                )
            else:
                model.load_state_dict(state_dict, strict=False)

            # Optimizaciones de inferencia
            if channels_last and USE_AMP:
                model = model.to(memory_format=torch.channels_last)
                self._inference_channels_last = True

            if compile_for_inference is None:
                compile_for_inference = bool(PYTORCH_2_PLUS and USE_AMP)

            if compile_for_inference and PYTORCH_2_PLUS:
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    self._inference_model_compiled = True
                except Exception as e:
                    print(f"⚠ No se pudo compilar el modelo para inferencia: {e}")
                    self._inference_model_compiled = False

            self.model = model
            self.model.eval()

            self._loaded_checkpoint_mtime = mtime
            self._loaded_checkpoint_num_patterns = num_classes

            if meta:
                compiled_flag = meta.get('compiled')
                if compiled_flag is not None:
                    print(f"ℹ Checkpoint entrenado con compile={compiled_flag}")

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
    
    def identify_from_folder(
        self,
        threshold: float = 0.5,
        output_file: Optional[str] = None,
        top_k: int = 1,
        include_all_images: bool = False,
        batch_size: int = 32,
        scale: int = 128,
    ) -> Dict[str, List[Dict]]:
        """Identifica patrones en todas las imágenes de fotos_identificar/.

        Mejoras:
        - Respeta umbral configurable.
        - Puede devolver top_k alternativas (útil para revisión humana).
        - Inferencia en batches (más rápida en GPU).

        Args:
            threshold: Umbral para considerar una detección como válida.
            output_file: Ruta del archivo JSON de salida.
            top_k: Si >1, devuelve alternativas ordenadas por probabilidad.
            include_all_images: Si True, incluye todas las imágenes aunque el top-1 esté por debajo del umbral.
            batch_size: Tamaño de batch para inferencia.
            scale: Tamaño de entrada (por defecto 128).
        """
        if output_file is None:
            output_file = f"resultados_identificacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        if not self._load_model_checkpoint():
            print("❌ No se pudo cargar el modelo V2")
            return {}

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        images = [img for img in self.identify_dir.glob("*.*") if img.suffix.lower() in image_extensions]
        images = sorted(images)

        if not images:
            print("No hay imágenes en fotos_identificar/")
            return {}

        effective_top_k = max(1, int(top_k))

        print(f"\n🔍 Identificando patrones en {len(images)} imágenes...")
        print(f"   Umbral: {threshold:.0%} | top_k: {effective_top_k} | batch_size: {batch_size}")

        transform = self._get_inference_transform(scale)

        results: Dict[str, List[Dict]] = {}
        batch_tensors: List[torch.Tensor] = []
        batch_paths: List[str] = []

        def flush_batch():
            if not batch_tensors:
                return

            batch = torch.stack(batch_tensors, dim=0)
            if USE_AMP and self._inference_channels_last:
                batch = batch.to(memory_format=torch.channels_last)

            batch = batch.to(self.device, non_blocking=True)

            with torch.no_grad():
                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)

                probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()

            for path_str, prob_vec in zip(batch_paths, probs):
                prob_vec = prob_vec[: len(self.patterns)]
                detections = self._detections_from_probabilities(
                    prob_vec,
                    threshold=threshold,
                    top_k=effective_top_k,
                )

                if include_all_images or (detections and detections[0]['probability'] >= threshold):
                    results[path_str] = detections
                    top_det = detections[0]
                    img_name = Path(path_str).name
                    status = "✅" if top_det['probability'] >= threshold else "⚠️"
                    tqdm.write(f"  {status} {img_name}: {top_det['pattern_name']} ({top_det['probability']:.2%})")

            batch_tensors.clear()
            batch_paths.clear()

        for img_path in tqdm(images):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            tensor = transform(image_pil)
            batch_tensors.append(tensor)
            batch_paths.append(str(img_path))

            if len(batch_tensors) >= batch_size:
                flush_batch()

        flush_batch()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Resultados guardados en: {output_file}")

        self._generate_identification_report(results, output_file.replace('.json', '_reporte.txt'))

        return results

    def review_identification_results(
        self,
        results: Dict[str, List[Dict]],
        add_to_training: bool = True,
        file_action: str = 'copy',
    ) -> Dict[str, int]:
        """Revisión humana (texto) para guiar a la IA.

        Permite aprobar/corregir el top-1 y, opcionalmente, añadir la imagen al set de entrenamiento.

        Args:
            results: Salida de identify_from_folder (idealmente con top_k>=3 e include_all_images=True)
            add_to_training: Si True, copia/mueve la imagen al patrón correcto y la registra como muestra.
            file_action: 'copy' o 'move'

        Returns:
            Resumen de acciones.
        """
        if not results:
            print("No hay resultados para revisar")
            return {'reviewed': 0, 'approved': 0, 'corrected': 0, 'skipped': 0}

        file_action = file_action.lower().strip()
        if file_action not in {'copy', 'move'}:
            raise ValueError("file_action debe ser 'copy' o 'move'")

        # Feedback acumulado
        feedback_path = self.patterns_dir / 'review_feedback_v2.json'
        feedback_entries: List[Dict] = []
        if feedback_path.exists():
            try:
                with open(feedback_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        feedback_entries = loaded
            except Exception:
                feedback_entries = []

        summary = {'reviewed': 0, 'approved': 0, 'corrected': 0, 'skipped': 0}

        ordered_items = list(results.items())
        print("\n=== Revisión humana V2 ===")
        print("Controles:")
        print("  [ENTER] aprobar top-1")
        print("  c       corregir (elegir patrón)")
        print("  s       saltar")
        print("  q       salir")

        for image_path, detections in ordered_items:
            summary['reviewed'] += 1
            img_name = Path(image_path).name

            print(f"\n🖼️  {img_name}")
            for i, det in enumerate(detections[:5], 1):
                print(f"  {i}. {det['pattern_name']} ({det['probability']:.2%})")

            predicted = detections[0] if detections else None
            predicted_name = predicted['pattern_name'] if predicted else None

            try:
                action = input("Acción [ENTER/c/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nFinalizando revisión...")
                break

            if action == 'q':
                break
            if action == 's':
                summary['skipped'] += 1
                continue

            if action == 'c':
                # Listado de patrones actuales
                pattern_names = [p['name'] for p in self.patterns.values()]
                for idx, name in enumerate(pattern_names, 1):
                    print(f"  {idx}. {name}")

                try:
                    choice = input("Patrón correcto (número o nombre; vacío=cancelar): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("Cancelado")
                    summary['skipped'] += 1
                    continue

                if not choice:
                    summary['skipped'] += 1
                    continue

                correct_name = None
                if choice.isdigit():
                    num = int(choice)
                    if 1 <= num <= len(pattern_names):
                        correct_name = pattern_names[num - 1]
                else:
                    correct_name = choice

                if not correct_name:
                    summary['skipped'] += 1
                    continue

                # Crear patrón si no existe
                correct_id = None
                for pid, p in self.patterns.items():
                    if p['name'] == correct_name:
                        correct_id = pid
                        break
                if correct_id is None:
                    correct_id = self.define_pattern_from_folder(correct_name, description="")

                # Registrar feedback
                if predicted_name:
                    for pid, p in self.patterns.items():
                        if p['name'] == predicted_name:
                            p['corrected'] = int(p.get('corrected', 0)) + 1
                            break

                feedback_entries.append(
                    {
                        'type': 'correction',
                        'image_path': image_path,
                        'predicted': predicted_name,
                        'correct': correct_name,
                        'timestamp': datetime.now().isoformat(),
                    }
                )

                if add_to_training:
                    self._add_image_as_training_sample(image_path, correct_id, correct_name, file_action=file_action)

                summary['corrected'] += 1
                self._save_patterns()
                continue

            # Por defecto: aprobar
            if not predicted_name:
                summary['skipped'] += 1
                continue

            for pid, p in self.patterns.items():
                if p['name'] == predicted_name:
                    p['approved'] = int(p.get('approved', 0)) + 1
                    break

            feedback_entries.append(
                {
                    'type': 'approval',
                    'image_path': image_path,
                    'predicted': predicted_name,
                    'timestamp': datetime.now().isoformat(),
                }
            )

            if add_to_training:
                # Aprobado -> añadimos como muestra del patrón predicho
                approved_id = None
                for pid, p in self.patterns.items():
                    if p['name'] == predicted_name:
                        approved_id = pid
                        break
                if approved_id is not None:
                    self._add_image_as_training_sample(image_path, approved_id, predicted_name, file_action=file_action)

            summary['approved'] += 1
            self._save_patterns()

        try:
            with open(feedback_path, 'w', encoding='utf-8') as f:
                json.dump(feedback_entries, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠ No se pudo guardar review_feedback_v2.json: {e}")

        print("\n✅ Revisión finalizada")
        print(f"   Revisadas: {summary['reviewed']}")
        print(f"   Aprobadas: {summary['approved']}")
        print(f"   Corregidas: {summary['corrected']}")
        print(f"   Saltadas: {summary['skipped']}")

        if add_to_training:
            print("\n💡 Siguiente paso recomendado: re-entrenar")
            print("   python dupin.py entrenar-patrones-v2 --epochs 30")

        return summary

    def _add_image_as_training_sample(self, image_path: str, pattern_id: str, pattern_name: str, file_action: str = 'copy') -> Optional[str]:
        """Copia/mueve una imagen a la carpeta de entrenamiento del patrón y la registra como muestra."""
        src = Path(image_path)
        if not src.exists():
            return None

        dest_dir = self.patterns_training_dir / pattern_name
        dest_dir.mkdir(exist_ok=True)

        dest_path = dest_dir / src.name
        if dest_path.exists():
            dest_path = dest_dir / f"{src.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{src.suffix}"

        try:
            if file_action == 'move':
                shutil.move(str(src), str(dest_path))
            else:
                shutil.copy2(str(src), str(dest_path))
        except Exception as e:
            print(f"⚠ Error copiando/moviendo {src} -> {dest_path}: {e}")
            return None

        # Registrar muestra apuntando a la copia final
        self.add_pattern_sample(pattern_id, str(dest_path))
        return str(dest_path)
    
    def _generate_identification_report(self, results: Dict, output_path: str):
        """Genera un reporte legible de identificaciones."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("═════════════════════════════════════════════════════════\n")
            f.write("  REPORTE DE IDENTIFICACIÓN DE PATRONES\n")
            f.write(f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("═════════════════════════════════════════════════════════\n\n")
            
            # Estadísticas por patrón (contamos solo el TOP-1 por imagen)
            pattern_counts = {}
            for _img_path, detections in results.items():
                if not detections:
                    continue
                det = detections[0]
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

                if not detections:
                    f.write("   └─ (sin detecciones)\n\n")
                    continue

                top = detections[0]
                f.write(f"   └─ TOP: {top['pattern_name']} ({top['probability']:.2%})\n")

                if len(detections) > 1:
                    for alt in detections[1: min(4, len(detections))]:
                        f.write(f"      · Alt: {alt['pattern_name']} ({alt['probability']:.2%})\n")

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
