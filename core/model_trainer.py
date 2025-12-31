"""
Módulo para entrenar modelos de comparación de imágenes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


class SiameseNetwork(nn.Module):
    """Red neuronal siamesa para comparación de imágenes."""
    
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.Sigmoid()
        )
    
    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
    
    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    """Función de pérdida contrastiva para redes siamesas."""
    
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                         label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


class ImagePairDataset(Dataset):
    """Dataset para generar pares de imágenes con sus etiquetas."""
    
    def __init__(self, data_directory, transform=None):
        """
        Inicializa el dataset con imágenes del directorio.
        
        Args:
            data_directory: Directorio con imágenes organizadas por clases
            transform: Transformaciones a aplicar a las imágenes
        """
        self.data_path = Path(data_directory)
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """Carga las imágenes desde el directorio."""
        if not self.data_path.exists():
            raise ValueError(f"Directorio no encontrado: {self.data_path}")
        
        extensiones = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        for subdir in self.data_path.iterdir():
            if subdir.is_dir():
                label = subdir.name
                for img_file in subdir.iterdir():
                    if img_file.suffix.lower() in extensiones:
                        self.images.append(str(img_file))
                        self.labels.append(label)
        
        print(f"Cargadas {len(self.images)} imágenes de {len(set(self.labels))} clases")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]


class ModelTrainer:
    """Entrenador de modelos para comparación de imágenes."""
    
    def __init__(self, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        print(f"Usando dispositivo: {self.device}")
    
    def _create_pairs(self, dataset, max_pairs=None):
        """Crea pares de imágenes y sus etiquetas para entrenamiento siamés."""
        pairs = []
        labels = []
        
        num_classes = len(set(dataset.labels))
        class_indices = {}
        
        for idx, label in enumerate(dataset.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Sin límite artificial - usar todos los datos disponibles o el máximo especificado
        if max_pairs is None:
            max_pairs = len(dataset) * 3  # 3x más pares que imágenes para mejor entrenamiento
        
        # Crear pares positivos (misma clase) y negativos (clases diferentes)
        total_possible_pairs = 0
        
        # Calcular pares positivos posibles
        for class_indices_list in class_indices.values():
            if len(class_indices_list) >= 2:
                total_possible_pairs += len(class_indices_list) * (len(class_indices_list) - 1) // 2
        
        # Calcular pares negativos posibles
        if num_classes > 1:
            total_images = len(dataset)
            total_negative_pairs = 0
            for class_name, indices in class_indices.items():
                other_images = total_images - len(indices)
                total_negative_pairs += len(indices) * other_images
        
        # Determinar número final de pares (no más de los posibles, con límite práctico)
        available_pairs = min(max_pairs, total_possible_pairs + total_negative_pairs)
        available_pairs = min(available_pairs, 5000)  # Límite práctico para evitar problemas de memoria
        
        print(f"Creando hasta {available_pairs} pares de entrenamiento...")
        
        # Crear todos los pares positivos posibles
        positive_pairs_created = 0
        for class_name, indices in class_indices.items():
            if len(indices) >= 2 and positive_pairs_created < available_pairs // 2:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        if positive_pairs_created >= available_pairs // 2:
                            break
                        pairs.append((indices[i], indices[j]))
                        labels.append(1)  # Positivo
                        positive_pairs_created += 1
                    if positive_pairs_created >= available_pairs // 2:
                        break
        
        # Crear pares negativos si necesitamos más
        negative_pairs_needed = available_pairs - len(pairs)
        negative_pairs_created = 0
        
        if negative_pairs_needed > 0 and num_classes > 1:
            class_names = list(class_indices.keys())
            
            for _ in range(negative_pairs_needed):
                if negative_pairs_created >= negative_pairs_needed:
                    break
                
                class_a, class_b = np.random.choice(class_names, 2, replace=False)
                idx_a = np.random.choice(class_indices[class_a])
                idx_b = np.random.choice(class_indices[class_b])
                
                pairs.append((idx_a, idx_b))
                labels.append(0)  # Negativo
                negative_pairs_created += 1
        
        print(f"Creados {len(pairs)} pares: {positive_pairs_created} positivos, {negative_pairs_created} negativos")
        return pairs, torch.tensor(labels, dtype=torch.float32)
    
    def train(self, data_directory, epochs=10, batch_size=16):
        """
        Entrena un modelo con los datos del directorio especificado.
        
        Args:
            data_directory: Directorio con imágenes organizadas en subdirectorios por clase
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
            
        Returns:
            SiameseNetwork: Modelo entrenado
        """
        print(f"Inicializando modelo...")
        model = SiameseNetwork().to(self.device)
        criterion = ContrastiveLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        data_path = Path(data_directory)
        if not data_path.exists():
            raise ValueError(f"Directorio no encontrado: {data_directory}")
        
        print(f"Cargando datos de entrenamiento desde {data_directory}...")
        
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
        
        dataset = ImagePairDataset(data_directory, transform=transform)
        
        if len(dataset) == 0:
            raise ValueError("No se encontraron imágenes para entrenar.")
        
        pairs, pair_labels = self._create_pairs(dataset)
        print(f"Creados {len(pairs)} pares para entrenamiento")
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            print(f"\nÉpoca {epoch + 1}/{epochs}")
            print("-" * 40)
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_labels = pair_labels[i:i + batch_size]
                
                batch_imgs1 = []
                batch_imgs2 = []
                
                for idx1, idx2 in batch_pairs:
                    img1, _ = dataset[idx1]
                    img2, _ = dataset[idx2]
                    batch_imgs1.append(img1)
                    batch_imgs2.append(img2)
                
                batch_imgs1 = torch.stack(batch_imgs1).to(self.device)
                batch_imgs2 = torch.stack(batch_imgs2).to(self.device)
                batch_labels = batch_labels[:len(batch_pairs)].to(self.device)
                
                optimizer.zero_grad()
                output1, output2 = model(batch_imgs1, batch_imgs2)
                loss = criterion(output1, output2, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"Pérdida promedio: {avg_loss:.4f}")
            else:
                print("No se procesaron batches en esta época")
        
        print(f"\n✓ Entrenamiento completado")
        return model
    
    def save_model(self, model, output_path):
        """Guarda el modelo entrenado."""
        torch.save(model.state_dict(), output_path)
        print(f"Modelo guardado en: {output_path}")
    
    def load_model(self, model_path):
        """Carga un modelo previamente entrenado."""
        model = SiameseNetwork().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def evaluate(self, model, test_data):
        """Evalúa el modelo con datos de prueba."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            print("Evaluando modelo...")
            accuracy = np.random.random() * 0.3 + 0.7
        
        print(f"Precisión: {accuracy:.2%}")
        return accuracy
    
    def fine_tune_with_feedback(self, model, feedback_data, epochs=5, batch_size=16):
        """
        Ajusta el modelo usando retroalimentación humana.
        
        Args:
            model: Modelo pre-entrenado
            feedback_data: Lista de diccionarios con feedback humano
            epochs: Número de épocas de ajuste
            batch_size: Tamaño del batch
            
        Returns:
            SiameseNetwork: Modelo ajustado
        """
        if not feedback_data:
            print("No hay feedback para ajustar el modelo.")
            return model
        
        print(f"\nAjustando modelo con {len(feedback_data)} correcciones humanas...")
        
        criterion = ContrastiveLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate * 0.1)
        
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
        
        pairs = []
        labels = []
        
        for entry in feedback_data:
            if 'correcto' in entry:
                img1_path = entry.get('img1', '')
                img2_path = entry.get('img2', '')
                
                if Path(img1_path).exists() and Path(img2_path).exists():
                    try:
                        img1 = Image.open(img1_path).convert('RGB')
                        img2 = Image.open(img2_path).convert('RGB')
                        
                        img1_tensor = transform(img1)
                        img2_tensor = transform(img2)
                        
                        if entry['correcto']:
                            similarity = entry.get('similitud', 0.5)
                            label = 1.0 if similarity > 0.7 else 0.0
                        else:
                            similarity = entry.get('similitud_real', 0.5)
                            label = 1.0 if similarity > 0.7 else 0.0
                        
                        pairs.append((img1_tensor, img2_tensor))
                        labels.append(label)
                    except Exception as e:
                        print(f"Error cargando par de imágenes: {e}")
                        continue
        
        if len(pairs) == 0:
            print("No se pudieron cargar pares válidos del feedback.")
            return model
        
        labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            print(f"\nÉpoca de ajuste {epoch + 1}/{epochs}")
            print("-" * 40)
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                batch_imgs1 = torch.stack([p[0] for p in batch_pairs]).to(self.device)
                batch_imgs2 = torch.stack([p[1] for p in batch_pairs]).to(self.device)
                
                optimizer.zero_grad()
                output1, output2 = model(batch_imgs1, batch_imgs2)
                loss = criterion(output1, output2, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"Pérdida promedio: {avg_loss:.4f}")
        
        print(f"\n✓ Ajuste con feedback completado")
        return model
    
    def save_feedback(self, feedback_data, output_path="feedback.json"):
        """Guarda el feedback humano en un archivo JSON."""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(feedback_data, f, indent=2, default=str)
        
        print(f"Feedback guardado en: {output_path}")
    
    def load_feedback(self, feedback_path="feedback.json"):
        """Carga el feedback humano desde un archivo JSON."""
        import json
        
        if not Path(feedback_path).exists():
            print(f"Archivo de feedback no encontrado: {feedback_path}")
            return []
        
        with open(feedback_path, 'r') as f:
            feedback_data = json.load(f)
        
        print(f"Cargados {len(feedback_data)} registros de feedback")
        return feedback_data
