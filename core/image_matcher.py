"""
Módulo para comparar y encontrar similitudes entre imágenes
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from typing import Dict, Optional, Tuple, List


class ImageMatcher:
    """Compara imágenes y calcula similitudes."""
    
    def __init__(self, metodo='orb', model_path='modelo.pth'):
        """
        Inicializa el matcher con el método especificado.
        
        Args:
            metodo: Método de comparación ('orb', 'sift', 'histogram', 'ssim', 'siamese')
            model_path: Ruta al modelo siamés entrenado
        """
        self.metodo = metodo
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        if metodo == 'orb':
            self.detector = cv2.ORB_create()
        elif metodo == 'sift':
            self.detector = cv2.SIFT_create()
        elif metodo == 'siamese':
            self._load_siamese_model()
    
    def _load_siamese_model(self):
        """Carga el modelo siamés para comparación."""
        from core.model_trainer import SiameseNetwork
        try:
            if Path(self.model_path).exists():
                self.model = SiameseNetwork().to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                print(f"✓ Modelo siamés cargado desde {self.model_path}")
            else:
                print(f"⚠️ Modelo siamés no encontrado en {self.model_path}. Usando ORB por defecto.")
                self.metodo = 'orb'
                self.detector = cv2.ORB_create()
        except Exception as e:
            print(f"❌ Error cargando modelo siamés: {e}")
            self.metodo = 'orb'
            self.detector = cv2.ORB_create()
    
    def cargar_imagen(self, ruta):
        """Carga una imagen desde la ruta especificada."""
        if not Path(ruta).exists():
            raise FileNotFoundError(f"Imagen no encontrada: {ruta}")
        
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta}")
        
        return imagen
    
    def compare(self, imagen1_path, imagen2_path, roi1=None, roi2=None):
        """
        Compara dos imágenes y retorna un score de similitud (0.0 - 1.0).
        
        Args:
            imagen1_path: Ruta a la primera imagen
            imagen2_path: Ruta a la segunda imagen
            roi1: Región de interés en imagen 1 (x, y, w, h) - opcional
            roi2: Región de interés en imagen 2 (x, y, w, h) - opcional
            
        Returns:
            float: Score de similitud entre 0.0 y 1.0
        """
        img1 = self.cargar_imagen(imagen1_path)
        img2 = self.cargar_imagen(imagen2_path)
        
        # Extraer ROIs si se especifican
        if roi1:
            x, y, w, h = roi1
            img1 = img1[y:y+h, x:x+w]
        
        if roi2:
            x, y, w, h = roi2
            img2 = img2[y:y+h, x:x+w]
        
        if self.metodo in ['orb', 'sift']:
            return self._compare_features(img1, img2)
        elif self.metodo == 'histogram':
            return self._compare_histogram(img1, img2)
        elif self.metodo == 'ssim':
            return self._compare_ssim(img1, img2)
        elif self.metodo == 'siamese':
            return self._compare_siamese(img1, img2)
        else:
            return self._compare_features(img1, img2)
    
    def _compare_siamese(self, img1, img2):
        """Compara imágenes usando la red neuronal siamesa."""
        if self.model is None:
            return 0.0
            
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
        
        # Convertir a PIL
        img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        
        # Transformar y mover al dispositivo
        tensor1 = transform(img1_pil).unsqueeze(0).to(self.device)
        tensor2 = transform(img2_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output1, output2 = self.model(tensor1, tensor2)
            # Calcular distancia euclidiana
            dist = torch.nn.functional.pairwise_distance(output1, output2).item()
            
            # Convertir distancia a similitud (la distancia suele estar entre 0 y 2)
            # Similitud = 1 / (1 + distancia) o algo similar
            similarity = max(0.0, 1.0 - (dist / 2.0))
            
        return similarity
    
    def compare_with_details(self, imagen1_path, imagen2_path, roi1=None, roi2=None):
        """
        Compara dos imágenes y retorna detalles completos de la comparación.
        
        Args:
            imagen1_path: Ruta a la primera imagen
            imagen2_path: Ruta a la segunda imagen
            roi1: Región de interés en imagen 1 (x, y, w, h) - opcional
            roi2: Región de interés en imagen 2 (x, y, w, h) - opcional
            
        Returns:
            dict: Detalles de la comparación incluyendo similitud, método usado, etc.
        """
        img1 = self.cargar_imagen(imagen1_path)
        img2 = self.cargar_imagen(imagen2_path)
        
        # Extraer ROIs si se especifican
        original_img1 = img1.copy()
        original_img2 = img2.copy()
        
        if roi1:
            x, y, w, h = roi1
            img1 = img1[y:y+h, x:x+w]
        
        if roi2:
            x, y, w, h = roi2
            img2 = img2[y:y+h, x:x+w]
        
        # Realizar comparación con cada método
        results = {
            'similarity': 0.0,
            'method': self.metodo,
            'image1_path': str(imagen1_path),
            'image2_path': str(imagen2_path),
            'roi1': roi1,
            'roi2': roi2,
            'details': {}
        }
        
        if self.metodo in ['orb', 'sift']:
            similarity, features_details = self._compare_features_with_details(img1, img2)
            results['similarity'] = similarity
            results['details'] = features_details
        elif self.metodo == 'histogram':
            similarity = self._compare_histogram(img1, img2)
            results['similarity'] = similarity
        elif self.metodo == 'ssim':
            similarity = self._compare_ssim(img1, img2)
            results['similarity'] = similarity
        elif self.metodo == 'siamese':
            similarity = self._compare_siamese(img1, img2)
            results['similarity'] = similarity
        else:
            similarity, features_details = self._compare_features_with_details(img1, img2)
            results['similarity'] = similarity
            results['details'] = features_details
        
        # Calcular probabilidad de similitud
        results['probability'] = self._calculate_probability(results['similarity'])
        
        return results
    
    def _compare_features(self, img1, img2):
        """Compara imágenes usando detección de características."""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) == 0:
            return 0.0
        
        good_matches = [m for m in matches if m.distance < 50]
        score = len(good_matches) / max(len(kp1), len(kp2))
        
        return min(score, 1.0)
    
    def _compare_histogram(self, img1, img2):
        """Compara imágenes usando histogramas de color."""
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, min(score, 1.0))
    
    def _compare_ssim(self, img1, img2):
        """Compara imágenes usando Structural Similarity Index."""
        size = (300, 300)
        img1_resized = cv2.resize(img1, size)
        img2_resized = cv2.resize(img2, size)
        
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        score = np.corrcoef(gray1.flatten(), gray2.flatten())[0, 1]
        return max(0.0, min(abs(score), 1.0))
    
    def find_matches(self, query_image, image_database, top_n=5):
        """
        Encuentra las imágenes más similares en una base de datos.
        
        Args:
            query_image: Ruta a la imagen de consulta
            image_database: Lista de rutas a imágenes en la base de datos
            top_n: Número de mejores coincidencias a retornar
            
        Returns:
            list: Lista de tuplas (ruta_imagen, score) ordenadas por similitud
        """
        resultados = []
        
        for db_image in image_database:
            try:
                score = self.compare(query_image, db_image)
                resultados.append((db_image, score))
            except Exception as e:
                print(f"Error comparando con {db_image}: {e}")
                continue
        
        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:top_n]
    
    def _compare_features_with_details(self, img1, img2, return_raw=False):
        """
        Compara imágenes usando detección de características y retorna detalles.

        Args:
            img1: Primera imagen
            img2: Segunda imagen
            return_raw: Si se deben retornar los keypoints y matches originales

        Returns:
            tuple: (similitud, detalles, raw_data si return_raw es True)
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        details = {
            'keypoints1': len(kp1) if kp1 is not None else 0,
            'keypoints2': len(kp2) if kp2 is not None else 0,
            'matches': 0,
            'good_matches': 0,
            'match_distance_avg': 0.0
        }

        raw_data = {'kp1': kp1, 'kp2': kp2, 'matches': [], 'good_matches': []}

        if des1 is None or des2 is None:
            return (0.0, details, raw_data) if return_raw else (0.0, details)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        details['matches'] = len(matches)
        raw_data['matches'] = matches

        if len(matches) == 0:
            return (0.0, details, raw_data) if return_raw else (0.0, details)

        good_matches = [m for m in matches if m.distance < 50]
        details['good_matches'] = len(good_matches)
        raw_data['good_matches'] = good_matches

        if good_matches:
            details['match_distance_avg'] = np.mean([m.distance for m in good_matches])

        score = len(good_matches) / max(len(kp1), len(kp2))

        if return_raw:
            return min(score, 1.0), details, raw_data
        return min(score, 1.0), details

    def _calculate_probability(self, similarity: float) -> Dict[str, float]:
        """
        Calcula probabilidades basadas en la similitud.
        
        Args:
            similarity: Score de similitud (0.0 - 1.0)
            
        Returns:
            dict: Diccionario con diferentes probabilidades
        """
        # Probabilidad de que sean similares
        prob_similar = similarity
        
        # Probabilidad de que sean idénticos
        prob_identical = min(similarity ** 2, 1.0)
        
        # Probabilidad de que sean diferentes
        prob_different = 1.0 - similarity
        
        # Nivel de confianza
        if similarity > 0.9:
            confidence_level = "muy alta"
        elif similarity > 0.7:
            confidence_level = "alta"
        elif similarity > 0.5:
            confidence_level = "media"
        elif similarity > 0.3:
            confidence_level = "baja"
        else:
            confidence_level = "muy baja"
        
        return {
            'similar': prob_similar,
            'identical': prob_identical,
            'different': prob_different,
            'confidence_level': confidence_level
        }
    
    def compare_multiple_rois(self, imagen1_path, imagen2_path, rois1, rois2):
        """
        Compara múltiples ROIs entre dos imágenes.
        
        Args:
            imagen1_path: Ruta a la primera imagen
            imagen2_path: Ruta a la segunda imagen
            rois1: Lista de ROIs para imagen 1 [(x,y,w,h), ...]
            rois2: Lista de ROIs para imagen 2 [(x,y,w,h), ...]
            
        Returns:
            list: Lista de resultados de comparación
        """
        results = []
        
        img1 = self.cargar_imagen(imagen1_path)
        img2 = self.cargar_imagen(imagen2_path)
        
        for i, (roi1, roi2) in enumerate(zip(rois1, rois2)):
            x1, y1, w1, h1 = roi1
            x2, y2, w2, h2 = roi2
            
            roi1_img = img1[y1:y1+h1, x1:x1+w1]
            roi2_img = img2[y2:y2+h2, x2:x2+w2]
            
            if self.metodo == 'siamese':
                similarity = self._compare_siamese(roi1_img, roi2_img)
            elif self.metodo == 'histogram':
                similarity = self._compare_histogram(roi1_img, roi2_img)
            elif self.metodo == 'ssim':
                similarity = self._compare_ssim(roi1_img, roi2_img)
            else:
                similarity = self._compare_features(roi1_img, roi2_img)
            probability = self._calculate_probability(similarity)
            
            results.append({
                'pair_id': i,
                'roi1': roi1,
                'roi2': roi2,
                'similarity': similarity,
                'probability': probability
            })
        
        return results
