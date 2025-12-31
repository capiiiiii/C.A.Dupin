"""
Módulo para comparar y encontrar similitudes entre imágenes
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image


class ImageMatcher:
    """Compara imágenes y calcula similitudes."""
    
    def __init__(self, metodo='orb'):
        """
        Inicializa el matcher con el método especificado.
        
        Args:
            metodo: Método de comparación ('orb', 'sift', 'histogram', 'ssim')
        """
        self.metodo = metodo
        
        if metodo == 'orb':
            self.detector = cv2.ORB_create()
        elif metodo == 'sift':
            self.detector = cv2.SIFT_create()
    
    def cargar_imagen(self, ruta):
        """Carga una imagen desde la ruta especificada."""
        if not Path(ruta).exists():
            raise FileNotFoundError(f"Imagen no encontrada: {ruta}")
        
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta}")
        
        return imagen
    
    def compare(self, imagen1_path, imagen2_path):
        """
        Compara dos imágenes y retorna un score de similitud (0.0 - 1.0).
        
        Args:
            imagen1_path: Ruta a la primera imagen
            imagen2_path: Ruta a la segunda imagen
            
        Returns:
            float: Score de similitud entre 0.0 y 1.0
        """
        img1 = self.cargar_imagen(imagen1_path)
        img2 = self.cargar_imagen(imagen2_path)
        
        if self.metodo in ['orb', 'sift']:
            return self._compare_features(img1, img2)
        elif self.metodo == 'histogram':
            return self._compare_histogram(img1, img2)
        elif self.metodo == 'ssim':
            return self._compare_ssim(img1, img2)
        else:
            return self._compare_features(img1, img2)
    
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
