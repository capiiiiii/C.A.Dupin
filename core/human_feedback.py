"""
Módulo para gestionar el loop de retroalimentación humana
"""

import os
from pathlib import Path
from .image_matcher import ImageMatcher


class HumanFeedbackLoop:
    """Gestiona el ciclo de retroalimentación humana para corrección de resultados."""
    
    def __init__(self, directorio_imagenes):
        self.directorio = Path(directorio_imagenes)
        self.matcher = ImageMatcher()
        self.feedback_history = []
        
        if not self.directorio.exists():
            raise ValueError(f"Directorio no encontrado: {directorio_imagenes}")
    
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
            print("4. Salir")
            
            try:
                opcion = input("\nSeleccione una opción (1-4): ").strip()
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
                        'img1': img1.name,
                        'img2': img2.name,
                        'similitud': similitud,
                        'correcto': True
                    })
                elif feedback in ['n', 'no']:
                    similitud_real = input("¿Cuál sería la similitud correcta? (0.0-1.0): ").strip()
                    try:
                        similitud_real = float(similitud_real)
                        print("✓ Feedback registrado: CORREGIDO")
                        self.feedback_history.append({
                            'img1': img1.name,
                            'img2': img2.name,
                            'similitud_calculada': similitud,
                            'similitud_real': similitud_real,
                            'correcto': False
                        })
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
