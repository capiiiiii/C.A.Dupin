#!/usr/bin/env python3
"""
C.A.Dupin - Herramienta de coincidencias visuales asistidas por humanos

Este programa permite entrenar modelos, comparar imágenes y corregir resultados
en tiempo real. La decisión final siempre es humana.
"""

import sys
import argparse
from pathlib import Path

from core.image_matcher import ImageMatcher
from core.model_trainer import ModelTrainer
from core.human_feedback import HumanFeedbackLoop


def comparar_imagenes(imagen1_path, imagen2_path, umbral=0.85):
    """Compara dos imágenes y muestra la similitud."""
    matcher = ImageMatcher()
    similitud = matcher.compare(imagen1_path, imagen2_path)
    
    print(f"\nComparación de imágenes:")
    print(f"  Imagen 1: {imagen1_path}")
    print(f"  Imagen 2: {imagen2_path}")
    print(f"  Similitud: {similitud:.2%}")
    
    if similitud >= umbral:
        print(f"  ✓ Las imágenes son similares (>= {umbral:.0%})")
    else:
        print(f"  ✗ Las imágenes no son similares (< {umbral:.0%})")
    
    return similitud


def entrenar_modelo(directorio_datos, epochs=10, output_path="modelo.pth"):
    """Entrena un modelo con los datos proporcionados."""
    trainer = ModelTrainer()
    print(f"\nEntrenando modelo con datos de: {directorio_datos}")
    print(f"Épocas: {epochs}")
    
    modelo = trainer.train(directorio_datos, epochs=epochs)
    trainer.save_model(modelo, output_path)
    
    print(f"\n✓ Modelo guardado en: {output_path}")
    return modelo


def modo_interactivo(directorio_imagenes):
    """Inicia el modo interactivo con retroalimentación humana."""
    print("\n=== C.A.Dupin - Modo Interactivo ===")
    print("Mostrando patrones y similitudes para revisión humana.\n")
    
    feedback_loop = HumanFeedbackLoop(directorio_imagenes)
    feedback_loop.start()


def ajustar_modelo(modelo_path, feedback_path, output_path="modelo_ajustado.pth"):
    """Ajusta un modelo usando retroalimentación humana."""
    print(f"\n=== Ajustando Modelo con Feedback Humano ===")
    
    trainer = ModelTrainer()
    
    print(f"Cargando modelo desde: {modelo_path}")
    modelo = trainer.load_model(modelo_path)
    
    print(f"Cargando feedback desde: {feedback_path}")
    feedback_data = trainer.load_feedback(feedback_path)
    
    if not feedback_data:
        print("No se encontró feedback para ajustar el modelo.")
        return
    
    modelo_ajustado = trainer.fine_tune_with_feedback(modelo, feedback_data)
    trainer.save_model(modelo_ajustado, output_path)
    
    print(f"\n✓ Modelo ajustado guardado en: {output_path}")
    return modelo_ajustado


def main():
    parser = argparse.ArgumentParser(
        description='C.A.Dupin - Herramienta de coincidencias visuales asistidas por humanos'
    )
    
    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponibles')
    
    comparar_parser = subparsers.add_parser('comparar', help='Comparar dos imágenes')
    comparar_parser.add_argument('imagen1', help='Ruta a la primera imagen')
    comparar_parser.add_argument('imagen2', help='Ruta a la segunda imagen')
    comparar_parser.add_argument('--umbral', type=float, default=0.85,
                                help='Umbral de similitud (0.0-1.0)')
    
    entrenar_parser = subparsers.add_parser('entrenar', help='Entrenar modelo')
    entrenar_parser.add_argument('directorio', help='Directorio con imágenes de entrenamiento')
    entrenar_parser.add_argument('--epochs', type=int, default=10,
                                help='Número de épocas de entrenamiento')
    entrenar_parser.add_argument('--output', default='modelo.pth',
                                help='Ruta para guardar el modelo')
    
    ajustar_parser = subparsers.add_parser('ajustar', help='Ajustar modelo con feedback humano')
    ajustar_parser.add_argument('--modelo', required=True,
                               help='Ruta al modelo entrenado')
    ajustar_parser.add_argument('--feedback', required=True,
                               help='Ruta al archivo JSON con feedback')
    ajustar_parser.add_argument('--output', default='modelo_ajustado.pth',
                               help='Ruta para guardar el modelo ajustado')
    
    interactivo_parser = subparsers.add_parser('interactivo',
                                               help='Modo interactivo con retroalimentación humana')
    interactivo_parser.add_argument('directorio', help='Directorio con imágenes a revisar')
    
    args = parser.parse_args()
    
    if args.comando == 'comparar':
        comparar_imagenes(args.imagen1, args.imagen2, args.umbral)
    elif args.comando == 'entrenar':
        entrenar_modelo(args.directorio, args.epochs, args.output)
    elif args.comando == 'ajustar':
        ajustar_modelo(args.modelo, args.feedback, args.output)
    elif args.comando == 'interactivo':
        modo_interactivo(args.directorio)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
