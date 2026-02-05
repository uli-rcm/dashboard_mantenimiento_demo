#!/usr/bin/env python3
"""
Script de prueba para verificar el rango de valores RMS generados
"""
import numpy as np

np.random.seed(42)

# Configuración
g = 9.81
samples_per_point = 1000
fs = 200.0

# Patrones
zone_patterns = [
    {'amp_min': 0.05*g, 'amp_max': 0.10*g, 'noise': 0.01*g, 'quality': 'buena'},      # 0: verde
    {'amp_min': 0.20*g, 'amp_max': 0.30*g, 'noise': 0.04*g, 'quality': 'regular'},    # 1: amarillo
    {'amp_min': 0.40*g, 'amp_max': 0.60*g, 'noise': 0.10*g, 'quality': 'mala'},       # 2: rojo
    {'amp_min': 0.08*g, 'amp_max': 0.15*g, 'noise': 0.015*g, 'quality': 'buena'},     # 3: verde
    {'amp_min': 0.25*g, 'amp_max': 0.35*g, 'noise': 0.05*g, 'quality': 'regular'},    # 4: amarillo
    {'amp_min': 0.45*g, 'amp_max': 0.65*g, 'noise': 0.12*g, 'quality': 'mala'},       # 5: rojo
    {'amp_min': 0.06*g, 'amp_max': 0.12*g, 'noise': 0.012*g, 'quality': 'buena'},     # 6: verde
    {'amp_min': 0.30*g, 'amp_max': 0.40*g, 'noise': 0.06*g, 'quality': 'regular'},    # 7: amarillo
    {'amp_min': 0.50*g, 'amp_max': 0.75*g, 'noise': 0.15*g, 'quality': 'mala'},       # 8: rojo
    {'amp_min': 0.10*g, 'amp_max': 0.20*g, 'noise': 0.02*g, 'quality': 'buena'},      # 9: verde
]

print("Análisis de RMS por zona:")
print("=" * 80)

for zone_idx, pattern in enumerate(zone_patterns):
    rms_values = []
    for _ in range(5):  # 5 muestras por zona
        t = np.arange(0, samples_per_point) / fs
        amp = pattern['amp_min'] + (pattern['amp_max'] - pattern['amp_min']) * np.random.rand()
        noise = pattern['noise']
        acc = amp * (np.sin(2*np.pi*1.5*t) + 0.5*np.sin(2*np.pi*6*t)) + noise * np.random.randn(samples_per_point)
        rms = np.sqrt(np.mean(acc**2))
        rms_values.append(rms)
    
    min_rms = min(rms_values)
    max_rms = max(rms_values)
    avg_rms = np.mean(rms_values)
    
    print(f"Zona {zone_idx} ({pattern['quality']:8s}): RMS min={min_rms:7.3f} m/s² max={max_rms:7.3f} m/s² avg={avg_rms:7.3f} m/s²")
    print(f"  Amplitud: {pattern['amp_min']/g:5.2f}g - {pattern['amp_max']/g:5.2f}g")

print("=" * 80)
print("\nUmbrales actuales:")
print(f"RMS_THRESHOLD_GOOD    = 0.25*g = {0.25*g:7.3f} m/s²")
print(f"RMS_THRESHOLD_REGULAR = 0.50*g = {0.50*g:7.3f} m/s²")
