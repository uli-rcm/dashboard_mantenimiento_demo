Demo dashboard para monitoreo de vía férrea

## Opción 1: Ejecutar script Python puro

```bash
pip install -r requirements.txt
python demo_dashboard.py
```

El script genera:
- `mapa_via.html` — Mapa interactivo con Folium (vía coloreada tipo semáforo)
- `perfil_vertical.html` — Gráfica interactiva del perfil vertical (Plotly)
- `datos_segmentos.csv` — Datos de calidad por segmento

## Opción 2: Ejecutar como Jupyter Notebook

```bash
pip install -r requirements.txt
jupyter notebook demo_dashboard.ipynb
```

Ejecuta las celdas en orden.

## Qué incluye

- **Mapa interactivo**: Visualiza la vía férrea con colores tipo semáforo:
  - 🟢 Verde = Calidad buena
  - 🟡 Amarillo = Calidad regular  
  - 🔴 Rojo = Calidad mala
  
- **Perfil vertical**: Estima el desplazamiento vertical integrando registros de aceleración
  - Procesado: detrend, filtrado, doble integración con corrección de deriva
  - Visualización interactiva con bandas de color que indican calidad

- **Datos sintéticos**: Genera una vía con ~120 puntos y registros de aceleración realistas

## Notas

- Los datos son completamente sintéticos para demostración.
- Para usar datos reales, adapte las secciones de generación de datos en el script o notebook.