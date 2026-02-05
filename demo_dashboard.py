#!/usr/bin/env python3
"""
Demo dashboard: Mapa de calidad de vía y perfil vertical desde registros de aceleración
Genera datos sintéticos y crea visualizaciones interactivas (Folium + Plotly)
"""

import numpy as np
import pandas as pd
import folium
import branca
import plotly.graph_objects as go
from scipy import signal

# Importar cumtrapz de forma compatible con diferentes versiones de SciPy
try:
    from scipy.integrate import cumtrapz
except ImportError:
    try:
        from scipy.integrate.quadrature import cumtrapz
    except ImportError:
        # Implementación manual compatible
        def cumtrapz(y, dx=1.0, initial=None):
            """Integración acumulativa usando regla trapezoidal"""
            result = np.zeros_like(y, dtype=float)
            result[1:] = np.cumsum((y[:-1] + y[1:]) * dx / 2.0)
            if initial is not None:
                result = result + initial
            return result

# ============================================================================
# 1. CONFIGURACIÓN
# ============================================================================
np.random.seed(42)
OUTPUT_DIR = '.'

# ============================================================================
# 2. GENERAR DATOS SINTÉTICOS DE LA VÍA
# ============================================================================
print("Generando datos sintéticos de la vía...")
n_points = 120
lats = 40.0 + 0.01 * np.linspace(0, 1, n_points) + 0.001 * np.sin(np.linspace(0, 6*np.pi, n_points))
lons = -3.7 + 0.01 * np.linspace(0, 1, n_points) + 0.001 * np.cos(np.linspace(0, 4*np.pi, n_points))
coords = list(zip(lats, lons))

# Dividir en segmentos
seg_size = 5
segments = []
for i in range(0, n_points-1, seg_size):
    seg_coords = coords[i:i+seg_size+1]
    segments.append({
        'start_idx': i,
        'end_idx': min(i+seg_size, n_points-1),
        'coords': seg_coords
    })

# ============================================================================
# 3. GENERAR REGISTROS DE ACELERACIÓN SINTÉTICOS
# ============================================================================
print("Generando registros de aceleración...")
samples_per_point = 200
fs = 100.0  # Hz
accel_data = []
for i in range(n_points):
    t = np.arange(0, samples_per_point) / fs
    amp = 0.005 + 0.02 * np.abs(np.sin(i / 10.0))
    acc = amp * (np.sin(2*np.pi*1.5*t) + 0.5*np.sin(2*np.pi*6*t)) + 0.002*np.random.randn(samples_per_point)
    accel_data.append(acc)

# Calcular RMS de aceleración
rms = np.array([np.sqrt(np.mean(a**2)) for a in accel_data])
pos_df = pd.DataFrame({'lat': lats, 'lon': lons, 'rms_acc': rms})
pos_df['chainage_m'] = np.linspace(0, 1000, n_points)

# ============================================================================
# 4. ASIGNAR CALIDAD POR SEGMENTO (SEMÁFORO)
# ============================================================================
print("Asignando calidad por segmento...")
seg_qualities = []
for seg in segments:
    idx0 = seg['start_idx']
    idx1 = seg['end_idx']
    seg_rms = pos_df['rms_acc'].iloc[idx0:idx1+1].mean()
    if seg_rms < 0.006:
        q = 'buena'
        color = 'green'
    elif seg_rms < 0.012:
        q = 'regular'
        color = 'orange'
    else:
        q = 'mala'
        color = 'red'
    seg_qualities.append({
        'start': idx0,
        'end': idx1,
        'rms': float(seg_rms),
        'quality': q,
        'color': color,
        'coords': seg['coords']
    })

seg_df = pd.DataFrame(seg_qualities)

# ============================================================================
# 5. PROCESAR REGISTROS DE ACELERACIÓN PARA PERFIL VERTICAL
# ============================================================================
print("Procesando aceleración → perfil vertical...")
displacements = []
for a in accel_data:
    a_detr = signal.detrend(a)
    # Filtro pasa-bajo antes de integrar
    b, c = signal.butter(3, 10.0/(fs/2), btype='low')
    a_f = signal.filtfilt(b, c, a_detr)
    # Doble integración
    v = cumtrapz(a_f, dx=1.0/fs, initial=0)
    d = cumtrapz(v, dx=1.0/fs, initial=0)
    # Corrección de deriva
    p = np.polyfit(np.arange(len(d)), d, 1)
    d_corr = d - (p[0]*np.arange(len(d)) + p[1])
    displacements.append(d_corr.mean())

pos_df['disp_mm'] = np.array(displacements) * 1000.0

# ============================================================================
# 6. CREAR MAPA INTERACTIVO CON FOLIUM
# ============================================================================
print("Creando mapa interactivo...")
m = folium.Map(
    location=[pos_df['lat'].mean(), pos_df['lon'].mean()],
    zoom_start=14,
    tiles='cartodbpositron'
)

# Añadir segmentos coloreados
for _, row in seg_df.iterrows():
    folium.PolyLine(
        locations=[(lat, lon) for lat, lon in row['coords']],
        color=row['color'],
        weight=6,
        opacity=0.8,
        tooltip=f"Calidad: {row['quality']} — RMS={row['rms']:.5f}"
    ).add_to(m)

# Marcadores de inicio/fin
folium.Marker(
    location=[lats[0], lons[0]],
    popup='Inicio',
    icon=folium.Icon(color='blue', icon='train')
).add_to(m)

folium.Marker(
    location=[lats[-1], lons[-1]],
    popup='Fin',
    icon=folium.Icon(color='blue', icon='flag')
).add_to(m)

# Leyenda personalizada
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 10px; width:150px; height:110px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
&nbsp;<b>Leyenda calidad</b><br>
&nbsp;<i style="background:green;color:green">....</i>&nbsp;Buena<br>
&nbsp;<i style="background:orange;color:orange">....</i>&nbsp;Regular<br>
&nbsp;<i style="background:red;color:red">....</i>&nbsp;Mala<br>
</div>
'''
m.get_root().html.add_child(branca.element.Element(legend_html))

map_file = f'{OUTPUT_DIR}/mapa_via.html'
m.save(map_file)
print(f"✓ Mapa guardado: {map_file}")

# ============================================================================
# 7. GRAFICAR PERFIL VERTICAL CON PLOTLY
# ============================================================================
print("Creando gráfica de perfil vertical...")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=pos_df['chainage_m'],
    y=pos_df['disp_mm'],
    mode='lines+markers',
    name='Perfil vertical (mm)',
    marker=dict(size=6, color='blue')
))

# Añadir regiones de color (semáforo) como bandas verticales
for _, s in seg_df.iterrows():
    x0 = pos_df['chainage_m'].iloc[s['start']]
    x1 = pos_df['chainage_m'].iloc[s['end']]
    if s['quality'] == 'buena':
        color = 'rgba(0,255,0,0.08)'
    elif s['quality'] == 'regular':
        color = 'rgba(255,165,0,0.08)'
    else:
        color = 'rgba(255,0,0,0.08)'
    
    fig.add_vrect(x0=x0, x1=x1, fillcolor=color, layer='below', line_width=0)

fig.update_layout(
    title='Perfil vertical estimado (mm) vs Chainage (m)',
    xaxis_title='Chainage (m)',
    yaxis_title='Desplazamiento vertical (mm)',
    height=500,
    hovermode='x unified'
)

graph_file = f'{OUTPUT_DIR}/perfil_vertical.html'
fig.write_html(graph_file)
print(f"✓ Gráfica guardada: {graph_file}")

# ============================================================================
# 8. EXPORTAR DATOS DE RESUMEN
# ============================================================================
print("\nResumen de calidad por segmento:")
print(seg_df[['start', 'end', 'rms', 'quality']].to_string(index=False))

csv_file = f'{OUTPUT_DIR}/datos_segmentos.csv'
seg_df[['start', 'end', 'rms', 'quality']].to_csv(csv_file, index=False)
print(f"\n✓ Datos de segmentos guardados: {csv_file}")

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "="*60)
print("DEMO FINALIZADA")
print("="*60)
print(f"📍 Mapa interactivo:   mapa_via.html")
print(f"📈 Perfil vertical:    perfil_vertical.html")
print(f"📊 Datos CSV:          datos_segmentos.csv")
print("\nAbre los archivos HTML en tu navegador para ver las visualizaciones.")
print("="*60)
