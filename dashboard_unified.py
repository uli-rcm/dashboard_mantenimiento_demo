#!/usr/bin/env python3
"""
Dashboard unificado: Mapa + Aceleraciones + Perfil vertical en una sola aplicación Dash
Con slider para visualizar 1 km a la vez
"""

import numpy as np
import pandas as pd
import folium
import plotly.graph_objects as go
from scipy import signal
from dash import Dash, dcc, html, Input, Output, State
import webbrowser
from pathlib import Path
import os

# Importar configuración modular de pestañas
from tabs_config import build_tab_geometrica, build_tab_estructural

# ============================================================================
# 1. CONFIGURACIÓN
# ============================================================================
np.random.seed(42)
PORT = 8050
WINDOW_SIZE = 1000  # metros (1 km)

# ============================================================================
# 2. CARGAR RUTA DEL TREN LIGERO DE CDMX - LÍNEA 1
# ============================================================================
print("Cargando ruta del Tren Ligero - Línea 1 (Tasqueña - Xochimilco)...")
# Coordenadas de las 18 estaciones de la Línea 1
stations = [
    (19.344168, -99.142685),  # Tasqueña (terminal) – junto a la estación Metro Tasqueña :contentReference[oaicite:1]{index=1}
    (19.339500, -99.140500),  # Las Torres (estimado)
    (19.335656, -99.141867),  # Ciudad Jardín (Wikipedia) :contentReference[oaicite:2]{index=2}
    (19.332000, -99.136000),  # La Virgen (estimado)
    (19.327500, -99.134000),  # Xotepingo (estimado)
    (19.323000, -99.130500),  # Nezahualpilli (estimado)
    (19.319000, -99.128000),  # Registro Federal (estimado)
    (19.315000, -99.126000),  # Textitlán (estimado)
    (19.311000, -99.123500),  # El Vergel (estimado)
    (19.302000, -99.124000),  # Estadio Azteca (estimado)
    (19.295000, -99.121000),  # Huipulco (estimado)
    (19.289000, -99.118000),  # Xomali (estimado)
    (19.284000, -99.115000),  # Periférico / Participación Ciudadana (estimado)
    (19.278000, -99.112000),  # Tepepan (estimado)
    (19.272000, -99.109000),  # La Noria (estimado)
    (19.268000, -99.107000),  # Huichapan (estimado)
    (19.263000, -99.110000),  # Francisco Goitia (estimado)
    (19.259455, -99.108042),  # Xochimilco (terminal) – coordenada SIG pública :contentReference[oaicite:3]{index=3}
]


# Interpolar puntos entre estaciones para mayor densidad de datos
lats, lons = [], []
for i in range(len(stations) - 1):
    lat1, lon1 = stations[i]
    lat2, lon2 = stations[i+1]
    for j in range(15):  # 15 puntos entre cada estación (optimizado para memoria)
        t = j / 15.0
        lats.append(lat1 + t * (lat2 - lat1))
        lons.append(lon1 + t * (lon2 - lon1))
# Añadir el último punto
lats.append(stations[-1][0])
lons.append(stations[-1][1])

n_points = len(lats)
coords = list(zip(lats, lons))
print(f"Ruta cargada con {n_points} puntos de {len(stations)} estaciones")

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

# Generar registros de aceleración con 10 zonas alternadas
samples_per_point = 10000  # Reducido de 100000 para optimizar memoria en deploy
fs = 1000.0  # Aumentado de 100.0 Hz
accel_data = []

# Dividir en 10 zonas con patrones alternados
num_zones = 10
zone_size = n_points / num_zones

# Definir patrones para cada zona (amplitudes en m/s²)
# g = 9.81 m/s², por lo que 50g ≈ 490 m/s²
g = 9.81
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

for i in range(n_points):
    t = np.arange(0, samples_per_point) / fs
    # Determinar a cuál zona pertenece este punto
    zone_idx = min(int(i / zone_size), num_zones - 1)
    pattern = zone_patterns[zone_idx]
    
    # Generar amplitud aleatoria dentro del rango de la zona
    amp = pattern['amp_min'] + (pattern['amp_max'] - pattern['amp_min']) * np.random.rand()
    noise = pattern['noise']
    
    # Señal con mayor variación: múltiples componentes sinusoidales con diferentes frecuencias
    # Esto simula mejor las vibraciones complejas en infraestructura ferroviaria
    signal_base = (np.sin(2*np.pi*0.8*t) +           # Componente baja frecuencia
                   0.6*np.sin(2*np.pi*1.5*t) +        # Componente dominante
                   0.4*np.sin(2*np.pi*3.2*t) +        # Componente media
                   0.5*np.sin(2*np.pi*6*t) +          # Componente más alta
                   0.3*np.sin(2*np.pi*11.5*t) +       # Componente de alta frecuencia
                   0.2*np.sin(2*np.pi*18*t))          # Componente muy alta
    
    acc = amp * signal_base + noise * np.random.randn(samples_per_point)
    accel_data.append(acc)

rms = np.array([np.sqrt(np.mean(a**2)) for a in accel_data])
pos_df = pd.DataFrame({'lat': lats, 'lon': lons, 'rms_acc': rms})
pos_df['chainage_m'] = np.linspace(0, 25000, n_points)  # ~25 km de ruta real

# Asignar calidad por segmento
# Umbrales ajustados para nuevas amplitudes (en m/s²)
# Basados en el análisis real de RMS generado:
# Verde:   RMS 0.45-1.25 m/s² (amplitudes 0.05-0.20g)
# Amarillo: RMS 1.6-2.9 m/s² (amplitudes 0.20-0.40g)
# Rojo:    RMS 3.3-4.6 m/s² (amplitudes 0.40-0.75g)
RMS_THRESHOLD_GOOD = 1.5        # < 1.5 m/s² = buena (verde)
RMS_THRESHOLD_REGULAR = 3.5     # < 3.5 m/s² = regular (amarillo)
seg_qualities = []
for seg in segments:
    idx0 = seg['start_idx']
    idx1 = seg['end_idx']
    seg_rms = pos_df['rms_acc'].iloc[idx0:idx1+1].mean()
    if seg_rms < RMS_THRESHOLD_GOOD:
        q = 'buena'
        color = '#27AE60'  # Verde semáforo
    elif seg_rms < RMS_THRESHOLD_REGULAR:
        q = 'regular'
        color = '#F39C12'  # Amarillo semáforo
    else:
        q = 'mala'
        color = '#E74C3C'  # Rojo semáforo
    seg_qualities.append({
        'start': idx0, 'end': idx1, 'rms': float(seg_rms),
        'quality': q, 'color': color, 'coords': seg['coords']
    })

seg_df = pd.DataFrame(seg_qualities)

# Procesar aceleración → perfil vertical
print("Procesando aceleración → perfil vertical...")

# Implementación compatible de cumtrapz
def cumtrapz(y, dx=1.0, initial=None):
    """Integración acumulativa usando regla trapezoidal"""
    result = np.zeros_like(y, dtype=float)
    result[1:] = np.cumsum((y[:-1] + y[1:]) * dx / 2.0)
    if initial is not None:
        result = result + initial
    return result

displacements = []
for a in accel_data:
    a_detr = signal.detrend(a)
    b, c = signal.butter(3, 10.0/(fs/2), btype='low')
    a_f = signal.filtfilt(b, c, a_detr)
    v = cumtrapz(a_f, dx=1.0/fs, initial=0)
    d = cumtrapz(v, dx=1.0/fs, initial=0)
    p = np.polyfit(np.arange(len(d)), d, 1)
    d_corr = d - (p[0]*np.arange(len(d)) + p[1])
    # d_corr está en metros, convertir a mm
    displacements.append(d_corr.mean() * 1000.0)

pos_df['disp_mm'] = np.array(displacements)  # Ya está en mm

# ============================================================================
# 3. CREAR FUNCIONES PARA GENERAR GRÁFICAS
# ============================================================================

def create_map_figure(quality_label):
    """Crea figura del mapa con Folium para una etiqueta de calidad"""
    # Crear mapa con Folium centrado en CDMX
    m = folium.Map(
        location=[19.259455, -99.108042],
        zoom_start=12,
        tiles='Cartodb Positron'
    )

    for _, row in seg_df.iterrows():
        folium.PolyLine(
            locations=[(lat, lon) for lat, lon in row['coords']],
            color=row['color'], weight=5, opacity=0.9,
            tooltip=f"{quality_label}: {row['quality']} — RMS={row['rms']:.5f}"
        ).add_to(m)

    folium.Marker(
        location=[lats[0], lons[0]], popup='Inicio (Tasqueña)',
        icon=folium.Icon(color='blue', icon='train', prefix='fa')
    ).add_to(m)

    folium.Marker(
        location=[lats[-1], lons[-1]], popup='Fin (Xochimilco)',
        icon=folium.Icon(color='blue', icon='flag', prefix='fa')
    ).add_to(m)

    return m

def create_profile_figure():
    """Crea figura del perfil vertical"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pos_df['chainage_m'], y=pos_df['disp_mm'],
        mode='lines+markers', name='Perfil vertical (mm)',
        marker=dict(size=6, color='blue'),
        hovertemplate='<b>Cadenamiento: %{x:.1f} m</b><br>Desplazamiento: %{y:.2f} mm<extra></extra>'
    ))
    
    for _, s in seg_df.iterrows():
        x0 = pos_df['chainage_m'].iloc[s['start']]
        x1 = pos_df['chainage_m'].iloc[s['end']]
        color = 'rgba(0,255,0,0.08)' if s['quality']=='buena' else (
            'rgba(255,165,0,0.08)' if s['quality']=='regular' else 'rgba(255,0,0,0.08)'
        )
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, layer='below', line_width=0)
    
    fig.update_layout(
        title='Perfil vertical estimado (mm) vs Cadenamiento (m)',
        xaxis_title='Cadenamiento (m)', yaxis_title='Desplazamiento vertical (mm)',
        height=500, hovermode='x unified'
    )
    
    return fig

# ============================================================================
# 4A. GENERAR DATOS GPR SINTÉTICOS
# ============================================================================
print("Generando datos GPR sintéticos...")

# Profundidades reales en metros (las antenas están 0.4m sobre el balasto)
# - Superficie del balasto (air-ballast): 0.4 m
# - Espesor de balasto: 0.3 m (0.4m a 0.7m)
# - Espesor de subbalasto: 0.3 m (0.7m a 1.0m)
# - Espesor de subrasante: 0.4 m (1.0m a 1.4m)

gpr_depth_samples = 400  # Muestras en profundidad
profundidad_total_m = 1.4  # Profundidad total en metros

# Convertir profundidades a índices de muestra
# Índice = (profundidad_m / profundidad_total_m) * gpr_depth_samples
surface_idx = int(0.4 / profundidad_total_m * gpr_depth_samples)  # ~114
interface1_idx = int(0.7 / profundidad_total_m * gpr_depth_samples)  # ~200 (balasto-subbalasto)
interface2_idx = int(1.0 / profundidad_total_m * gpr_depth_samples)  # ~286 (subbalasto-subrasante)

print(f"  • Superficie balasto: índice {surface_idx}")
print(f"  • Interfaz balasto-subbalasto: índice {interface1_idx}")
print(f"  • Interfaz subbalasto-subrasante: índice {interface2_idx}")

# Función para crear wavelet tipo Ricker (simula radargrama)
def ricker_wavelet(length, amplitude=1.0, freq=0.1):
    """
    Crea un wavelet tipo Ricker normalizado
    Simula la oscilación típica en un radargrama GPR
    """
    t = np.linspace(-length/2, length/2, length)
    # Wavelet de Ricker: (1 - 2*pi^2*freq^2*t^2) * exp(-pi^2*freq^2*t^2)
    ricker = (1 - 2 * np.pi**2 * freq**2 * t**2) * np.exp(-np.pi**2 * freq**2 * t**2)
    return amplitude * ricker / np.max(np.abs(ricker))

# Generar matriz GPR: (n_points, gpr_depth_samples)
gpr_raw = np.zeros((n_points, gpr_depth_samples))

for i in range(n_points):
    # Determinar si este punto está en zona "fouled" (contaminada)
    zone_idx = min(int(i / zone_size), num_zones - 1)
    pattern = zone_patterns[zone_idx]
    is_fouled = (pattern['quality'] == 'mala')  # Fouled ballast en zonas con mala calidad
    
    # Generar ruido base (siempre presente)
    noise = np.random.randn(gpr_depth_samples) * 0.015
    gpr_raw[i, :] = noise
    
    # Reflexión 1: Interfaz aire-balasto (superficie)
    # Reflexión muy fuerte en la superficie
    width_surface = 25
    amplitude_surface = 0.95
    surface_wavelet = ricker_wavelet(width_surface * 2, amplitude_surface, freq=0.15)
    idx_surface_start = max(0, surface_idx - width_surface)
    idx_surface_end = min(gpr_depth_samples, surface_idx + width_surface)
    surf_len = idx_surface_end - idx_surface_start
    gpr_raw[i, idx_surface_start:idx_surface_end] += surface_wavelet[-surf_len:] if surf_len <= len(surface_wavelet) else surface_wavelet[:surf_len]
    
    if is_fouled:
        # Fouled ballast: la interfaz balasto-subbalasto NO tiene reflexión clara
        # Solo ruido y reverberación del balasto contaminado
        # Atenuación extrema en la interfaz balasto-subbalasto
        weak_signal = 0.05 * np.exp(-np.arange(50)**2 / 100) * (0.3 + 0.4*np.random.rand())
        idx_weak_start = max(0, interface1_idx - 25)
        idx_weak_end = min(gpr_depth_samples, interface1_idx + 25)
        weak_len = idx_weak_end - idx_weak_start
        if weak_len <= len(weak_signal):
            gpr_raw[i, idx_weak_start:idx_weak_end] += weak_signal[-weak_len:]
    else:
        # Reflexión 2: Interfaz balasto-subbalasto (clara)
        # Segundo pico más débil pero clara osculación
        width_inter1 = 30
        amplitude_inter1 = 0.7 + 0.1 * np.random.randn()
        amplitude_inter1 = np.clip(amplitude_inter1, 0.5, 0.85)
        inter1_wavelet = ricker_wavelet(width_inter1 * 2, amplitude_inter1, freq=0.12)
        idx_inter1_start = max(0, interface1_idx - width_inter1)
        idx_inter1_end = min(gpr_depth_samples, interface1_idx + width_inter1)
        inter1_len = idx_inter1_end - idx_inter1_start
        gpr_raw[i, idx_inter1_start:idx_inter1_end] += inter1_wavelet[-inter1_len:] if inter1_len <= len(inter1_wavelet) else inter1_wavelet[:inter1_len]
    
    # Reflexión 3: Interfaz subbalasto-subrasante
    # Tercera reflexión más débil por atenuación
    width_inter2 = 35
    amplitude_inter2 = 0.5 + 0.08 * np.random.randn()
    amplitude_inter2 = np.clip(amplitude_inter2, 0.35, 0.65)
    inter2_wavelet = ricker_wavelet(width_inter2 * 2, amplitude_inter2, freq=0.10)
    idx_inter2_start = max(0, interface2_idx - width_inter2)
    idx_inter2_end = min(gpr_depth_samples, interface2_idx + width_inter2)
    inter2_len = idx_inter2_end - idx_inter2_start
    gpr_raw[i, idx_inter2_start:idx_inter2_end] += inter2_wavelet[-inter2_len:] if inter2_len <= len(inter2_wavelet) else inter2_wavelet[:inter2_len]
    
    # Aplicar atenuación exponencial con profundidad (pérdida realista de energía EM)
    attenuation = np.exp(-np.arange(gpr_depth_samples) / 120.0)
    gpr_raw[i, :] *= attenuation
    
    # Normalizar entre 0-1
    gpr_max = np.max(np.abs(gpr_raw[i, :]))
    if gpr_max > 0:
        gpr_raw[i, :] = np.abs(gpr_raw[i, :]) / gpr_max

# Crear DataFrame con datos GPR
gpr_data = []
for i in range(n_points):
    gpr_data.append({
        'chainage_m': pos_df['chainage_m'].iloc[i],
        'gpr_profile': gpr_raw[i, :]  # Array de profundidad
    })

gpr_df = pd.DataFrame(gpr_data)

print(f"✓ Datos GPR generados: {n_points} perfiles con {gpr_depth_samples} muestras de profundidad")

print("Iniciando aplicación Dash...")
app = Dash(__name__)

# =================== Lazy loading: cachear los mapas HTML ===================
_map_html_cache = {
    'geometrica': None,
    'estructural': None
}

def get_map_html(map_kind, quality_label):
    """Genera el HTML del mapa solo cuando se necesita (lazy loading)"""
    if _map_html_cache.get(map_kind) is None:
        map_obj = create_map_figure(quality_label)
        # Marcadores de estaciones con ícono personalizado
        station_icon_path = os.path.join(os.getcwd(), 'assets', 'station_icon.png')
        for idx, (lat, lon) in enumerate(stations):
            folium.Marker(
                location=[lat, lon],
                popup=f'Estación {idx+1}',
                icon=folium.CustomIcon(station_icon_path, icon_size=(40, 40))
            ).add_to(map_obj)
        _map_html_cache[map_kind] = map_obj._repr_html_()
    return _map_html_cache[map_kind]

# Calcular rango del slider
max_chainage = pos_df['chainage_m'].max()
slider_max = int(max_chainage - WINDOW_SIZE)

# Construir pestañas de forma modular
tab_geometrica = build_tab_geometrica(
    app, pos_df, seg_df, 
    RMS_THRESHOLD_GOOD, RMS_THRESHOLD_REGULAR, 
    WINDOW_SIZE, slider_max, get_map_html
)

tab_estructural = build_tab_estructural(
    app, pos_df, seg_df, gpr_df, gpr_depth_samples,
    WINDOW_SIZE, slider_max, get_map_html
)

# =================== Encabezado con logos ===================
header = html.Div(
    style={
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'background': 'white',
        'boxShadow': '0 10px 30px rgba(0,0,0,0.12)',
        'borderBottom': '4px solid #27AE60',
        'padding': '18px 40px 18px 40px',
        'marginBottom': '0px',
        'minHeight': '80px'
    },
    children=[
        html.Img(src=app.get_asset_url('foglia.png'), style={'height': '80px', 'marginRight': '30px'}),
        html.Div([
            html.H1('Monitoreo de Infraestructura Ferroviaria', style={
                'textAlign': 'center',
                'margin': '0',
                'color': '#1565C0',
                'fontSize': '2.2rem',
                'fontWeight': '700',
                'letterSpacing': '-0.5px',
                'fontFamily': 'inherit',
            }),
            html.P('Tren Ligero CDMX - Línea 1: Tasqueña → Xochimilco', style={
                'textAlign': 'center',
                'margin': '0',
                'color': '#666',
                'fontSize': '1rem',
                'fontWeight': '300',
                'letterSpacing': '0.5px',
                'fontFamily': 'inherit',
            })
        ], style={'flex': '1'}),
        html.Img(src=app.get_asset_url('te.png'), style={'height': '80px', 'marginLeft': '30px'}),
    ]
)

# Crear layout con encabezado y pestañas modulares
app.layout = html.Div([
    header,
    dcc.Tabs(
        value='tab-geometrica',
        children=[tab_geometrica, tab_estructural],
        style={'padding': '10px 25px'}
    )
], style={
    'fontFamily': '"Inter", "Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
    'backgroundColor': '#F8F9FA',
    'minHeight': '100vh',
    'color': '#333'
})

# ============================================================================
# 6. EJECUTAR APLICACIÓN
# ============================================================================

# Exponer el servidor para Gunicorn (necesario para deployment)
server = app.server

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("DASHBOARD INICIADO")
    print(f"{'='*60}")
    print(f"\nAbre tu navegador en: http://localhost:{PORT}")
    print("Presiona CTRL+C para detener el servidor.\n")
    
    # Abrir navegador automáticamente solo en desarrollo local
    import time
    import threading
    
    def open_browser():
        time.sleep(1)
        webbrowser.open(f'http://localhost:{PORT}')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(debug=False, port=PORT)
