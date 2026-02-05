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

def create_map_figure():
    """Crea figura del mapa con Folium convertida a Plotly"""
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
            tooltip=f"Calidad: {row['quality']} — RMS={row['rms']:.5f}"
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
        hovertemplate='<b>Chainage: %{x:.1f} m</b><br>Desplazamiento: %{y:.2f} mm<extra></extra>'
    ))
    
    for _, s in seg_df.iterrows():
        x0 = pos_df['chainage_m'].iloc[s['start']]
        x1 = pos_df['chainage_m'].iloc[s['end']]
        color = 'rgba(0,255,0,0.08)' if s['quality']=='buena' else (
            'rgba(255,165,0,0.08)' if s['quality']=='regular' else 'rgba(255,0,0,0.08)'
        )
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, layer='below', line_width=0)
    
    fig.update_layout(
        title='Perfil vertical estimado (mm) vs Chainage (m)',
        xaxis_title='Chainage (m)', yaxis_title='Desplazamiento vertical (mm)',
        height=500, hovermode='x unified'
    )
    
    return fig

def create_window_figures(chainage_min):
    """Crea dos figuras para una ventana de chainage (1 km)"""
    chainage_max = chainage_min + WINDOW_SIZE
    
    # Filtrar datos dentro de la ventana
    mask = (pos_df['chainage_m'] >= chainage_min) & (pos_df['chainage_m'] <= chainage_max)
    df_window = pos_df[mask].copy()
    
    if len(df_window) == 0:
        # Ventana vacía
        fig_acc = go.Figure()
        fig_prof = go.Figure()
        fig_acc.add_annotation(text="Sin datos en esta ventana")
        fig_prof.add_annotation(text="Sin datos en esta ventana")
        return fig_acc, fig_prof
    
    # Figura 1: Aceleraciones (RMS)
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=df_window['chainage_m'], y=df_window['rms_acc']*1000,  # convertir a mm/s²
        mode='lines+markers', name='Aceleración RMS (mm/s²)',
        marker=dict(size=8, color='red'),
        hovertemplate='<b>Chainage: %{x:.1f} m</b><br>RMS: %{y:.3f} mm/s²<extra></extra>'
    ))
    
    # Colorear bandas según calidad
    for _, s in seg_df.iterrows():
        x0 = pos_df['chainage_m'].iloc[s['start']]
        x1 = pos_df['chainage_m'].iloc[s['end']]
        # Restringir a ventana
        x0_win = max(x0, chainage_min)
        x1_win = min(x1, chainage_max)
        if x0_win < x1_win:
            # Colores semáforo suaves para las bandas
            if s['quality']=='buena':
                color = 'rgba(39,174,96,0.12)'    # Verde suave
            elif s['quality']=='regular':
                color = 'rgba(243,156,18,0.12)'   # Amarillo suave
            else:
                color = 'rgba(231,76,60,0.12)'    # Rojo suave
            fig_acc.add_vrect(x0=x0_win, x1=x1_win, fillcolor=color, layer='below', line_width=0)
    
    fig_acc.update_layout(
        title=dict(text=f'Aceleraciones (RMS): {chainage_min:.0f} - {chainage_max:.0f} m', font=dict(size=14, color='#333', family='Arial')),
        xaxis_title='Chainage (m)',
        yaxis_title='RMS Aceleración (mm/s²)',
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(245,243,240,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=False),
        margin=dict(l=60, r=20, t=40, b=60)
    )
    
    # Figura 2: Perfil vertical
    fig_prof = go.Figure()
    fig_prof.add_trace(go.Scatter(
        x=df_window['chainage_m'], y=df_window['disp_mm'],
        mode='lines+markers', name='Perfil vertical (mm)',
        marker=dict(size=8, color='blue'),
        hovertemplate='<b>Chainage: %{x:.1f} m</b><br>Desplazamiento: %{y:.2f} mm<extra></extra>'
    ))
    
    # Colorear bandas según calidad
    for _, s in seg_df.iterrows():
        x0 = pos_df['chainage_m'].iloc[s['start']]
        x1 = pos_df['chainage_m'].iloc[s['end']]
        x0_win = max(x0, chainage_min)
        x1_win = min(x1, chainage_max)
        if x0_win < x1_win:
            # Colores semáforo suaves para las bandas
            if s['quality']=='buena':
                color = 'rgba(39,174,96,0.12)'    # Verde suave
            elif s['quality']=='regular':
                color = 'rgba(243,156,18,0.12)'   # Amarillo suave
            else:
                color = 'rgba(231,76,60,0.12)'    # Rojo suave
            
            fig_prof.add_vrect(x0=x0_win, x1=x1_win, fillcolor=color, layer='below', line_width=0)
    
    fig_prof.update_layout(
        title=dict(text=f'Perfil vertical: {chainage_min:.0f} - {chainage_max:.0f} m', font=dict(size=14, color='#333', family='Arial')),
        xaxis_title='Chainage (m)',
        yaxis_title='Desplazamiento vertical (mm)',
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(245,243,240,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=False),
        margin=dict(l=60, r=20, t=40, b=60)
    )
    
    return fig_acc, fig_prof

def create_window_table(chainage_min):
    """Crea una tabla con un único renglón para la ventana de 1 km actual"""
    # P.K. Inicial = valor actual del slider
    # P.K. Final = P.K. Inicial + 1000 metros
    chainage_start = chainage_min
    chainage_end = chainage_min + 1000
    
    # Filtrar datos dentro de este rango
    mask = (pos_df['chainage_m'] >= chainage_start) & (pos_df['chainage_m'] < chainage_end)
    data_in_window = pos_df[mask]
    
    if len(data_in_window) == 0:
        # Sin datos en esta ventana
        return html.Table([
            html.Thead(
                html.Tr([
                    html.Th('P.K. Inicial (m)', style={'padding': '10px', 'borderBottom': '2px solid #ddd'}),
                    html.Th('P.K. Final (m)', style={'padding': '10px', 'borderBottom': '2px solid #ddd'}),
                    html.Th('RMS (m/s²)', style={'padding': '10px', 'borderBottom': '2px solid #ddd'}),
                    html.Th('Calidad', style={'padding': '10px', 'borderBottom': '2px solid #ddd'}),
                ])
            ),
            html.Tbody(
                html.Tr(
                    html.Td("Sin datos", colSpan=4, style={'padding': '10px', 'textAlign': 'center'})
                )
            )
        ], style={
            'width': '100%', 'borderCollapse': 'collapse',
            'backgroundColor': 'white', 'borderRadius': '5px'
        })
    
    # Calcular RMS promedio para esta ventana
    avg_rms = data_in_window['rms_acc'].mean()
    
    # Determinar calidad según RMS
    if avg_rms < RMS_THRESHOLD_GOOD:
        quality = 'buena'
        color = "#08DA5F"  # Verde semáforo
    elif avg_rms < RMS_THRESHOLD_REGULAR:
        quality = 'regular'
        color = "#F3EF12"  # Amarillo semáforo
    else:
        quality = 'mala'
        color = "#F51E06"  # Rojo semáforo
    
    # Crear tabla con un solo renglón - diseño moderno
    row = html.Tr([
        html.Td(f"{chainage_start:.0f}", style={'padding': '16px 18px', 'border': 'none', 'borderBottom': '1px solid #E0E0E0', 'fontSize': '14px', 'fontWeight': '500', 'color': '#333'}),
        html.Td(f"{chainage_end:.0f}", style={'padding': '16px 18px', 'border': 'none', 'borderBottom': '1px solid #E0E0E0', 'fontSize': '14px', 'fontWeight': '500', 'color': '#333'}),
        html.Td(f"{avg_rms:.5f}", style={'padding': '16px 18px', 'border': 'none', 'borderBottom': '1px solid #E0E0E0', 'fontSize': '14px', 'fontWeight': '600', 'color': '#333', 'fontFamily': 'monospace'}),
        html.Td(
            f"● {quality.upper()}",
            style={
                'padding': '16px 18px',
                'border': 'none',
                'borderBottom': '1px solid #E0E0E0',
                'color': 'white',
                'backgroundColor': color,
                'fontWeight': '700',
                'textAlign': 'center',
                'borderRadius': '6px',
                'fontSize': '13px',
                'letterSpacing': '0.5px'
            }
        )
    ])
    
    return html.Table([
        html.Thead(
        html.Tr([
                html.Th('P.K. Inicial (m)', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'left', 'fontSize': '13px', 'letterSpacing': '0.3px'}),
                html.Th('P.K. Final (m)', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'left', 'fontSize': '13px', 'letterSpacing': '0.3px'}),
                html.Th('RMS (m/s²)', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'left', 'fontSize': '13px', 'letterSpacing': '0.3px'}),
                html.Th('Calidad', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'center', 'fontSize': '13px', 'letterSpacing': '0.3px'}),
            ])
        ),
        html.Tbody(row)
    ], style={
        'width': '100%', 'borderCollapse': 'collapse',
        'backgroundColor': 'white', 'borderRadius': '6px'
    })

print("Iniciando aplicación Dash...")
app = Dash(__name__)

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
        html.Img(src=app.get_asset_url('foglia.png'), style={'height': '60px', 'marginRight': '30px'}),
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
        html.Img(src=app.get_asset_url('te.png'), style={'height': '60px', 'marginLeft': '30px'}),
    ]
)

# Generar mapa
def create_map_figure():
    """Crea figura del mapa con Folium convertida a Plotly"""
    m = folium.Map(
        location=[19.259455, -99.108042],
        zoom_start=12,
        tiles='Cartodb Positron'
    )
    for _, row in seg_df.iterrows():
        folium.PolyLine(
            locations=[(lat, lon) for lat, lon in row['coords']],
            color=row['color'], weight=5, opacity=0.9,
            tooltip=f"Calidad: {row['quality']} — RMS={row['rms']:.5f}"
        ).add_to(m)
    # Marcadores de estaciones con ícono personalizado
    station_icon_path = os.path.join(os.getcwd(), 'assets', 'station_icon.png')
    for idx, (lat, lon) in enumerate(stations):
        folium.Marker(
            location=[lat, lon],
            popup=f'Estación {idx+1}',
            icon=folium.CustomIcon(station_icon_path, icon_size=(40, 40))
        ).add_to(m)
    # Inicio y fin (opcional, ya están marcados)
    return m

# Lazy loading: cachear el mapa HTML para evitar regenerarlo
_map_html_cache = None

def get_map_html():
    """Genera el HTML del mapa solo cuando se necesita (lazy loading)"""
    global _map_html_cache
    if _map_html_cache is None:
        map_obj = create_map_figure()
        _map_html_cache = map_obj._repr_html_()
    return _map_html_cache

# Calcular rango del slider
max_chainage = pos_df['chainage_m'].max()
slider_max = int(max_chainage - WINDOW_SIZE)

# Crear layout con encabezado de logos
app.layout = html.Div([
    header,
    # Contenido principal
    html.Div([
        # Fila 1: Mapa con icono
        html.Div([
            html.Div([
                html.Div([
                    html.H3('🗺️ Mapa de Calidad de Vía', 
                            style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 18, 'margin': '0 0 15px 0'}),
                    html.P('Visualización geoespacial del estado de la infraestructura',
                           style={'color': '#666', 'fontSize': 12, 'margin': 0})
                ]),
                html.Div([
                    dcc.Graph(id='mapa-via', style={'display': 'none'}),
                    html.Iframe(srcDoc=get_map_html(), width='100%', height=500, style={'border': 'none', 'borderRadius': '8px'})
                ], style={'marginTop': 15})
            ], style={
                'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 
                'borderRadius': '12px', 
                'padding': 25,
                'backgroundColor': 'white',
                'transition': 'box-shadow 0.3s ease'
            })
        ], style={'marginBottom': 40}),
        
        # Fila 2: Dos gráficas lado a lado con headers mejorados
        html.Div([
            html.Div([
                html.Div([
                    html.H3('📈 Aceleración RMS', 
                            style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 16, 'margin': '0 0 8px 0'}),
                    html.P('Energía acumulada de vibración (mm/s²)',
                           style={'color': '#666', 'fontSize': 11, 'margin': 0})
                ]),
                dcc.Graph(id='grafica-aceleraciones', style={'marginTop': 15})
            ], style={
                'flex': '1', 
                'marginRight': '20px', 
                'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 
                'borderRadius': '12px', 
                'padding': 25,
                'backgroundColor': 'white'
            }),
            
            html.Div([
                html.Div([
                    html.H3('📉 Perfil Vertical', 
                            style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 16, 'margin': '0 0 8px 0'}),
                    html.P('Desplazamiento vertical estimado (mm)',
                           style={'color': '#666', 'fontSize': 11, 'margin': 0})
                ]),
                dcc.Graph(id='grafica-perfil', style={'marginTop': 15})
            ], style={
                'flex': '1', 
                'marginLeft': '20px', 
                'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 
                'borderRadius': '12px', 
                'padding': 25,
                'backgroundColor': 'white'
            })
        ], style={'display': 'flex', 'gap': '0px', 'marginBottom': 40}),
        
        # Fila 3: Control del slider
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H3('🎯 Control de Ventana', 
                                style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 16, 'margin': '0 0 8px 0'}),
                        html.P('Desplaza la ventana de visualización a lo largo de la vía',
                               style={'color': '#666', 'fontSize': 11, 'margin': 0})
                    ]),
                    html.Div([
                        dcc.Slider(
                            id='chainage-slider',
                            min=0,
                            max=slider_max,
                            step=100,
                            value=0,
                            marks={i: f'{i} m' for i in range(0, int(slider_max)+1, int(slider_max/4))},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Div(id='slider-output', style={
                            'marginTop': 20, 
                            'fontSize': 13, 
                            'color': '#1565C0',
                            'fontWeight': '600',
                            'textAlign': 'center',
                            'padding': '15px',
                            'backgroundColor': '#E3F2FD',
                            'borderRadius': '6px',
                            'border': '1px solid #90CAF9'
                        })
                    ], style={'marginTop': 20})
                ])
            ], style={
                'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 
                'borderRadius': '12px', 
                'padding': 25,
                'backgroundColor': 'white'
            })
        ], style={'marginBottom': 40}),
        
        # Fila 4: Tabla de datos con diseño mejorado
        html.Div([
            html.Div([
                html.Div([
                    html.H3('📋 Datos del Kilómetro Actual', 
                            style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 16, 'margin': '0 0 8px 0'}),
                    html.P('Resumen agregado de calidad y vibraciones',
                           style={'color': '#666', 'fontSize': 11, 'margin': 0})
                ], style={'marginBottom': 20}),
                html.Div(id='tabla-segmentos')
            ], style={
                'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 
                'borderRadius': '12px', 
                'padding': 25,
                'backgroundColor': 'white'
            })
        ], style={'marginBottom': 40}),
        
    ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '40px 25px'})
    
], style={
    'fontFamily': '"Inter", "Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
    'backgroundColor': '#F8F9FA',
    'minHeight': '100vh',
    'color': '#333'
})

# ============================================================================
# 5. CALLBACKS PARA ACTUALIZAR GRÁFICAS CON SLIDER
# ============================================================================

@app.callback(
    [Output('grafica-aceleraciones', 'figure'),
     Output('grafica-perfil', 'figure'),
     Output('slider-output', 'children'),
     Output('tabla-segmentos', 'children')],
    Input('chainage-slider', 'value')
)
def update_graphs(chainage_min):
    fig_acc, fig_prof = create_window_figures(chainage_min)
    chainage_max = chainage_min + WINDOW_SIZE
    info_text = f"Visualizando: {chainage_min:.0f} - {chainage_max:.0f} metros ({WINDOW_SIZE/1000:.1f} km)"
    tabla = create_window_table(chainage_min)
    return fig_acc, fig_prof, info_text, tabla

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
