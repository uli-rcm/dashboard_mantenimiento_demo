#!/usr/bin/env python3
"""
Configuración modular de pestañas para el dashboard ferroviario
Cada pestaña se define como una función independiente que retorna:
- layout: El componente dcc.Tab
- callbacks: Lista de funciones callback
"""

from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ============================================================================
# TAB 1: CALIDAD GEOMÉTRICA (Aceleraciones + Perfil Vertical)
# ============================================================================

def build_tab_geometrica(app, pos_df, seg_df, RMS_THRESHOLD_GOOD, RMS_THRESHOLD_REGULAR, WINDOW_SIZE, slider_max, get_map_html):
    """
    Construye la pestaña de Calidad Geométrica
    
    Args:
        app: Instancia de Dash app
        pos_df: DataFrame con datos de posición y aceleración
        seg_df: DataFrame con segmentos y calidades
        RMS_THRESHOLD_GOOD: Umbral de buena calidad
        RMS_THRESHOLD_REGULAR: Umbral de calidad regular
        WINDOW_SIZE: Tamaño de ventana en metros
        slider_max: Valor máximo del slider
        get_map_html: Función para obtener HTML del mapa
    
    Returns:
        dcc.Tab: Componente de la pestaña
    """
    
    def cumtrapz(y, dx=1.0, initial=None):
        """Integración acumulativa usando regla trapezoidal"""
        result = np.zeros_like(y, dtype=float)
        result[1:] = np.cumsum((y[:-1] + y[1:]) * dx / 2.0)
        if initial is not None:
            result = result + initial
        return result
    
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
            x=df_window['chainage_m'], y=df_window['rms_acc']*1000,
            mode='lines+markers', name='Aceleración RMS (mm/s²)',
            marker=dict(size=8, color='red'),
            hovertemplate='<b>Chainage: %{x:.1f} m</b><br>RMS: %{y:.3f} mm/s²<extra></extra>'
        ))
        
        # Colorear bandas según calidad
        for _, s in seg_df.iterrows():
            x0 = pos_df['chainage_m'].iloc[s['start']]
            x1 = pos_df['chainage_m'].iloc[s['end']]
            x0_win = max(x0, chainage_min)
            x1_win = min(x1, chainage_max)
            if x0_win < x1_win:
                if s['quality']=='buena':
                    color = 'rgba(39,174,96,0.12)'
                elif s['quality']=='regular':
                    color = 'rgba(243,156,18,0.12)'
                else:
                    color = 'rgba(231,76,60,0.12)'
                fig_acc.add_vrect(x0=x0_win, x1=x1_win, fillcolor=color, layer='below', line_width=0)
        
        fig_acc.update_layout(
            title=dict(text=f'Aceleraciones (RMS): {chainage_min:.0f} - {chainage_max:.0f} m', font=dict(size=14, color='#333', family='Arial')),
            xaxis_title='Cadenamiento (m)',
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
                if s['quality']=='buena':
                    color = 'rgba(39,174,96,0.12)'
                elif s['quality']=='regular':
                    color = 'rgba(243,156,18,0.12)'
                else:
                    color = 'rgba(231,76,60,0.12)'
                fig_prof.add_vrect(x0=x0_win, x1=x1_win, fillcolor=color, layer='below', line_width=0)
        
        fig_prof.update_layout(
            title=dict(text=f'Perfil vertical: {chainage_min:.0f} - {chainage_max:.0f} m', font=dict(size=14, color='#333', family='Arial')),
            xaxis_title='Cadenamiento (m)',
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
        """Crea tabla con datos de ventana actual"""
        chainage_start = chainage_min
        chainage_end = chainage_min + 1000
        
        mask = (pos_df['chainage_m'] >= chainage_start) & (pos_df['chainage_m'] < chainage_end)
        data_in_window = pos_df[mask]
        
        if len(data_in_window) == 0:
            return html.Table([
                html.Thead(html.Tr([
                    html.Th('P.K. Inicial (m)', style={'padding': '10px', 'borderBottom': '2px solid #ddd'}),
                    html.Th('P.K. Final (m)', style={'padding': '10px', 'borderBottom': '2px solid #ddd'}),
                ])),
                html.Tbody(html.Tr(html.Td("Sin datos", colSpan=2, style={'padding': '10px', 'textAlign': 'center'})))
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': 'white', 'borderRadius': '5px'})
        
        avg_rms = data_in_window['rms_acc'].mean()
        
        if avg_rms < RMS_THRESHOLD_GOOD:
            quality = 'buena'
            color = "#08DA5F"
        elif avg_rms < RMS_THRESHOLD_REGULAR:
            quality = 'regular'
            color = "#F3EF12"
        else:
            quality = 'mala'
            color = "#F51E06"
        
        row = html.Tr([
            html.Td(f"{chainage_start:.0f}", style={'padding': '16px 18px', 'border': 'none', 'borderBottom': '1px solid #E0E0E0', 'fontSize': '14px', 'fontWeight': '500', 'color': '#333'}),
            html.Td(f"{chainage_end:.0f}", style={'padding': '16px 18px', 'border': 'none', 'borderBottom': '1px solid #E0E0E0', 'fontSize': '14px', 'fontWeight': '500', 'color': '#333'}),
            html.Td(f"{avg_rms:.5f}", style={'padding': '16px 18px', 'border': 'none', 'borderBottom': '1px solid #E0E0E0', 'fontSize': '14px', 'fontWeight': '600', 'color': '#333', 'fontFamily': 'monospace'}),
            html.Td(f"● {quality.upper()}", style={'padding': '16px 18px', 'border': 'none', 'borderBottom': '1px solid #E0E0E0', 'color': 'white', 'backgroundColor': color, 'fontWeight': '700', 'textAlign': 'center', 'borderRadius': '6px', 'fontSize': '13px', 'letterSpacing': '0.5px'})
        ])
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th('P.K. Inicial (m)', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'left', 'fontSize': '13px', 'letterSpacing': '0.3px'}),
                html.Th('P.K. Final (m)', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'left', 'fontSize': '13px', 'letterSpacing': '0.3px'}),
                html.Th('RMS (m/s²)', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'left', 'fontSize': '13px', 'letterSpacing': '0.3px'}),
                html.Th('Calidad', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'center', 'fontSize': '13px', 'letterSpacing': '0.3px'}),
            ])),
            html.Tbody(row)
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': 'white', 'borderRadius': '6px'})
    
    # Layout de la pestaña
    tab_layout = dcc.Tab(
        label='Calidad Geometrica',
        value='tab-geometrica',
        children=[
            html.Div([
                # Fila 1: Mapa
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3('🗺️ Mapa de Calidad Geometrica', 
                                    style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 22, 'margin': '0 0 15px 0'}),
                            html.P('Visualizacion geoespacial del estado de la infraestructura',
                                   style={'color': '#666', 'fontSize': 14, 'margin': 0})
                        ]),
                        html.Div([
                            dcc.Graph(id='mapa-via', style={'display': 'none'}),
                            html.Iframe(srcDoc=get_map_html('geometrica', 'Calidad Geometrica'), width='100%', height=500, style={'border': 'none', 'borderRadius': '8px'})
                        ], style={'marginTop': 15})
                    ], style={'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 'borderRadius': '12px', 'padding': 25, 'backgroundColor': 'white', 'transition': 'box-shadow 0.3s ease'})
                ], style={'marginBottom': 40}),

                # Fila 2: Dos gráficas
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3('📈 Aceleracion RMS', 
                                    style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 20, 'margin': '0 0 8px 0'}),
                            html.P('Energia acumulada de vibracion (mm/s2)',
                                   style={'color': '#666', 'fontSize': 13, 'margin': 0})
                        ]),
                        dcc.Graph(id='grafica-aceleraciones', style={'marginTop': 15})
                    ], style={'flex': '1', 'marginRight': '20px', 'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 'borderRadius': '12px', 'padding': 25, 'backgroundColor': 'white'}),

                    html.Div([
                        html.Div([
                            html.H3('📉 Perfil Vertical', 
                                    style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 20, 'margin': '0 0 8px 0'}),
                            html.P('Desplazamiento vertical estimado (mm)',
                                   style={'color': '#666', 'fontSize': 13, 'margin': 0})
                        ]),
                        dcc.Graph(id='grafica-perfil', style={'marginTop': 15})
                    ], style={'flex': '1', 'marginLeft': '20px', 'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 'borderRadius': '12px', 'padding': 25, 'backgroundColor': 'white'})
                ], style={'display': 'flex', 'gap': '0px', 'marginBottom': 40}),

                # Fila 3: Slider
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3('🎯 Control de Ventana', 
                                    style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 20, 'margin': '0 0 8px 0'}),
                            html.P('Desplaza la ventana de visualizacion a lo largo de la via',
                                   style={'color': '#666', 'fontSize': 13, 'margin': 0}),
                            dcc.Slider(
                                id='chainage-slider',
                                min=0,
                                max=slider_max,
                                step=100,
                                value=0,
                                marks={i: f'{i} m' for i in range(0, int(slider_max)+1, int(slider_max/4))},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Div(id='slider-output', style={'marginTop': 20, 'fontSize': 13, 'color': '#1565C0', 'fontWeight': '600', 'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#E3F2FD', 'borderRadius': '6px', 'border': '1px solid #90CAF9'})
                        ])
                    ], style={'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 'borderRadius': '12px', 'padding': 25, 'backgroundColor': 'white'})
                ], style={'marginBottom': 40}),

                # Fila 4: Tabla
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3('📋 Datos del Kilometro Actual', 
                                    style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 20, 'margin': '0 0 8px 0'}),
                            html.P('Resumen agregado de calidad y vibraciones',
                                   style={'color': '#666', 'fontSize': 13, 'margin': 0})
                        ], style={'marginBottom': 20}),
                        html.Div(id='tabla-segmentos')
                    ], style={'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 'borderRadius': '12px', 'padding': 25, 'backgroundColor': 'white'})
                ], style={'marginBottom': 40})

            ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '40px 25px'})
        ]
    )
    
    # Registrar callback para esta pestaña
    @app.callback(
        [Output('grafica-aceleraciones', 'figure'),
         Output('grafica-perfil', 'figure'),
         Output('slider-output', 'children'),
         Output('tabla-segmentos', 'children')],
        Input('chainage-slider', 'value')
    )
    def update_geometrica(chainage_min):
        fig_acc, fig_prof = create_window_figures(chainage_min)
        chainage_max = chainage_min + WINDOW_SIZE
        info_text = f"Visualizando: {chainage_min:.0f} - {chainage_max:.0f} metros ({WINDOW_SIZE/1000:.1f} km)"
        tabla = create_window_table(chainage_min)
        return fig_acc, fig_prof, info_text, tabla
    
    return tab_layout


# ============================================================================
# TAB 2: CALIDAD ESTRUCTURAL (GPR)
# ============================================================================

def build_tab_estructural(app, pos_df, seg_df, gpr_df, gpr_depth_samples, WINDOW_SIZE, slider_max, get_map_html):
    """
    Construye la pestaña de Calidad Estructural (GPR)
    
    Args:
        app: Instancia de Dash app
        pos_df: DataFrame con datos de posición
        seg_df: DataFrame con segmentos
        gpr_df: DataFrame con datos GPR
        gpr_depth_samples: Número de muestras en profundidad
        WINDOW_SIZE: Tamaño de ventana en metros
        slider_max: Valor máximo del slider
        get_map_html: Función para obtener HTML del mapa
    
    Returns:
        dcc.Tab: Componente de la pestaña
    """
    
    def create_gpr_window_figure(chainage_min):
        """Crea figura GPR tipo B-scan"""
        chainage_max = chainage_min + WINDOW_SIZE
        
        mask = (gpr_df['chainage_m'] >= chainage_min) & (gpr_df['chainage_m'] <= chainage_max)
        gpr_window = gpr_df[mask].copy()
        
        if len(gpr_window) == 0:
            fig = go.Figure()
            fig.add_annotation(text="Sin datos GPR en esta ventana")
            return fig
        
        bscan_matrix = np.array([profile for profile in gpr_window['gpr_profile'].values]).T
        
        profundidad_total_mm = 1400
        depth_scale = np.linspace(0, profundidad_total_mm, gpr_depth_samples)
        chainage_window = gpr_window['chainage_m'].values
        
        fig = go.Figure(data=go.Heatmap(
            z=bscan_matrix,
            x=chainage_window,
            y=depth_scale,
            colorscale='Greys_r',
            showscale=True,
            colorbar=dict(
                title="Amplitud Reflexión",
                thickness=20,
                len=0.6,
                x=0.5,
                y=-0.15,
                xanchor='center',
                yanchor='top',
                orientation='h'
            ),
            hovertemplate='<b>Chainage: %{x:.1f} m</b><br>Profundidad: %{y:.0f} mm<br>Amplitud: %{z:.3f}<extra></extra>'
        ))
        
        fig.add_hline(y=400, line_dash="dash", line_color="cyan", opacity=0.6, annotation_text="Superficie balasto", annotation_position="right")
        fig.add_hline(y=700, line_dash="dash", line_color="yellow", opacity=0.6, annotation_text="Interfaz balasto-subbalasto (fouled aquí)", annotation_position="right")
        fig.add_hline(y=1000, line_dash="dash", line_color="orange", opacity=0.6, annotation_text="Interfaz subbalasto-subrasante", annotation_position="right")
        fig.add_hline(y=1400, line_dash="dash", line_color="red", opacity=0.6, annotation_text="Base subrasante", annotation_position="right")
        
        for _, s in seg_df.iterrows():
            x0 = pos_df['chainage_m'].iloc[s['start']]
            x1 = pos_df['chainage_m'].iloc[s['end']]
            x0_win = max(x0, chainage_min)
            x1_win = min(x1, chainage_max)
            if x0_win < x1_win and s['quality'] == 'mala':
                fig.add_vrect(x0=x0_win, x1=x1_win, fillcolor='rgba(231,76,60,0.08)', layer='below', line_width=0)
        
        fig.update_layout(
            title=dict(text=f'Perfil GPR (B-scan): {chainage_min:.0f} - {chainage_max:.0f} m<br><sub>Radargrama con reflexiones en interfaces de capas</sub>', font=dict(size=14, color='#333', family='Arial')),
            xaxis_title='Cadenamiento (m)',
            yaxis_title='Profundidad (mm)',
            height=600,
            hovermode='closest',
            plot_bgcolor='rgba(245,243,240,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=False),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', zeroline=False, autorange='reversed'),
            margin=dict(l=70, r=280, t=100, b=150)
        )
        
        return fig
    
    # Layout de la pestaña
    tab_layout = dcc.Tab(
        label='Calidad Estructural',
        value='tab-estructural',
        children=[
            html.Div([
                # Mapa
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3('🧱 Mapa de Calidad Estructural', 
                                    style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 22, 'margin': '0 0 15px 0'}),
                            html.P('Resultados tipo GPR sobre la via y su condicion estructural',
                                   style={'color': '#666', 'fontSize': 14, 'margin': 0})
                        ]),
                        html.Div([
                            dcc.Graph(id='mapa-via-estructural', style={'display': 'none'}),
                            html.Iframe(srcDoc=get_map_html('estructural', 'Calidad Estructural'), width='100%', height=500, style={'border': 'none', 'borderRadius': '8px'})
                        ], style={'marginTop': 15})
                    ], style={'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 'borderRadius': '12px', 'padding': 25, 'backgroundColor': 'white', 'transition': 'box-shadow 0.3s ease'})
                ], style={'marginBottom': 40}),

                # Gráfica GPR
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3('📡 Perfil GPR (Ground Penetrating Radar)', 
                                    style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 20, 'margin': '0 0 8px 0'}),
                            html.P('Deteccion de reflexiones en interfaces: balasto, subbalasto y subrasante. Ausencia de reflexiones = fouled ballast.',
                                   style={'color': '#666', 'fontSize': 13, 'margin': 0})
                        ]),
                        dcc.Graph(id='grafica-gpr', style={'marginTop': 15})
                    ], style={'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 'borderRadius': '12px', 'padding': 25, 'backgroundColor': 'white'})
                ], style={'marginBottom': 40}),

                # Slider
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3('🎯 Control de Ventana', 
                                    style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 20, 'margin': '0 0 8px 0'}),
                            html.P('Desplaza la ventana de visualizacion a lo largo de la via',
                                   style={'color': '#666', 'fontSize': 13, 'margin': 0}),
                            dcc.Slider(
                                id='chainage-slider-gpr',
                                min=0,
                                max=slider_max,
                                step=100,
                                value=0,
                                marks={i: f'{i} m' for i in range(0, int(slider_max)+1, int(slider_max/4))},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Div(id='slider-output-gpr', style={'marginTop': 20, 'fontSize': 13, 'color': '#1565C0', 'fontWeight': '600', 'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#E3F2FD', 'borderRadius': '6px', 'border': '1px solid #90CAF9'})
                        ])
                    ], style={'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 'borderRadius': '12px', 'padding': 25, 'backgroundColor': 'white'})
                ], style={'marginBottom': 40}),

                # Tabla GPR
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3('📋 Evaluacion Estructural del Kilometro Actual', 
                                    style={'color': '#1565C0', 'fontWeight': '700', 'fontSize': 20, 'margin': '0 0 8px 0'}),
                            html.P('Resumen de condiciones estructurales detectadas por GPR',
                                   style={'color': '#666', 'fontSize': 13, 'margin': 0})
                        ], style={'marginBottom': 20}),
                        html.Div(id='tabla-gpr')
                    ], style={'boxShadow': '0 8px 24px rgba(0,0,0,0.12)', 'borderRadius': '12px', 'padding': 25, 'backgroundColor': 'white'})
                ], style={'marginBottom': 40})

            ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '40px 25px'})
        ]
    )
    
    # Registrar callback para esta pestaña
    @app.callback(
        [Output('grafica-gpr', 'figure'),
         Output('slider-output-gpr', 'children'),
         Output('tabla-gpr', 'children')],
        Input('chainage-slider-gpr', 'value')
    )
    def update_estructural(chainage_min):
        fig_gpr = create_gpr_window_figure(chainage_min)
        chainage_max = chainage_min + WINDOW_SIZE
        info_text = f"Visualizando: {chainage_min:.0f} - {chainage_max:.0f} metros ({WINDOW_SIZE/1000:.1f} km)"
        
        chainage_start = chainage_min
        chainage_end = chainage_min + 1000
        
        fouled_count = 0
        clean_count = 0
        for _, s in seg_df.iterrows():
            x0 = pos_df['chainage_m'].iloc[s['start']]
            x1 = pos_df['chainage_m'].iloc[s['end']]
            x0_win = max(x0, chainage_start)
            x1_win = min(x1, chainage_end)
            if x0_win < x1_win:
                if s['quality'] == 'mala':
                    fouled_count += 1
                else:
                    clean_count += 1
        
        fouled_percent = (fouled_count / (fouled_count + clean_count) * 100) if (fouled_count + clean_count) > 0 else 0
        
        if fouled_percent > 30:
            condition = 'CRITICA'
            color = '#F51E06'
        elif fouled_percent > 10:
            condition = 'REGULAR'
            color = '#F3EF12'
        else:
            condition = 'BUENA'
            color = '#08DA5F'
        
        tabla_html = html.Table([
            html.Thead(html.Tr([
                html.Th('P.K. Inicial (m)', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'left', 'fontSize': '13px'}),
                html.Th('P.K. Final (m)', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'left', 'fontSize': '13px'}),
                html.Th('Fouled Ballast (%)', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'left', 'fontSize': '13px'}),
                html.Th('Condicion Estructural', style={'padding': '14px 18px', 'backgroundColor': '#1565C0', 'color': 'white', 'fontWeight': '700', 'borderBottom': '3px solid #0D47A1', 'textAlign': 'center', 'fontSize': '13px'}),
            ])),
            html.Tbody(html.Tr([
                html.Td(f"{chainage_start:.0f}", style={'padding': '16px 18px', 'border': 'none', 'borderBottom': '1px solid #E0E0E0', 'fontSize': '14px', 'fontWeight': '500', 'color': '#333'}),
                html.Td(f"{chainage_end:.0f}", style={'padding': '16px 18px', 'border': 'none', 'borderBottom': '1px solid #E0E0E0', 'fontSize': '14px', 'fontWeight': '500', 'color': '#333'}),
                html.Td(f"{fouled_percent:.1f}%", style={'padding': '16px 18px', 'border': 'none', 'borderBottom': '1px solid #E0E0E0', 'fontSize': '14px', 'fontWeight': '600', 'color': '#333', 'fontFamily': 'monospace'}),
                html.Td(f"● {condition}", style={'padding': '16px 18px', 'border': 'none', 'borderBottom': '1px solid #E0E0E0', 'color': 'white', 'backgroundColor': color, 'fontWeight': '700', 'textAlign': 'center', 'borderRadius': '6px', 'fontSize': '13px', 'letterSpacing': '0.5px'})
            ]))
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'backgroundColor': 'white', 'borderRadius': '6px'})
        
        return fig_gpr, info_text, tabla_html
    
    return tab_layout
