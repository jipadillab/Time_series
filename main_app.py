import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objects as go
import holidays
from datetime import date
from dateutil.relativedelta import relativedelta

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Predicción TRM Dólar Colombia", layout="wide", initial_sidebar_state="expanded")

# --- Funciones de Carga y Preprocesamiento de Datos ---

@st.cache_data
def load_data(file):
    """Carga y preprocesa los datos del archivo CSV."""
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'VIGENCIADESDE': 'fecha', 'VALOR': 'valor'})
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')
    df['valor'] = df['valor'].replace({'\$': '', ' ': '', ',': ''}, regex=True).astype(float)
    df = df[['fecha', 'valor']].sort_values('fecha').reset_index(drop=True)
    return df

def get_colombian_holidays(year_list):
    """Obtiene los festivos de Colombia para una lista de años."""
    return holidays.Colombia(years=year_list)

def add_features(df):
    """Agrega características de ingeniería de tiempo al DataFrame."""
    df_featured = df.copy()
    colombian_holidays = get_colombian_holidays(df_featured['fecha'].dt.year.unique())
    df_featured['dia_semana'] = df_featured['fecha'].dt.dayofweek
    df_featured['dia_mes'] = df_featured['fecha'].dt.day
    df_featured['mes'] = df_featured['fecha'].dt.month
    df_featured['anio'] = df_featured['fecha'].dt.year
    df_featured['festivo'] = df_featured['fecha'].apply(lambda x: x in colombian_holidays).astype(int)
    for lag in [1, 5, 10]:
        df_featured[f'valor_lag_{lag}'] = df_featured['valor'].shift(lag)
    df_featured['valor_roll_mean_5'] = df_featured['valor'].rolling(window=5).mean()
    df_featured = df_featured.dropna().reset_index(drop=True)
    return df_featured

# --- Funciones de Entrenamiento y Evaluación ---

def train_ml_model_with_error_margin(df, model_type='RandomForest'):
    """Entrena un modelo de ML y calcula un margen de error basado en los residuos."""
    df_train = df.copy()
    df_train['target'] = df_train['valor'].shift(-1)
    df_train = df_train.dropna()

    features = [col for col in df_train.columns if col not in ['fecha', 'target', 'valor']]
    X = df_train[features]
    y = df_train['target']

    # Dividir para calcular el error
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'XGBoost':
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    elif model_type == 'LightGBM':
        model = lgb.LGBMRegressor(objective='regression', n_estimators=100, random_state=42)
    else:
        model = LinearRegression()

    # Entrenar en el subconjunto de entrenamiento
    model.fit(X_train, y_train)
    
    # Calcular residuos en el conjunto de validación
    predictions_val = model.predict(X_val)
    residuals = y_val - predictions_val
    error_margin = residuals.std() * 1.96  # Para un intervalo de confianza del 95%

    # Re-entrenar el modelo con todos los datos para máxima precisión
    model.fit(X, y)
    
    return model, features, error_margin

# --- Interfaz de Streamlit ---

st.title('💵 Plataforma de Predicción de la TRM en Colombia')

with st.expander("Ver Instrucciones de Uso", expanded=False):
    st.markdown("""
    Esta plataforma te ofrece tres formas de analizar la TRM, ahora con **gráficas de incertidumbre**:
    1.  **🔮 Predicción para el Siguiente Día Hábil**: Estima el valor de la TRM para mañana y muestra una banda de confianza.
    2.  **🔍 Verificación en Fecha Pasada**: Compara la predicción con el valor real en una fecha histórica, incluyendo su margen de error.
    3.  **🗓️ Proyección a Futuro**: Proyecta el valor de la TRM hasta 3 meses en el futuro con su respectivo intervalo de confianza.
    
    **Pasos:**
    - **Carga tus datos** y **selecciona un rango** en la barra lateral para entrenar los modelos.
    - Elige la sección que deseas utilizar y presiona el botón correspondiente.
    """)

uploaded_file = st.file_uploader("Carga tu archivo CSV con el histórico de la TRM", type="csv")

if uploaded_file is not None:
    data_raw = load_data(uploaded_file)

    st.sidebar.header('⚙️ Configuración de Entrenamiento')
    
    min_date = data_raw['fecha'].min().date()
    max_date = data_raw['fecha'].max().date()

    start_date = st.sidebar.date_input('Fecha de inicio para entrenamiento', min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input('Fecha de fin para entrenamiento', max_date, min_value=start_date, max_value=max_date)

    training_data_raw = data_raw[(data_raw['fecha'].dt.date >= start_date) & (data_raw['fecha'].dt.date <= end_date)]

    if len(training_data_raw) < 50: # Aumentado para tener un conjunto de validación decente
        st.sidebar.warning(f"El rango seleccionado tiene {len(training_data_raw)} registros. Se necesitan al menos 50 para un entrenamiento y validación confiables.")
    else:
        st.sidebar.success(f"Rango de entrenamiento: {len(training_data_raw)} registros.")
        training_data_featured = add_features(training_data_raw.copy())

        model_choice = st.sidebar.selectbox(
            'Elige un modelo de ML (para predicción a 1 día y verificación):',
            ('Random Forest', 'XGBoost', 'LightGBM', 'Regresión Lineal')
        )

        st.markdown(f"#### Rango de Datos de Entrenamiento: `{start_date.strftime('%d/%m/%Y')}` al `{end_date.strftime('%d/%m/%Y')}`")
        st.markdown("---")

        # 1. Predicción para el Siguiente Día
        with st.container():
            st.header('🔮 Predicción para el Siguiente Día Hábil')
            if st.button('Calcular Predicción del Siguiente Día'):
                with st.spinner('Entrenando y prediciendo...'):
                    model_trm, features_trm, error_margin = train_ml_model_with_error_margin(training_data_featured, model_choice)
                    prediction_input = training_data_featured[features_trm].tail(1)
                    predicted_trm = model_trm.predict(prediction_input)[0]
                    
                    lower_bound = predicted_trm - error_margin
                    upper_bound = predicted_trm + error_margin

                    st.metric(label=f"Predicción TRM para el día siguiente con {model_choice}", value=f"${predicted_trm:,.2f}")
                    st.info(f"Intervalo de confianza (95%): Se espera que el valor se encuentre entre ${lower_bound:,.2f} y ${upper_bound:,.2f}.")

                    # Gráfica
                    fig = go.Figure()
                    last_30_days = training_data_raw.tail(30)
                    next_day = last_30_days['fecha'].iloc[-1] + pd.Timedelta(days=1)
                    fig.add_trace(go.Scatter(x=last_30_days['fecha'], y=last_30_days['valor'], mode='lines', name='TRM Histórica'))
                    fig.add_trace(go.Scatter(x=[next_day], y=[upper_bound], mode='lines', line=dict(color='rgba(0,0,0,0)')))
                    fig.add_trace(go.Scatter(x=[next_day], y=[lower_bound], fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'), name='Incertidumbre'))
                    fig.add_trace(go.Scatter(x=[next_day], y=[predicted_trm], mode='markers', name='Predicción', marker=dict(color='red', size=10)))
                    fig.update_layout(title='Predicción del Siguiente Día con Intervalo de Confianza', showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 2. Verificación en Fecha Pasada
        with st.container():
            st.header('🔍 Verificación en Fecha Pasada')
            min_selectable_date = training_data_featured['fecha'].min().date()
            max_selectable_date = training_data_featured['fecha'].max().date()

            specific_date = st.date_input('Elige una fecha para verificar', value=max_selectable_date, min_value=min_selectable_date, max_value=max_selectable_date)
            if st.button('Verificar en Fecha Seleccionada'):
                with st.spinner('Calculando...'):
                    date_index = training_data_featured[training_data_featured['fecha'].dt.date == specific_date].index
                    if not date_index.empty and date_index[0] > 0:
                        input_data = training_data_featured.iloc[[date_index[0] - 1]]
                        actual_value = training_data_featured.loc[date_index[0], 'valor']
                        
                        model_trm, features_trm, error_margin = train_ml_model_with_error_margin(training_data_featured, model_choice)
                        prediction = model_trm.predict(input_data[features_trm])[0]
                        diff = prediction - actual_value
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Predicción del Modelo", f"${prediction:,.2f}")
                        c2.metric("Valor Real", f"${actual_value:,.2f}")
                        c3.metric("Diferencia", f"${diff:,.2f}", delta_color="inverse")

                        # Gráfica
                        fig = go.Figure()
                        date_range_for_plot = training_data_raw[(training_data_raw['fecha'] >= pd.to_datetime(specific_date) - pd.Timedelta(days=15)) & 
                                                               (training_data_raw['fecha'] <= pd.to_datetime(specific_date) + pd.Timedelta(days=15))]
                        fig.add_trace(go.Scatter(x=date_range_for_plot['fecha'], y=date_range_for_plot['valor'], mode='lines', name='TRM Real'))
                        fig.add_trace(go.Scatter(x=[specific_date], y=[prediction], mode='markers', marker=dict(color='red', size=12), name='Predicción'))
                        fig.add_trace(go.Scatter(x=[specific_date, specific_date], y=[prediction - error_margin, prediction + error_margin], 
                                                 mode='lines', line=dict(color='red', width=2), name='Error Bar'))
                        fig.update_layout(title=f'Verificación para el {specific_date.strftime("%d/%m/%Y")}', showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay datos del día anterior para predecir esta fecha. Elige una fecha posterior.")

        st.markdown("---")

        # 3. Proyección a Futuro
        with st.container():
            st.header('🗓️ Proyección a Futuro (con Prophet)')
            future_date = st.date_input('Elige una fecha futura (hasta 3 meses)', value=end_date + relativedelta(months=1), min_value=end_date, max_value=end_date + relativedelta(months=3))
            if st.button('Realizar Proyección a Futuro'):
                with st.spinner('Generando proyección a largo plazo...'):
                    prophet_df = training_data_raw[['fecha', 'valor']].rename(columns={'fecha': 'ds', 'valor': 'y'})
                    model_prophet = Prophet().fit(prophet_df)
                    future_df = model_prophet.make_future_dataframe(periods=(future_date - end_date).days)
                    forecast = model_prophet.predict(future_df)
                    predicted_value = forecast[forecast['ds'].dt.date == future_date]
                    
                    if not predicted_value.empty:
                        yhat, yhat_lower, yhat_upper = predicted_value[['yhat', 'yhat_lower', 'yhat_upper']].values[0]
                        st.metric(f"Proyección para el {future_date.strftime('%d/%m/%Y')}", f"${yhat:,.2f}")
                        st.info(f"Rango de confianza: Se espera que el valor se encuentre entre ${yhat_lower:,.2f} y ${yhat_upper:,.2f}.")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='Límite Superior'))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='Límite Inferior'))
                        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', line_color='rgb(0,100,80)', name='Proyección'))
                        fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='markers', marker=dict(size=4, color='black'), name='Datos Históricos'))
                        fig.add_trace(go.Scatter(x=[pd.to_datetime(future_date)], y=[yhat], mode='markers', marker=dict(size=12, color='red', symbol='star'), name='Fecha Proyectada'))
                        fig.update_layout(title='Proyección de la TRM a Futuro', xaxis_title='Fecha', yaxis_title='Valor (COP)', showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("No se pudo generar una predicción para la fecha seleccionada.")

else:
    st.info('👈 Por favor, carga un archivo CSV para comenzar.')
