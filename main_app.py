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

def train_ml_model(df, model_type='RandomForest'):
    """Entrena un modelo de Machine Learning (RF, XGB, LGBM, LR)."""
    df_train = df.copy()
    df_train['target'] = df_train['valor'].shift(-1)
    df_train = df_train.dropna()

    features = [col for col in df_train.columns if col not in ['fecha', 'target', 'valor']]
    X = df_train[features]
    y = df_train['target']

    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'XGBoost':
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    elif model_type == 'LightGBM':
        model = lgb.LGBMRegressor(objective='regression', n_estimators=100, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X, y)
    return model, features

# --- Interfaz de Streamlit ---

st.title('💵 Plataforma de Predicción de la TRM en Colombia')

with st.expander("Ver Instrucciones de Uso", expanded=False):
    st.markdown("""
    Esta plataforma te ofrece tres formas de analizar la TRM:
    1.  **🔮 Predicción para el Siguiente Día Hábil**: Estima el valor de la TRM para mañana usando modelos de Machine Learning.
    2.  **🔍 Verificación en Fecha Pasada**: Compara la predicción de un modelo con el valor real en una fecha que elijas del histórico.
    3.  **🗓️ Proyección a Futuro**: Utiliza un modelo de series de tiempo (Prophet) para proyectar el valor de la TRM hasta 3 meses en el futuro.
    
    **Pasos:**
    - **Carga tus datos** y **selecciona un rango** en la barra lateral para entrenar los modelos.
    - Elige la sección que deseas utilizar y sigue las instrucciones.
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

    if len(training_data_raw) < 30:
        st.sidebar.warning(f"El rango seleccionado tiene {len(training_data_raw)} registros. Se necesitan al menos 30 para un entrenamiento confiable.")
    else:
        st.sidebar.success(f"Rango de entrenamiento: {len(training_data_raw)} registros.")
        training_data_featured = add_features(training_data_raw.copy())

        model_choice = st.sidebar.selectbox(
            'Elige un modelo de ML (para predicción a 1 día y verificación):',
            ('Random Forest', 'XGBoost', 'LightGBM', 'Regresión Lineal')
        )

        st.markdown(f"#### Rango de Datos de Entrenamiento: `{start_date.strftime('%d/%m/%Y')}` al `{end_date.strftime('%d/%m/%Y')}`")
        st.markdown("---")

        # --- TRES SECCIONES DE PREDICCIÓN ---
        
        # 1. Predicción para el Siguiente Día
        with st.container():
            st.header('🔮 Predicción para el Siguiente Día Hábil')
            if st.button('Calcular Predicción del Siguiente Día'):
                with st.spinner('Entrenando y prediciendo...'):
                    model_trm, features_trm = train_ml_model(training_data_featured, model_choice)
                    prediction_input = training_data_featured[features_trm].tail(1)
                    predicted_trm = model_trm.predict(prediction_input)[0]
                    st.metric(label=f"Predicción TRM para el día siguiente con {model_choice}", value=f"${predicted_trm:,.2f}")

        st.markdown("---")

        # 2. Verificación en Fecha Pasada
        with st.container():
            st.header('🔍 Verificación en Fecha Pasada')
            min_selectable_date = training_data_featured['fecha'].min().date()
            max_selectable_date = training_data_featured['fecha'].max().date()

            specific_date = st.date_input(
                'Elige una fecha dentro del rango para verificar la predicción',
                value=max_selectable_date, min_value=min_selectable_date, max_value=max_selectable_date
            )
            if st.button('Verificar en Fecha Seleccionada'):
                with st.spinner('Calculando...'):
                    date_index = training_data_featured[training_data_featured['fecha'].dt.date == specific_date].index
                    if not date_index.empty and date_index[0] > 0:
                        input_data = training_data_featured.iloc[[date_index[0] - 1]]
                        actual_value = training_data_featured.loc[date_index[0], 'valor']
                        
                        model_trm, features_trm = train_ml_model(training_data_featured, model_choice)
                        prediction = model_trm.predict(input_data[features_trm])[0]
                        diff = prediction - actual_value
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Predicción del Modelo", f"${prediction:,.2f}")
                        c2.metric("Valor Real", f"${actual_value:,.2f}")
                        c3.metric("Diferencia", f"${diff:,.2f}", delta_color="inverse")
                    else:
                        st.warning("No hay datos del día anterior para predecir esta fecha. Elige una fecha posterior.")

        st.markdown("---")

        # 3. Proyección a Futuro
        with st.container():
            st.header('🗓️ Proyección a Futuro (con Prophet)')
            future_date = st.date_input(
                'Elige una fecha futura (hasta 3 meses)',
                value=end_date + relativedelta(months=1),
                min_value=end_date,
                max_value=end_date + relativedelta(months=3)
            )
            if st.button('Realizar Proyección a Futuro'):
                with st.spinner('Generando proyección a largo plazo...'):
                    prophet_df = training_data_raw[['fecha', 'valor']].rename(columns={'fecha': 'ds', 'valor': 'y'})
                    model_prophet = Prophet().fit(prophet_df)
                    
                    future_df = model_prophet.make_future_dataframe(periods=(future_date - end_date).days)
                    forecast = model_prophet.predict(future_df)
                    
                    predicted_value = forecast[forecast['ds'].dt.date == future_date]
                    
                    if not predicted_value.empty:
                        yhat = predicted_value['yhat'].values[0]
                        yhat_lower = predicted_value['yhat_lower'].values[0]
                        yhat_upper = predicted_value['yhat_upper'].values[0]
                        
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
