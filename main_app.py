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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import holidays
from datetime import date, timedelta

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
    df['centavos'] = (df['valor'] - np.floor(df['valor'])) * 100
    return df

def get_colombian_holidays(year_list):
    """Obtiene los festivos de Colombia para una lista de años."""
    return holidays.Colombia(years=year_list)

def add_features(df):
    """Agrega características de ingeniería de tiempo al DataFrame."""
    colombian_holidays = get_colombian_holidays(df['fecha'].dt.year.unique())
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['dia_mes'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['anio'] = df['fecha'].dt.year
    df['festivo'] = df['fecha'].apply(lambda x: x in colombian_holidays).astype(int)
    for lag in [1, 5, 10]:
        df[f'valor_lag_{lag}'] = df['valor'].shift(lag)
        df[f'centavos_lag_{lag}'] = df['centavos'].shift(lag)
    df['valor_roll_mean_5'] = df['valor'].rolling(window=5).mean()
    df = df.dropna().reset_index(drop=True)
    return df

# --- Funciones de Entrenamiento y Evaluación ---

def train_ml_model(df, model_type='RandomForest', target='valor'):
    """Entrena un modelo de Machine Learning (RF, XGB, LGBM, LR)."""
    df_train = df.copy()
    df_train['target'] = df_train[target].shift(-1)
    df_train = df_train.dropna()

    features = [col for col in df_train.columns if col not in ['fecha', 'target', 'valor', 'centavos']]
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

# --- Nueva Función para Encontrar el Mejor Modelo ---
@st.cache_data
def find_best_model(data_featured):
    """Evalúa varios modelos y devuelve el mejor basado en MAPE."""
    models_to_evaluate = ['Random Forest', 'XGBoost', 'LightGBM', 'Regresión Lineal']
    performance = {}

    train_data, test_data = train_test_split(data_featured, test_size=0.2, shuffle=False)
    
    for model_name in models_to_evaluate:
        model_eval, features_eval = train_ml_model(train_data, model_name, target='valor')
        
        # Asegurarse de que X_test tenga las mismas dimensiones que los datos de entrenamiento
        X_test = test_data.shift(1).dropna()[features_eval]
        y_test = test_data.iloc[1:]['valor']
        
        # Alinear y_test con X_test
        y_test = y_test.tail(len(X_test))

        if not X_test.empty:
            predictions = model_eval.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, predictions) * 100
            performance[model_name] = mape

    if not performance:
        return None, None

    best_model_name = min(performance, key=performance.get)
    min_error = performance[best_model_name]
    
    return best_model_name, min_error

# --- Interfaz de Streamlit ---

st.title('💵 Predicción Avanzada de la TRM del Dólar en Colombia')

with st.expander("Ver Instrucciones de Uso", expanded=False):
    st.markdown("""
    1.  **Carga tus Datos**: Sube tu archivo CSV con el histórico de la TRM.
    2.  **Selecciona el Rango de Entrenamiento**: En la barra lateral, elige las fechas de inicio y fin para entrenar los modelos. **Necesitas al menos 30 días de datos.**
    3.  **Obtén una Recomendación**: Haz clic en "Sugerir Mejor Modelo" para que la app analice y te diga cuál modelo tiene el menor error para el rango de fechas seleccionado.
    4.  **Configura y Realiza la Predicción**: Elige el modelo que prefieras (¡quizás el recomendado!) y presiona "Realizar Predicción".
    5.  **Analiza y Evalúa**: Revisa los resultados y, si quieres, explora la sección de "Evaluar Desempeño" para un análisis más profundo.
    """)

uploaded_file = st.file_uploader("Carga tu archivo CSV con el histórico de la TRM", type="csv")

if uploaded_file is not None:
    data_raw = load_data(uploaded_file)

    # --- Sidebar de Opciones ---
    st.sidebar.header('⚙️ Configuración de Entrenamiento y Predicción')
    
    # --- Selección de Rango de Fechas ---
    min_date = data_raw['fecha'].min().date()
    max_date = data_raw['fecha'].max().date()

    start_date = st.sidebar.date_input('Fecha de inicio para entrenamiento', min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input('Fecha de fin para entrenamiento', max_date, min_value=start_date, max_value=max_date)

    # Filtrar datos según el rango seleccionado
    training_data = data_raw[(data_raw['fecha'].dt.date >= start_date) & (data_raw['fecha'].dt.date <= end_date)]

    if len(training_data) < 30:
        st.sidebar.warning(f"El rango seleccionado contiene solo {len(training_data)} registros. Por favor, selecciona un rango con al menos 30 registros para un entrenamiento confiable.")
    else:
        st.sidebar.success(f"Rango seleccionado contiene {len(training_data)} registros para entrenar.")
        data_featured = add_features(training_data.copy())

        # --- Sugerencia de Modelo ---
        if st.sidebar.button('Sugerir Mejor Modelo'):
            with st.spinner('Analizando modelos...'):
                best_model, min_error = find_best_model(data_featured)
                if best_model:
                    st.sidebar.success(f"🏆 **Recomendado:**\n**{best_model}**\n(Error: {min_error:.2f}%)")
                else:
                    st.sidebar.error("No se pudo determinar el mejor modelo. Intenta con un rango de fechas más amplio.")

        model_choice = st.sidebar.selectbox(
            'Elige un modelo de predicción:',
            ('Random Forest', 'XGBoost', 'LightGBM', 'Regresión Lineal'), # Simplificado para la recomendación
            help="Elige el modelo para realizar la predicción. Usa el botón de sugerencia para una recomendación."
        )

        st.subheader(f'Datos de Entrenamiento: {start_date.strftime("%d/%m/%Y")} al {end_date.strftime("%d/%m/%Y")}')
        
        st.markdown("---")

        # --- Sección de Predicción ---
        st.header(f'🔮 Resultados de la Predicción con {model_choice}')
        
        if st.button('Realizar Predicción'):
            with st.spinner('Entrenando y prediciendo...'):
                # Predicción TRM
                model_trm, features_trm = train_ml_model(data_featured, model_choice, target='valor')
                prediction_input = data_featured[features_trm].tail(1)
                predicted_trm = model_trm.predict(prediction_input)[0]

                # Predicción Centavos
                model_cents, features_cents = train_ml_model(data_featured, 'LightGBM', target='centavos')
                prediction_input_cents = data_featured[features_cents].tail(1)
                predicted_cents = model_cents.predict(prediction_input_cents)[0]
                predicted_cents = max(0, min(99.99, predicted_cents))

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label=f"Predicción TRM (Valor Completo)", value=f"${predicted_trm:,.2f}")
                with col2:
                    st.metric(label="Predicción de Centavos", value=f"{predicted_cents:.2f} ¢")
                
                st.subheader('Tendencia Histórica y Punto de Predicción')
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(x=training_data['fecha'], y=training_data['valor'], mode='lines', name='TRM Histórica (Entrenamiento)'))
                next_day = training_data['fecha'].iloc[-1] + pd.Timedelta(days=1)
                fig_hist.add_trace(go.Scatter(x=[next_day], y=[predicted_trm], mode='markers', name='Predicción', 
                                              marker=dict(color='red', size=12, symbol='star')))
                fig_hist.update_layout(title='Histórico de la TRM y Valor Predicho', xaxis_title='Fecha', yaxis_title='Valor (COP)')
                st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # --- Sección de Desempeño del Modelo ---
        with st.expander("📊 Evaluar Desempeño del Modelo Seleccionado"):
            if st.button('Calcular Desempeño del Modelo'):
                with st.spinner('Evaluando...'):
                    train_data, test_data = train_test_split(data_featured, test_size=0.2, shuffle=False)
                    
                    if len(test_data) > 1:
                        model_eval, features_eval = train_ml_model(train_data, model_choice, target='valor')
                        X_test = test_data.shift(1).dropna()[features_eval]
                        y_test = test_data.iloc[1:]['valor']
                        y_test = y_test.tail(len(X_test))

                        predictions = model_eval.predict(X_test)

                        mae = mean_absolute_error(y_test, predictions)
                        rmse = np.sqrt(mean_squared_error(y_test, predictions))
                        mape = mean_absolute_percentage_error(y_test, predictions) * 100

                        st.subheader(f'Métricas de Desempeño para {model_choice}')
                        c1, c2, c3 = st.columns(3)
                        c1.metric("MAE", f"${mae:,.2f}")
                        c2.metric("RMSE", f"${rmse:,.2f}")
                        c3.metric("MAPE", f"{mape:.2f}%")

                        fig_eval = go.Figure()
                        fig_eval.add_trace(go.Scatter(x=test_data.iloc[1:]['fecha'], y=y_test, mode='lines', name='Valor Real'))
                        fig_eval.add_trace(go.Scatter(x=test_data.iloc[1:]['fecha'], y=predictions, mode='lines', name='Predicción', line=dict(dash='dot')))
                        fig_eval.update_layout(title='Comparación Real vs. Predicción (Datos de Prueba)',
                                               xaxis_title='Fecha', yaxis_title='Valor (COP)')
                        st.plotly_chart(fig_eval, use_container_width=True)
                    else:
                        st.warning("No hay suficientes datos en el rango seleccionado para crear un conjunto de prueba. Por favor, elige un rango de fechas más amplio.")

else:
    st.info('👈 Por favor, carga un archivo CSV para comenzar.')
