import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import holidays
from datetime import date

# --- Configuración de la página de Streamlit ---
st.set_page_config(page_title="Predicción TRM Dólar Colombia", layout="wide", initial_sidebar_state="expanded")

# --- Funciones de Carga y Preprocesamiento de Datos ---

@st.cache_data
def load_data(file):
    """Carga y preprocesa los datos del archivo CSV."""
    df = pd.read_csv(file)
    # Limpieza de nombres de columna (eliminar espacios)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'VIGENCIADESDE': 'fecha', 'VALOR': 'valor'})
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')
    df['valor'] = df['valor'].replace({'\$': '', ' ': '', ',': ''}, regex=True).astype(float)
    df = df[['fecha', 'valor']].sort_values('fecha').reset_index(drop=True)
    df['centavos'] = (df['valor'] - np.floor(df['valor'])) * 100
    return df

def get_colombian_holidays(year):
    """Obtiene los festivos de Colombia para un año específico."""
    return holidays.Colombia(years=year)

def add_features(df):
    """Agrega características de ingeniería de tiempo al DataFrame."""
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['dia_mes'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['anio'] = df['fecha'].dt.year
    df['festivo'] = df['fecha'].apply(lambda x: x in get_colombian_holidays(x.year)).astype(int)
    # Lags y promedios móviles
    for lag in [1, 5, 10]:
        df[f'valor_lag_{lag}'] = df['valor'].shift(lag)
        df[f'centavos_lag_{lag}'] = df['centavos'].shift(lag)
    df['valor_roll_mean_5'] = df['valor'].rolling(window=5).mean()
    df = df.dropna().reset_index(drop=True)
    return df

# --- Funciones de Entrenamiento de Modelos ---

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
    else: # Regresión Lineal
        model = LinearRegression()

    model.fit(X, y)
    return model, features

def create_dataset_for_lstm(dataset, look_back=10):
    """Crea secuencias para el modelo LSTM."""
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def train_lstm_model(df, look_back=10):
    """Entrena un modelo LSTM."""
    dataset = df['valor'].values.reshape(-1, 1)
    X, y = create_dataset_for_lstm(dataset, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    return model

# --- Interfaz de Streamlit ---

st.title('💵 Predicción Avanzada de la TRM del Dólar en Colombia')

# --- Instrucciones de Uso ---
with st.expander("Ver Instrucciones de Uso", expanded=True):
    st.markdown("""
    **¡Bienvenido al predictor de la TRM! Sigue estos sencillos pasos:**

    1.  **Carga tus Datos**: Haz clic en el botón "Cargar archivo" y selecciona tu archivo CSV con el histórico de la TRM. El formato debe ser el proporcionado por el Banco de la República.
    2.  **Configura la Predicción**: En la barra lateral izquierda, elige el modelo de predicción que deseas utilizar. Tienes desde modelos clásicos hasta redes neuronales (LSTM).
    3.  **Realiza la Predicción**: Presiona el botón "Realizar Predicción". La aplicación calculará y mostrará el valor estimado de la TRM y los centavos para el siguiente día hábil.
    4.  **Analiza los Resultados**: Revisa las métricas y las gráficas para entender la predicción.
    5.  **Evalúa el Desempeño (Opcional)**: Si quieres saber qué tan preciso es el modelo, despliega la sección "Evaluar Desempeño de los Modelos" en la parte inferior. Verás métricas de error y una comparación gráfica entre los valores reales y los predichos.
    """)

# --- Carga de datos ---
uploaded_file = st.file_uploader("Carga tu archivo CSV con el histórico de la TRM", type="csv")

if uploaded_file is not None:
    data_raw = load_data(uploaded_file)
    data_featured = add_features(data_raw.copy())

    # --- Sidebar de Opciones ---
    st.sidebar.header('⚙️ Configuración de Predicción')
    model_choice = st.sidebar.selectbox(
        'Elige un modelo de predicción:',
        ('Random Forest', 'XGBoost', 'LightGBM', 'LSTM', 'Prophet', 'Regresión Lineal')
    )

    # --- Mostrar Últimos Datos ---
    st.subheader('Últimos 15 Días Registrados de la TRM')
    last_15_days = data_raw.tail(15).sort_values('fecha', ascending=False).reset_index(drop=True)
    st.dataframe(last_15_days[['fecha', 'valor']].style.format({'valor': '{:,.2f}'}), use_container_width=True)

    st.markdown("---")

    # --- Sección de Predicción ---
    st.header(f'🔮 Resultados de la Predicción con {model_choice}')
    
    if st.button('Realizar Predicción'):
        with st.spinner('Entrenando modelos y realizando predicciones... ¡Esto puede tardar un momento!'):
            # --- Predicción del Valor Completo de la TRM ---
            if model_choice == 'Prophet':
                model_trm = Prophet().fit(data_raw.rename(columns={'fecha': 'ds', 'valor': 'y'}))
                future = model_trm.make_future_dataframe(periods=1)
                forecast = model_trm.predict(future)
                predicted_trm = forecast['yhat'].iloc[-1]
            elif model_choice == 'LSTM':
                model_trm = train_lstm_model(data_raw)
                last_data_points = data_raw['valor'].values[-10:].reshape(1, 10, 1)
                predicted_trm = model_trm.predict(last_data_points)[0][0]
            else:
                model_trm, features_trm = train_ml_model(data_featured, model_choice, target='valor')
                prediction_input = data_featured[features_trm].tail(1)
                predicted_trm = model_trm.predict(prediction_input)[0]

            # --- Predicción Específica de Centavos ---
            # Usaremos un modelo robusto como LightGBM para los centavos
            model_cents, features_cents = train_ml_model(data_featured, 'LightGBM', target='centavos')
            prediction_input_cents = data_featured[features_cents].tail(1)
            predicted_cents = model_cents.predict(prediction_input_cents)[0]
            # Asegurarse que los centavos estén en el rango [0, 99]
            predicted_cents = max(0, min(99.99, predicted_cents))

            # --- Mostrar Resultados ---
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label=f"Predicción TRM (Valor Completo)",
                    value=f"${predicted_trm:,.2f}",
                    help="Valor estimado del dólar para el siguiente día hábil."
                )
            with col2:
                st.metric(
                    label="Predicción de Centavos",
                    value=f"{predicted_cents:.2f} ¢",
                    help="Estimación específica de la porción de centavos de la TRM."
                )
            
            # --- Gráfica de Tendencia con Predicción ---
            st.subheader('Tendencia Histórica y Punto de Predicción')
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=data_raw['fecha'], y=data_raw['valor'], mode='lines', name='TRM Histórica'))
            # Añadir el punto de predicción
            next_day = data_raw['fecha'].iloc[-1] + pd.Timedelta(days=1)
            fig_hist.add_trace(go.Scatter(x=[next_day], y=[predicted_trm], mode='markers', name='Predicción', 
                                          marker=dict(color='red', size=12, symbol='star')))
            fig_hist.update_layout(title='Histórico de la TRM y Valor Predicho', xaxis_title='Fecha', yaxis_title='Valor (COP)')
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # --- Sección de Desempeño del Modelo (Colapsable) ---
    with st.expander("📊 Evaluar Desempeño de los Modelos"):
        st.markdown("""
        Aquí puedes ver qué tan precisos son los modelos. Dividimos los datos históricos en un conjunto de **entrenamiento (80%)** y uno de **prueba (20%)**. 
        El modelo aprende de los datos de entrenamiento y luego intentamos predecir los datos de prueba para compararlos con los valores reales.
        """)
        
        if st.button('Calcular Desempeño'):
            with st.spinner('Evaluando el modelo...'):
                if model_choice not in ['Prophet', 'LSTM']:
                    train_data, test_data = train_test_split(data_featured, test_size=0.2, shuffle=False)
                    
                    # Evaluar modelo TRM
                    model_eval, features_eval = train_ml_model(train_data, model_choice, target='valor')
                    X_test = test_data.shift(1).dropna()[features_eval]
                    y_test = test_data.iloc[1:]['valor']
                    predictions = model_eval.predict(X_test)

                    mae = mean_absolute_error(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

                    st.subheader(f'Métricas de Desempeño para {model_choice}')
                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE (Error Absoluto Medio)", f"${mae:,.2f}", help="El error promedio en pesos.")
                    c2.metric("RMSE (Raíz del Error Cuadrático)", f"${rmse:,.2f}", help="Similar al MAE pero penaliza más los errores grandes.")
                    c3.metric("MAPE (Error Porcentual Absoluto)", f"{mape:.2f}%", help="El error promedio en términos de porcentaje.")

                    # Gráfica de comparación
                    fig_eval = go.Figure()
                    fig_eval.add_trace(go.Scatter(x=test_data.iloc[1:]['fecha'], y=y_test, mode='lines', name='Valor Real'))
                    fig_eval.add_trace(go.Scatter(x=test_data.iloc[1:]['fecha'], y=predictions, mode='lines', name='Predicción', line=dict(dash='dot')))
                    fig_eval.update_layout(title='Comparación de Predicciones vs. Valores Reales (Datos de Prueba)',
                                           xaxis_title='Fecha', yaxis_title='Valor (COP)')
                    st.plotly_chart(fig_eval, use_container_width=True)
                else:
                    st.info(f"La evaluación de desempeño detallada para {model_choice} es más compleja y se implementará en futuras versiones. ¡Pero es uno de los modelos más potentes!")

else:
    st.info('👈 Por favor, carga un archivo CSV para comenzar.')
