import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import holidays
from datetime import date, timedelta

# --- Configuración de la página ---
st.set_page_config(page_title="Predicción TRM Dólar Colombia", layout="wide")

# --- Funciones ---

@st.cache_data
def load_data(file):
    """Carga y preprocesa los datos del archivo CSV."""
    df = pd.read_csv(file)
    df['VIGENCIADESDE'] = pd.to_datetime(df['VIGENCIADESDE'], format='%d/%m/%Y')
    df['VALOR'] = df['VALOR'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    df = df.rename(columns={'VIGENCIADESDE': 'fecha', 'VALOR': 'valor'})
    df = df[['fecha', 'valor']].sort_values('fecha').reset_index(drop=True)
    df['centavos'] = (df['valor'] - np.floor(df['valor'])) * 100
    return df

def get_colombian_holidays(year):
    """Obtiene los festivos de Colombia para un año específico."""
    return holidays.Colombia(years=year)

def add_features(df):
    """Agrega características adicionales al DataFrame."""
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['dia_mes'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['anio'] = df['fecha'].dt.year
    df['festivo'] = df['fecha'].apply(lambda x: x in get_colombian_holidays(x.year))
    return df

def train_prophet_model(df):
    """Entrena un modelo Prophet."""
    model_df = df.rename(columns={'fecha': 'ds', 'valor': 'y'})
    model = Prophet()
    model.fit(model_df)
    return model

def train_ml_model(df, model_type='RandomForest'):
    """Entrena un modelo de Machine Learning."""
    df_train = df.copy()
    df_train['target'] = df_train['valor'].shift(-1)
    df_train = df_train.dropna()

    features = ['valor', 'dia_semana', 'dia_mes', 'mes', 'anio', 'festivo']
    X = df_train[features]
    y = df_train['target']

    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else: # LinearRegression
        model = LinearRegression()

    model.fit(X, y)
    return model

# --- Interfaz de Streamlit ---

st.title('📈 Predicción de la TRM del Dólar en Colombia')
st.markdown("""
Esta aplicación permite predecir el valor de la Tasa Representativa del Mercado (TRM) del dólar en Colombia.
Puedes subir tu propio histórico de datos, seleccionar un modelo de predicción y visualizar los resultados.
""")

# --- Carga de datos ---
uploaded_file = st.file_uploader("Carga tu archivo CSV con el histórico de la TRM", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    data_with_features = add_features(data.copy())

    # --- Sidebar de opciones ---
    st.sidebar.header('Opciones de Predicción')
    model_choice = st.sidebar.selectbox(
        'Elige un modelo de predicción:',
        ('Prophet', 'Random Forest', 'Regresión Lineal')
    )
    prediction_date = st.sidebar.date_input(
        "Selecciona una fecha para predecir el siguiente día hábil:",
        date.today()
    )

    # --- Mostrar últimos datos ---
    st.header('Últimos 15 días de la TRM')
    last_15_days = data_with_features.tail(15).sort_values('fecha', ascending=False)
    st.dataframe(last_15_days[['fecha', 'valor', 'festivo']].style.format({'valor': '{:,.2f}'}))

    # --- Predicción ---
    if st.sidebar.button('Realizar Predicción'):
        st.header(f'Resultados de la Predicción con {model_choice}')

        # Lógica de predicción
        if model_choice == 'Prophet':
            model = train_prophet_model(data)
            future = model.make_future_dataframe(periods=1)
            forecast = model.predict(future)
            predicted_value = forecast['yhat'].iloc[-1]
        else:
            ml_model_name = 'RandomForest' if model_choice == 'Random Forest' else 'LinearRegression'
            model = train_ml_model(data_with_features, ml_model_name)
            last_data = data_with_features.tail(1)
            features = ['valor', 'dia_semana', 'dia_mes', 'mes', 'anio', 'festivo']
            prediction_input = last_data[features]
            predicted_value = model.predict(prediction_input)[0]

        predicted_cents = (predicted_value - np.floor(predicted_value)) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=f"Predicción TRM para el siguiente día hábil",
                value=f"${predicted_value:,.2f}"
            )
        with col2:
            st.metric(
                label="Predicción de Centavos",
                value=f"{predicted_cents:.2f} centavos"
            )

    # --- Visualizaciones ---
    st.header('Visualización de Datos y Tendencias')

    # Gráfica de tendencia histórica
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=data['fecha'], y=data['valor'], mode='lines', name='TRM Histórica'))
    fig_hist.update_layout(title='Histórico de la TRM', xaxis_title='Fecha', yaxis_title='Valor (COP)')
    st.plotly_chart(fig_hist, use_container_width=True)

    # Gráfica de predicción de centavos (ejemplo simple)
    st.subheader('Análisis de los Centavos')
    fig_cents = go.Figure()
    fig_cents.add_trace(go.Histogram(x=data['centavos'], name='Distribución de Centavos'))
    fig_cents.update_layout(title='Distribución Histórica de los Centavos de la TRM',
                            xaxis_title='Centavos', yaxis_title='Frecuencia')
    st.plotly_chart(fig_cents, use_container_width=True)


    # --- Desempeño del modelo ---
    st.header('Desempeño del Modelo')
    st.markdown("""
    Para evaluar el desempeño, dividimos los datos en un conjunto de entrenamiento (80%) y de prueba (20%).
    Luego, entrenamos el modelo con los datos de entrenamiento y lo evaluamos con los datos de prueba.
    """)

    train_data, test_data = train_test_split(data_with_features, test_size=0.2, shuffle=False)

    if model_choice != 'Prophet':
        ml_model_name = 'RandomForest' if model_choice == 'Random Forest' else 'LinearRegression'
        model_eval = train_ml_model(train_data, ml_model_name)
        
        last_train_data = train_data.tail(len(test_data))
        features = ['valor', 'dia_semana', 'dia_mes', 'mes', 'anio', 'festivo']
        X_test = last_train_data[features]
        y_test = test_data['valor']
        
        predictions = model_eval.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        col1, col2 = st.columns(2)
        col1.metric("Error Absoluto Medio (MAE)", f"${mae:,.2f}")
        col2.metric("Raíz del Error Cuadrático Medio (RMSE)", f"${rmse:,.2f}")

        # Gráfica de comparación
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=test_data['fecha'], y=y_test, mode='lines', name='Valor Real'))
        fig_eval.add_trace(go.Scatter(x=test_data['fecha'], y=predictions, mode='lines', name='Predicción'))
        fig_eval.update_layout(title='Comparación de Predicciones vs. Valores Reales',
                               xaxis_title='Fecha', yaxis_title='Valor (COP)')
        st.plotly_chart(fig_eval, use_container_width=True)
    else:
        st.info("La evaluación de desempeño para Prophet se implementará en una futura versión.")


else:
    st.info('Por favor, carga un archivo CSV para comenzar.')
