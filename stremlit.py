import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer

# Cargar datos
file_path = "datos/Dados_totais.csv"
df = pd.read_csv(file_path)

# --- Menú de configuración ---
st.sidebar.title("Configuración de Datos")

# Selección de columnas a eliminar
all_columns = list(df.columns)
default_irrelevant = ["key", "mode", "loudness", "liveness", "acousticness", "instrumentalness", "speechiness", "duration_ms"]
columns_to_remove = st.sidebar.multiselect("Selecciona columnas a eliminar:", all_columns, default=default_irrelevant)

# Ajuste del rango de normalización
st.sidebar.subheader("Rango de Normalización")
min_range = st.sidebar.slider("Valor mínimo", -1.0, 0.0, 0.0)
max_range = st.sidebar.slider("Valor máximo", 0.1, 1.0, 1.0)

# Opciones de visualización de gráficos
st.sidebar.subheader("Opciones de visualización")
show_histograms = st.sidebar.checkbox("Mostrar histogramas", value=True)
show_bar_chart = st.sidebar.checkbox("Mostrar comparación de columnas", value=True)

# Mostrar encabezado del dataframe
st.title("Análisis y Transformación de Datos")
st.subheader("Datos Originales")
st.write(df.head(10))

# Mostrar el número de valores únicos por columna
st.subheader("Número de valores únicos por columna")
st.write(df.nunique())

# --- Limpieza de Datos ---
# Comparación del número de columnas antes y después de la eliminación
st.subheader("Comparación del número de columnas antes y después de la limpieza")
columns_before = len(df.columns)
irrelevant_columns = ["key", "mode", "loudness", "liveness", "acousticness", "instrumentalness", "speechiness", "duration_ms"]
df_cleaned = df.drop(columns=irrelevant_columns, errors='ignore')
columns_after = len(df_cleaned.columns)

if show_bar_chart:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Antes"], [columns_before], color='red', alpha=0.7, label="Antes")
    ax.bar(["Después"], [columns_after], color='green', alpha=0.7, label="Después")
    ax.set_title("Número de columnas antes y después de la limpieza")
    ax.set_ylabel("Cantidad de columnas")
    ax.legend()
    st.pyplot(fig)

# Tabla después de la eliminación de columnas irrelevantes
st.subheader("Tabla después de la eliminación de columnas irrelevantes")
st.write(df_cleaned.head(10))

# --- Cálculo de Gender Index ---
st.subheader("Tabla para sacar el índice de género")
columnas_ = ["id", "danceability", "valence", "liveness", "instrumentalness", "loudness"]
gender_index_df = df[columnas_]
st.write(gender_index_df.head(10))

# Normalización de datos
st.subheader("Normalización de los datos")
scaler = MinMaxScaler()
columns_to_normalize = ['danceability', 'valence', 'liveness', 'instrumentalness']
gender_index_df[columns_to_normalize] = scaler.fit_transform(gender_index_df[columns_to_normalize])

# Normalizar loudness por separado para valores entre -1 y 0
loudness_scaler = MinMaxScaler(feature_range=(-1, 0))
gender_index_df['loudness'] = loudness_scaler.fit_transform(gender_index_df[['loudness']])

# Función para calcular Gender Index
def get_gender_index(danceability, valence, liveness, instrumentalness, loudness):
    danceability_index = danceability * 0.161
    valence_index = valence * 0.181
    liveness_index = liveness * 0.153
    instrumentalness_index = instrumentalness * 0.238
    loudness_index = loudness * 0.267
    return danceability_index + valence_index + liveness_index + instrumentalness_index - loudness_index

class GenderIndexCalculator:
    def __init__(self, id, danceability, valence, liveness, instrumentalness, loudness):
        self.id = id
        self.danceability = danceability
        self.valence = valence
        self.liveness = liveness
        self.instrumentalness = instrumentalness
        self.loudness = loudness
    
    def __str__(self):
        return f"GenderIndex(id={self.id}, danceability={self.danceability}, valence={self.valence}, liveness={self.liveness}, instrumentalness={self.instrumentalness}, loudness={self.loudness})"
    
    def get_gender_index(self):
        return get_gender_index(self.danceability, self.valence, self.liveness, self.instrumentalness, self.loudness)

# Calcular Gender Index
arr_gender_index = []
for i in range(gender_index_df.shape[0]):
    gender_index = GenderIndexCalculator(gender_index_df.iloc[i]['id'], 
                                       gender_index_df.iloc[i]['danceability'],
                                       gender_index_df.iloc[i]['valence'], 
                                       gender_index_df.iloc[i]['liveness'],
                                       gender_index_df.iloc[i]['instrumentalness'],
                                       gender_index_df.iloc[i]['loudness'])
    arr_gender_index.append(gender_index.get_gender_index())

gender_index_df['gender_index'] = arr_gender_index

# Normalizar Gender Index
scaler = MinMaxScaler(feature_range=(min_range, max_range))
gender_index_df['gender_index'] = scaler.fit_transform(gender_index_df[['gender_index']])

# Mostrar tabla con los valores normalizados
st.subheader("Tabla con los valores normalizados")
st.write(gender_index_df.head(10))

# Crear un DataFrame con solo 'id' y 'gender_index'
gender_index_simple = gender_index_df[['id', 'gender_index']]

# Unir los datos originales con los datos de género
st.subheader("Tabla combinada de datos originales y el índice de género")
resultado = pd.merge(df_cleaned, gender_index_simple, on='id')
st.write(resultado.head(10))

# Clase para normalizar canciones
class Cancion_normalizada: 
    def __init__ (self, valence, danceability, energy, id, popularity, tempo, genre_index):
        self.valence = valence
        self.danceability = danceability
        self.energy = energy
        self.id = id
        self.popularity = popularity
        self.tempo = tempo
        self.genre_index = genre_index
    
    def __str__(self):
        return f"Cancion(valence={self.valence}, danceability={self.danceability}, energy={self.energy}, id={self.id}, popularity={self.popularity}, tempo={self.tempo}, genre_index={self.genre_index})"
    
    def compare_song_index(self, other):
        dist = (
            0.4 * (abs(self.genre_index - other.genre_index) / 10) +
            0.1 * abs(self.valence - other.valence) +
            0.1 * abs(self.danceability - other.danceability) +
            0.1 * abs(self.energy - other.energy) +
            0.1 * (abs(self.popularity - other.popularity) / 100) +
            0.1 * (abs(self.tempo - other.tempo) / 250) 
        )
        return 1/(1+dist)

# Normalización de resultado
scaler = MinMaxScaler(feature_range=(min_range, max_range))
resultado_normalizado = resultado.copy()
resultado_normalizado = resultado_normalizado.drop(columns=['name', 'artists', 'year', 'explicit', 'artists_song'])
resultado_normalizado.set_index('id', inplace=True)
resultado_normalizado[resultado_normalizado.columns] = scaler.fit_transform(resultado_normalizado[resultado_normalizado.columns])

st.subheader("Tabla con los datos normalizados de canciones")
st.write(resultado_normalizado.head(10))
if show_histograms:
    st.subheader("Histogramas de las características normalizadas")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, col in enumerate(columns_to_normalize):
        ax = axes[i // 2, i % 2]
        sns.histplot(gender_index_df[col], bins=20, kde=True, ax=ax)
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    st.pyplot(fig)

st.write("Transformaciones aplicadas con éxito. Explora los gráficos para visualizar los cambios en los datos.")
