import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
import heapq

# Cargar datos
file_path = "datos/Dados_totais.csv"
df = pd.read_csv(file_path)

# --- Menú de configuración ---
st.sidebar.title("Configuración de Datos")

st.sidebar.subheader("Seleccionar cancion")
selected_song = st.sidebar.selectbox("Selecciona una cancion:", df['artists_song'].unique())

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
genre_index_df = df[columnas_]
st.write(genre_index_df.head(10))

# Normalización de datos
st.subheader("Normalización de los datos")
scaler = MinMaxScaler()
columns_to_normalize = ['danceability', 'valence', 'liveness', 'instrumentalness']
genre_index_df[columns_to_normalize] = scaler.fit_transform(genre_index_df[columns_to_normalize])

# Normalizar loudness por separado para valores entre -1 y 0
loudness_scaler = MinMaxScaler(feature_range=(-1, 0))
genre_index_df['loudness'] = loudness_scaler.fit_transform(genre_index_df[['loudness']])

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
arr_genre_index = []
for i in range(genre_index_df.shape[0]):
    genre_index = GenderIndexCalculator(genre_index_df.iloc[i]['id'], 
                                       genre_index_df.iloc[i]['danceability'],
                                       genre_index_df.iloc[i]['valence'], 
                                       genre_index_df.iloc[i]['liveness'],
                                       genre_index_df.iloc[i]['instrumentalness'],
                                       genre_index_df.iloc[i]['loudness'])
    arr_genre_index.append(genre_index.get_gender_index())

genre_index_df['genre_index'] = arr_genre_index

# Normalizar Gender Index
scaler = MinMaxScaler(feature_range=(min_range, max_range))
genre_index_df['genre_index'] = scaler.fit_transform(genre_index_df[['genre_index']])

# Mostrar tabla con los valores normalizados
st.subheader("Tabla con los valores normalizados")
st.write(genre_index_df.head(10))

# Crear un DataFrame con solo 'id' y 'gender_index'
genre_index_simple = genre_index_df[['id', 'genre_index']]

# Unir los datos originales con los datos de género
st.subheader("Tabla combinada de datos originales y el índice de género")
resultado = pd.merge(df_cleaned, genre_index_simple, on='id')
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
resultado_normalizado = resultado_normalizado.drop(columns=['name', 'artists', 'year', 'explicit', 'id'])
resultado_normalizado.set_index('artists_song', inplace=True)
resultado_normalizado[resultado_normalizado.columns] = scaler.fit_transform(resultado_normalizado[resultado_normalizado.columns])

st.subheader("Tabla con los datos normalizados de canciones")
st.write(resultado_normalizado.head(10))
if show_histograms:
    st.subheader("Histogramas de las características normalizadas")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, col in enumerate(columns_to_normalize):
        ax = axes[i // 2, i % 2]
        sns.histplot(genre_index_df[col], bins=20, kde=True, ax=ax)
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    st.pyplot(fig)

def calcular_similitud(resultado_normalizado, artists_song):
    if artists_song not in resultado_normalizado.index:
        raise KeyError(f"cancion '{artists_song}' no encontrado en el DataFrame.")
    target_row = resultado_normalizado.loc[artists_song]

    target = Cancion_normalizada(
        target_row['valence'],
        target_row['danceability'],
        target_row['energy'],
        artists_song,
        target_row['popularity'],
        target_row['tempo'],
        target_row['genre_index']
    )

    similitudes = {}

    for other_artists_song, row in resultado_normalizado.iterrows():
        other = Cancion_normalizada(
            row['valence'],
            row['danceability'],
            row['energy'],
            other_artists_song,
            row['popularity'],
            row['tempo'],
            row['genre_index']
        )
        similitudes[other_artists_song] = target.compare_song_index(other)

    return similitudes

similitudes = calcular_similitud(resultado_normalizado, selected_song)
st.header(f"Similitudes respecto a '{selected_song}'")


# Obtener los 5 elementos con mayor similitud sin ordenar todo el diccionario
top5_similitudes = heapq.nlargest(6, similitudes.items(), key=lambda item: item[1])

# Crear un DataFrame con las canciones y sus índices de similitud
top5_df = pd.DataFrame(top5_similitudes, columns=["Canción", "Similitud"])
st.write("Tabla con el índice de similitud:")
st.table(top5_df)

# Separar las etiquetas (canciones) y los valores de similitud para graficar
top5_canciones, top5_valores = zip(*top5_similitudes)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(top5_canciones, top5_valores)
ax.set_ylim(0, 1)
ax.set_ylabel('Similitud')
ax.set_xlabel('Canción')
ax.set_title(f"Top 5 canciones más similares a '{selected_song}'")

# Escapar '$' en las etiquetas si es necesario
escaped_labels = [cancion.replace('$', r'\$') for cancion in top5_canciones]
ax.set_xticklabels(escaped_labels, rotation=90, fontsize=8)

plt.tight_layout()
st.pyplot(fig)
st.write("Transformaciones aplicadas con éxito. Explora los gráficos para visualizar los cambios en los datos.")
