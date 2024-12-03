# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Cargar el dataset
dataDeathRate = pd.read_csv('../data/death_rate_of_countries_and_its_causes.csv')

# Normalizar nombres de columnas
dataDeathRate.columns = dataDeathRate.columns.str.strip().str.replace(' ', '_').str.replace('.', '').str.lower()

# --- Análisis inicial ---
# Filtrar columnas relevantes (por ejemplo, relacionadas con condiciones ambientales y salud pública)
variables = [
    "outdoor_air_pollution",
    "high_systolic_blood_pressure",
    "diet_high_in_sodium",
    "diet_low_in_whole_grains",
    "alochol_use",
    "smoking",
    "high_body_mass_index",
    "unsafe_sanitation",
    "unsafe_water_source",
    "child_wasting"
]

# Inspeccionar datos seleccionados
print(dataDeathRate[variables].describe().T.round(2))

# --- Limpieza de datos ---
# Eliminar duplicados y manejar valores nulos
dataDeathRate = dataDeathRate.drop_duplicates()
dataDeathRate.dropna(subset=variables, inplace=True)

# --- Escalamiento ---
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(dataDeathRate[variables])

# --- Determinar el número óptimo de clusters ---
inertia = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))

# Graficar el método del codo y puntajes de silueta
plt.figure(figsize=(12, 6))

# Método del codo
plt.subplot(1, 2, 1)
plt.plot(K, inertia, marker='o')
plt.title("Método del codo")
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")

# Puntaje de silueta
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, marker='o', color='orange')
plt.title("Puntaje de silueta")
plt.xlabel("Número de clusters")
plt.ylabel("Silhouette Score")

plt.tight_layout()
plt.show()


# --- K-Means Clustering ---
# Determinar el número óptimo de clusters usando 4
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Agregar los clusters al dataset original
dataDeathRate["Cluster"] = clusters

# --- Visualización interactiva con Plotly (scatter matrix reducido) ---
selected_variables = [
    "outdoor_air_pollution",
    "high_systolic_blood_pressure",
    "smoking",
    "high_body_mass_index"
]

fig = px.scatter_matrix(
    dataDeathRate,
    dimensions=selected_variables,
    color="Cluster",
    title="Clusters basados en factores de mortalidad (variables seleccionadas)",
    labels={col: col.replace("_", " ").capitalize() for col in selected_variables},
    hover_name="entity"
)
fig.show()

# --- Centroides y análisis ---
centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=variables)
print("\nCentroides por cluster:")
print(centroids)

# Visualizar centroides
centroids["Cluster"] = range(optimal_clusters)
fig_centroids = px.scatter(
    centroids,
    x="high_systolic_blood_pressure",
    y="outdoor_air_pollution",
    size="high_body_mass_index",
    color="Cluster",
    title="Centroides de clusters",
    labels={"high_systolic_blood_pressure": "High Systolic Blood Pressure", "outdoor_air_pollution": "Outdoor Air Pollution"}
)
fig_centroids.show()


fig_bubble_interactive = px.scatter(
    dataDeathRate,
    x="high_systolic_blood_pressure",
    y="outdoor_air_pollution",
    size="high_body_mass_index",
    color="Cluster",
    hover_name="entity",  # Aquí mostrará el nombre del país
    hover_data=["diet_high_in_sodium", "child_wasting", "unsafe_water_source"],
    title="Relación entre variables clave con países individuales",
    labels={
        "high_systolic_blood_pressure": "High Systolic Blood Pressure",
        "outdoor_air_pollution": "Outdoor Air Pollution",
        "high_body_mass_index": "High Body Mass Index"
    },
    size_max=50
)
fig_bubble_interactive.show()



# --- Gráfico de burbujas interactivo con países ---
fig_bubble_interactive = px.scatter(
    dataDeathRate,
    x="outdoor_air_pollution",
    y="high_systolic_blood_pressure",
    size="high_body_mass_index",
    color="Cluster",
    hover_name="entity",
    hover_data={
        "year": True,
        "smoking": True,
        "unsafe_sanitation": True,
    },
    title="Gráfico de burbujas interactivo: Factores de mortalidad por cluster",
    labels={
        "outdoor_air_pollution": "Outdoor Air Pollution",
        "high_systolic_blood_pressure": "High Systolic Blood Pressure",
        "high_body_mass_index": "High Body Mass Index"
    },
    size_max=50
)
fig_bubble_interactive.show()

# --- Relación entre consumo de alcohol y obesidad ---
bubble_chart_alcohol_obesity = px.scatter(
    dataDeathRate,
    x="alochol_use",
    y="high_body_mass_index",
    color="Cluster",
    size="high_body_mass_index",
    hover_name="entity",
    title="Relación entre Consumo de Alcohol y Obesidad (High Body Mass Index) por Cluster",
    labels={
        "alochol_use": "Alcohol Use",
        "high_body_mass_index": "High Body Mass Index"
    },
    color_continuous_scale="Viridis"
)
bubble_chart_alcohol_obesity.show()

# --- Relación entre agua insegura y desnutrición infantil ---
fig_unsafe_water_child_wasting = px.scatter(
    dataDeathRate,
    x="unsafe_water_source",
    y="child_wasting",
    color="Cluster",
    size="child_wasting",
    hover_name="entity",
    hover_data=["unsafe_water_source", "child_wasting"],
    title="Relación entre Unsafe Water y Child Wasting",
    labels={"unsafe_water_source": "Unsafe Water Source", "child_wasting": "Child Wasting"}
)
fig_unsafe_water_child_wasting.show()



# Generar un gráfico de dispersión para analizar la relación entre sodio y presión arterial alta
fig_sodium_pressure = px.scatter(
    dataDeathRate,
    x="diet_high_in_sodium",
    y="high_systolic_blood_pressure",
    color="Cluster",
    size="high_body_mass_index",  # Tamaño de las burbujas según índice de masa corporal
    hover_name="entity",
    hover_data=["diet_high_in_sodium", "high_systolic_blood_pressure", "high_body_mass_index"],
    title="Relación entre Dieta Alta en Sodio y Presión Arterial Alta",
    labels={
        "diet_high_in_sodium": "Dieta Alta en Sodio",
        "high_systolic_blood_pressure": "Presión Arterial Sistólica Alta",
        "high_body_mass_index": "Índice de Masa Corporal"
    }
)

fig_sodium_pressure.show()