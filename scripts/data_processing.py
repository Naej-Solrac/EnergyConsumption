# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Cargar los datos desde el archivo CSV
data_path = '../data/death_rate_of_countries_and_its_causes.csv'  # Reemplaza con la ruta correcta
data = pd.read_csv(data_path)

# Mostrar las primeras filas del dataset
print("Primeras filas del dataset:")
print(data.head())
print("\nColumnas del dataset:")
print(data.columns)

# --- Limpieza de nombres de columnas ---
# Eliminar espacios y corregir errores en nombres de columnas
data.columns = data.columns.str.strip()  # Elimina espacios al inicio o final
data.rename(columns={"Alochol use": "Alcohol use"}, inplace=True)  # Corrige error tipográfico

# Revisar columnas limpias
print("\nColumnas disponibles después de la limpieza:")
print(data.columns)

# --- Exploración ---
# Identificar columnas numéricas
numerical_columns = data.select_dtypes(include=np.number).columns
print("\nColumnas numéricas:")
print(numerical_columns)

# Seleccionar al menos 8 variables numéricas relevantes
selected_columns = [
    "Outdoor air pollution",
    "High systolic blood pressure",
    "Diet high in sodium",
    "Diet low in whole grains",
    "Alcohol use",
    "Smoking",
    "High body mass index",
    "Unsafe sanitation"
]
print("\nVariables seleccionadas para el análisis:")
print(selected_columns)

# Verificar si todas las columnas seleccionadas están en el DataFrame
missing_columns = [col for col in selected_columns if col not in data.columns]
if missing_columns:
    raise KeyError(f"Las siguientes columnas faltan en el DataFrame: {missing_columns}")

# Crear un DataFrame con las columnas seleccionadas
data_numeric = data[selected_columns]

# --- Preprocesamiento ---
# Manejar valores nulos (si los hay)
data_numeric = data_numeric.dropna()

# Escalar los datos para K-Means
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# --- Aplicar K-Means ---
# Determinar el número óptimo de clusters usando el método del codo
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo para determinar el número óptimo de clusters')
plt.show()

# Entrenar el modelo con el número óptimo de clusters (por ejemplo, 4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Agregar las etiquetas de cluster al DataFrame original
data["Cluster"] = clusters

# --- Visualización ---
# Graficar los clusters usando dos dimensiones principales
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=data[selected_columns[0]],
    y=data[selected_columns[1]],
    hue=data["Cluster"],
    palette='viridis',
    s=100
)
plt.title('Visualización de Clusters')
plt.xlabel(selected_columns[0])
plt.ylabel(selected_columns[1])
plt.legend(title='Cluster')
plt.show()

# Calcular y mostrar el puntaje de silueta para evaluar la calidad de los clusters
silhouette_avg = silhouette_score(data_scaled, clusters)
print(f"Puntaje de silueta para {optimal_clusters} clusters: {silhouette_avg}")

# Crear un pairplot para analizar relaciones entre variables
pairplot_data = data[selected_columns + ["Cluster"]].copy()
pairplot_data["Cluster"] = pairplot_data["Cluster"].astype(int).astype(str)

sns.pairplot(
    pairplot_data,
    hue="Cluster",
    palette="viridis",
    diag_kind="kde",
    markers=["o", "s", "D", "^"]
)
plt.suptitle("Pairplot de Variables por Cluster", y=1.02)
plt.show()
