# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Cargar los datos desde el archivo CSV
data_path = '../data/global_air_pollution_data.csv'
pollution_data = pd.read_csv(data_path)

# Mostrar las primeras filas del dataset
print('print data.head:')
print(pollution_data.head())
print('print pollution_data.columns:')
print(pollution_data.columns)

# eliminar espacios y caracteres de tabulación
pollution_data.rename(columns=lambda x: x.strip(), inplace=True)
print('===========================')


# --- Preprocesamiento ---
# Seleccionar columnas relevantes para el análisis numérico
numerical_columns = ['aqi_value', 'co_aqi_value', 'ozone_aqi_value', 'no2_aqi_value', 'pm2.5_aqi_value']
pollution_numeric = pollution_data[numerical_columns]

# Manejar valores nulos (si los hay)
pollution_numeric = pollution_numeric.dropna()

# Escalar los datos para K-Means
scaler = StandardScaler()
pollution_scaled = scaler.fit_transform(pollution_numeric)

# --- Aplicar K-Means ---
# Determinar el número óptimo de clusters usando el método del codo
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pollution_scaled)
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
clusters = kmeans.fit_predict(pollution_scaled)

# Agregar las etiquetas de cluster al DataFrame original
pollution_data['Cluster'] = clusters

# --- Visualización ---
# Graficar los clusters usando dos dimensiones principales
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=pollution_data['aqi_value'],
    y=pollution_data['pm2.5_aqi_value'],
    hue=pollution_data['Cluster'],
    palette='viridis',
    s=100
)
plt.title('Visualización de Clusters por Calidad del Aire')
plt.xlabel('AQI Value')
plt.ylabel('PM2.5 AQI Value')
plt.legend(title='Cluster')
plt.show()

# Calcular y mostrar el puntaje de silueta para evaluar la calidad de los clusters
silhouette_avg = silhouette_score(pollution_scaled, clusters)
print(f"Puntaje de silueta para {optimal_clusters} clusters: {silhouette_avg}")



# Graficar los clusters con los centroides
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=pollution_data['aqi_value'],
    y=pollution_data['pm2.5_aqi_value'],
    hue=pollution_data['Cluster'],
    palette='viridis',
    s=100
)
# Añadir centroides al gráfico
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0], centroids[:, 4],  # Coordenadas X e Y de los centroides
    c='black', s=200, marker='X', label='Centroides'
)
plt.title('Clusters con Centroides')
plt.xlabel('AQI Value')
plt.ylabel('PM2.5 AQI Value')
plt.legend(title='Cluster')
plt.show()





# Crear un pairplot para analizar relaciones entre variables
pairplot_data = pollution_data[['aqi_value', 'pm2.5_aqi_value', 'ozone_aqi_value', 'no2_aqi_value', 'Cluster']].copy()
pairplot_data['Cluster'] = pairplot_data['Cluster'].astype(int).astype(str)

sns.pairplot(
    pairplot_data,
    hue='Cluster',  # Diferenciar por cluster
    palette='viridis',
    diag_kind='kde',  # Añadir distribuciones en la diagonal
    markers=['o', 's', 'D', '^']  # Diferentes marcadores para cada cluster
)
plt.suptitle('Pairplot de Variables por Cluster', y=1.02)
plt.show()
