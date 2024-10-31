# scripts/data_processing.py

# Importación de bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from prophet import Prophet

# Paso 1: Especificar la ruta del archivo de datos
# Esta es la ubicación del archivo de datos que contiene el consumo energético por hora.
filepath = '../data/AEP_hourly.csv'
print(f"Cargando datos desde: {filepath}")

# Paso 2: Cargar el dataset
# Carga el conjunto de datos en un DataFrame de Pandas y muestra las primeras filas para ver su estructura.
data = pd.read_csv(filepath)
print("Primeras filas del dataset:")
print(data.head())

# Paso 3: Convertir la columna de fecha y hora a datetime
# Convierte la columna 'Datetime' a tipo datetime para facilitar el análisis de series temporales.
print("\nConvirtiendo la columna 'Datetime' a tipo datetime...")
data['Datetime'] = pd.to_datetime(data['Datetime'])
print("Tipos de datos después de la conversión:")
print(data.dtypes)

# Paso 4: Configurar 'Datetime' como índice para análisis de series de tiempo
# Al usar 'Datetime' como índice, se facilita el uso de herramientas de series temporales en Pandas.
print("\nConfigurando 'Datetime' como índice...")
data.set_index('Datetime', inplace=True)
print("Índice actual del dataset:")
print(data.index)

# Paso 5: Eliminar valores nulos, si existen
# Elimina cualquier valor nulo en el conjunto de datos para evitar problemas en el análisis.
print("\nEliminando valores nulos...")
data = data.dropna()
print(f"Total de filas después de eliminar nulos: {len(data)}")

# Paso 6: Visualización inicial del consumo energético
# Grafica el consumo energético por hora para observar el comportamiento general de la serie temporal.
print("\nGenerando la gráfica del consumo energético...")
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['AEP_MW'], color='blue')
plt.title('Consumo Energético por Hora')
plt.xlabel('Fecha')
plt.ylabel('Consumo (MW)')
plt.show()

# Paso 7: Aplicar regresión lineal para identificar la tendencia
# Utilizamos un modelo de regresión lineal para identificar la tendencia a largo plazo del consumo energético.
print("\nAplicando regresión lineal para analizar la tendencia...")
data['Time'] = np.arange(len(data))  # Crear una variable de tiempo en formato numérico
model = LinearRegression()
model.fit(data[['Time']], data['AEP_MW'])  # Ajustar el modelo de regresión
data['Trend'] = model.predict(data[['Time']])  # Obtener la predicción de tendencia

# Grafica el consumo energético junto con la tendencia lineal
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['AEP_MW'], label='Consumo Energético')
plt.plot(data.index, data['Trend'], label='Tendencia Lineal', color='red')
plt.xlabel('Fecha')
plt.ylabel('Consumo (MW)')
plt.legend()
plt.title('Consumo Energético con Tendencia Lineal')
plt.show()

# Paso 8: Proyección a futuro usando Prophet
# Prophet se utiliza para hacer predicciones basadas en series temporales, teniendo en cuenta la estacionalidad y la tendencia.
print("\nRealizando proyección a futuro usando Prophet hasta el año 2030...")
data_prophet = data.reset_index()[['Datetime', 'AEP_MW']]
data_prophet.columns = ['ds', 'y']  # Prophet requiere las columnas 'ds' (fecha) y 'y' (valor)

# Creación y ajuste del modelo Prophet
prophet_model = Prophet()
prophet_model.fit(data_prophet)

# Configuración de la proyección hasta el año 2030 usando frecuencia mensual de fin de mes ('ME')
years_to_project = 2030 - data.index[-1].year
future = prophet_model.make_future_dataframe(periods=years_to_project * 12, freq='ME')

# Realizar predicciones con el modelo Prophet
forecast = prophet_model.predict(future)

# Graficar la proyección
prophet_model.plot(forecast)
plt.title('Proyección del Consumo Energético hasta 2030')
plt.xlabel('Fecha')
plt.ylabel('Consumo (MW)')
plt.show()

print("Proyección completada.")
