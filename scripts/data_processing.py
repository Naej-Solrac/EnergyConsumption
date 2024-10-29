import pandas as pd
import matplotlib.pyplot as plt

def load_and_process_data(filepath):
    # Cargar el dataset
    data = pd.read_csv(filepath)

    # Convertir la columna de fecha y hora a datetime
    data['Datetime'] = pd.to_datetime(data['Datetime'])

    # Configurar 'Datetime' como índice para análisis de series de tiempo
    data.set_index('Datetime', inplace=True)

    # Eliminar valores nulos, si existen
    data = data.dropna()

    # Visualización inicial del consumo energético
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['AEP_MW'], color='blue')
    plt.title('Consumo Energético por Hora')
    plt.xlabel('Fecha')
    plt.ylabel('Consumo (MW)')
    plt.show()

    return data

# Ejecuta la función con el archivo de datos como argumento
if __name__ == "__main__":
    data = load_and_process_data('../data/AEP_hourly.csv')
