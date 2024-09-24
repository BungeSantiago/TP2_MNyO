import pandas as pd
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

data = pd.read_csv('city_temperature.csv')

def filter_country(country:str):
    return data[data['Country'] == country]

def filter_year(country, year):
    return country[country['Year'] == year]

def filter_month(country, month):
    return country[country['Month'] == month]

def foward_diff(country):
    return np.diff(country['AvgTemperature'])/ np.diff(np.arrange(len(country['AvgTemperature'])))

def rate_of_change_normalized(series):
    """
    Calcula la tasa de cambio normalizada de una serie temporal.
    """
    diffs = np.diff(series)
    # Normalizar cada cambio por el valor absoluto del punto anterior para independizar de la magnitud
    normalized_changes = diffs / np.abs(series[:-1])
    return normalized_changes

def cosine_similarity(series1, series2):
    """
    Calcula la medida de similaridad de coseno entre dos series de tasas de cambio normalizadas.
    """
    dot_product = np.dot(series1, series2)
    norm1 = np.linalg.norm(series1)
    norm2 = np.linalg.norm(series2)
    return dot_product / (norm1 * norm2)

def eucledean_distance(series1, series2):
    """
    Calcula la distancia euclídea entre dos series de tasas de cambio normalizadas.
    """
    return np.linalg.norm(series1 - series2)


Argentina = filter_country('Argentina')
Argentina_1995 = filter_year(Argentina, 1995)
daysarg = np.arange(1, len(Argentina_1995) + 1)
interpolArg = spi.interp1d(daysarg, Argentina_1995['AvgTemperature'], kind='linear')
Austria = filter_country('Austria')
Austria_1995 = filter_year(Austria, 1995)
daysaut = np.arange(1, len(Austria_1995) + 1)
interpolAut = spi.interp1d(daysaut, Austria_1995['AvgTemperature'], kind='linear')

normArg = rate_of_change_normalized(Argentina_1995['AvgTemperature'])
normAut = rate_of_change_normalized(Austria_1995['AvgTemperature'])
similarity = cosine_similarity(normArg, normAut)
eulidean = eucledean_distance(normArg, normAut)

print('Similarity:', similarity)
print('Eucledean:', eulidean)
plt.plot(daysarg, interpolArg(daysarg), 'r-', label='Interpolated')
plt.plot(daysarg, interpolAut(daysarg), 'b-', label='Interpolated')
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.title('Argentina vs Austria')
plt.grid()
plt.show()
