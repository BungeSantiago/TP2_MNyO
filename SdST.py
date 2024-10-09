import pandas as pd
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

data = pd.read_csv('city_temperature.csv')

def filter_country(country:str):
    return data[data['Country'] == country]

def filter_year(country, year):
    return country[country['Year'] == year].copy()  # Copia para no modificar el original

def filter_month(country, month):
    return country[country['Month'] == month]

def clean_info(country):
    country = country.copy()  # Aseguramos que no se modifique la vista original
    country.drop(['State', 'Region'], axis=1, inplace=True)

def days_to_weeks(country):
    weeks_temperatures = []
    week_temperature = []
    for i in range(len(country['AvgTemperature'])):
        week_temperature.append(float(country['AvgTemperature'].iloc[i]))
        if (i + 1) % 7 == 0:
            weeks_temperatures.append(np.average(week_temperature))
            week_temperature = []
    if len(week_temperature) > 0:
        weeks_temperatures.append(np.average(week_temperature))
    return np.array(weeks_temperatures)

def zero_derivative_weeks(country):
    derivatives = np.diff(country['AvgTemperature'])
    zero_derivative_weeks = [i + 1 for i, deriv in enumerate(derivatives) if deriv == 0]
    return zero_derivative_weeks

def cambio_de_signo(series):
  derivada = np.diff(series)
  semanas_extremos = []
  for i in range(len(derivada) - 1):
      if derivada[i] * derivada[i + 1] < 0:  # Cambio de signo
          semana = i + 2 
          if derivada[i] > 0 and derivada[i + 1] < 0:
              tipo = 'subio'
          elif derivada[i] < 0 and derivada[i + 1] > 0:
              tipo = 'bajo'
          semanas_extremos.append((semana, tipo))
  return semanas_extremos

def analisis(series):
  derivada = np.diff(series)
  semanas_extremos = []
  for i in range(len(derivada) - 1):
      if derivada[i] * derivada[i + 1] < 0:  # Cambio de signo
          semana = i + 2 
          if derivada[i] > 0 and derivada[i + 1] < 0:
              tipo = 'subio'
          elif derivada[i] < 0 and derivada[i + 1] > 0:
              tipo = 'bajo'
          semanas_extremos.append((semana, tipo))
      else:
        semana = i + 2
        if derivada[i] > 0:
            tipo = 'subio'
        else:
            tipo = 'bajo'
        semanas_extremos.append((semana, tipo))
  return semanas_extremos

def similaritud(country1, country2):
    if len(country1) == len(country2):
      equal_weeks = 0
      for i in range(len(country1)):
        if country1[i][1] == country2[i][1]:
          equal_weeks += 1
      return equal_weeks / len(country1)

def foward_diff(country):
    return np.diff(country['AvgTemperature']) / np.diff(np.arange(len(country['AvgTemperature'])))

def rate_of_change_normalized(series):
    diffs = np.diff(series)
    normalized_changes = diffs / np.abs(series[:-1])
    return normalized_changes

def cosine_similarity(series1, series2):
    dot_product = np.dot(series1, series2)
    norm1 = np.linalg.norm(series1)
    norm2 = np.linalg.norm(series2)
    return dot_product / (norm1 * norm2)

def euclidean_distance(series1, series2):
    return np.linalg.norm(series1 - series2)

# Obtener la información relevante de Argentina
Argentina = filter_country('Argentina')
Argentina_1995 = filter_year(Argentina, 1995)
daysarg = np.arange(1, len(Argentina_1995) + 1)
clean_info(Argentina_1995)
Argentina_1995 = Argentina_1995.dropna(subset=['AvgTemperature'])  # Eliminar valores nulos
temperature_in_weeks_arg = np.array(days_to_weeks(Argentina_1995))
interpolArg = spi.interp1d(daysarg, Argentina_1995['AvgTemperature'], kind='linear')

# Obtener la información relevante de Austria
Austria = filter_country('Austria')
Austria_1995 = filter_year(Austria, 1995)
daysaut = np.arange(1, len(Austria_1995) + 1)
clean_info(Austria_1995)
Austria_1995 = Austria_1995.dropna(subset=['AvgTemperature'])  # Eliminar valores nulos
temperature_in_weeks_aut = np.array(days_to_weeks(Austria_1995))
interpolAut = spi.interp1d(daysaut, Austria_1995['AvgTemperature'], kind='linear')

# Comparación por días
days_normArg = rate_of_change_normalized(Argentina_1995['AvgTemperature'])
days_normAut = rate_of_change_normalized(Austria_1995['AvgTemperature'])
days_similarity = cosine_similarity(days_normArg, days_normAut)
days_euclidean = euclidean_distance(days_normArg, days_normAut)
#print('Similarity per days:', days_similarity)
#print('Euclidean per days:', days_euclidean)

# Comparación por semanas
weeks_normArg = rate_of_change_normalized(temperature_in_weeks_arg)
weeks_normAut = rate_of_change_normalized(temperature_in_weeks_aut)
weeks_similarity = cosine_similarity(weeks_normArg, weeks_normAut)
weeks_euclidean = euclidean_distance(weeks_normArg, weeks_normAut)
#print('Similarity per weeks:', weeks_similarity)
#print('Euclidean per weeks:', weeks_euclidean)


#semanas con derivada = 0
a= cambio_de_signo(temperature_in_weeks_arg)
print('Weeks with zero derivative:', a)
b= cambio_de_signo(temperature_in_weeks_aut)
print('Weeks with zero derivative:', b)

#semanas analisadas
aa = analisis(temperature_in_weeks_arg)
print('weeks analised:', aa)
bb = analisis(temperature_in_weeks_aut)
print('weeks analised:', bb)
print('Similaridad:', similaritud(aa, bb))

# Gráfico de temperaturas promedio semanales
plt.plot(np.arange(1, len(temperature_in_weeks_arg) + 1), temperature_in_weeks_arg, 'r-', label='Argentina')
plt.plot(np.arange(1, len(temperature_in_weeks_aut) + 1), temperature_in_weeks_aut, 'b-', label='Austria')
plt.xlabel('Weeks')
plt.ylabel('Temperature')
plt.legend()
plt.grid()
plt.show()

# Gráfico de interpolación por días
plt.plot(daysarg, interpolArg(daysarg), 'r-', label='Argentina Interpolated')
plt.plot(daysaut, interpolAut(daysaut), 'b-', label='Austria Interpolated')
plt.xlabel('Days')
plt.ylabel('Temperature')
plt.legend()
plt.grid()
plt.show()