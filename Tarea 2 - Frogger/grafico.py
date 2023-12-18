import csv
import matplotlib.pyplot as plt

# Lee el archivo CSV y extrae la tercera columna (índice 2)
archivo_csv = '1702155263.csv'  # Reemplaza con el nombre de tu archivo CSV
columna_y = []  # Cambié el nombre a 'columna_y' para que refleje mejor que es la columna de las ordenadas

with open(archivo_csv, 'r') as archivo:
    lector_csv = csv.reader(archivo)
    # Omite la primera fila que contiene los encabezados
    next(lector_csv)
    for fila in lector_csv:
        # Suponiendo que la tercera columna contiene datos numéricos (índice 2)
        valor = float(fila[2])
        columna_y.append(valor)

# Crea el gráfico
plt.plot(columna_y)

# Añade etiquetas y título
plt.xlabel('Iteración')  # Corregí la ortografía de 'Iteracion' a 'Iteración'
plt.ylabel('Recompensa')  # Corregí la ortografía de 'Recomensa' a 'Recompensa'
plt.title('Recompensas Greedy')

# Muestra el gráfico
plt.savefig("graf.png")