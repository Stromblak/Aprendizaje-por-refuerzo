import csv
import matplotlib.pyplot as plt
import os





dir = "PandaReach-CSV"
files = os.listdir(dir)


colors = ['C0', 'C1', 'C2', 'C3', 'C6', 'C7', 'C9']
c = 0

for f in files: 
    columna_x = []
    columna_y = [] 

    with open(dir + "/" + f, 'r') as archivo:
        lector_csv = csv.reader(archivo)
        next(lector_csv)

        for fila in lector_csv:
            columna_x.append(int(fila[1]))
            columna_y.append(float(fila[2]))
    
    
    plt.plot(columna_x, columna_y, label = f, color = colors[c])
    c += 1




# Crea el gráfico

# Añade etiquetas y título
plt.xlabel('Steps')  # Corregí la ortografía de 'Iteracion' a 'Iteración'
plt.ylabel('Recompensa')  # Corregí la ortografía de 'Recomensa' a 'Recompensa'
plt.title('Recompensas en Panda Reach')
plt.xlim(0, 120_000)
plt.legend(loc="upper right")

# Muestra el gráfico
plt.savefig("graf.png")