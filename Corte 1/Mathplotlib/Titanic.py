import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np

# Cargar el archivo CSV
titanicData = pd.read_csv("./Mathplotlib/titanic_upgrade.csv")

# Extraer las columnas Cabin_BowDistance_m (X), Cabin_DeckHeight_m (Y) y Pclass
# Eliminar valores NaN en las columnas de cabina
data = titanicData[['Cabin_BowDistance_m', 'Cabin_DeckHeight_m', 'Pclass']].dropna(subset=['Cabin_BowDistance_m', 'Cabin_DeckHeight_m'])

# ============= GRÁFICA 1: Datos originales por clase =============
fig1 = plt.figure(figsize=(12, 10))
ax1 = fig1.add_subplot(111)

colores_originales = {1: 'blue', 2: 'orange', 3: 'green'}
labels_originales = {1: 'Clase 1', 2: 'Clase 2', 3: 'Clase 3'}

for clase in [1, 2, 3]:
    clase_data = data[data['Pclass'] == clase]
    ax1.scatter(clase_data['Cabin_BowDistance_m'], 
                clase_data['Cabin_DeckHeight_m'], 
                alpha=0.6, s=50, 
                color=colores_originales[clase], 
                label=labels_originales[clase])

ax1.set_xlabel('Cabin Bow Distance (m)', fontsize=12)
ax1.set_ylabel('Cabin Deck Height (m)', fontsize=12)
ax1.set_title('Datos Originales por Clase', fontweight='bold', fontsize=16)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Puntos de referencia
puntos_ref = {
    'Punto 1': (113, 16),
    'Punto 2': (221, 13),
    'Punto 3': (61, 10)
}

# Colores según la clase predicha
colores_clase_predicha = {1: 'blue', 2: 'orange', 3: 'green'}

# Función para calcular distancia euclidiana
def calcular_distancia(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Valores de k a probar
valores_k = [3, 5, 7, 13]

# ============= GRÁFICAS 2-5: KNN con diferentes valores de k =============
for idx_k, k in enumerate(valores_k):
    # Crear una figura nueva para cada valor de k
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    # Graficar todos los puntos en gris claro
    ax.scatter(data['Cabin_BowDistance_m'], 
               data['Cabin_DeckHeight_m'], 
               alpha=0.2, s=20, color='black')
    
    print(f"\n{'='*50}")
    print(f"KNN con k = {k}")
    print('='*50)
    
    for nombre, (px, py) in puntos_ref.items():
        # Calcular distancias de todos los puntos al punto de referencia
        data['distancia'] = data.apply(
            lambda row: calcular_distancia(px, py, 
                                          row['Cabin_BowDistance_m'], 
                                          row['Cabin_DeckHeight_m']), 
            axis=1
        )
        
        # Obtener los k vecinos más cercanos
        vecinos = data.nsmallest(k, 'distancia')
        
        # Clasificación por mayoría de votos (KNN)
        clases_vecinos = vecinos['Pclass'].value_counts()
        clase_predicha = clases_vecinos.idxmax()
        cantidad_mayoria = clases_vecinos.max()
        
        # Imprimir resultado en consola
        print(f"\n{nombre} ({px}, {py}):")
        print(f"  Vecinos: {list(vecinos['Pclass'].values)}")
        print(f"  Clasificación: Clase {clase_predicha} ({cantidad_mayoria}/{k} vecinos)")
        
        # Color del punto según la clase predicha
        color_punto = colores_clase_predicha[clase_predicha]
        
        # Graficar el punto de referencia con el color de su clase predicha
        ax.scatter(px, py, s=300, marker='*', 
                  color=color_punto, 
                  edgecolors='black', linewidths=2.5,
                  zorder=5)
        
        # Agregar etiqueta al punto
        ax.text(px, py + 1.5, nombre, 
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Graficar los vecinos más cercanos
        ax.scatter(vecinos['Cabin_BowDistance_m'], 
                  vecinos['Cabin_DeckHeight_m'], 
                  s=80, color=color_punto, 
                  alpha=0.5, edgecolors='black', linewidths=1,
                  zorder=3)
        
        # Dibujar líneas a los vecinos
        for _, vecino in vecinos.iterrows():
            ax.plot([px, vecino['Cabin_BowDistance_m']], 
                   [py, vecino['Cabin_DeckHeight_m']], 
                   color=color_punto, alpha=0.3, linestyle='--', linewidth=1)
    
    ax.set_xlabel('Cabin Bow Distance (m)', fontsize=12)
    ax.set_ylabel('Cabin Deck Height (m)', fontsize=12)
    ax.set_title(f'KNN con k = {k}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Agregar leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Clase 1'),
        Patch(facecolor='orange', edgecolor='black', label='Clase 2'),
        Patch(facecolor='green', edgecolor='black', label='Clase 3')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
              title='Color según clase predicha')

plt.show()
