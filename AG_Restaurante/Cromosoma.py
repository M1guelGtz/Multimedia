import random
import math
import copy
from dataclasses import dataclass
from typing import List, Optional

# ==========================================
# 1. DEFINICIÓN DE ESTRUCTURAS
# ==========================================
@dataclass
class Obstaculo:
    nombre: str
    x: float
    y: float
    ancho: float
    largo: float

@dataclass
class Reserva:
    id_reserva: int
    num_personas: int

@dataclass
class DefinicionMesa:
    id_mesa: int
    capacidad_base: int
    ancho: float
    largo: float
    es_fija: bool
    x_fija: Optional[float] = None
    y_fija: Optional[float] = None
    orientacion_fija: Optional[str] = 'H'

@dataclass
class GenMesa:
    id_mesa: int
    id_reserva: Optional[int]
    x: float
    y: float
    orientacion: str

class Cromosoma:
    def __init__(self, mesas_def: List[DefinicionMesa] = None):
        self.genes: List[GenMesa] = []
        self.fitness: float = float('inf') # Infinito = Pésima acomodación
        
        if mesas_def:
            for mesa in mesas_def:
                if mesa.es_fija:
                    gen = GenMesa(mesa.id_mesa, None, mesa.x_fija, mesa.y_fija, mesa.orientacion_fija)
                else:
                    gen = GenMesa(mesa.id_mesa, None, random.uniform(0, 9), random.uniform(0, 9), random.choice(['H', 'V']))
                self.genes.append(gen)

    def clonar(self):
        """Crea una copia exacta del individuo para la reproducción"""
        nuevo = Cromosoma()
        nuevo.genes = copy.deepcopy(self.genes)
        nuevo.fitness = self.fitness
        return nuevo

# ==========================================
# 2. CONFIGURACIÓN DEL LOCAL (10x10)
# ==========================================
ANCHO_LOCAL = 10.0
LARGO_LOCAL = 10.0

obstaculos = [
    Obstaculo("Maceta_1", 8.0, 3.0, 1.0, 1.0),
    Obstaculo("Maceta_2", 8.0, 5.0, 1.0, 1.0),
    Obstaculo("Barra", 0.0, 3.0, 3.0, 9.0 ),
    Obstaculo("Escenario", 5,0, 4 ,1.5)
]

mesas_def = []
id_contador = 1
def crear_mesas(cantidad, cap, w, h, es_fija=False, x=None, y=None):
    global id_contador
    for _ in range(cantidad):
        mesas_def.append(DefinicionMesa(id_contador, cap, w, h, es_fija, x, y))
        id_contador += 1

# Inventario de mesas solicitado:
crear_mesas(1, 12, 1.2, 3.6, True, 8.0, 6.0) # Mesa fija 2
crear_mesas(1, 12, 0.8, 3.6, True, 5.0, 6.0) # Mesa fija 1
crear_mesas(2, 10, 0.8, 3.0)
crear_mesas(2, 6,  0.8, 1.8)
crear_mesas(4, 4,  0.8, 1.2)
crear_mesas(4, 2,  0.8, 0.8)

# Reservas de ejemplo
reservas = [
    Reserva(101, 18), 
    Reserva(102, 10), 
    Reserva(103, 4), 
    Reserva(104, 2)
]

# ==========================================
# 3. LÓGICA DE EVALUACIÓN (APTITUD / FITNESS)
# ==========================================
def obtener_dimensiones_reales(gen):
    mesa_base = next(m for m in mesas_def if m.id_mesa == gen.id_mesa)
    return (mesa_base.largo, mesa_base.ancho) if gen.orientacion == 'V' else (mesa_base.ancho, mesa_base.largo)

def hay_interseccion(x1, y1, w1, h1, x2, y2, w2, h2):
    # Agregamos un pequeño margen (0.2m) para que las mesas no estén literalmente pegadas sin pasillo
    margen = 0.2 
    return (x1 < x2 + w2 + margen and x1 + w1 + margen > x2 and
            y1 < y2 + h2 + margen and y1 + h1 + margen > y2)

def evaluar_fitness(cromosoma):
    penalizacion = 0
    
    # 1. EVALUAR COLISIONES ESPACIALES
    cajas = []
    for gen in cromosoma.genes:
        w, h = obtener_dimensiones_reales(gen)
        cajas.append({'id': gen.id_mesa, 'x': gen.x, 'y': gen.y, 'w': w, 'h': h})
        
        # Fuera del local
        if gen.x < 0 or gen.y < 0 or gen.x + w > ANCHO_LOCAL or gen.y + h > LARGO_LOCAL:
            penalizacion += 1000

        # Choque con obstáculos
        for obs in obstaculos:
            if hay_interseccion(gen.x, gen.y, w, h, obs.x, obs.y, obs.ancho, obs.largo):
                penalizacion += 1000

    # Choque entre mesas (Si son de la misma reserva, perdonamos que estén muy cerca)
    # Choque entre mesas (¡NUEVA LÓGICA!)
    # Ahora NINGUNA mesa puede encimarse, ni siquiera las de la misma reserva.
    for i in range(len(cajas)):
        for j in range(i + 1, len(cajas)):
            # Siempre revisamos si chocan
            if hay_interseccion(cajas[i]['x'], cajas[i]['y'], cajas[i]['w'], cajas[i]['h'],
                                cajas[j]['x'], cajas[j]['y'], cajas[j]['w'], cajas[j]['h']):
                
                gen_a = cromosoma.genes[i]
                gen_b = cromosoma.genes[j]
                
                # Si son de la misma reserva, el castigo es alto, 
                # pero si son de distinta reserva, el castigo es EXTREMO.
                if gen_a.id_reserva == gen_b.id_reserva:
                    penalizacion += 800  # Castigo por encimarse
                else:
                    penalizacion += 2000 # Castigo por chocar con desconocidos
    # 2. LÓGICA DE UNIONES Y RESERVAS
    ids_reservas_asignadas = set([g.id_reserva for g in cromosoma.genes if g.id_reserva is not None])
    
    # Castigo severo si una reserva se quedó sin ninguna mesa asignada
    for res in reservas:
        if res.id_reserva not in ids_reservas_asignadas:
            penalizacion += 5000 

    for res in reservas:
        genes_asignados = [g for g in cromosoma.genes if g.id_reserva == res.id_reserva]
        if not genes_asignados: continue
        
        # Calcular capacidad con pérdida de uniones
        capacidades_base = [next(m.capacidad_base for m in mesas_def if m.id_mesa == g.id_mesa) for g in genes_asignados]
        capacidad_total = sum(capacidades_base)
        num_uniones = len(genes_asignados) - 1
        capacidad_efectiva = capacidad_total - (2 * num_uniones)
        
        # Evaluar falta de sillas o exceso de espacio
        if capacidad_efectiva < res.num_personas:
            penalizacion += 1000 * (res.num_personas - capacidad_efectiva) # Castigo brutal por no caber
        else:
            sillas_vacias = capacidad_efectiva - res.num_personas
            penalizacion += 10 * sillas_vacias # Castigo leve por desperdiciar espacio
            
        # Penalización por dispersión (obligar a que mesas de la misma reserva estén juntas)
        if len(genes_asignados) > 1:
            xs = [g.x for g in genes_asignados]
            ys = [g.y for g in genes_asignados]
            distancia_maxima = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if distancia_maxima > 3.0: # Si están a más de 3 metros, se castiga
                penalizacion += 100 * distancia_maxima

    cromosoma.fitness = penalizacion

# ==========================================
# 4. OPERADORES GENÉTICOS
# ==========================================
def cruzar(padre1, padre2):
    hijo = Cromosoma()
    # Cruce uniforme: para cada mesa, elige al azar de qué padre hereda la configuración
    for i in range(len(padre1.genes)):
        if random.random() < 0.5:
            hijo.genes.append(copy.deepcopy(padre1.genes[i]))
        else:
            hijo.genes.append(copy.deepcopy(padre2.genes[i]))
    return hijo

def mutar(cromosoma):
    TASA_MUTACION = 0.1 # 10% de probabilidad de mutar un gen
    ids_reservas_validas = [r.id_reserva for r in reservas] + [None]
    
    for gen in cromosoma.genes:
        if random.random() < TASA_MUTACION:
            # Mutar asignación de reserva
            if random.random() < 0.5:
                gen.id_reserva = random.choice(ids_reservas_validas)
            
            # Mutar posición/rotación (SOLO SI ES MÓVIL)
            # Mutar posición/rotación (SOLO SI ES MÓVIL)
            mesa_base = next(m for m in mesas_def if m.id_mesa == gen.id_mesa)
            if not mesa_base.es_fija:
                # Movemos en pasos fijos de 0.5 metros
                gen.x += random.choice([-1.0, -0.5, 0.5, 1.0])
                gen.y += random.choice([-1.0, -0.5, 0.5, 1.0])
                
                # Redondear al 0.5 más cercano para mantener la cuadrícula
                gen.x = round(gen.x * 2) / 2
                gen.y = round(gen.y * 2) / 2
                
                # Mantener dentro del local
                gen.x = max(0, min(gen.x, ANCHO_LOCAL - mesa_base.ancho))
                gen.y = max(0, min(gen.y, LARGO_LOCAL - mesa_base.largo))

def torneo(poblacion, k=3):
    seleccionados = random.sample(poblacion, k)
    return min(seleccionados, key=lambda ind: ind.fitness)

# ==========================================
# 5. CICLO PRINCIPAL (EL ALGORITMO)
# ==========================================
def ejecutar_algoritmo():
    TAM_POBLACION = 100
    GENERACIONES = 500
    
    # 1. Generar población inicial al azar
    poblacion = []
    for _ in range(TAM_POBLACION):
        ind = Cromosoma(mesas_def)
        # Asignar reservas al azar para empezar
        ids_res = [r.id_reserva for r in reservas] + [None]*10
        for gen in ind.genes:
            gen.id_reserva = random.choice(ids_res)
        poblacion.append(ind)

    # 2. Evolucionar
    for gen in range(GENERACIONES):
        # Evaluar a todos
        for ind in poblacion:
            evaluar_fitness(ind)
            
        poblacion.sort(key=lambda x: x.fitness)
        mejor_actual = poblacion[0]
        
        # Imprimir progreso cada 20 generaciones
        if gen % 20 == 0 or gen == GENERACIONES - 1:
            print(f"Generación {gen:3d} | Mejor Fitness (Menor es mejor): {mejor_actual.fitness:.2f}")
            
        # Crear nueva generación (Elitismo: guardamos al mejor intacto)
        nueva_poblacion = [mejor_actual.clonar()]
        
        while len(nueva_poblacion) < TAM_POBLACION:
            padre1 = torneo(poblacion)
            padre2 = torneo(poblacion)
            hijo = cruzar(padre1, padre2)
            mutar(hijo)
            nueva_poblacion.append(hijo)
            
        poblacion = nueva_poblacion

    # 3. Mostrar el mejor resultado final
    print("\n" + "="*50)
    print("🏆 MEJOR ACOMODACIÓN ENCONTRADA")
    print("="*50)
    mejor_final = poblacion[0]
    
    for res in reservas:
        mesas_asignadas = [g for g in mejor_final.genes if g.id_reserva == res.id_reserva]
        caps_base = [next(m.capacidad_base for m in mesas_def if m.id_mesa == g.id_mesa) for g in mesas_asignadas]
        cap_efectiva = sum(caps_base) - 2 * (max(0, len(mesas_asignadas) - 1))
        
        print(f"🍽️ Reserva {res.id_reserva} ({res.num_personas} personas):")
        print(f"   Mesas asignadas: {[g.id_mesa for g in mesas_asignadas]}")
        print(f"   Capacidad lograda con uniones: {cap_efectiva} sillas.")
        for g in mesas_asignadas:
            print(f"   -> Mesa {g.id_mesa}: Pos(x:{g.x:.1f}, y:{g.y:.1f}), Orientación: {g.orientacion}")
        print("-" * 30)
    graficar_acomodacion(cromosoma=mejor_final, mesas_def=mesas_def, obstaculos=obstaculos, reservas= reservas, ancho_local=ANCHO_LOCAL, largo_local=LARGO_LOCAL)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

def graficar_acomodacion(cromosoma, mesas_def, obstaculos, reservas, ancho_local, largo_local):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Dibujar el local (Fondo)
    ax.set_xlim(0, ancho_local)
    ax.set_ylim(0, largo_local)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("Plano del Restaurante - Mejor Acomodación", fontsize=14, fontweight='bold')
    ax.set_xlabel("Ancho (metros)")
    ax.set_ylabel("Largo (metros)")

    # 2. Dibujar Obstáculos (Macetas / Decoraciones)
    for obs in obstaculos:
        rect = patches.Rectangle((obs.x, obs.y), obs.ancho, obs.largo, 
                                 linewidth=1, edgecolor='darkgreen', facecolor='lightgreen', hatch='//')
        ax.add_patch(rect)
        ax.text(obs.x + obs.ancho/2, obs.y + obs.largo/2, obs.nombre, 
                color='darkgreen', weight='bold', fontsize=8, ha='center', va='center')

    # 3. Asignar un color único a cada reserva
    colores_reservas = list(mcolors.TABLEAU_COLORS.values())
    mapa_colores = {}
    for i, res in enumerate(reservas):
        # Asignamos un color de la paleta a cada ID de reserva
        mapa_colores[res.id_reserva] = colores_reservas[i % len(colores_reservas)]

    # 4. Dibujar las Mesas
    for gen in cromosoma.genes:
        # Obtener dimensiones según rotación
        mesa_base = next(m for m in mesas_def if m.id_mesa == gen.id_mesa)
        w, h = (mesa_base.largo, mesa_base.ancho) if gen.orientacion == 'V' else (mesa_base.ancho, mesa_base.largo)
        
        # Determinar color: Gris si está vacía, Color asignado si tiene reserva
        color_mesa = mapa_colores.get(gen.id_reserva, 'lightgray')
        borde_mesa = 'black' if gen.id_reserva is None else 'darkblue'

        # Dibujar el rectángulo de la mesa
        rect = patches.Rectangle((gen.x, gen.y), w, h, 
                                 linewidth=1.5, edgecolor=borde_mesa, facecolor=color_mesa, alpha=0.8)
        ax.add_patch(rect)
        
        # Etiqueta de la mesa (ID y Reserva)
        texto_reserva = f"R:{gen.id_reserva}" if gen.id_reserva else "Vacía"
        etiqueta = f"M{gen.id_mesa}\n({texto_reserva})"
        ax.text(gen.x + w/2, gen.y + h/2, etiqueta, 
                color='black', weight='bold', fontsize=7, ha='center', va='center')

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    ejecutar_algoritmo()