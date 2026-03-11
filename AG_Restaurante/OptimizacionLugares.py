import random
import math
import copy
from dataclasses import dataclass
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# ==========================================
# 1. DEFINICIÓN DE ESTRUCTURAS
# ==========================================
@dataclass
class Obstaculo:
    nombre: str; x: float; y: float; ancho: float; largo: float

@dataclass
class Reserva:
    id_reserva: int; num_personas: int

@dataclass
class DefinicionMesa:
    id_mesa: int
    capacidad_base: int
    ancho: float
    largo: float
    es_fija: bool
    # AHORA TODAS LAS MESAS TIENEN UN ORIGEN (Su lugar por defecto en el restaurante)
    x_origen: float 
    y_origen: float
    orientacion_origen: str = 'H'

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
        self.fitness: float = float('inf')
        
        if mesas_def:
            for mesa in mesas_def:
                # TODAS las mesas nacen en su posición original, no al azar.
                gen = GenMesa(mesa.id_mesa, None, mesa.x_origen, mesa.y_origen, mesa.orientacion_origen)
                self.genes.append(gen)

    def clonar(self):
        nuevo = Cromosoma()
        nuevo.genes = copy.deepcopy(self.genes)
        nuevo.fitness = self.fitness
        return nuevo

# ==========================================
# 2. CONFIGURACIÓN DEL LOCAL (10x10) Y ORDEN INICIAL
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
def crear_mesa(cap, w, h, es_fija, x_orig, y_orig, orient='H'):
    global id_contador
    mesas_def.append(DefinicionMesa(id_contador, cap, w, h, es_fija, x_orig, y_orig, orient))
    id_contador += 1

# Inventario con su "Acomodo Original" (He puesto coordenadas lógicas para que no choquen al inicio)
crear_mesa(12, 1.6, 4.6, True,  5.7, 5.6 ) # M1 (Fija)
crear_mesa(12, 2.2, 4.0, True,  7.8, 5.6) # M2 (Fija)

crear_mesa(10, 1.6, 4.0, True, 3.6, 5.0) # M3 (Móvil)
#crear_mesa(10, 0.8, 3.0, False, 1.0, 6.0, 'V') # M4 (Móvil)

#crear_mesa(6,  0.8, 1.8, False, 3.0, 2.0, 'V') # M5 (Móvil)
#crear_mesa(6,  0.8, 1.8, False, 3.0, 5.0, 'V') # M6 (Móvil)
#crear_mesa(6,  0.8, 1.8, False, 3.0, 5.0, 'V')

#crear_mesa(4,  0.8, 1.2, False, 7.5, 1.0) # M7 (Móvil)
#crear_mesa(4,  0.8, 1.2, False, 7.5, 3.0) # M8 (Móvil)
#crear_mesa(4,  0.8, 1.2, False, 7.5, 5.0) # M9 (Móvil)
#crear_mesa(4,  0.8, 1.2, False, 7.5, 7.0) # M10 (Móvil)

#crear_mesa(2,  0.8, 0.8, False, 9.0, 2.0) # M11 (Móvil)
#crear_mesa(2,  0.8, 0.8, False, 9.0, 4.0) # M12 (Móvil)
#crear_mesa(2,  0.8, 0.8, False, 9.0, 6.0) # M13 (Móvil)
#crear_mesa(2,  0.8, 0.8, False, 9.0, 8.0) # M14 (Móvil)

reservas = [
    Reserva(101, 18), 
    Reserva(102, 10), 
    Reserva(103, 4), 
    Reserva(104, 2)
]

# ==========================================
# 3. LÓGICA DE EVALUACIÓN (APTITUD / FITNESS)
# ==========================================
def obtener_dimensiones(gen):
    mb = next(m for m in mesas_def if m.id_mesa == gen.id_mesa)
    return (mb.largo, mb.ancho) if gen.orientacion == 'V' else (mb.ancho, mb.largo)

def hay_interseccion(x1, y1, w1, h1, x2, y2, w2, h2):
    margen = 0.1 
    return (x1 < x2 + w2 + margen and x1 + w1 + margen > x2 and
            y1 < y2 + h2 + margen and y1 + h1 + margen > y2)

def evaluar_fitness(cromosoma):
    penalizacion = 0
    cajas = []
    
    for gen in cromosoma.genes:
        mb = next(m for m in mesas_def if m.id_mesa == gen.id_mesa)
        w, h = obtener_dimensiones(gen)
        cajas.append({'id': gen.id_mesa, 'x': gen.x, 'y': gen.y, 'w': w, 'h': h, 'res': gen.id_reserva})
        
        # 1. Castigo por salirse del local
        if gen.x < 0 or gen.y < 0 or gen.x + w > ANCHO_LOCAL or gen.y + h > LARGO_LOCAL:
            penalizacion += 2000

        # 2. Castigo por chocar con obstáculos
        for obs in obstaculos:
            if hay_interseccion(gen.x, gen.y, w, h, obs.x, obs.y, obs.ancho, obs.largo):
                penalizacion += 2000
                
        # 3. EL COSTO DE MOVERSE (Castigo por pereza)
        # Si la mesa se movió de su origen, cobramos 5 puntos por metro de distancia.
        distancia_origen = math.hypot(gen.x - mb.x_origen, gen.y - mb.y_origen)
        penalizacion += (distancia_origen * 5)
        
        # Si rotó la mesa y no era necesario, castigo leve
        if gen.orientacion != mb.orientacion_origen:
            penalizacion += 2

    # 4. Castigo por encimar mesas (¡Corregido para que no hagan el "Sándwich"!)
    for i in range(len(cajas)):
        for j in range(i + 1, len(cajas)):
            if hay_interseccion(cajas[i]['x'], cajas[i]['y'], cajas[i]['w'], cajas[i]['h'],
                                cajas[j]['x'], cajas[j]['y'], cajas[j]['w'], cajas[j]['h']):
                if cajas[i]['res'] == cajas[j]['res'] and cajas[i]['res'] is not None:
                    penalizacion += 800  # Castigo alto si son de la misma reserva y se enciman
                else:
                    penalizacion += 3000 # Castigo EXTREMO si chocan con extraños o vacías

    # 5. Lógica de Reservas y Sillas
    ids_asignadas = set([g.id_reserva for g in cromosoma.genes if g.id_reserva is not None])
    for res in reservas:
        if res.id_reserva not in ids_asignadas:
            penalizacion += 5000 

    for res in reservas:
        genes_asig = [g for g in cromosoma.genes if g.id_reserva == res.id_reserva]
        if not genes_asig: continue
        
        caps_base = [next(m.capacidad_base for m in mesas_def if m.id_mesa == g.id_mesa) for g in genes_asig]
        cap_efectiva = sum(caps_base) - (2 * (len(genes_asig) - 1))
        
        if cap_efectiva < res.num_personas:
            penalizacion += 1000 * (res.num_personas - cap_efectiva) 
        else:
            penalizacion += 10 * (cap_efectiva - res.num_personas)
            
        # Penalización por dispersión (Tienen que estar juntas si son varias mesas)
        if len(genes_asig) > 1:
            xs = [g.x for g in genes_asig]
            ys = [g.y for g in genes_asig]
            dist_max = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if dist_max > 2.0:
                penalizacion += 200 * dist_max

    cromosoma.fitness = penalizacion

# ==========================================
# 4. OPERADORES GENÉTICOS
# ==========================================
def cruzar(padre1, padre2):
    hijo = Cromosoma()
    for i in range(len(padre1.genes)):
        hijo.genes.append(copy.deepcopy(padre1.genes[i] if random.random() < 0.5 else padre2.genes[i]))
    return hijo

def mutar(cromosoma):
    TASA_MUTACION = 0.15 
    ids_res_validas = [r.id_reserva for r in reservas] + [None]
    
    for gen in cromosoma.genes:
        if random.random() < TASA_MUTACION:
            # Mutar reserva
            if random.random() < 0.5:
                gen.id_reserva = random.choice(ids_res_validas)
            
            # Mutar posición (Solo móviles y en cuadrícula de 0.5m)
            mb = next(m for m in mesas_def if m.id_mesa == gen.id_mesa)
            if not mb.es_fija:
                gen.x += random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                gen.y += random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
                
                # Ajustar a la cuadrícula más cercana (Grid Snapping)
                gen.x = round(gen.x * 2) / 2
                gen.y = round(gen.y * 2) / 2
                
                gen.x = max(0, min(gen.x, ANCHO_LOCAL - mb.ancho))
                gen.y = max(0, min(gen.y, LARGO_LOCAL - mb.largo))
                
                if random.random() < 0.2:
                    gen.orientacion = 'V' if gen.orientacion == 'H' else 'H'

def torneo(poblacion, k=3):
    return min(random.sample(poblacion, k), key=lambda ind: ind.fitness)

# ==========================================
# 5. GRAFICADOR Y CICLO PRINCIPAL
# ==========================================
def graficar_acomodacion(cromosoma):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, ANCHO_LOCAL); ax.set_ylim(0, LARGO_LOCAL)
    ax.grid(True, linestyle='--', alpha=0.6); ax.set_aspect('equal')
    ax.set_title("Plano del Restaurante - Inteligencia de Movimiento", fontsize=14, fontweight='bold')

    for obs in obstaculos:
        ax.add_patch(patches.Rectangle((obs.x, obs.y), obs.ancho, obs.largo, facecolor='lightgreen', edgecolor='darkgreen', hatch='//'))
        ax.text(obs.x + obs.ancho/2, obs.y + obs.largo/2, obs.nombre, ha='center', va='center', fontsize=8)

    colores = list(mcolors.TABLEAU_COLORS.values())
    mapa_colores = {res.id_reserva: colores[i % len(colores)] for i, res in enumerate(reservas)}

    for gen in cromosoma.genes:
        w, h = obtener_dimensiones(gen)
        color = mapa_colores.get(gen.id_reserva, '#E0E0E0')
        borde = 'black' if gen.id_reserva is None else 'darkblue'
        
        ax.add_patch(patches.Rectangle((gen.x, gen.y), w, h, facecolor=color, edgecolor=borde, linewidth=1.5, alpha=0.9))
        
        # Etiqueta con ID de mesa y Reserva
        txt = f"M{gen.id_mesa}\n" + (f"(R:{gen.id_reserva})" if gen.id_reserva else "(Libre)")
        ax.text(gen.x + w/2, gen.y + h/2, txt, ha='center', va='center', fontsize=7, weight='bold')

    plt.tight_layout()
    plt.show()

def ejecutar_algoritmo():
    TAM_POBLACION = 100
    GENERACIONES = 400 # Aumentamos generaciones para que piense mejor
    
    # Iniciar población con las mesas en sus lugares de origen
    poblacion = []
    for _ in range(TAM_POBLACION):
        ind = Cromosoma(mesas_def)
        # Solo asignamos reservas al azar, la posición ya viene del origen
        for gen in ind.genes:
            gen.id_reserva = random.choice([r.id_reserva for r in reservas] + [None]*10)
        poblacion.append(ind)

    print("Calculando la mejor acomodación (Evaluando movimiento y uniones)...")
    for gen in range(GENERACIONES):
        for ind in poblacion: evaluar_fitness(ind)
        poblacion.sort(key=lambda x: x.fitness)
        
        if gen % 50 == 0:
            print(f"Generación {gen:3d} | Mejor Fitness: {poblacion[0].fitness:.2f}")
            
        nueva_poblacion = [poblacion[0].clonar()]
        while len(nueva_poblacion) < TAM_POBLACION:
            hijo = cruzar(torneo(poblacion), torneo(poblacion))
            mutar(hijo)
            nueva_poblacion.append(hijo)
        poblacion = nueva_poblacion

    mejor = poblacion[0]
    print(f"\n¡Listo! Mejor Fitness final: {mejor.fitness:.2f}")
    graficar_acomodacion(mejor)

if __name__ == "__main__":
    ejecutar_algoritmo()