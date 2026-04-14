import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURACIÓN GPU (evita OOM en GPUs con poca VRAM)
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU detectada: {[g.name for g in gpus]}")
else:
    print("GPU no detectada. Usando CPU.")


# ==========================================
# CARGA INTERACTIVA DEL DATASET
# ==========================================
def cargar_dataset():
    print("\n" + "=" * 60)
    print("CARGA DE DATASET")
    print("=" * 60)

    ruta = input("Ruta del archivo CSV: ").strip().strip('"')

    tiene_header = input("¿El archivo tiene encabezado? (s/n): ").strip().lower()
    header = 0 if tiene_header == 's' else None

    df = pd.read_csv(ruta, header=header)

    print(f"\nDataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("\nPrimeras filas:")
    print(df.head(3).to_string())
    print(f"\nColumnas: {list(df.columns)}")

    col_target = input("\n¿Cuál es la columna target? (nombre o índice numérico): ").strip()

    # Aceptar nombre o índice
    if col_target.lstrip('-').isdigit():
        idx = int(col_target)
        target_col = df.columns[idx]
    else:
        target_col = col_target

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Eliminar columnas no numéricas
    cols_no_numericas = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cols_no_numericas:
        print(f"Columnas no numéricas eliminadas automáticamente: {cols_no_numericas}")
        X = X.select_dtypes(include=[np.number])

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print(f"\nFeatures ({X.shape[1]}): {list(X.columns)}")
    print(f"Target: '{target_col}' | Rango: [{y.min():.4f}, {y.max():.4f}]")

    return X, y


# ==========================================
# CLASE PERCEPTRÓN MULTICAPA (TensorFlow)
# ==========================================
class PerceptronMulticapa:
    def __init__(self, capas_ocultas, tasa_aprendizaje, epocas, func_activacion='relu'):
        self.capas_ocultas   = capas_ocultas
        self.tasa            = tasa_aprendizaje
        self.epocas          = epocas
        self.func_activacion = func_activacion
        self.modelo          = None
        self.historial       = None

    def construir_modelo(self, n_features):
        modelo = keras.Sequential()
        modelo.add(keras.Input(shape=(n_features,)))
        for neuronas in self.capas_ocultas:
            modelo.add(keras.layers.Dense(neuronas, activation=self.func_activacion))
        modelo.add(keras.layers.Dense(1))  # salida lineal para regresión
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.tasa),
            loss='mse',
            metrics=['mae']
        )
        self.modelo = modelo

    def entrenar(self, X_train, y_train, X_val, y_val):
        self.construir_modelo(X_train.shape[1])
        self.historial = self.modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epocas,
            batch_size=32,
            verbose=0
        )

    def predecir(self, X_nuevos):
        return self.modelo.predict(X_nuevos, verbose=0)


# ==========================================
# VALIDACIÓN CRUZADA K-FOLD
# ==========================================
def validacion_cruzada_mlp(X, y, k=4, epocas=150, tasa=0.001,
                            capas=[64, 32], func_act='relu'):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X_shuf  = X.iloc[indices].reset_index(drop=True)
    y_shuf  = y.iloc[indices].reset_index(drop=True)

    n_muestras  = len(X)
    tamaño_base = n_muestras // k
    residuo     = n_muestras % k

    # Sobrante repartido de a 1 en los primeros bloques
    tamaños = [tamaño_base + 1 if i < residuo else tamaño_base for i in range(k)]
    inicios = [sum(tamaños[:i]) for i in range(k)]

    nombres_bloques = [chr(ord('A') + j) for j in range(k)]

    modelos          = []
    todos_hist_train = []
    todos_hist_val   = []

    print(f"\n--- INICIANDO CREACIÓN DE {k} MODELOS (VALIDACIÓN CRUZADA) ---")

    # Iterar en orden inverso: validación D → C → B → A
    for num_modelo, idx in enumerate(range(k - 1, -1, -1), start=1):
        inicio_val = inicios[idx]
        fin_val    = inicios[idx] + tamaños[idx]

        X_val_raw   = X_shuf.iloc[inicio_val:fin_val].values
        y_val_raw   = y_shuf.iloc[inicio_val:fin_val].values

        X_train_raw = np.vstack([X_shuf.iloc[:inicio_val].values,
                                 X_shuf.iloc[fin_val:].values])
        y_train_raw = np.concatenate([y_shuf.iloc[:inicio_val].values,
                                      y_shuf.iloc[fin_val:].values])

        scaler      = StandardScaler()
        X_train_n   = scaler.fit_transform(X_train_raw)
        X_val_n     = scaler.transform(X_val_raw)

        mlp = PerceptronMulticapa(capas, tasa, epocas, func_act)
        mlp.entrenar(X_train_n, y_train_raw, X_val_n, y_val_raw)

        hist_train = mlp.historial.history['loss']
        hist_val   = mlp.historial.history['val_loss']

        epoca_opt     = int(np.argmin(hist_val))
        error_train_f = hist_train[-1]
        error_val_f   = hist_val[-1]
        obs_train     = len(X_train_raw)
        obs_val       = len(X_val_raw)

        bloque_val    = nombres_bloques[idx]
        bloques_train = ",".join([b for b in nombres_bloques if b != bloque_val])

        # Promedio ponderado de MSE
        error_total = ((obs_train * error_train_f) + (obs_val * error_val_f)) / (obs_train + obs_val)

        modelos.append({
            'Num_Modelo':    num_modelo,
            'Modelo':        mlp,
            'Scaler':        scaler,
            'Bloque_Val':    bloque_val,
            'Val_Size':      obs_val,
            'Bloques_Train': bloques_train,
            'Train_Size':    obs_train,
            'Iter_Optima':   epoca_opt,
            'Error_Train':   error_train_f,
            'Error_Val':     error_val_f,
            'Error_Total':   error_total,
        })

        todos_hist_train.append(hist_train)
        todos_hist_val.append(hist_val)

        print(f"Modelo {num_modelo} | Val: {bloque_val} | Err Total: {error_total:.4f} | Punto Óptimo: Época {epoca_opt}")

    return modelos, todos_hist_train, todos_hist_val


# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
X_raw, y_raw = cargar_dataset()

print("\n" + "=" * 60)
print("CONFIGURACIÓN DEL MODELO")
print("=" * 60)

def pedir_int(msg, default):
    val = input(f"{msg} [default: {default}]: ").strip()
    return int(val) if val else default

def pedir_float(msg, default):
    val = input(f"{msg} [default: {default}]: ").strip()
    return float(val) if val else default

def pedir_lista(msg, default):
    val = input(f"{msg} [default: {default}]: ").strip()
    if not val:
        return default
    return [int(x.strip()) for x in val.split(',')]

K_BLOQUES     = pedir_int("Número de folds (k)",                    4)
EPOCAS        = pedir_int("Épocas de entrenamiento",               150)
TASA          = pedir_float("Tasa de aprendizaje",               0.001)
CAPAS_OCULTAS = pedir_lista("Capas ocultas (neuronas separadas por coma)", [64, 32])

print(f"\nConfiguración: k={K_BLOQUES} | Épocas={EPOCAS} | Tasa={TASA} | Capas={CAPAS_OCULTAS}")

modelos_cv, todos_hist_train, todos_hist_val = validacion_cruzada_mlp(
    X_raw, y_raw,
    k=K_BLOQUES,
    epocas=EPOCAS,
    tasa=TASA,
    capas=CAPAS_OCULTAS
)


# ==========================================
# GRÁFICAS POR FOLD
# ==========================================
for i in range(K_BLOQUES):
    info    = modelos_cv[i]
    ep_opt  = info['Iter_Optima']
    min_val = todos_hist_val[i][ep_opt]

    plt.figure(f"Modelo {i+1} - Validación Cruzada MLP", figsize=(9, 5))
    plt.plot(todos_hist_train[i], label='Error Entrenamiento (MSE)', color='steelblue')
    plt.plot(todos_hist_val[i],   label='Error Validación (MSE)',    color='orange')
    plt.axvline(x=ep_opt, color='red', linestyle='--',
                label=f'Punto Óptimo (Época {ep_opt})')
    plt.scatter(ep_opt, min_val, color='red', zorder=5)
    plt.title(f"Modelo {i+1} — Train: {info['Bloques_Train']} | Val: {info['Bloque_Val']}")
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)


# ==========================================
# TABLA RESUMEN K-FOLD
# ==========================================
modelos_ordenados = sorted(modelos_cv, key=lambda x: x['Error_Total'])
mejor_num         = modelos_ordenados[0]['Num_Modelo']

COL_K     = 6
COL_TRAIN = 32
COL_VAL   = 18
COL_ERR   = 14
ANCHO     = COL_K + COL_TRAIN + COL_VAL + COL_ERR * 3 + 14

print("\n" + "=" * ANCHO)
print(f"{'RESUMEN DE LOS MODELOS ENTRENADOS (K-FOLD) — MLP TensorFlow':^{ANCHO}}")
print(f"{'Arquitectura: ' + str(CAPAS_OCULTAS) + ' | Épocas: ' + str(EPOCAS) + ' | Tasa: ' + str(TASA):^{ANCHO}}")
print("=" * ANCHO)
print(f"{'k':<{COL_K}} | {'Entrenamiento (grupos + obs)':<{COL_TRAIN}} | "
      f"{'Validación (grupo + obs)':<{COL_VAL}} | {'Err Train':<{COL_ERR}} | "
      f"{'Err Val':<{COL_ERR}} | {'Error Total':<{COL_ERR}}")
print("-" * ANCHO)

for info in modelos_cv:
    marcador  = " <-- MEJOR" if info['Num_Modelo'] == mejor_num else ""
    train_str = f"{info['Bloques_Train']} (n={info['Train_Size']})"
    val_str   = f"{info['Bloque_Val']} (n={info['Val_Size']})"
    e_train   = f"{info['Error_Train']:.6f}"
    e_val     = f"{info['Error_Val']:.6f}"
    e_total   = f"{info['Error_Total']:.6f}"
    print(f"{info['Num_Modelo']:<{COL_K}} | {train_str:<{COL_TRAIN}} | {val_str:<{COL_VAL}} | "
          f"{e_train:<{COL_ERR}} | {e_val:<{COL_ERR}} | {e_total:<{COL_ERR}}{marcador}")

print("=" * ANCHO)


# ==========================================
# SISTEMA DE PREDICCIÓN
# ==========================================
mejor_info   = modelos_ordenados[0]
mejor_modelo = mejor_info['Modelo']
mejor_scaler = mejor_info['Scaler']
nombres_feat = list(X_raw.columns)
n_features   = len(nombres_feat)

print("\n" + "=" * 60)
print("SISTEMA DE PREDICCIÓN ACTIVADO — MLP TensorFlow")
print(f"Modelo seleccionado: Modelo {mejor_info['Num_Modelo']} (Menor Error Total)")
print(f"Features requeridos: {nombres_feat}")
print("=" * 60)

while True:
    print(f"\nIngresa {n_features} valores separados por comas (o escribe 'salir'):")
    print(f"Orden: {', '.join(nombres_feat)}")
    entrada = input("Valores X: ")

    if entrada.lower() == 'salir':
        print("Saliendo del sistema.")
        break

    try:
        valores = [float(v.strip()) for v in entrada.split(',')]

        if len(valores) != n_features:
            print(f"Error: Debes ingresar exactamente {n_features} valores.")
            continue

        X_nuevo   = np.array(valores).reshape(1, -1)
        X_nuevo_n = mejor_scaler.transform(X_nuevo)
        pred      = mejor_modelo.predecir(X_nuevo_n)

        print("-" * 40)
        print(f"Entrada: {valores}")
        print(f"Predicción: {np.squeeze(pred):.4f}")
        print("-" * 40)

    except ValueError:
        print("Error: Ingresa números válidos.")
