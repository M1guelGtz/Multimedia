import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import os

# ==========================================
# CONFIGURACIÓN GPU
# ==========================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ==========================================
# CARGA DEL DATASET
# ==========================================
def cargar_dataset(ruta_csv=None):
    if ruta_csv is None:
        ruta_csv = input("Ruta del archivo CSV: ").strip().strip('"')

    df = pd.read_csv(ruta_csv)

    print("=" * 60)
    print("1. EL DATASET")
    print("=" * 60)
    print(f"\nArchivo: {os.path.basename(ruta_csv)}")
    print(f"Primeras filas:\n{df.head(3).to_string()}\n")
    print(f"Columnas: {list(df.columns)}")

    # La ultima columna es el target
    target_col = df.columns[-1]
    X = df.iloc[:, :-1]
    y = df[target_col]

    # Eliminar columnas no numericas de X
    cols_no_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cols_no_num:
        print(f"Columnas no numericas eliminadas de features: {cols_no_num}")
        X = X.select_dtypes(include=[np.number])

    # Detectar tipo de problema
    if y.dtype == object or y.nunique() <= 20:
        tipo = "clasificacion"
        clases = sorted(y.unique())
        n_clases = len(clases)
        label_encoder = LabelEncoder()
        label_encoder.fit(clases)
        y_encoded = label_encoder.transform(y)
        print(f"\nEl dataset consta de {X.shape[0]} observaciones.")
        print(f"Cada observacion tiene {X.shape[1]} caracteristicas.")
        print(f"Dado los datos de la salida, se concluye que la salida es de tipo CATEGORICA (clasificacion).")
        print(f"Clases ({n_clases}): {clases}")
    else:
        tipo = "regresion"
        label_encoder = None
        y_encoded = y.values.astype(float)
        n_clases = 1
        clases = None
        print(f"\nEl dataset consta de {X.shape[0]} observaciones.")
        print(f"Cada observacion tiene {X.shape[1]} caracteristicas.")
        print(f"Dado los datos de la salida, se concluye que la salida es de tipo NUMERICA (regresion).")

    X = X.reset_index(drop=True)

    return X, y_encoded, tipo, n_clases, clases, label_encoder, list(X.columns)


# ==========================================
# CLASE PERCEPTRON MULTICAPA
# ==========================================
class PerceptronMulticapa:
    def __init__(self, capas_ocultas, tasa_aprendizaje, epocas,
                 func_activacion='relu', tipo='clasificacion', n_clases=3):
        self.capas_ocultas = capas_ocultas
        self.tasa = tasa_aprendizaje
        self.epocas = epocas
        self.func_activacion = func_activacion
        self.tipo = tipo
        self.n_clases = n_clases
        self.modelo = None
        self.historial = None

    def construir_modelo(self, n_features):
        modelo = keras.Sequential()
        modelo.add(keras.Input(shape=(n_features,)))
        for neuronas in self.capas_ocultas:
            modelo.add(keras.layers.Dense(neuronas, activation=self.func_activacion))

        if self.tipo == 'clasificacion':
            modelo.add(keras.layers.Dense(self.n_clases, activation='softmax'))
            modelo.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.tasa),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            modelo.add(keras.layers.Dense(1))
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
# VALIDACION CRUZADA K-FOLD
# ==========================================
def validacion_cruzada_mlp(X, y, k=4, epocas=150, tasa=0.001,
                           capas=[64, 32], func_act='relu',
                           tipo='clasificacion', n_clases=3):
    np.random.seed(42)
    tf.random.set_seed(42)

    indices = np.random.permutation(len(X))
    X_shuf = X.iloc[indices].reset_index(drop=True)
    y_shuf = y[indices]

    n_muestras = len(X)
    tam_base = n_muestras // k
    residuo = n_muestras % k

    tamanos = [tam_base + 1 if i < residuo else tam_base for i in range(k)]
    inicios = [sum(tamanos[:i]) for i in range(k)]

    nombres_bloques = [chr(ord('A') + j) for j in range(k)]

    modelos = []
    todos_hist_train = []
    todos_hist_val = []

    print(f"\n{'=' * 60}")
    print(f"3. VALIDACION CRUZADA (K = {k})")
    print(f"{'=' * 60}")
    print(f"\n--- INICIANDO CREACION DE {k} MODELOS ---\n")

    for num_modelo, idx in enumerate(range(k - 1, -1, -1), start=1):
        inicio_val = inicios[idx]
        fin_val = inicios[idx] + tamanos[idx]

        X_val_raw = X_shuf.iloc[inicio_val:fin_val].values
        y_val_raw = y_shuf[inicio_val:fin_val]

        X_train_raw = np.vstack([X_shuf.iloc[:inicio_val].values,
                                 X_shuf.iloc[fin_val:].values])
        y_train_raw = np.concatenate([y_shuf[:inicio_val],
                                      y_shuf[fin_val:]])

        scaler = StandardScaler()
        X_train_n = scaler.fit_transform(X_train_raw)
        X_val_n = scaler.transform(X_val_raw)

        mlp = PerceptronMulticapa(capas, tasa, epocas, func_act, tipo, n_clases)
        mlp.entrenar(X_train_n, y_train_raw, X_val_n, y_val_raw)

        hist_train = mlp.historial.history['loss']
        hist_val = mlp.historial.history['val_loss']

        epoca_opt = int(np.argmin(hist_val))
        error_train_f = hist_train[-1]
        error_val_f = hist_val[-1]
        obs_train = len(X_train_raw)
        obs_val = len(X_val_raw)

        bloque_val = nombres_bloques[idx]
        bloques_train = ",".join([b for b in nombres_bloques if b != bloque_val])

        error_total = ((obs_train * error_train_f) + (obs_val * error_val_f)) / (obs_train + obs_val)

        # Accuracy si es clasificacion
        if tipo == 'clasificacion':
            acc_train = mlp.historial.history['accuracy'][-1]
            acc_val = mlp.historial.history['val_accuracy'][-1]
        else:
            acc_train = None
            acc_val = None

        modelos.append({
            'Num_Modelo': num_modelo,
            'Modelo': mlp,
            'Scaler': scaler,
            'Bloque_Val': bloque_val,
            'Val_Size': obs_val,
            'Bloques_Train': bloques_train,
            'Train_Size': obs_train,
            'Iter_Optima': epoca_opt,
            'Error_Train': error_train_f,
            'Error_Val': error_val_f,
            'Error_Total': error_total,
            'Acc_Train': acc_train,
            'Acc_Val': acc_val,
        })

        todos_hist_train.append(hist_train)
        todos_hist_val.append(hist_val)

        extra = f" | Acc Train: {acc_train:.4f} | Acc Val: {acc_val:.4f}" if acc_train else ""
        print(f"  Modelo {num_modelo} | Val: {bloque_val} | Err Total: {error_total:.6f}{extra}")

    return modelos, todos_hist_train, todos_hist_val


# ==========================================
# EJECUCION PRINCIPAL
# ==========================================
if __name__ == '__main__':
    # Ruta CSV por argumento o input
    ruta = sys.argv[1] if len(sys.argv) > 1 else None
    X_raw, y_raw, tipo, n_clases, clases, label_enc, nombres_feat = cargar_dataset(ruta)

    # ---- Configuracion ----
    print(f"\n{'=' * 60}")
    print("2. MODELO DE RED IMPLEMENTADA")
    print("=" * 60)

    K_BLOQUES = 4
    EPOCAS = 150
    TASA = 0.001
    CAPAS_OCULTAS = [64, 32]

    metrica_loss = "categorical_crossentropy" if tipo == 'clasificacion' else "MSE"
    salida_desc = f"softmax ({n_clases} neuronas)" if tipo == 'clasificacion' else "lineal (1 neurona)"

    print(f"\n  Tipo de problema: {tipo.upper()}")
    print(f"  Entrada: {len(nombres_feat)} neuronas ({', '.join(nombres_feat)})")
    print(f"  Capas ocultas: {CAPAS_OCULTAS} con activacion ReLU")
    print(f"  Salida: {salida_desc}")
    print(f"  Funcion de perdida: {metrica_loss}")
    print(f"  Optimizador: Adam (lr={TASA})")
    print(f"  Epocas: {EPOCAS} | K-Folds: {K_BLOQUES}")

    if tipo == 'clasificacion' and clases is not None:
        print(f"  Clases de salida: {clases}")

    # ---- Validacion Cruzada ----
    modelos_cv, todos_hist_train, todos_hist_val = validacion_cruzada_mlp(
        X_raw, y_raw,
        k=K_BLOQUES, epocas=EPOCAS, tasa=TASA,
        capas=CAPAS_OCULTAS, tipo=tipo, n_clases=n_clases
    )

    # ---- Tabla Resumen ----
    modelos_ordenados = sorted(modelos_cv, key=lambda x: x['Error_Total'])
    mejor_num = modelos_ordenados[0]['Num_Modelo']

    print(f"\n{'=' * 110}")
    print(f"{'TABLA DE VALIDACION CRUZADA':^110}")
    print(f"{'=' * 110}")

    if tipo == 'clasificacion':
        print(f"{'k':<4} | {'# Obs Entrenamiento':<22} | {'# Obs Validacion':<18} | "
              f"{'Err Entren.':<14} | {'Err Valid.':<14} | {'Error Total':<14} | {'Acc Val':<10}")
        print("-" * 110)
        for info in modelos_cv:
            marcador = " <-- MEJOR" if info['Num_Modelo'] == mejor_num else ""
            print(f"{info['Num_Modelo']:<4} | {info['Train_Size']:<22} | {info['Val_Size']:<18} | "
                  f"{info['Error_Train']:<14.6f} | {info['Error_Val']:<14.6f} | "
                  f"{info['Error_Total']:<14.6f} | {info['Acc_Val']:<10.4f}{marcador}")
    else:
        print(f"{'k':<4} | {'# Obs Entrenamiento':<22} | {'# Obs Validacion':<18} | "
              f"{'Err Entren.':<14} | {'Err Valid.':<14} | {'Error Total':<14}")
        print("-" * 110)
        for info in modelos_cv:
            marcador = " <-- MEJOR" if info['Num_Modelo'] == mejor_num else ""
            print(f"{info['Num_Modelo']:<4} | {info['Train_Size']:<22} | {info['Val_Size']:<18} | "
                  f"{info['Error_Train']:<14.6f} | {info['Error_Val']:<14.6f} | "
                  f"{info['Error_Total']:<14.6f}{marcador}")

    print(f"{'=' * 110}")
    print(f"\nEl mejor modelo es el Modelo {mejor_num} (menor error total: {modelos_ordenados[0]['Error_Total']:.6f})")

    # ---- Grafica: Evolucion del error para el MEJOR modelo ----
    print(f"\n{'=' * 60}")
    print("4. ENTRENAMIENTO")
    print("=" * 60)
    print("4.1 Evolucion del error de entrenamiento (mejor modelo)")

    mejor_idx = None
    for i, m in enumerate(modelos_cv):
        if m['Num_Modelo'] == mejor_num:
            mejor_idx = i
            break

    info_mejor = modelos_cv[mejor_idx]
    ep_opt = info_mejor['Iter_Optima']

    plt.figure("Evolucion Error - Mejor Modelo", figsize=(10, 6))
    plt.plot(todos_hist_train[mejor_idx], label='Error Entrenamiento', color='steelblue', linewidth=2)
    plt.plot(todos_hist_val[mejor_idx], label='Error Validacion', color='orange', linewidth=2)
    plt.axvline(x=ep_opt, color='red', linestyle='--', linewidth=1.5,
                label=f'Mejor epoca (val) = {ep_opt}')
    plt.scatter(ep_opt, todos_hist_val[mejor_idx][ep_opt], color='red', s=80, zorder=5)
    plt.title(f"Modelo {mejor_num} (MEJOR) — Evolucion del Error de Entrenamiento\n"
              f"Train: {info_mejor['Bloques_Train']} | Val: {info_mejor['Bloque_Val']}")
    plt.xlabel('Epocas')
    plt.ylabel('Loss (Error)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('grafica_mejor_modelo_error.png', dpi=150)
    print("  -> Grafica guardada: grafica_mejor_modelo_error.png")
    plt.show(block=False)

    # ---- Graficas individuales por fold ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evolucion del Error por Fold (Validacion Cruzada)', fontsize=14, fontweight='bold')

    for i in range(K_BLOQUES):
        ax = axes[i // 2][i % 2]
        info = modelos_cv[i]
        ep = info['Iter_Optima']

        ax.plot(todos_hist_train[i], label='Entrenamiento', color='steelblue')
        ax.plot(todos_hist_val[i], label='Validacion', color='orange')
        ax.axvline(x=ep, color='red', linestyle='--', alpha=0.7, label=f'Opt: Ep {ep}')
        ax.scatter(ep, todos_hist_val[i][ep], color='red', s=50, zorder=5)

        marcador = " (MEJOR)" if info['Num_Modelo'] == mejor_num else ""
        ax.set_title(f"Modelo {info['Num_Modelo']}{marcador} | Val: {info['Bloque_Val']}")
        ax.set_xlabel('Epocas')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grafica_todos_folds.png', dpi=150)
    print("  -> Grafica guardada: grafica_todos_folds.png")
    plt.show(block=False)

    # ---- Grafica de accuracy (solo clasificacion) ----
    if tipo == 'clasificacion':
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        fig2.suptitle('Evolucion del Accuracy por Fold', fontsize=14, fontweight='bold')

        for i in range(K_BLOQUES):
            ax = axes2[i // 2][i % 2]
            info = modelos_cv[i]

            acc_t = modelos_cv[i]['Modelo'].historial.history['accuracy']
            acc_v = modelos_cv[i]['Modelo'].historial.history['val_accuracy']

            ax.plot(acc_t, label='Acc Entrenamiento', color='steelblue')
            ax.plot(acc_v, label='Acc Validacion', color='orange')

            marcador = " (MEJOR)" if info['Num_Modelo'] == mejor_num else ""
            ax.set_title(f"Modelo {info['Num_Modelo']}{marcador} | Val: {info['Bloque_Val']}")
            ax.set_xlabel('Epocas')
            ax.set_ylabel('Accuracy')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('grafica_accuracy_folds.png', dpi=150)
        print("  -> Grafica guardada: grafica_accuracy_folds.png")
        plt.show(block=False)

        # Grafica accuracy solo del mejor modelo
        plt.figure("Accuracy - Mejor Modelo", figsize=(10, 6))
        acc_t = modelos_cv[mejor_idx]['Modelo'].historial.history['accuracy']
        acc_v = modelos_cv[mejor_idx]['Modelo'].historial.history['val_accuracy']
        plt.plot(acc_t, label='Accuracy Entrenamiento', color='steelblue', linewidth=2)
        plt.plot(acc_v, label='Accuracy Validacion', color='orange', linewidth=2)
        plt.title(f"Modelo {mejor_num} (MEJOR) — Evolucion del Accuracy\n"
                  f"Train: {info_mejor['Bloques_Train']} | Val: {info_mejor['Bloque_Val']}")
        plt.xlabel('Epocas')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('grafica_mejor_modelo_accuracy.png', dpi=150)
        print("  -> Grafica guardada: grafica_mejor_modelo_accuracy.png")
        plt.show(block=False)

    # ---- Resumen Final ----
    print(f"\n{'=' * 60}")
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"  Dataset: {X_raw.shape[0]} observaciones, {X_raw.shape[1]} features")
    print(f"  Tipo: {tipo}")
    print(f"  Red: {len(nombres_feat)} -> {' -> '.join(map(str, CAPAS_OCULTAS))} -> {n_clases if tipo == 'clasificacion' else 1}")
    print(f"  K-Folds: {K_BLOQUES}")
    print(f"  Mejor modelo: Modelo {mejor_num}")
    print(f"  Error total del mejor: {modelos_ordenados[0]['Error_Total']:.6f}")
    if tipo == 'clasificacion':
        print(f"  Accuracy validacion del mejor: {modelos_ordenados[0]['Acc_Val']:.4f}")

    # Mantener graficas abiertas
    plt.show()
