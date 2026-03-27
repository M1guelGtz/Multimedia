import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def umbral_otsu(img_gris: np.ndarray) -> int:
    hist = np.bincount(img_gris.ravel(), minlength=256).astype(float)
    prob = hist / hist.sum()

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_total = mu[-1]

    var_entre_clases = ((mu_total * omega - mu) ** 2) / (omega * (1 - omega) + 1e-12)
    return int(np.argmax(var_entre_clases))


def suavizar_gauss_5x5(img: np.ndarray) -> np.ndarray:
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel /= kernel.sum()

    h, w = img.shape
    pad_h = np.pad(img, ((0, 0), (2, 2)), mode="edge")
    tmp = np.zeros_like(img, dtype=np.float32)
    for i, kv in enumerate(kernel):
        tmp += kv * pad_h[:, i : i + w]

    pad_v = np.pad(tmp, ((2, 2), (0, 0)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for i, kv in enumerate(kernel):
        out += kv * pad_v[i : i + h, :]

    return out


def gradiente_sobel(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    p = np.pad(img, 1, mode="edge")

    gx = (
        p[0:h, 2 : w + 2]
        + 2 * p[1 : h + 1, 2 : w + 2]
        + p[2 : h + 2, 2 : w + 2]
        - p[0:h, 0:w]
        - 2 * p[1 : h + 1, 0:w]
        - p[2 : h + 2, 0:w]
    )
    gy = (
        p[2 : h + 2, 0:w]
        + 2 * p[2 : h + 2, 1 : w + 1]
        + p[2 : h + 2, 2 : w + 2]
        - p[0:h, 0:w]
        - 2 * p[0:h, 1 : w + 1]
        - p[0:h, 2 : w + 2]
    )

    return np.hypot(gx, gy)


def detectar_monedas_hough(img_gris: np.ndarray):
    img_suave = suavizar_gauss_5x5(img_gris.astype(np.float32))
    bordes = gradiente_sobel(img_suave)

    umbral_borde = np.percentile(bordes, 90)
    ys, xs = np.where(bordes >= umbral_borde)

    h, w = img_gris.shape
    r_min = max(24, int(min(h, w) * 0.09))
    r_max = max(r_min + 2, int(min(h, w) * 0.14))
    radios = np.arange(r_min, r_max + 1)

    angulos = np.deg2rad(np.arange(0, 360, 10))
    cos_t = np.cos(angulos)
    sin_t = np.sin(angulos)

    acumulador = np.zeros((h, w), dtype=np.uint16)
    for y, x in zip(ys, xs):
        for r in radios:
            cx = np.rint(x - r * cos_t).astype(int)
            cy = np.rint(y - r * sin_t).astype(int)
            m = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
            acumulador[cy[m], cx[m]] += 1

    umbral_votos = max(30, int(acumulador.max() * 0.62))
    candidatos = np.argwhere(acumulador >= umbral_votos)
    if len(candidatos) == 0:
        return [], radios, acumulador

    votos = acumulador[candidatos[:, 0], candidatos[:, 1]]
    orden = np.argsort(votos)[::-1]

    distancia_min = int(r_min * 1.1)
    centros = []
    for idx in orden:
        y, x = candidatos[idx]
        if x < r_min or x >= w - r_min or y < r_min or y >= h - r_min:
            continue

        if all((x - cx) ** 2 + (y - cy) ** 2 >= distancia_min**2 for cy, cx, _, _ in centros):
            centros.append((int(y), int(x), int(votos[idx]), int(np.median(radios))))

    centros = sorted(centros, key=lambda c: (c[0], c[1]))
    return centros, radios, acumulador


def dibujar_resultado(ax, imagen_rgb, centros, titulo):
    ax.imshow(imagen_rgb)

    for i, (y_c, x_c, _, radio) in enumerate(centros, start=1):
        ax.plot(x_c, y_c, "r+", markersize=9, markeredgewidth=2)
        ax.text(
            x_c + 6,
            y_c,
            str(i),
            color="yellow",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1),
        )

        ang = np.deg2rad(np.arange(0, 360, 10))
        circ_x = x_c + radio * np.cos(ang)
        circ_y = y_c + radio * np.sin(ang)
        ax.plot(circ_x, circ_y, color="cyan", linewidth=1)

    ax.set_title(titulo)
    ax.axis("off")


def main():
    ruta_imagen = "./Corte 3/Monedas2 (1).jpg"

    imagen_original = Image.open(ruta_imagen).convert("RGB")
    img_rgb = np.array(imagen_original)
    img_gris = np.array(imagen_original.convert("L"))

    umbral = umbral_otsu(img_gris)
    binaria = img_gris > umbral
    centros, radios, _ = detectar_monedas_hough(img_gris)

    print("=== Etiquetado de componentes conectados ===")
    print(f"Imagen: {ruta_imagen}")
    print(f"Umbral Otsu: {umbral}")
    print(f"Rango de radios evaluado: {radios[0]} a {radios[-1]} px")
    print(f"Monedas detectadas: {len(centros)}")

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Imagen original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(binaria, cmap="gray")
    plt.title(f"Binarizada (Otsu={umbral})")
    plt.axis("off")

    ax = plt.subplot(1, 3, 3)
    dibujar_resultado(
        ax,
        img_rgb,
        centros,
        f"Etiquetas de monedas\nMonedas={len(centros)}",
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()