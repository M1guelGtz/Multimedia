import matplotlib.pyplot as plt

calificaciones = [
    78, 23, 7, 3, 4, 68, 77, 49, 45, 50,
    74, 6, 23, 90, 67, 76, 75, 12, 77, 14,
    23, 90, 36, 24, 36, 34, 48, 64, 97, 19,
    64, 96, 65, 23, 7, 74, 34, 45, 33, 28
]

aprobados = [c for c in calificaciones if c >= 60]
reprobados = [c for c in calificaciones if c < 60]
indices_aprobados = [i+1 for i, c in enumerate(calificaciones) if c >= 60]
indices_reprobados = [i+1 for i, c in enumerate(calificaciones) if c < 60]
rangos = ['0-59', '60-69', '70-79', '80-89', '90-100']
conteo_rangos = [
    len([c for c in calificaciones if 0 <= c < 60]),
    len([c for c in calificaciones if 60 <= c < 70]),
    len([c for c in calificaciones if 70 <= c < 80]),
    len([c for c in calificaciones if 80 <= c < 90]),
    len([c for c in calificaciones if 90 <= c <= 100])
]

total = len(calificaciones)
porcentaje_aprobados = (len(aprobados) / total) * 100
porcentaje_reprobados = (len(reprobados) / total) * 100
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
#scatter de calificaciones
ax1.scatter(indices_reprobados, reprobados, color='orange', alpha=0.6, s=100, label='No pasa')
ax1.scatter(indices_aprobados, aprobados, color='blue', alpha=0.6, s=100, label='Pasa (>=60)')
ax1.axhline(y=60, color='gray', linestyle='--', linewidth=1)
ax1.set_xlabel('Alumno')
ax1.set_ylabel('Calificación')
ax1.set_title('Pasa vs No pasa')
ax1.legend()
ax1.grid(True, alpha=0.3)
#pie de pasa/no pasa
colores = ['#5B9BD5', '#ED7D31']
etiquetas = [f'Aprobados: {porcentaje_aprobados:.1f}%', f'No pasa: {porcentaje_reprobados:.1f}%']
ax2.pie([len(aprobados), len(reprobados)], labels=['Pasa', 'No pasa'], 
        autopct='%1.1f%%', colors=colores, startangle=90)
ax2.set_title('Pasa/No pasa')
ax3.bar(rangos, conteo_rangos, color='#5B9BD5', alpha=0.7)
ax3.set_xlabel('Rango')
ax3.set_ylabel('Número de alumnos')
ax3.set_title('Conteo por rango')
ax3.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(conteo_rangos):
    ax3.text(i, v + 0.2, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()
