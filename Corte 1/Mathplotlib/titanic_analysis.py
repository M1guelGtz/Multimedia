import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Mathplotlib/titanic_upgrade.csv')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

survival_by_sex = df.groupby('Sex')['Survived'].agg(['sum', 'count'])
survival_by_sex['survival_rate'] = (survival_by_sex['sum'] / survival_by_sex['count'] * 100).round(2)

ax1 = axes[0]
colors = ['#FF6B6B', '#4ECDC4']
labels = [f"{sex.capitalize()}\n{rate:.1f}%" for sex, rate in survival_by_sex['survival_rate'].items()]

wedges, texts, autotexts = ax1.pie(survival_by_sex['survival_rate'], 
                                     labels=labels, 
                                     colors=colors, 
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     textprops={'fontweight': 'bold', 'fontsize': 10},
                                     explode=(0.05, 0.05),
                                     shadow=True)

ax1.set_title('¿Quién tuvo mayor tasa de supervivencia:\nhombres o mujeres?', fontsize=12, fontweight='bold')

girls_not_survived = df[(df['Sex'] == 'female') & 
                        (df['Age'] < 18) & 
                        (df['Survived'] == 0)]

girls_by_class = girls_not_survived['Pclass'].value_counts().sort_index()

ax2 = axes[1]
colors2 = ['#FFD93D', '#FF8C42', '#A23B72']
bars2 = ax2.bar(girls_by_class.index.astype(str), girls_by_class.values, color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Cantidad de niñas no sobrevivientes', fontsize=11, fontweight='bold')
ax2.set_xlabel('Clase de pasaje', fontsize=11, fontweight='bold')
ax2.set_title('¿Cuál fue la clase con más niñas (<18)\nno sobrevivientes?', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold')

survival_by_class = df.groupby('Pclass')['Survived'].agg(['sum', 'count'])
survival_by_class['survival_rate'] = (survival_by_class['sum'] / survival_by_class['count'] * 100).round(2)

ax3 = axes[2]
colors3 = ['#6C5CE7', '#A29BFE', '#81ECEC']
bars3 = ax3.bar(survival_by_class.index.astype(str), survival_by_class['survival_rate'], color=colors3, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Porcentaje de Supervivencia (%)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Clase de pasaje', fontsize=11, fontweight='bold')
ax3.set_title('¿Cuál es el porcentaje de los\nsobrevivientes por clase?', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3)

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()