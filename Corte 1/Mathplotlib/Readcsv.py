#Libreria para manejo de datos y graficas
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 

data = pd.read_csv('./Mathplotlib/datos.csv')

dataC = data[['Cabin_DeckHeight_m', 'Cabin_BowDistance_m', 'Pclass']].dropna()
    