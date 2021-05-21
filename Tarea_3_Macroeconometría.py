# -*- coding: utf-8 -*-
"""
Created on Sun May 16 09:47:46 2021

@author: David Mora Salazar
"""
#Pregunta 5
from bccr import SW
from bccr import GUI
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import xlrd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import norm
from bccr import GUI
GUI()
IMAE = SW(IMAE=35553)
SW.quien(35553)
#a
tasa_int = (100*np.log(IMAE).diff(12)).rename(columns= {'IMAE':'Interanual'})
tasa_anual= (100*np.log(IMAE).diff()*12).rename(columns= {'IMAE':'Anualizada'})

tasa_int['Anualizada'] = tasa_anual
fecha = tasa_int['2018':'2020']

def figura1(dato1, dato2, leyenda1, leyenda2, titulo, y):
    variable_de_apoyo1 = dato1
    variable_de_apoyo2 = dato2
    IPCrespaldo = fecha.copy()
    IPCrespaldo['variable_de_apoyo1'] =variable_de_apoyo1
    IPCrespaldo['variable_de_apoyo2'] =variable_de_apoyo2
    IPCrespaldo['variable_de_apoyo1'].plot(color="green", label=leyenda1)
    IPCrespaldo['variable_de_apoyo2'].plot( color = "red", label=leyenda2)
    plt.rcParams["figure.figsize"] = (15,15)
    plt.legend(fontsize=10)
    plt.title(titulo, fontsize=20)
    plt.xlabel(xlabel= "Año",fontsize=15)
    plt.ylabel(ylabel= y,fontsize=15)
    plt.show()        
    return
figura1(fecha['Interanual'], fecha['Anualizada'], "Tasa de crecimiento interanual del PIB de Costa Rica", 'Tasa de crecimiento mensual anualizada del PIB de Costa Rica',"Tasa de crecimiento interanual y la tasa de crecimiento mensual anualizada",'por ciento')   
#c
#Creación del criterio de selección
#Akaike
anualizada_2019 =tasa_anual[:'2019'].dropna()
pmax = 4
qmax = 4
P = np.arange(pmax+1)
Q = np.arange(qmax+1)
aic = [[ARIMA(anualizada_2019, order=[p,0,q]).fit().aic for q in Q ] for p in P ]
AIC = pd.DataFrame(aic, index=[f'p={p}' for p in P], columns=[f'q={q}' for q in Q])
AIC
# Recomienda p=4 y q=4
#Bayesiano
bic = [[ARIMA(anualizada_2019, order=[p,0,q]).fit().bic for q in Q ] for p in P ]
BIC = pd.DataFrame(bic, index=[f'p={p}' for p in P], columns=[f'q={q}' for q in Q])
BIC
#Recomienda p=1 y q=3
#En este caso, los criterios no coinciden, por lo que se recomienda el de Bayes.
#Estimación con p=1 y q=3
res = ARIMA(anualizada_2019, order=[1,0,3]).fit()
res.summary()
#d
horizon = 12 
ff= res.forecast(steps=horizon, alpha=0.1)#alpha de significancia
std = res.forecast(steps=horizon, alpha=0.1)#alpha de significancia
conf = res.forecast(steps=horizon, alpha=0.1)#alpha de significancia
alpha = np.arange(1,6)/10
zvalues = norm(0, 1).isf(np.array(alpha)/2)
# Datos pronosticados
fcast = pd.DataFrame({'Anualizada pronosticada':ff,'std':std}, index=pd.period_range(anualizada_2019.index[-1]+1, periods=horizon, freq='M'))

# Concatenar los datos observados con los pronosticados
fcast2 = pd.concat([anualizada_2019,fcast], sort=False)
fcast2['$\mu$'] = anualizada_2019.values.mean()

# Graficar la serie y el pronóstico
fig, ax =plt.subplots(figsize=[12,4]) 
fcast2.loc['2007-01':'2020-12'][["Anualizada",'Anualizada pronosticada','$\mu$']].plot(ax=ax)
plt.title("Crecimiento pronosticado del IMAE tendencia-ciclo", fontsize=20)
plt.xlabel(xlabel= "Año",fontsize=15)
plt.ylabel(ylabel= "Crecimiento pronosticado del IMAE",fontsize=15)

 

def intervalo(z):
    """
    Para calcular los límites superior e inferior del intervalo de confianza,
    dado el valor crítico de la distribución normal
    """
    return fcast2['Anualizada pronosticada']+z*fcast2['std'],  fcast2['Anualizada pronosticada']-z*fcast2['std']

# fechas para graficar los intervalos
d = fcast2.index.values

# Graficar los intervalos de confianza
for z in zvalues:
    ax.fill_between(d, *intervalo(z), facecolor='blue', alpha=0.12, interpolate=True) #alpha gráfico
#e
figura1(fcast2.loc['2020-01':'2020-11'][['Anualizada pronosticada']], fecha.loc['2020-01':'2020-11']["Anualizada"], "Pronóstico de la tasa de crecimiento interanual del PIB", 'Tasa de crecimiento mensual anualizada del PIB de Costa Rica',"Pronóstico de la tasa de crecimiento para 2020 y la verdadera tasa de crecimiento mensual anualizada",'por ciento')   
#Errores del pronóstico
datos_pronosticados =fcast2.loc['2020-01':'2020-11'][['Anualizada pronosticada']]
datos_observados = fecha.loc['2020-01':'2020-11']["Anualizada"]
datos_pronosticados = pd.DataFrame(datos_pronosticados)
datos_observados = pd.DataFrame(datos_observados)
error =datos_observados["Anualizada"] - datos_pronosticados["Anualizada pronosticada"]
error.plot(kind='line',x='Fecha',y='Errores',color='red')
plt.title("Errores del pronóstico de la tasa de crecimiento del IMAE para el año 2020")
plt.show()



    





