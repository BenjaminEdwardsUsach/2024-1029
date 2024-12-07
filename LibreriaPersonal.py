# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 22:24:26 2024

@author: benja
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib.ticker import AutoMinorLocator
from scipy import stats as st

def promedio(lista):
    """
    Calcula el promedio de una lista de números.

    Parameters
    ----------
    lista : list
        Lista de números a promediar.

    Returns
    -------
    float
        Promedio aritmético de los elementos en la lista o None si la lista está vacía.

    """
    return sum(lista) / len(lista)

def leerArchivo(nombre):
    """
    Lee un archivo de texto con encabezados y organiza los datos en un diccionario,
    donde cada clave es un encabezado y cada valor es una lista de datos de esa columna.

    Parameters
    ----------
    nombre : str
        Nombre del archivo a leer.

    Returns
    -------
    dict
        Diccionario con los encabezados como claves y los datos como listas.
        ejemplo de uso: kmag = np.array(datos["K"]).
        "K" Es el nombre del encabezado a buscar
    """
    with open(nombre, "r") as file:
        lineas = file.readlines()  
    
    encabezados = lineas[0].strip("# \n").split()
    datos = {encabezado: [] for encabezado in encabezados}
    
    for linea in lineas[1:]:
        info = linea.strip().split()
        for i, valor in enumerate(info):
            print(valor)
            try:
                datos[encabezados[i]].append(float(valor) if valor != "nan" else np.nan)
            except ValueError:
                datos[encabezados[i]].append(valor)
    return datos#añadir al final los sobrantes

def mediana(lista):
    """
    Calcula la mediana de una lista de números.

    Parameters
    ----------
    lista : list
        Lista de números.

    Returns
    -------
    float
        La mediana de la lista.

    """
    lista_ordenada = sorted(lista)  # Ordena la lista sin modificar la original
    largo = len(lista_ordenada)

    if largo % 2 == 0:
        # Si la lista tiene un número par de elementos, promedia los dos del medio
        mid1 = largo // 2 - 1
        mid2 = largo // 2
        return (lista_ordenada[mid1] + lista_ordenada[mid2]) / 2
    else:
        # Si la lista tiene un número impar de elementos, devuelve el del medio
        mid = largo // 2
        return lista_ordenada[mid]

def moda(lista):
    """
    Calcula la moda de una lista de números.

    Parameters
    ----------
    lista : list
        Lista de números.

    Returns
    -------
    int/float
        La moda de la lista.

    """
    conteo = Counter(lista)
    return max(conteo, key=conteo.get)  # Devuelve el elemento más frecuente

def factorial(n):
    """
    Calcula el factorial de un número entero.

    Parameters
    ----------
    n : int
        Número entero no negativo.

    Returns
    -------
    int
        Factorial de n.

    """
    resultado = 1
    for i in range(2, n + 1):
        resultado *= i
    return resultado

def serieDeTaylor(x, n):
    """
    Calcula la serie de Taylor para e**x hasta el n-ésimo término.

    Parameters
    ----------
    x : float
        El valor en el que se evalúa la serie de Taylor.
    n : int
        Número de términos de la serie de Taylor.

    Returns
    -------
    float
        Aproximación de e**x usando n términos de la serie de Taylor.

    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("El número de términos n debe ser un entero no negativo.")

    resultado = 0
    for i in range(n + 1):
        resultado += (x**i) / factorial(i)
    return resultado

def varianza(lista):
    """
    Calcula la varianza de una lista de números.

    Parameters
    ----------
    lista : list
        Lista de números.

    Returns
    -------
    float
        Varianza de los datos en la lista.

    """
    media = promedio(lista)
    return sum((x - media) ** 2 for x in lista) / len(lista)

def desvi(lista):
    """
    Calcula la desviación estándar de una lista de números.

    Parameters
    ----------
    lista : list
        Lista de números.

    Returns
    -------
    float
        Desviación estándar.

    """
    return math.sqrt(varianza(lista))

def MAD(datos):
    """
    Calcula la desviación absoluta media (MAD) de una lista de datos.

    Parameters
    ----------
    datos : list
        Lista de datos numéricos.

    Returns
    -------
    float
        Desviación absoluta media.

    """
    datos_filtrados = [x for x in datos if math.isfinite(x)]
    if not datos_filtrados:
        return None

    media = promedio(datos_filtrados)
    desviaciones_absolutas = [abs(x - media) for x in datos_filtrados]
    return sum(desviaciones_absolutas) / len(desviaciones_absolutas)

def percentil(vals_in, q):
    """
    Calcula el percentil q de una lista de datos.

    Parameters
    ----------
    vals_in : list
        Lista de datos numéricos.
    q : float
        Percentil a calcular (0-100).

    Returns
    -------
    float
        Valor del percentil.

    """
    vals = [v for v in vals_in if math.isfinite(v)]
    if not vals:
        return None

    vals.sort()
    pos = (len(vals) - 1) * (q / 100.0)
    base = int(pos)
    resto = pos - base

    if base + 1 < len(vals):
        return vals[base] + resto * (vals[base + 1] - vals[base])
    else:
        return vals[base]

def iqr(vals_in):
    """
    Calcula el rango intercuartil (IQR) de una lista de datos.

    Parameters
    ----------
    vals_in : list
        Lista de datos numéricos.

    Returns
    -------
    float
        Rango intercuartil.

    """
    q75 = percentil(vals_in, 75)
    q25 = percentil(vals_in, 25)
    if q75 is not None and q25 is not None:
        return q75 - q25
    return None



# ////////////////////////////////////////////////////
# Funciones matemáticas y estadísticas revisadas arriba.
# ////////////////////////////////////////////////////

def strugels(lista):
    """
    Calcula el número óptimo de bins para un histograma 
    usando la regla de Sturges.

    Parameters
    ----------
    lista : list
        Lista de datos numéricos.

    Returns
    -------
    float
        Número óptimo de bins.
    """
    return np.log2(len(lista)) + 1

def scot(lista):
    """
    Calcula el ancho de bin óptimo para un histograma 
    usando la regla de Scott.

    Parameters
    ----------
    lista : list
        Lista de datos numéricos.

    Returns
    -------
    float
        Ancho de bin óptimo.
    """
    return 3.49 * desvi(lista) * len(lista) ** (-1 / 3)

# ////////////////////////////////////////////////////
# Procesamiento de archivos y generación de histogramas
# ////////////////////////////////////////////////////

class estadistica:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
    
    def covalenciaT(self):
        n = len(self.x)
        sumatoriaG = np.nansum((self.x - np.nanmean(self.x)) * (self.y - np.nanmean(self.y)))
        final_covarianceT = sumatoriaG / (n-1) 
        
        return final_covarianceT, sumatoriaG
    
    def covalenciaF(self):
        n = len(self.x)
        sumatoriaG = np.nansum((self.x - np.nanmean(self.x)) * (self.y - np.nanmean(self.y)))
        final_covarianceF = sumatoriaG / n
        
        return final_covarianceF, sumatoriaG
    
    def correlacion(self, sumatoriaG):
        numerator = sumatoriaG
        denominator = (np.nansum((self.x - np.nanmean(self.x))**2)**0.5) * (np.nansum((self.y - np.nanmean(self.y))**2)**0.5)
        final_corre = numerator / denominator
        return final_corre