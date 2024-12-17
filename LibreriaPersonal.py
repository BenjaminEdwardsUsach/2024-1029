# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 22:24:26 2024

@author: benja
"""

import math
import numpy as np
from collections import Counter

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
    Lee un archivo de texto con encabezados y organiza los datos en un diccionario.
    Maneja valores con espacios encerrados en comillas dobles y asegura que las columnas
    se alineen correctamente con los encabezados.

    Parameters
    ----------
    nombre : str
        Nombre del archivo a leer.

    Returns
    -------
    dict
        Diccionario con los encabezados como claves y los datos como listas.
        Ejemplo de uso: 
        datos = leerArchivo("archivo.dat")
        kmag = np.array(datos["K"])  # "K" es un encabezado en el archivo.

    Notas
    -----
    - Los valores encerrados entre comillas dobles y con espacios son tratados como una única entrada.
    - Se maneja la posibilidad de líneas con más datos de los que indican los encabezados.
    - Los valores numéricos se convierten a float. Si hay un error o es texto, el valor se conserva como cadena.
    - Los valores "nan" (sin importar mayúsculas) se convierten a np.nan.
    """
    with open(nombre, "r") as file:
        # Leer todas las líneas del archivo
        lineas = file.readlines()  
    
    # Extraer los encabezados desde la primera línea
    encabezados = lineas[0].strip("# \n").split()
    datos = {encabezado: [] for encabezado in encabezados}  # Crear diccionario vacío para almacenar datos
    
    for linea in lineas[1:]:
        # Dividir la línea en palabras considerando espacios y comillas
        elementos = linea.strip().split()
        info_combinada = []  # Almacena los elementos procesados
        i = 0

        # Combinar elementos que están encerrados entre comillas dobles
        while i < len(elementos):
            if elementos[i].startswith('"') and not elementos[i].endswith('"'):
                # Combina elementos hasta cerrar las comillas
                j = i + 1
                while j < len(elementos) and not elementos[j].endswith('"'):
                    j += 1
                combinado = " ".join(elementos[i:j+1]).replace('"', '')  # Remover comillas
                info_combinada.append(combinado)
                i = j + 1
            else:
                # Elemento individual sin comillas
                info_combinada.append(elementos[i])
                i += 1

        # Ajustar la lista para evitar exceder el número de encabezados
        if len(info_combinada) > len(encabezados):
            info_combinada = info_combinada[:len(encabezados)]

        # Añadir cada valor a la columna correspondiente en el diccionario
        for i, valor in enumerate(info_combinada):
            try:
                # Convertir a float si es posible; manejar "nan" como np.nan
                datos[encabezados[i]].append(float(valor) if valor.lower() != "nan" else np.nan)
            except ValueError:
                # Si no es numérico, almacenar como texto
                datos[encabezados[i]].append(valor)

    return datos

def rango(lista):
  """
  Función que calcula el rango de una lista
  -----
  Parámetros:
  lista: lista a la que calcular el rango de
  -----
  Retorna: el valor del rango
  """
  lista.sort()
  return lista[-1] - lista[0]



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


#///////////////////////////////////////////////////////////////////////////////

def r_squared(y,theta,X):
    y_prom=np.mean(y)
    suma_total_cuadrados=sum([v**2 for v in y-y_prom])
    #suma de errores cuadraticos
    sum_sqerrors=sum([(theta[0]*xv[0]+theta[1]*xv[1]+theta[2]*xv[2]-yv)**2 for xv,yv in zip(X,y)])
    sum_errors=sum([(theta[0]*xv[0]+theta[1]*xv[1]+theta[2]*xv[2]-yv) for xv,yv in zip(X,y)])
    #calculo de R-squared
    r=1-sum_errors/suma_total_cuadrados
    r_squared=1-sum_sqerrors/suma_total_cuadrados
    print("R-squared = ",r_squared)
    print("R = ", r)


def mse(x,y,theta):
  """
  Función que calcula el minimo error cuadrado de(...)
  """
  m,b = theta
  res = [(y_i - (m*x_i+b)**2) for x_i, y_i in zip(x,y)]
  mse = sum(res) / len(x)
  return mse

def lim_de_cuo(x, y, f, v, i, h=0.0001):
    """
    límimte de cuociente

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.
    h : TYPE, optional
        DESCRIPTION. The default is 0.0001.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    w = [v_j + (h if j==i else 0) for j,v_j in enumerate(v)]
    return (f(x,y,w) - f(x,y,v)) / h

def estimate_gradient(x, y, f, v, h = 0.0001):
    """
    estimado de la gradiente

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    h : TYPE, optional
        DESCRIPTION. The default is 0.0001.

    Returns
    -------
    list
        DESCRIPTION.

    """
    return [lim_de_cuo(x, y, f, v, j) for j in range(len(v))]

def paso_en_gradiente(v, gradiente, step_size):
    """
        Paso de la gradiente

    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    gradiente : TYPE
        DESCRIPTION.
    step_size : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    step = [step_size * g_i for g_i in gradiente]
    return [a + b for a, b in zip(v, step)]

def gradiente_mse(x, y, theta):
    """
    minimmos cuadradros de la gradiente

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    pendiente, intercepto = theta
    y_pred = [pendiente * xv + intercepto for xv in x]

    g1 = 2 / len(x) + sum([(y_p - y_d) * x_d for x_d, y_d, y_p in zip(x, y_pred, y)])
    g2 = 2 / len(x) + sum([(y_p - y_d) for x_d, y_d, y_p in zip(x, y, y_pred)])

    return [g1, g2]

def gradiente_mse_pol(x,y,theta):
    #calcuar predicicones del modelo con parametros actuales (input)
    y_pred=[]
    for xa in x:
        yp=sum([t*xv for xv,t in zip(xa,theta)])
        y_pred.append(yp)
    #calcular derivada parciales para atributo 
    derivadas=[]
    for i in range(len(x[0])):
        g=2/len(y)*sum([(y_p-y_d)*x_d for x_d,y_d,y_p in zip(x[:,i],y,y_pred)])
        derivadas.append(g)
    return derivadas

def derivada(f,x,h = 0.0001):
  return (f(x+h)-f(x))/h


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

def FYD(lista):
    """
    Calcula el ancho de los bines con la regla de Freeman & Diaconi

    Parameters
    ----------
    lista : List
        Lista a la cual se le vana calcular el ancho de los bines.

    Returns
    -------
    float
        ancho de los bines.

    """
    return 2*iqr(lista)*len(lista)**(-1/3)

def modas(lista):
  """
  Función que calcula la moda general de una lista de datos, o sea, 
  la cantidad de cada valor dentro de la lista.
  -------
  Parámetros:
  lista: lista de datos a calcluar las modas
  -------
  Retorna: todos los datos y sus cantidades respectiva en la lista como diccionario
  """
  dic = {}
  for elemento in lista:
    if elemento in dic:
      dic[elemento] += 1
    else:
      dic[elemento] = 1
  return dic

def media_aritmetica(lista):
  """
  Código para calcular el promedio de una lista de números
  ----------
  parámetros:
    lista: lista de números
    mitad_list: mitad de la lista
    mediana: numerador de la fracción de la mediana
  ----------
  retorna: mediana de la lista
  """
  mitad_list = len(lista)//2
  lista.sort()
  if len(lista)%2 == 0:
    mediana = lista[mitad_list] + lista[mitad_list-1]
    return mediana/2
  else:
    mediana = lista[mitad_list]
  return mediana




# ////////////////////////////////////////////////////
# Procesamiento de archivos y generación de histogramas
# ////////////////////////////////////////////////////

class estadistica:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.sumatoriaG = np.nansum((self.x - np.nanmean(self.x)) * (self.y - np.nanmean(self.y)))
    
    def covalenciaT(self):
        n = len(self.x)
        sumatoriaG = np.nansum((self.x - np.nanmean(self.x)) * (self.y - np.nanmean(self.y)))
        final_covarianceT = sumatoriaG / (n-1) 
        
        return final_covarianceT, sumatoriaG
    
    def covalenciaF(self):
        n = len(self.x)
        final_covarianceF = self.sumatoriaG / n
        
        return final_covarianceF
    
    def correlacion(self):
        numerator = self.sumatoriaG
        denominator = (np.nansum((self.x - np.nanmean(self.x))**2)**0.5) * (np.nansum((self.y - np.nanmean(self.y))**2)**0.5)
        final_corre = numerator / denominator
        return final_corre
