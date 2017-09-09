import pandas as pd 
import scipy.io as sio
import scipy.signal as ss
import matplotlib.pyplot as plt
import numpy as np
import functools
import operator
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns

def flip(func):
    @functools.wraps(func)
    def newfunc(x, y):
      return func(y, x)
    return newfunc

def foldr(func, acc, xs):
  return functools.reduce(flip(func), reversed(xs), acc)

a = sio.loadmat("P01.mat")
a = a["data"]

def convertToDataFrame(data):
   epochs, electrodos, tiempos =  data.shape
   data = data.reshape((electrodos*epochs, tiempos))

   l1 = []
   l2 = []
   datos = []
   for x in range(0, electrodos) : 
       for y in range(0, epochs) : 
           l1.append(x)
           l2.append(y)

   l =  [l1,l2]
   tuples = list(zip(*l))

   index = pd.MultiIndex.from_tuples(tuples, names=['electrodos', 'epochs'])

   return pd.DataFrame(data, index=index)


def plotear_a1(a):
  electrodos = [7, 43, 79, 130, 184]
  #electrodos = [7]
  potencias = []
  intensidades = None
  seteado = False
  f = lambda x , y : x + y
  for electrodo in electrodos : 
    epoch = 0
    fs = []
    ps = []
    while epoch < 894 : 
      frecuencia, p = ss.welch(a.loc[electrodo, epoch], fs = 250)
      ps.append(p)
      epoch = epoch + 1
      #print epoch 

    intensity = ps
    #convert intensity (list of lists) to a numpy array for plotting
    intensity = np.matrix(intensity)
    intensity = np.array(intensity.T)
    if seteado  : 
      intensidades = (intensidades + intensity) / 2
    else : 
      intensidades = intensity
      seteado = True

  x = range(0, 894)
  y = frecuencia


  #now just plug the data into pcolormesh, it's that easy!
  plt.pcolormesh(x, y, intensidades)
  plt.colorbar() #need a colorbar to show the intensity scale
  plt.show() #boom

def plotear_a2(a):

  i = 0
  electrodo = 0
  while electrodo < 256 :
    print electrodo 
    i = 0 
    ps = []
    while i < 894 : 
      frecuencia , p =  ss.welch(a.loc[electrodo,i], fs = 250) 
      ps.append(p)
      i = i + 1


    ps = np.asarray(ps)
    longitud_potencias = len(ps[0])
    l = [0] * longitud_potencias
    f = lambda x , y : x + y
    ps = foldr(f, l, ps) / longitud_potencias    
    plt.plot(frecuencia , ps) 

    electrodo = electrodo + 1
  
  plt.show()    



a = convertToDataFrame(a)
plotear_a1(a)
#plotear_a2(a)

