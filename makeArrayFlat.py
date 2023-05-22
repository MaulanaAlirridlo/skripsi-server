import numpy as np

def flatArray(array):
  newArray = []
  for i,v in enumerate(array) :
    newData = [item for sublist in v for item in ([sublist] if (isinstance(sublist, int) or isinstance(sublist, float)) else sublist)]
    newArray.append(newData)
  newArray = np.array(newArray, dtype=object)
  return newArray