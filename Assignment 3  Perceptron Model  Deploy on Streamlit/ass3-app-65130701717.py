import pickle
import numpy as np
from sklearn.linear_model import Perceptron

model = pickle.load(open('per_model-65130701717.sav','rb'))

x1 = float(input('Enter x1: ')) #4.5
x2 = float(input('Enter x2: ')) #5
x3 = float(input('Enter x3: ')) #1.7
x4 = float(input('Enter x4: ')) #2.3

xnew = np.array([x1,x2,x3,x4]).reshape(1, -1)
y_pred = model.predict(xnew)
print(y_pred[0])