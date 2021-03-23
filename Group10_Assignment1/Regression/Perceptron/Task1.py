import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

file = pd.read_csv(r"C:\Users\Shikhar Kushwah\Desktop\projects-and-learnings\CS671-Deep-Learning-and-its-Applications\Assignment1\Group10\Regression\UnivariateData\10.csv")
file1= open(r"C:\Users\Shikhar Kushwah\Desktop\projects-and-learnings\CS671-Deep-Learning-and-its-Applications\Assignment1\Group10\Regression\UnivariateData\10.csv")
file1 = file1.read()
file1=file1.split()
x=[]
y=[]
for i in file1:
    i = i.split(",")
    x.append(float(i[0]))
    y.append(float(i[1]))

x = np.array(x)
y = np.array(y)
#print(x.shape)
#print(y.shape)
class Perceptron:
  def __init__ (self):
    self.w = None
    self.b = None
    self.yy = None
    self.prev_error = None

  #def sigmoid(x):
   # return 1/(1+ math.exp(-1*x))
 
  def model(self, x):
    a = np.dot(self.w,x) + self.b
    return a
  
  def predict(self, X):
    Y = []
    for x in X:
      temp = 1 if self.model(x)>0.5 else 0
      Y.append(temp)
    return np.array(Y)

  def average_error(self,y_calc , y_actual):
    sum_error = 0
    for i in range(len(y_actual)):
      sum_error  = sum_error + 0.5*((y_actual[i]-y_calc[i])*(y_actual[i]-y_calc[i]))
    return sum_error/(len(y_actual))  
    
  def fit(self, X, Y, epochs = 1000, lr = 0.01):
    self.w = np.ones(1)
    self.b = 0
    accuracy = {}
    wt_matrix = []
    b_matrix = []
    for i in range(epochs):
      temp_y = []
      for x, y in zip(X, Y):
        y_pred = self.model(x)
        temp_y.append(self.model(x))
        self.w = self.w + lr * x * (y-y_pred)
        self.b = self.b + lr * 1 * (y-y_pred)
      if(i == 0):
          self.prev_error = self.average_error(temp_y,Y)
      else:
          curr_error = self.average_error(temp_y,Y)
          if((self.prev_error - curr_error)<0.00001 ):
              epochs = i+1
              break

          else:
              self.prev_error = curr_error
          
      wt_matrix.append(self.w) 
      b_matrix.append(self.b)   
      #accuracy[i] = accuracy_score(self.predict(X), Y)
      accuracy[i] = self.average_error(temp_y,Y)
      if (accuracy[i] == 1):
        self.yy = temp_y
        break
    plt.plot(accuracy.values())
    plt.xlabel("Epoch #")
    plt.ylabel("MSE")
    #plt.ylim([0, 1])
    plt.xticks(np.arange(0,epochs,1))
    plt.show()
    return temp_y,np.array(wt_matrix),np.array(b_matrix)

x_train = []
x_val = []
x_test = []
y_train = []
y_val = []
y_test = []

for i in range(int(0.6*len(x))):
  x_train.append(x[i])
  y_train.append(y[i])

for i in range(int(0.6*len(x)),int(0.8*len(x))):
  x_val.append(x[i])
  y_val.append(y[i])

for i in range(int(0.8*len(x)),len(x)):
  x_test.append(x[i])
  y_test.append(y[i])

x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

model = Perceptron()
a,b,c = model.fit(x_train,y_train)
def mse(y_calc , y_actual):
    sum_error = 0
    for i in range(len(y_actual)):
      sum_error  = sum_error + 0.5*((y_actual[i]-y_calc[i])*(y_actual[i]-y_calc[i]))
    return sum_error[0]/(len(y_actual))
#checking on train data
y_train_pred = []
for x in x_train:
  y_train_pred.append(np.dot(b[-1],x) + c[-1])

y_train_pred = np.array(y_train_pred)

print("MSE on train  data is : ", mse(y_train_pred,y_train))
#checking on validation data
y_val_pred = []
for x in x_val:
  y_val_pred.append(np.dot(b[-1],x) + c[-1])

y_val_pred = np.array(y_val_pred)



print("MSE on validation data is : ", mse(y_val_pred,y_val))

y_test_pred = []
for x in x_test:
  y_test_pred.append(np.dot(b[-1],x) + c[-1])

y_test_pred = np.array(y_test_pred)

print("MSE on test data is : ", mse(y_test,y_test_pred))

plt.plot(y_train)
plt.plot(y_train_pred)
plt.title("plot of model output and target output for train data")
plt.show()
plt.plot(y_val)
plt.plot(y_val_pred)
plt.title("plot of model output and target output for validation data")
plt.show()
plt.plot(y_test)
plt.plot(y_test_pred)
plt.title("plot of model output and target output for test data")
plt.show()

plt.scatter(y_train,y_train_pred)
plt.title("scatter plot between target output and model output for train data")
plt.show()
plt.scatter(y_val,y_val_pred)
plt.title("scatter plot between target output and model output for validation data")
plt.show()
plt.scatter(y_test,y_test_pred)
plt.title("scatter plot between target output and model output for test data")
plt.show()