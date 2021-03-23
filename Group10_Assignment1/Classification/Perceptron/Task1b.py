import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

file = open(r"C:\Users\Shikhar Kushwah\Desktop\projects-and-learnings\CS671-Deep-Learning-and-its-Applications\Assignment1\Group10\Classification\NLS_Group10.txt")
file = file.read()
file = file.split()

file1 = []
file2 = []
file3 = []

for i in range(18,1018,2):
    temp = []
    temp.append(float(file[i]))
    temp.append(float(file[i+1]))
    file1.append(temp)
file1 = np.array(file1)
#print(file1.shape)

for i in range(1018,2018,2):
    temp = []
    temp.append(float(file[i]))
    temp.append(float(file[i+1]))
    file2.append(temp)
file2 = np.array(file2)
#print(file2.shape)

for i in range(2018,3018,2):
    temp = []
    temp.append(float(file[i]))
    temp.append(float(file[i+1]))
    file3.append(temp)
file3 = np.array(file3)
#print(file3.shape)

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
    return 1/(1+ math.exp(-1*a))
  
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
    
  def fit(self, X, Y, epochs = 1000, lr = 0.001):
    self.w = np.ones(X.shape[1])
    self.b = 0
    accuracy = {}
    wt_matrix = []
    b_matrix = []
    for i in range(epochs):
      temp_y = []
      for x, y in zip(X, Y):
        y_pred = self.model(x)
        temp_y.append(self.model(x))
        self.w = self.w + lr * x * (y-y_pred)*y_pred*(1-y_pred)
        self.b = self.b + lr * 1 * (y-y_pred)*y_pred*(1-y_pred)
      if(i == 0):
          self.prev_error = self.average_error(temp_y,Y)
      else:
          curr_error = self.average_error(temp_y,Y)
          if((self.prev_error - curr_error)<0.0001 ):
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
    plt.ylim([0, 1])
    plt.xticks(np.arange(0,epochs,5))
    plt.show()
    return temp_y,np.array(wt_matrix),np.array(b_matrix)

x_train = []
x_val = []
x_test = []
y_train = []
y_val = []
y_test = []

for i in range(int(0.6*len(file1))):
    x_train.append(file1[i])
    x_train.append(file2[i])
    x_train.append(file3[i])
    y_train.append(int(1))
    y_train.append(int(2))
    y_train.append(int(3))

x_train = np.array(x_train)
y_train = np.array(y_train)
#print(x_train.shape,y_train.shape)

for i in range(int(0.6*len(file1)),int(0.8*len(file1))):
    x_val.append(file1[i])
    x_val.append(file2[i])
    x_val.append(file3[i])
    y_val.append(int(1))
    y_val.append(int(2))
    y_val.append(int(3))

x_val = np.array(x_val)
y_val = np.array(y_val)
#print(x_val.shape,y_val.shape)
#print(y_val)

for i in range(int(0.8*len(file1)),int(1*len(file1))):
    x_test.append(file1[i])
    x_test.append(file2[i])
    x_test.append(file3[i])
    y_test.append(int(1))
    y_test.append(int(2))
    y_test.append(int(3))

x_test = np.array(x_test)
y_test = np.array(y_test)
#print(x_test.shape,y_test.shape)

#One-against-the-rest-approach

#class1 and rest
y_train1 = []
for i in y_train:
  if(i == 1):
    y_train1.append(1)
  else:
    y_train1.append(0)
Perceptron1 = Perceptron()
a1 , w1 , b1 = Perceptron1.fit(x_train,y_train1)
#print(len(a1),len(w1),len(b1))
#print(b1)
#class2 and rest
y_train2 = []
for i in y_train:
  if(i == 2):
    y_train2.append(1)
  else:
    y_train2.append(0)
Perceptron2 = Perceptron()
a2 , w2 , b2 = Perceptron2.fit(x_train,y_train2)
#print(len(a2),len(w2),len(b2))
#print(b1)

#class3 and rest
y_train3 = []
for i in y_train:
  if(i == 3):
    y_train3.append(1)
  else:
    y_train3.append(0)
Perceptron3 = Perceptron()
a3 , w3 , b3 = Perceptron3.fit(x_train,y_train3)
#print(len(a3),len(w3),len(b3))
#print(b3)

def sigmoid(x,w,b):
  t = np.dot(w[-1],x) + b[-1]
  return 1/(1+math.exp(-1*t))

#testing on train data
y_val_pred = []
for i in  x_train:
  temp1 = sigmoid(i,w1,b1)
  temp2 = sigmoid(i,w2,b2)
  temp3 = sigmoid(i,w3,b3)
  if(temp1>temp2 and temp1>temp3):
    y_val_pred.append(1)
  elif(temp2>temp1 and temp2>temp3):
    y_val_pred.append(2)
  else:
    y_val_pred.append(3)

print("Accuracy on train data : ",accuracy_score(y_train , np.array(y_val_pred))*100)
print("Confusion Matrix train data: \n", confusion_matrix(y_train,y_val_pred,labels=[1,2,3]))
#testing on validation data
y_val_pred = []
for i in  x_val:
  temp1 = sigmoid(i,w1,b1)
  temp2 = sigmoid(i,w2,b2)
  temp3 = sigmoid(i,w3,b3)
  if(temp1>temp2 and temp1>temp3):
    y_val_pred.append(1)
  elif(temp2>temp1 and temp2>temp3):
    y_val_pred.append(2)
  else:
    y_val_pred.append(3)

print("Accuracy on validation data : ",accuracy_score(y_val , np.array(y_val_pred))*100)
print("Confusion Matrix on val data : \n",confusion_matrix(y_val,y_val_pred,labels=[1,2,3]))
#esting on test data now
y_val_pred = []
for i in  x_test:
  temp1 = sigmoid(i,w1,b1)
  temp2 = sigmoid(i,w2,b2)
  temp3 = sigmoid(i,w3,b3)
  if(temp1>temp2 and temp1>temp3):
    y_val_pred.append(1)
  elif(temp2>temp1 and temp2>temp3):
    y_val_pred.append(2)
  else:
    y_val_pred.append(3)

print("Accuracy on test data : ",accuracy_score(y_test , np.array(y_val_pred))*100)
print("Confusion Matrix on test data : \n",confusion_matrix(y_test,y_val_pred,labels=[1,2,3]))
#Decision boundary diagram
k_factor = 0.1 #this controls how fine are the boundaries of diagram
i = -5.0
while(i<=5):
  j = -5.0
  while(j<=5):
    temp1 = sigmoid([i,j],w1,b1)
    temp2 = sigmoid([i,j],w2,b2)
    temp3 = sigmoid([i,j],w3,b3)
    if(temp1>temp2 and temp1>temp3):
      plt.scatter(i,j,c='orange')
    elif(temp2>temp1 and temp2>temp3):
      plt.scatter(i,j,c='yellow')
    else:
      plt.scatter(i,j,c='brown')
    j = j + k_factor
  i = i + k_factor

for i in range(0,len(x_train),3):
  plt.scatter(x_train[i][0],x_train[i][1],c='r')
  plt.scatter(x_train[i+1][0],x_train[i+1][1],c='g')
  plt.scatter(x_train[i+2][0],x_train[i+2][1],c='b')
plt.show()