import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import math

class Perceptron:
  def __init__ (self):
    self.w = None
    self.b = None
    self.yy = None

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
    
  def fit(self, X, Y, epochs = 1000, lr = 0.001):
    self.w = np.ones(X.shape[1])
    self.b = 0
    accuracy = {}
    wt_matrix = []
    b_matrix = []
    for i in range(epochs):
      temp_y = []
      for x, y in zip(X, Y):
        y_pred = 1 if self.model(x) > 0.5 else 0
        temp_y.append(self.model(x))
        self.w = self.w + lr * x * (y-y_pred)
        self.b = self.b + lr * 1 * (y-y_pred)
      
        
          
      wt_matrix.append(self.w) 
      b_matrix.append(self.b)   
      accuracy[i] = accuracy_score(self.predict(X), Y)
      if (accuracy[i] == 1):
        self.yy = temp_y
        break
    plt.plot(accuracy.values())
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy Score")
    plt.ylim([0.8, 1.1])
    plt.show()
    return temp_y,np.array(wt_matrix),np.array(b_matrix)

file1 = open(r"C:\Users\Shikhar Kushwah\Desktop\projects-and-learnings\CS671-Deep-Learning-and-its-Applications\Assignment1\Group10\Classification\LS_Group10\Class1.txt")
file2 = open(r"C:\Users\Shikhar Kushwah\Desktop\projects-and-learnings\CS671-Deep-Learning-and-its-Applications\Assignment1\Group10\Classification\LS_Group10\Class2.txt")
file3 = open(r"C:\Users\Shikhar Kushwah\Desktop\projects-and-learnings\CS671-Deep-Learning-and-its-Applications\Assignment1\Group10\Classification\LS_Group10\Class3.txt")

file1 = file1.read()
file2 = file2.read()
file3 = file3.read()

file1 = file1.split()
file2 = file2.split()
file3 = file3.split()

train_end = int(0.6*len(file1))
val_end = int(0.8*len(file1))
test_end = int(1*len(file1))

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

for i in range(0,train_end,2):
    temp = []
    temp.append(float(file1[i]))
    temp.append(float(file1[i+1]))
    x_train.append(temp)
    y_train.append(int(1))

    temp = []
    temp.append(float(file2[i]))
    temp.append(float(file2[i+1]))
    x_train.append(temp)
    y_train.append(int(2))

    temp = []
    temp.append(float(file3[i]))
    temp.append(float(file3[i+1]))
    x_train.append(temp)
    y_train.append(int(3))

for i in range(train_end,val_end,2):
    temp = []
    temp.append(float(file1[i]))
    temp.append(float(file1[i+1]))
    x_val.append(temp)
    y_val.append(int(1))

    temp = []
    temp.append(float(file2[i]))
    temp.append(float(file2[i+1]))
    x_val.append(temp)
    y_val.append(int(2))

    temp = []
    temp.append(float(file3[i]))
    temp.append(float(file3[i+1]))
    x_val.append(temp)
    y_val.append(int(3))

for i in range(val_end,test_end,2):
    temp = []
    temp.append(float(file1[i]))
    temp.append(float(file1[i+1]))
    x_test.append(temp)
    y_test.append(int(1))

    temp = []
    temp.append(float(file2[i]))
    temp.append(float(file2[i+1]))
    x_test.append(temp)
    y_test.append(int(2))

    temp = []
    temp.append(float(file3[i]))
    temp.append(float(file3[i+1]))
    x_test.append(temp)
    y_test.append(int(3))

x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

# One-against-the-rest-Approch

#class 1 and rest
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

#class 2 and rest
y_train2 = []
for i in y_train:
  if(i == 2):
    y_train2.append(1)
  else:
    y_train2.append(0)
Perceptron2 = Perceptron()
a2 , w2 , b2 = Perceptron2.fit(x_train,y_train2)
#print(len(a2),len(w2),len(b2))
#print(b2)

#class 3 and rest
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
print("Train data confusion matrix : \n", confusion_matrix(y_train,y_val_pred,labels=[1,2,3]))
#testing on validation test
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
print("Confusion Matrix validation data :\n ",confusion_matrix(y_val,y_val_pred,labels=[1,2,3]))

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
print("Confusion Matrix on test data :\n ", confusion_matrix(y_test,y_val_pred,labels=[1,2,3]))
#Decision region diagram

k_factor = 0.5 #this controls how fine are the boundaries of diagram
i = -10.0
while(i<=25):
  j = -15.0
  while(j<=20):
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



