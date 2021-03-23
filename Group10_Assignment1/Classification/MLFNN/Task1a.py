import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
class dnn:
    def __init__ (self):
        self.d = None
        self.J = None
        self.K = None
        self.prev_error = 0.0
        
    def sigmoid(self,t):
        return 1/(1 + np.exp(-1*t))
    def derivative(self,t):
        return t*(1-t)
    def predict(self,X):
        y = []
        for n in range(len(X)):
            x = []
            x.append(1)
            for i in X[n]:
                x.append(i)
            a=[]
            s=[]
            for j in range(0,int(self.J)):
                a.append(np.dot(self.Wh[j],x))
                s.append(self.sigmoid(np.dot(self.Wh[j],x)))
            ao=[]
            so=[]
            s.insert(0,1)
            for k in range(0,self.K):
                ao.append(np.dot(self.Wk[k],s))
                so.append(self.sigmoid(np.dot(self.Wk[k],s)))
            #print(so)
            y.append(np.argmax(so)+1)
        return np.array(y)
            
    def train(self,architecture,X,Y,x_val,y_val,epochs=1000,lr=0.01):
        self.d = architecture[0]
        self.J = architecture[1]
        self.K = architecture[2]
        np.random.seed(1)
        self.Wh = np.random.randn(self.J,self.d + 1) * 0.01
        self.Wk = np.random.randn(self.K,self.J + 1) *0.01
        mse = {}
        hidden_outputs = []
        output_outputs = []
        tX = X
        tY = Y
        for epoch in range(epochs):
            #print(self.Wh[0][1])
            #print("epoch : ",epoch)
            #print(self.Wh)
            #print(self.Wk)
            Ex = 0
            tX = shuffle(X,random_state = epoch)
            tY = shuffle(Y,random_state = epoch)
            hidden_outputs = []
            output_outputs = []
            for n in range(len(X)):
                x = []
                x.append(1)
                for p in tX[n]:
                    x.append(p)
                x = np.array(x)
                a=[]
                s=[]
                for j in range(0,int(self.J)):
                    a.append(np.dot(self.Wh[j],x))
                    s.append(self.sigmoid(np.dot(self.Wh[j],x)))
                hidden_outputs.append(s)
                ao=[]
                so=[]
                s.insert(0,1)
                for k in range(0,self.K):
                    ao.append(np.dot(self.Wk[k],s))
                    so.append(self.sigmoid(np.dot(self.Wk[k],s)))
                    
                    Ex = Ex + 0.5*(tY[n][k] - so[k])*(tY[n][k] - so[k])
                output_outputs.append(so)
                #Backpropagation
                #Weight Updation for hidden to output layer
                delta_xk = []
                for k in range(self.K):
                    temp = (tY[n][k] - so[k])*(self.derivative(so[k]))
                    delta_xk.append(temp)
                for j in range(0,self.J + 1):
                    for k in range(self.K):
                        
                        self.Wk[k][j]  = self.Wk[k][j] + lr*delta_xk[k]*s[j]
                        
                #Weight Updation for input to hidden layer
                for j in range(self.J):
                    sigma_term = 0
                    for k in range(self.K):
                        sigma_term = sigma_term + delta_xk[k]*(self.Wk[k][j])
                    for i in range(self.d + 1):
                        delta_xj = sigma_term*(self.derivative(s[j+1]))
                        self.Wh[j][i] = self.Wh[j][i] + lr*delta_xj*x[i]
            #print(Ex)
            curr_error = Ex/(len(X))
            if(epoch == 0):
                self.prev_error = curr_error
            else:
                if(self.prev_error - curr_error < 0.001  ):
                   break
                else:
                   self.prev_error = curr_error
            mse[epoch] = Ex/len(X)
            print(epoch,mse[epoch])
            
        plt.plot(mse.values())
        plt.xlabel("Epoch #")
        plt.ylabel("MSE")
        #plt.ylim([0, 1])
        plt.xticks(np.arange(0,epoch,1))
        plt.show()
        #print(len(hidden_outputs),hidden_outputs[0])
        #print(len(output_outputs),output_outputs[0])
        #plotting outputs of hidden layer
        for i in range(self.J):
            fig = plt.figure(figsize=(10,7))
            ax = plt.axes(projection = '3d')
            x = []
            y = []
            z = []
            for p in tX:
                x.append(p[0])
                y.append(p[1])
            for p in hidden_outputs:
                z.append(p[i+1])
            ax.scatter3D(x,y,z)
            title = "Hidden Neuron : " + str(i+1)
            plt.title(title)
            plt.show()
        #plotting of output nodes output
        for i in range(self.K):
            fig = plt.figure(figsize=(10,7))
            ax = plt.axes(projection = '3d')
            x = []
            y = []
            z = []
            for p in tX:
                x.append(p[0])
                y.append(p[1])
            for p in output_outputs:
                z.append(p[i])
            ax.scatter3D(x,y,z)
            title = "Output Neuron : " + str(i+1)
            plt.title(title)
            plt.show()
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


u = []
for i in range(0,len(y_train),3):
    u.append([1,0,0])
    u.append([0,1,0])
    u.append([0,0,1])


model = dnn()
model.train([2,5,3],x_train,u,x_val,y_val) 

#testing on train data
y_pred_train =  model.predict(x_train)    
print("ACcuracy score on train data : ",accuracy_score(y_train,y_pred_train)*100)
print("Confusion Matrix train data \n",confusion_matrix(y_train,y_pred_train,labels=[1,2,3]))
#testing on val data
y_pred_val =  model.predict(x_val)    
print("ACcuracy score on val data : ",accuracy_score(y_val,y_pred_val)*100)
print("Confusion Matrix val data \n",confusion_matrix(y_val,y_pred_val,labels=[1,2,3]))
#testing on test data
y_pred_test =  model.predict(x_test)    
print("ACcuracy score on test data : ",accuracy_score(y_test,y_pred_test)*100)
print("Confusion Matrix test data \n",confusion_matrix(y_test,y_pred_test,labels=[1,2,3]))
k_factor = 0.5 #this controls how fine are the boundaries of diagram
i = -10.0
while(i<=25):
  j = -15.0
  while(j<=20):
    a = []
    b = []
    b.append(i)
    b.append(j)
    a.append(b)
    pred_value = (model.predict(a))[0]
    if(pred_value == 1):
        plt.scatter(i,j,c='yellow')
    elif(pred_value == 2):
        plt.scatter(i,j,c='orange')
    else:
        plt.scatter(i,j,c='brown')
    j = j + k_factor
  i = i + k_factor

for i in range(0,len(x_train),3):
  plt.scatter(x_train[i][0],x_train[i][1],c='r')
  plt.scatter(x_train[i+1][0],x_train[i+1][1],c='g')
  plt.scatter(x_train[i+2][0],x_train[i+2][1],c='b')
plt.show()