import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

#file = pd.read_csv(r"C:\Users\Shikhar Kushwah\Desktop\projects-and-learnings\CS671-Deep-Learning-and-its-Applications\Assignment1\Group10\Regression\UnivariateData\10.csv")
file1= open(r"C:\Users\Shikhar Kushwah\Desktop\projects-and-learnings\CS671-Deep-Learning-and-its-Applications\Assignment1\Group10\Regression\BivariateData\10.csv")
file1 = file1.read()
file1=file1.split()
x=[]
y=[]
for i in file1:
    i = i.split(",")
    temp = []
    temp.append(float(i[0]))
    temp.append(float(i[1]))
    x.append(temp)
    y.append(float(i[2]))

x = np.array(x)
y = np.array(y)
class dnn:
    def __init__ (self):
        self.d = None
        self.J = None
        self.K = None
        self.prev_error = 0.0
        
    def sigmoid(self,t):
        return t
    def derivative(self,t):
        return 1
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
            y.append(so[0])
        return np.array(y)
            
    def train(self,architecture,X,Y,epochs=1000,lr=0.01):
        self.d = architecture[0]
        self.J = architecture[1]
        self.K = architecture[2]
        np.random.seed(1)
        self.Wh = np.random.randn(self.J,self.d + 1) * 0.01
        self.Wk = np.random.randn(self.K,self.J + 1) *0.01
        mse = {}
        l = []
        for epoch in range(epochs):
            #print(self.Wh[0][1])
            #print("epoch : ",epoch)
            #print(self.Wh)
            #print(self.Wk)
            Ex = 0
            tX = shuffle(X,random_state = epoch)
            tY = shuffle(Y,random_state = epoch)
            l = []
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
                    output_outputs.append(self.sigmoid(np.dot(self.Wk[k],s)))
                    Ex = Ex + 0.5*(tY[n] - so[k])*(tY[n] - so[k])
                
                #print(np.argmax(Y[n]),np.argmax(so))
                #print(Ex)
                #Backpropagation
                #Weight Updation for hidden to output layer
                delta_xk = []
                for k in range(self.K):
                    temp = (tY[n] - so[k])*(self.derivative(so[k]))
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
            
            for h in range(self.J):
                tt = []
                for p in hidden_outputs:
                    tt.append(p[h+1])
                plt.plot(tt)
                title = "Output of hidden neuron : " + str(h+1) + "after epoch = " + str(epoch+1)
                plt.title(title)
                plt.show()
            plt.plot(output_outputs)
            title = "Output of output node after epoch = " + str(epoch + 1)
            plt.title(title)
            plt.show()
            curr_error = Ex/(len(X))
            if(epoch == 0):
                self.prev_error = curr_error
            else:
                if(self.prev_error - curr_error < 0.001):
                   
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

model = dnn()
model.train([2,2,1],x_train,y_train)

y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
y_test_pred  = model.predict(x_test)
def mse(y_calc , y_actual):
    sum_error = 0
    for i in range(len(y_actual)):
      sum_error  = sum_error + 0.5*((y_actual[i]-y_calc[i])*(y_actual[i]-y_calc[i]))
    return sum_error/(len(y_actual))

print("MSE on train  data is : ", mse(y_train_pred,y_train))
print("MSE on validation data is : ", mse(y_val_pred,y_val))
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


