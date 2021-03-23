import numpy as np
import math
import matplotlib.pyplot as plt
from fermulerpy.constants import accuracy
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


class dnn:
    def __init__ (self):
        self.d = None
        self.L = None
        self.J = None
        self.K = None
        self.prev_error = 0.0
        
    def sigmoid(self,t):
        return 1/(1 + np.exp(-1*t))
    def derivative(self,t):
        return t*(1-t)
    def predict(self,X):
        
        lp = []
        tX = X
        for n in range(len(tX)):
            x = []
            x.append(1)
            for p in tX[n]:
                x.append(p)
            x = np.array(x)
            a=[]
            s=[]
            for l in range(0,int(self.L)):
                a.append(np.dot(self.Wl[l],x))
                s.append(self.sigmoid(np.dot(self.Wl[l],x)))
            s.insert(0,1)
            a1=[]
            s1=[]
            for j in range(0,int(self.J)):
                a1.append(np.dot(self.Wh[j],s))
                s1.append(self.sigmoid(np.dot(self.Wh[j],s)))
            s1.insert(0,1)
            ao=[]
            so=[]
                
                
            for k in range(0,self.K):
                ao.append(np.dot(self.Wk[k],s1))
                so.append(self.sigmoid(np.dot(self.Wk[k],s1)))

            lp.append(np.argmax(so) +1)
        return lp
    def train(self,architecture,X,Y,tt,epochs=30,lr=0.1):
        self.d = architecture[0]
        self.L = architecture[1]
        self.J = architecture[2]
        self.K = architecture[3]
        np.random.seed(10)
        self.Wl = np.random.randn(self.L,self.d+1) 
        self.Wh = np.random.randn(self.J,self.L + 1) 
        self.Wk = np.random.randn(self.K,self.J + 1) 
        mse = {}
        l = []
        for epoch in range(epochs):
            lr = ((0.9/epochs)*(epoch)) + 0.1
            Ex = 0
            tX = X
            tY = Y
            #tX = shuffle(X,random_state = epoch)
            #tY = shuffle(Y,random_state = epoch) 
            for n in range(len(tX)):
                
                x = []
                x.append(1)
                for p in tX[n]:
                    x.append(p)
                x = np.array(x)
                a=[]
                s=[]
                for l in range(0,int(self.L)):
                    a.append(np.dot(self.Wl[l],x))
                    s.append(self.sigmoid(np.dot(self.Wl[l],x)))
                s.insert(0,1)
                a1=[]
                s1=[]
                for j in range(0,int(self.J)):
                    a1.append(np.dot(self.Wh[j],s))
                    s1.append(self.sigmoid(np.dot(self.Wh[j],s)))
                s1.insert(0,1)
                ao=[]
                so=[]
                
                
                for k in range(0,self.K):
                    ao.append(np.dot(self.Wk[k],s1))
                    so.append(self.sigmoid(np.dot(self.Wk[k],s1)))
                    
                    Ex = Ex + 0.5*(tY[n][k] - so[k])*(tY[n][k] - so[k])
                

                #print(np.argmax(Y[n]),np.argmax(so))
                #print(Ex)
                #Backpropagation
                #Weight Updation for hidden to output layer
                delta_xk = []
                for k in range(self.K):
                    temp = (tY[n][k] - so[k])*(self.derivative(so[k]))
                    delta_xk.append(temp)
                for j in range(0,self.J + 1):
                    for k in range(self.K):
                        
                        self.Wk[k][j]  = self.Wk[k][j] + lr*delta_xk[k]*s1[j]
                        
                #Weight Updation for hidden to hidden layer
                delta_xj = []
                for j in range(self.J):
                    sigma_term = 0
                    for k in range(self.K):
                        sigma_term = sigma_term + delta_xk[k]*(self.Wk[k][j])
                    for l in range(self.L + 1):
                        temp = sigma_term*(self.derivative(s1[j+1]))
                        delta_xj.append(temp)
                        self.Wh[j][l] = self.Wh[j][l] + lr*temp*s[l]
                
                for l in range(self.L):
                    sigma_term = 0
                    for j in range(self.J):
                        sigma_term = sigma_term + delta_xj[j]*(self.Wh[j][l])
                    for i in range(self.d + 1):
                        temp = sigma_term*(self.derivative(s[l+1]))
                        self.Wl[l][i] = self.Wl[l][i] + lr*temp*x[i]
            
            #print(Ex)
            curr_error = Ex/(len(X))
            mse[epoch] = Ex/len(X)
            print(epoch,mse[epoch])
            if(epoch == 0):
                self.prev_error = curr_error
            else:
                if(epoch == epochs-1 ):
                   
                   break
                else:
                   self.prev_error = curr_error
            
           
        plt.plot(mse.values())
        plt.xlabel("Epoch #")
        plt.ylabel("MSE")
        #plt.ylim([0, 1])
        plt.xticks(np.arange(0,epoch,2))
        plt.show()
        
train_x = np.loadtxt('x_train.txt')
val_x = np.loadtxt('x_val.txt')
test_x = np.loadtxt('x_test.txt')
train_y=[]
val_y = []
test_y = []

for i in range(40):
    train_y.append(1)
for i in range(40):
    train_y.append(2)
for i in range(40):
    train_y.append(3)
for i in range(10):
    val_y.append(1)
for i in range(10):
    val_y.append(2)
for i in range(10):
    val_y.append(3)
for i in range(50):
    test_y.append(1)
for i in range(50):
    test_y.append(2)
for i in range(50):
    test_y.append(3)
train_y = np.array(train_y)
val_y = np.array(val_y)
test_y = np.array(test_y)

t1 = []
for i in train_y:
    if(i==1):
        t1.append([1,0,0])
    elif(i==2):
        t1.append([0,1,0])
    else:
        t1.append([0,0,1])

t1 = np.array(t1)


dnn_model = dnn()
architecture = [32,40,50,3]
dnn_model.train(architecture,train_x,t1,train_y)
print("accuracy score on train data : ",accuracy(train_y,train_x,architecture))
print("accuracy score on val data : ",accuracy(val_y,val_x,architecture))
print("accuracy score on test data : ",accuracy(test_y,test_x,architecture))




































