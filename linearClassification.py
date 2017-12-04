# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt

def loss(X,y,w,b,C):
    m = y.shape[0]
    hinge = sum(list(map(lambda x:max(0,x[0]),(1-y.A*(X*w+b).A))))
    w_2 = sum(w.A**2)[0]
    return (0.5*w_2+C*hinge)/m
    
def gradient(X,y,w,C,b):
    m = y.shape[0]
    dw = np.zeros((X.shape[1],1))
    db = 0
    db = 0
    indicator = 1-y.A*((X*w+b).A)
    for i in range(m):
        if indicator[i]>=0:
            dw += w - C*(y[i]*X[i]).T
            db += -C*y[i]
        else:
            dw += w 
    return [dw,db]

def checkGradient(X,y,w,C,b):
    delta = 1e-6
    dw = (loss(X,y,w+delta,b,C)-loss(X,y,w-delta,b,C))/(np.ones(w.shape)*delta*2)
    db = (loss(X,y,w,b+delta,C)-loss(X,y,w,b-delta,C))/delta*2
    return dw,db
    
def gradientDecent(X,y,w,C,b,alpha,num_rounds,val_x,val_y): 
    m = y.shape[0]
    train_loss_history = []
    val_loss_history = []
    print("origin train loss:%f"%loss(X,y,w,b,C))
    train_loss_history.append(loss(X,y,w,b,C))
    print("origin validation loss:%f"%loss(val_x,val_y,w,b,C))
    val_loss_history.append(loss(val_x,val_y,w,b,C))    
    print("")
    
    for i in range(num_rounds):
        new_w = w - gradient(X,y,w,C,b)[0]*alpha/m
        new_b = b - gradient(X,y,w,C,b)[1]*alpha/m
        w = new_w
        b = new_b
        train_loss_history.append(loss(X,y,w,b,C))
        val_loss_history.append(loss(val_x,val_y,w,b,C))
        
    return w,b,train_loss_history,val_loss_history

def predict(X,y,w,b):
    pred = X*w+b
    pred_y = list(map(lambda x:1 if x[0]>0 else -1,pred.A))
    acc = (y.A1==pred_y).sum()/len(y.A)
    print("acc:%f"%acc)

def train(X,y,val_x,val_y):
    m = X.shape[1]
    init_w = np.matrix(np.zeros(m)).T
    print("begin to train")
    C=1
    b = 1
    alpha = 0.1
    num_rounds=10
    print("C:%f"%C)
    print("learning rate:%f"%alpha)
    print("number of rounds:%d"%num_rounds)
    print("")
    
    #print("check gradient")
    #print(gradient(X,y,init_w,C,b))
    #print(checkGradient(X,y,init_w,C,b))
    #print("")
    
    
    w,b,train_loss_history,val_loss_history = gradientDecent(X,y,init_w,C,b,alpha,num_rounds,val_x,val_y)
    plt.plot(np.arange(num_rounds+1),train_loss_history,label='train loss')
    plt.plot(np.arange(num_rounds+1),val_loss_history,label='validation loss')
    plt.legend(loc=1)
    plt.xlabel('number_of_rounds')
    plt.ylabel('loss')
    return w,b,train_loss_history,val_loss_history
    

def getData():
    X,y = datasets.load_svmlight_file('./australian_scale',n_features=14)
    X = np.matrix(X.toarray())
    y = np.matrix(y).T
    train_x,test_x,train_y,test_y = model_selection.train_test_split(X,y,test_size=0.2)   
    return train_x,test_x,train_y,test_y
    

train_x,test_x,train_y,test_y = getData()
w,b,train_loss,val_loss = train(train_x,train_y,test_x,test_y)
print("final train loss:%f"%train_loss.pop())
print("final validation loss:%f"%val_loss.pop())
print("train:")
predict(train_x,train_y,w,b)
print("test:")
predict(test_x,test_y,w,b)


