
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from joblib import Parallel, delayed
import time
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score


def training(X,Y):
    model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    model.fit(X, Y)
    return model

def trainingS(X,Y):
    model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    model.fit(X, Y)
    return model

def trainingSparbest(X,Y):
    model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    model.fit(X, Y)
    return model

def trainingSbest(X,Y):
    model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    model.fit(X, Y)
    return model


def trainingSR5(X,Y):
    model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    model.fit(X, Y)
    return model

def trainingSRD(X,Y):
    model = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
    model.fit(X, Y)
    return model

def scoring(model,X,Y):
    score = model.score(X, Y)
    predictions = model.predict(X)
    mcc = matthews_corrcoef(Y,predictions)
    f1 = f1_score(Y,predictions,average='micro')
    # print('F1:%.2f,MCC:%.2f,Score:%.2f' % (f1,mcc,score))
    return score,f1,mcc


def trainingRF(X,Y):
    model = RandomForestClassifier(n_estimators=300,criterion='entropy',max_depth=None, random_state=0)
    model.fit(X, Y)
    return model



def inference(S,model,X,Y, ppvmax_mode):
    # print(len(S.shape))
    if len(S.shape)==3:
        # print('i am in')
        NS = S.shape[2] # number of state vectors (population size)

        if ppvmax_mode == 'ppvmax':    
            ### For ppv and max
            ## For ppv and max
            S_ppvmax = np.zeros((X.shape[0],X.shape[1],X.shape[2]))
            S_ppvmax[:,0::2,:] = S
            S_ppvmax[:,1::2,:] = S
            X = S_ppvmax*X # apply states to the inputs   X = S*X
        elif ppvmax_mode == 'ppv':
            ## For only        
            X = S*X # apply states to the inputs   X = S*X


        ts_time = time.time()
        score = Parallel(n_jobs=int(NS/4),backend="threading")(delayed(scoring)(model,X[:,:,i],Y) for i in range(NS))
        score = np.asarray(score)
        f1 = score[:,1]
        mcc = score[:,2]
        score = score[:,0]
    elif len(S.shape)==2:
        ts_time = time.time()
        score,f1,mcc = scoring(model,X,Y)
    return np.asarray(score),np.asarray(f1),np.asarray(mcc)

def costting(S,score):
    dcost_S  = np.sum(S[0,:,:],axis=0)/S.shape[1] # normalizing 
    cost = 1 - (score - dcost_S) 
    return cost, dcost_S