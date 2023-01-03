from scipy.spatial.distance import cdist
import numpy as np
import sys, copy
from scipy.special import softmax


def evolution(S,F_thr,crossover_rate):
    ## Evolution for the population of sample target only
    NS = S.shape[2] # pop size
    population = S[0,:,:] # D,NS ; taking it out from tiling
    D = S.shape[1] 
    
    parents = np.asarray([np.random.permutation(np.arange(NS))[:3] for i in range(NS)])
    F_thr = F_thr
    F = np.random.random((D,NS))<F_thr
    mask = np.logical_and(F,np.logical_or(population[:,parents[:,1]], population[:,parents[:,2]]))
    ev = (1-population[:,parents[:,0]])*mask + population[:,parents[:,0]]*np.logical_not(mask)

    crossover_rate = crossover_rate                # Recombination rate [0,1] - larger more diveristy
    cr = (np.random.rand(D,NS)<crossover_rate)
    mut_keep = ev*cr
    pop_keep = population*np.logical_not(cr)
    new_population = mut_keep + pop_keep
    new_population = np.expand_dims(new_population,axis=0)
    new_population = np.tile(new_population,(S.shape[0],1,1))
    return new_population


def selection(S,CS,cost_S,cost_CS,score_S,score_CS):
    best_indx = cost_CS<cost_S
    best_indx = np.where(best_indx==True)[0]
    S[:,:,best_indx] = CS[:,:,best_indx]
    cost_S[best_indx] = cost_CS[best_indx]
    score_S[best_indx]=score_CS[best_indx]
    cost_S_best = np.min(cost_S)
    S_best = S[0,:,np.argmin(cost_S)]
    S_score_best = score_S[np.argmin(cost_S)]
 
    return S, S_best, cost_S, cost_S_best, np.mean(cost_S),score_S,S_score_best
