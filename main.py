
import argparse, copy
import numpy as np
import pandas as pd
import time, os, sys, json
from scipy.spatial.distance import cdist
sys.path.insert(0, "utils")

import optimization, classifier, audiomnistreader, freespokenreader, utils
from sklearn.linear_model import RidgeClassifierCV
import matplotlib.pyplot as plt
from rocket_functions import generate_kernels, apply_kernels
# from pygsp import graphs, filters, plotting
# import signal_energy
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# == parse arguments ===========================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--ds_name", required = True)
parser.add_argument("-i", "--input_path", required = True)
# parser.add_argument("-o", "--output_path", required = True)
parser.add_argument("-cv", "--num_runs", type = int, default = 1)
parser.add_argument("-m", "--model", required = True)
parser.add_argument("-k", "--num_kernels", type = int, default = 10000)
parser.add_argument("-p", "--pop_size", type = int, default = 10000)
parser.add_argument("-e", "--num_epochs", type = int, default = 100)
parser.add_argument("-pm", "--ppvmax", type = str, default = 'ppv') # ppv ppvmax
parser.add_argument("-bm", "--benchmark", type = str, default = 'UCR') # UCR AudioMNIST FreeSpoken
parser.add_argument("-s", "--samplingstep", type = int, default = 1) # UCR AudioMNIST FreeSpoken
parser.add_argument("-cr", "--crossover", type =float, default = 0.9) # UCR AudioMNIST FreeSpoken
parser.add_argument("-F", "--mutation_rate", type = float, default = 0.9) # UCR AudioMNIST FreeSpoken



parser.add_argument("-t", "--edge_threshold", type = float, default = 0)
arguments = parser.parse_args()

auto_save = False
n_epochs = arguments.num_epochs
NS = arguments.pop_size
Ncv = arguments.num_runs
benchmark = arguments.benchmark
ppvmax_mode = arguments.ppvmax
ds_name = arguments.ds_name
sampling_step = arguments.samplingstep
model_ = arguments.model #  R:Rocket(original); SR: SRocket
F_thr=arguments.mutation_rate
crossover_rate=arguments.crossover





# -- read data -------------------------------------------------------------
print("Loading data ...")
if benchmark == 'UCR':
    dataset_path = os.path.join(arguments.input_path,ds_name)
    print(dataset_path)
    # == run =======================================================================
    # dataset_names = np.loadtxt(arguments.dataset_names, "str")
    # dataset_name = dataset_path.split('/')[1]
    print(f"{ds_name}".center(80, "-"))
    results = pd.DataFrame(index = [ds_name],
                        columns = ["accuracy_mean",
                                    "accuracy_standard_deviation",
                                    "time_training_seconds",
                                    "time_test_seconds"],
                        data = 0)
    results.index.name = "dataset"
    # results_output_name = dataset_path+arguments.model+'_K'+str(arguments.num_kernels)+'_Th'+str(arguments.edge_threshold)+'ppv.csv'

    print(f"RUNNING".center(80, "="))

    print(os.path.join(dataset_path,ds_name+"_TRAIN.tsv"))
    training_data = np.loadtxt(os.path.join(dataset_path,ds_name+"_TRAIN.tsv"))
    Y_training, X_training = training_data[:, 0].astype(np.int32), training_data[:, 1:]
    # print(X_training.shape)
    test_data = np.loadtxt(os.path.join(dataset_path,ds_name+"_TEST.tsv"))
    Y_test, X_test = test_data[:, 0].astype(np.int32), test_data[:, 1:]
    # print(Y_training.shape, X_training.shape,Y_test.shape, X_test.shape)
    # (20,) (20, 512) (20,) (20, 512)
    C = np.max(Y_training)

print("Done.")

if sampling_step!=1:
    X_training = utils.sampler(X_training,sampling_step)
    X_test = utils.sampler(X_test,sampling_step)

input_length = X_training.shape[-1]
 # number of classes


### Results collection

_timings = np.zeros([4, Ncv]) # trans. tr., trans. te., training, test


scores_collection = np.zeros((Ncv,n_epochs))
avg_scores_collection = np.zeros((Ncv,n_epochs))

sparsity_collection = np.zeros((Ncv,n_epochs))
avg_sparsity_collection = np.zeros((Ncv,n_epochs))

cost_collection = np.zeros((Ncv,n_epochs))
avg_cost_collection = np.zeros((Ncv,n_epochs))
    
test_results_ = np.zeros((Ncv,26))

# -- run -------------------------------------------------------------------

### Saving results
file_path = 'ParetoResultsNP'+str(NS)+'_F'+str(F_thr)+'_Cr'+str(crossover_rate)+ds_name

if not os.path.exists(file_path):
    os.makedirs(file_path)

print(f"Performing runs...")



for cv_indx in range(Ncv):
    stime_kg = time.time()
    print('Generating kernels...')
    kernels = generate_kernels(input_length, arguments.num_kernels)
    print((kernels[0].shape))
    kernels_g = kernels[-2]
    kernels_norm1 = kernels[-1] # l1 norm of kernels
    kernels = kernels[0:-2]
    # print('kernels',kernels[0],kernels[0].shape)
    # print('lengths',kernels[1],kernels[1].shape)
    # -- Transform Training ------------------------------------------------
    # time_a = time.perf_counter()
    print('Applying kernels...')
    X_training_transform = apply_kernels(X_training, kernels) # N_Samples * 2*N_kernels (ppv,max)
    X_test_transform = apply_kernels(X_test, kernels)
    print('kernel gen and apply time:',time.time()-stime_kg)


    ## Features
    graph_signal_tr = np.expand_dims(X_training_transform,axis=2) # n_samples x : x feature_indx (d=2) print(graph_signal_tr[0,:,0])
    graph_signal_tr = np.reshape(graph_signal_tr,(X_training_transform.shape[0],arguments.num_kernels,2))        
    
    if ppvmax_mode == 'ppvmax':
        ## For PPV and Max
        graph_signal_tr = np.reshape(graph_signal_tr,(graph_signal_tr.shape[0],graph_signal_tr.shape[1]*graph_signal_tr.shape[2]))
    elif ppvmax_mode == 'ppv':    
        ## For PPV Only
        graph_signal_tr = graph_signal_tr[:,:,0] # ppv data N,N_kernels
    # print(graph_signal_tr.shape)
    np.savetxt(file_path+'/'+ppvmax_mode+'_featuresTr.txt',graph_signal_tr, fmt="%f",delimiter=',')
    np.savetxt(file_path+'/'+ppvmax_mode+'_labelsTr.txt',Y_training, fmt="%f",delimiter=',')

    graph_signal_tr_tiled = np.expand_dims(graph_signal_tr,axis=2)
    graph_signal_tr_tiled = np.tile(graph_signal_tr_tiled,(1,1,NS)) # N_Sample,N_Kernels,N_pop    
    n_samples = graph_signal_tr.shape[0]

    ## Test
    graph_signal_ts = np.expand_dims(X_test_transform,axis=2) # n_samples x : x feature_indx (d=2) print(graph_signal_tr[0,:,0])
    graph_signal_ts = np.reshape(graph_signal_ts,(X_test_transform.shape[0],arguments.num_kernels,2))
    
    if ppvmax_mode == 'ppvmax':
        ## For PPV and Max
        graph_signal_ts = np.reshape(graph_signal_ts,(graph_signal_ts.shape[0],graph_signal_ts.shape[1]*graph_signal_ts.shape[2]))
    elif ppvmax_mode == 'ppv':    
        # For PPV Only
        graph_signal_ts = graph_signal_ts[:,:,0] # ppv data N,N_kernels

    np.savetxt(file_path+'/'+ppvmax_mode+'_featuresTs.txt',graph_signal_ts, fmt="%f",delimiter=',')
    np.savetxt(file_path+'/'+ppvmax_mode+'_labelsTs.txt',Y_test, fmt="%f",delimiter=',')

    graph_signal_ts_tiled = np.expand_dims(graph_signal_ts,axis=2)
    graph_signal_ts_tiled = np.tile(graph_signal_ts_tiled,(1,1,NS)) # N_Sample,N_Kernels,N_pop

    ## States Initialization
    # S2 = np.round(np.random.random((arguments.num_kernels,NS))) # Num_classes x Num_kernels x Population size
    S = np.ones((arguments.num_kernels,NS)) # Num_classes x Num_kernels x Population size
    # S[:,:int(NS/2)] = S2[:,:int(NS/2)]
    # S = S2
    S = np.expand_dims(S,axis=0)
    S = np.tile(S,(n_samples,1,1))
    S_full = np.ones((arguments.num_kernels,NS)) # full states
    S_full = np.expand_dims(S_full,axis=0)

    ### Best with respect to individual metrics
    best_score_sofar = [0,0,0] # first element is score, secod is sparsity, third is cost
    best_state_scorewise = []
    best_sparsity_sofar = [0,1,0] # first element is score, secod is sparsity, third is cost
    best_state_sparsitywise = []
    best_score_epoch = 0
    best_sparsity_epoch = 0





    ## Training with full states
    print('Pre-Training ...')
    trtime = time.time()
    trained_model = classifier.training(graph_signal_tr,Y_training)
    print('training time:',time.time()-trtime)

    infftime = time.time()
    score_full,_,_ = classifier.inference(S_full,trained_model,graph_signal_tr_tiled,Y_training,ppvmax_mode) # inference performance of full states
    print('inf full time:',time.time()-infftime)
        
 
    score_S,_,_ = classifier.inference(S,trained_model,graph_signal_tr_tiled,Y_training,ppvmax_mode) # inference performance of initialized sparse states


    cost_S, dcost_S = classifier.costting(S,score_S) # cost_S: total cost  dcost_S: dimensionality cost
    # print(cost_S, dcost_S,score_S,score_full)
    print(20*'-')
    print('Max score initial S:',np.max(score_S),' *** Mean score initial S:',np.mean(score_S))
    print('Min cost initial S:',np.min(cost_S),'*** Mean cost initial S:',np.mean(cost_S))
    print('Max score Full S:',np.max(score_full),'*** Mean score full S:',np.mean(score_full))
    print(20*'-')

    if model_ == 'R':
        print('Original Rocket Model')
        score_full,_,_ = classifier.inference(S_full,trained_model,graph_signal_ts_tiled,Y_test,ppvmax_mode) # inference performance of full states
        print(np.mean(score_full)) # mean because it is a population

        break
    

    print('Optimization ...')
    timeColl = []
    ## Optimization
    break_flag = False
    for epoch in range(n_epochs):
        if break_flag==True:
            break
        tr_time = time.time()

        CS = optimization.evolution(S,F_thr,crossover_rate) # candidate states        
        score_CS,_,_ = classifier.inference(CS,trained_model,graph_signal_tr_tiled,Y_training,ppvmax_mode)
        # print('score_CS',score_CS)
        cost_CS, dcost_CS = classifier.costting(CS,score_CS) # cost_S: total cost  dcost_S: dimensionality cost
        # print(cost_CS, dcost_CS,score_CS)
        S, S_best, cost_S, cost_S_best, avg_cost_S, score_S, S_score_best = optimization.selection(S,CS,cost_S,cost_CS,score_S,score_CS)
        sparsity_value = np.sum(S_best)/S.shape[1]
        print('opt time per epoch:', time.time()-tr_time)
        timeColl.append(time.time()-tr_time)
        ## Picking best so far
        if epoch == 0:
            best_state_sparsitywise = S_best
            best_state_scorewise = S_best

        if S_score_best>best_score_sofar[0]:
            best_score_sofar[0] = S_score_best
            best_score_sofar[1] = sparsity_value
            best_score_sofar[2] = cost_S_best
            best_state_scorewise = S_best
            best_score_epoch = epoch
        if sparsity_value < best_sparsity_sofar[1]:            
            best_sparsity_sofar[0] = S_score_best
            best_sparsity_sofar[1] = sparsity_value
            best_sparsity_sofar[2] = cost_S_best
            best_state_sparsitywise = S_best
            best_sparsity_epoch = epoch


        print('Epoch:',epoch,'Best Cost:',cost_S_best,'Avg Cost:',avg_cost_S)
        print('Best score S:',S_score_best,' *** Mean score S:',np.mean(score_S),'*** D:',sparsity_value)        
        print(20*'-')
        
    
        scores_collection[cv_indx,epoch] = S_score_best
        avg_scores_collection[cv_indx,epoch]  = np.mean(score_S)

        sparsity_collection[cv_indx,epoch]  = sparsity_value
        avg_sparsity_collection[cv_indx,epoch]  = np.mean(S)

        cost_collection[cv_indx,epoch]  = cost_S_best
        avg_cost_collection[cv_indx,epoch]  = avg_cost_S
    

    S_best_tiled = np.tile(np.expand_dims(S_best,axis=0),(S.shape[0],1))
    print(np.mean(np.array(timeColl)))
    if ppvmax_mode == 'ppvmax':    
        ### For ppv and max
        S_best_tiled_pm = np.zeros((S_best_tiled.shape[0],2*S_best_tiled.shape[1]))
        S_best_tiled_pm[:,0::2] = S_best_tiled
        S_best_tiled_pm[:,1::2] = S_best_tiled
        S_best_tiled = S_best_tiled_pm

    # print(S_best.shape,graph_signal_tr.shape,S_best_tiled.shape)
    sparse_graph_signal_tr = S_best_tiled*graph_signal_tr
    post_trained_model = classifier.trainingS(sparse_graph_signal_tr,Y_training) ## Training the classifier with best sparse state vector
    score_sparse,_,_ = classifier.inference(S_best_tiled,post_trained_model,sparse_graph_signal_tr,Y_training,ppvmax_mode) # inference performance of best sparse state vector
    
    ####### Testing
    np.savetxt(file_path+'/sbest.txt',S_best, fmt="%f",delimiter=',')
    ## Proposed ERocket Sparsty
    S_best_tiled = np.tile(np.expand_dims(S_best,axis=0),(graph_signal_ts.shape[0],1))

    if ppvmax_mode == 'ppvmax':    
        ### For ppv and max
        S_best_tiled_pm = np.zeros((S_best_tiled.shape[0],2*S_best_tiled.shape[1]))
        S_best_tiled_pm[:,0::2] = S_best_tiled
        S_best_tiled_pm[:,1::2] = S_best_tiled
        S_best_tiled = S_best_tiled_pm


    sparse_graph_signal_ts = S_best_tiled*graph_signal_ts
    # ts_time = time.time()
    mcc_f1_acc = []
    print(10*'~')
    print('SRocket')
    test_score_sparse_post,f1,mcc = classifier.inference(S_best_tiled,post_trained_model,sparse_graph_signal_ts,Y_test,ppvmax_mode) # inference performance of best sparse state vector on the test dataset using post-trained model
    print('F1:%.2f,MCC:%.2f,Score:%.2f' % (f1,mcc,test_score_sparse_post))
    print(10*'~')
    mcc_f1_acc.append(['SRocket',np.round(f1,2),np.round(mcc,2),np.round(test_score_sparse_post,2)])
    # print('was sparse inference time')
    test_score_sparse_pre,_,_ = classifier.inference(S_best_tiled,trained_model,sparse_graph_signal_ts,Y_test,ppvmax_mode)    
    
    # Best Sparse and Score vectors independently
    ## Best Score
    S_best_score_tiled = np.tile(np.expand_dims(best_state_scorewise,axis=0),(graph_signal_ts.shape[0],1))

    if ppvmax_mode == 'ppvmax':    
        ### For ppv and max
        S_best_tiled_pm = np.zeros((S_best_score_tiled.shape[0],2*S_best_score_tiled.shape[1]))
        S_best_tiled_pm[:,0::2] = S_best_score_tiled
        S_best_tiled_pm[:,1::2] = S_best_score_tiled
        S_best_score_tiled = S_best_tiled_pm


    sparse_graph_signal_ts_bestscore = S_best_score_tiled*graph_signal_ts
    test_best_score_sparse_post,_,_ = classifier.inference(S_best_score_tiled,post_trained_model,sparse_graph_signal_ts_bestscore,Y_test,ppvmax_mode) # inference performance of best sparse state vector on the test dataset using post-trained model
    test_best_score_sparse_pre,_,_ = classifier.inference(S_best_score_tiled,trained_model,sparse_graph_signal_ts_bestscore,Y_test,ppvmax_mode)    
    #### Training with best score and evaluating
    S_best_score_tiledt = np.tile(np.expand_dims(best_state_scorewise,axis=0),(graph_signal_tr.shape[0],1))

    if ppvmax_mode == 'ppvmax':    
        ### For ppv and max
        S_best_tiled_pm = np.zeros((S_best_score_tiledt.shape[0],2*S_best_score_tiledt.shape[1]))
        S_best_tiled_pm[:,0::2] = S_best_score_tiledt
        S_best_tiled_pm[:,1::2] = S_best_score_tiledt
        S_best_score_tiledt = S_best_tiled_pm

    
    sparse_graph_signal_tr = S_best_score_tiledt*graph_signal_tr
    post_trained_model_bestscore = classifier.trainingSbest(sparse_graph_signal_tr,Y_training) ## Training the classifier with best sparse state vector
    score_bestscore,_,_ = classifier.inference(S_best_score_tiled,post_trained_model_bestscore,sparse_graph_signal_ts_bestscore,Y_test,ppvmax_mode) # inference performance of best sparse state vector
    
    ## Best Sparsity
    S_best_sparsity_tiled = np.tile(np.expand_dims(best_state_sparsitywise,axis=0),(graph_signal_ts.shape[0],1))

    if ppvmax_mode == 'ppvmax':    
        ### For ppv and max
        S_best_tiled_pm = np.zeros((S_best_sparsity_tiled.shape[0],2*S_best_sparsity_tiled.shape[1]))
        S_best_tiled_pm[:,0::2] = S_best_sparsity_tiled
        S_best_tiled_pm[:,1::2] = S_best_sparsity_tiled
        S_best_sparsity_tiled = S_best_tiled_pm



    sparse_graph_signal_ts_bestsparsity = S_best_sparsity_tiled*graph_signal_ts
    test_best_score_sparse_post,_,_ = classifier.inference(S_best_sparsity_tiled,post_trained_model,sparse_graph_signal_ts_bestsparsity,Y_test,ppvmax_mode) # inference performance of best sparse state vector on the test dataset using post-trained model
    test_best_score_sparse_pre,_,_ = classifier.inference(S_best_sparsity_tiled,trained_model,sparse_graph_signal_ts_bestsparsity,Y_test,ppvmax_mode)    
   
    #### Training with best sparsity and evaluating
    S_best_sparsity_tiledt = np.tile(np.expand_dims(best_state_sparsitywise,axis=0),(graph_signal_tr.shape[0],1))
   
    if ppvmax_mode == 'ppvmax':    
        ### For ppv and max
        S_best_tiled_pm = np.zeros((S_best_sparsity_tiledt.shape[0],2*S_best_sparsity_tiledt.shape[1]))
        S_best_tiled_pm[:,0::2] = S_best_sparsity_tiledt
        S_best_tiled_pm[:,1::2] = S_best_sparsity_tiledt
        S_best_sparsity_tiledt = S_best_tiled_pm
   
    sparse_graph_signal_tr = S_best_sparsity_tiledt*graph_signal_tr
    spttime = time.time()
    post_trained_model_bestsparsity = classifier.trainingSparbest(sparse_graph_signal_tr,Y_training) ## Training the classifier with best sparse state vector
    inftime = time.time()
    score_bestsparse,_,_ = classifier.inference(S_best_score_tiled,post_trained_model_bestsparsity,sparse_graph_signal_ts_bestsparsity,Y_test,ppvmax_mode) # inference performance of best sparse state vector

    ## Full states
    S_full = np.ones((1,arguments.num_kernels))
    S_full = np.tile(S_full,(graph_signal_ts.shape[0],1))
    test_score_full_pre,f1,mcc = classifier.inference(S_full,trained_model,graph_signal_ts,Y_test,ppvmax_mode) # inference performance of full state vector on the test dataset using pre-trained model
    print(10*'~')
    print('Full state:')
    print('F1:%.2f,MCC:%.2f,Score:%.2f' % (f1,mcc,test_score_full_pre))
    print(10*'~')
    mcc_f1_acc.append(['Rocket',np.round(f1,2),np.round(mcc,2),np.round(test_score_full_pre,2)])

    # ts_time = time.time()
    test_score_full_post,_,_ = classifier.inference(S_full,post_trained_model,graph_signal_ts,Y_test,ppvmax_mode) # inference performance of full state vector on the test dataset using post-trained model
    # print('was full inference time full')


    ## Random Sparsity of States
    S_random1 = np.round(np.random.random((1,arguments.num_kernels)))
    S_random = np.tile(S_random1,(graph_signal_ts.shape[0],1))
    
    if ppvmax_mode == 'ppvmax':    
        ### For ppv and max
        S_best_tiled_pm = np.zeros((S_random.shape[0],2*S_random.shape[1]))
        S_best_tiled_pm[:,0::2] = S_random
        S_best_tiled_pm[:,1::2] = S_random
        S_random = S_best_tiled_pm   
    
    random_sparse_graph_signal_ts = S_random*graph_signal_ts
    test_score_random_post,_,_= classifier.inference(S_random,post_trained_model,random_sparse_graph_signal_ts,Y_test,ppvmax_mode) # inference performance of best sparse state vector on the test dataset using post-trained model
    test_score_random_pre,_,_ = classifier.inference(S_random,trained_model,random_sparse_graph_signal_ts,Y_test,ppvmax_mode) # inference performance of full state vector on the test dataset using pre-trained model
    
    # Training with random state vector 0.5
    S_random = np.tile(S_random1,(graph_signal_tr.shape[0],1))

    if ppvmax_mode == 'ppvmax':    
        ### For ppv and max
        S_best_tiled_pm = np.zeros((S_random.shape[0],2*S_random.shape[1]))
        S_best_tiled_pm[:,0::2] = S_random
        S_best_tiled_pm[:,1::2] = S_random
        S_random = S_best_tiled_pm  

    randomsparse_graph_signal_tr = S_random*graph_signal_tr
    random_trained_model = classifier.trainingSR5(randomsparse_graph_signal_tr,Y_training) ## Training the classifier with best sparse state vector
    test_score_random_randmodel05,_,_ = classifier.inference(S_random,random_trained_model,random_sparse_graph_signal_ts,Y_test,ppvmax_mode) # inference performance of best sparse state vector on the test dataset using post-trained model
    
    # Training with random state vector length of S_best
    S_randomD1 = np.zeros((1,arguments.num_kernels))
    S_randomD1[0,:int(np.sum(S_best))] = 1
    np.random.shuffle(S_randomD1)
    S_randomD = np.tile(S_randomD1,(graph_signal_tr.shape[0],1))

    if ppvmax_mode == 'ppvmax':    
        ### For ppv and max
        S_best_tiled_pm = np.zeros((S_randomD.shape[0],2*S_randomD.shape[1]))
        S_best_tiled_pm[:,0::2] = S_randomD
        S_best_tiled_pm[:,1::2] = S_randomD
        S_randomD = S_best_tiled_pm  

    randomsparse_graph_signal_trD = S_randomD*graph_signal_tr
    random_trained_modelD = classifier.trainingSRD(randomsparse_graph_signal_trD,Y_training) ## Training the classifier with best sparse state vector
    test_score_random_randmodelD,f1,mcc = classifier.inference(S_randomD,random_trained_modelD,graph_signal_ts,Y_test,ppvmax_mode) # inference performance of best sparse state vector on the test dataset using post-trained model
    print(10*'~')
    print('Random@ S-rocket rate:')
    print('F1:%.2f,MCC:%.2f,Score:%.2f' % (f1,mcc,test_score_random_randmodelD))
    mcc_f1_acc.append(['Random',np.round(f1,2),np.round(mcc,2),np.round(test_score_random_randmodelD,2)])
    print(10*'~')


    ## Paretofront 
    kept_kernel_count = int(np.sum(S_best))# number of active kernels srocket found; applying as l1 threshold
    sorted_index = np.argsort(kernels_norm1[0,:])[-kept_kernel_count:]
    S_l1 = np.ones((1,arguments.num_kernels))
    S_l1[0,sorted_index] = 0

    for s_i in range(8):
        print(S.shape)
        S_l1 = S[:,:,s_i] #np.reshape(S[:,:,s_i],(1,10000))
        l1_signal = S_l1*graph_signal_tr
        l1_model = classifier.trainingSRD(l1_signal,Y_training) ## Training the classifier with best sparse state vector
        fff = np.tile(S_l1[0,:],(graph_signal_ts.shape[0],1))
        print(fff.shape)
        l1_signal = np.tile(S_l1[0,:],(graph_signal_ts.shape[0],1))*graph_signal_ts
        test_score_l1,f1,mcc = classifier.inference(l1_signal,l1_model,graph_signal_ts,Y_test,ppvmax_mode) # inference performance of full state vector on the test dataset using pre-trained model
        print(10*'~')
        print('Paretofront:',s_i)
        print('F1:%.2f,MCC:%.2f,Score:%.2f,D:%.2f' % (f1,mcc,test_score_l1,np.sum(S[0,:,s_i])/10000))
        print(10*'~')
    # sys.exit()




    ## l1 norm pruning
    kept_kernel_count = int(np.sum(S_best))# number of active kernels srocket found; applying as l1 threshold
    sorted_index = np.argsort(kernels_norm1[0,:])[-kept_kernel_count:]
    S_l1 = np.ones((1,arguments.num_kernels))
    S_l1[0,sorted_index] = 0

    l1_signal = S_l1*graph_signal_tr
    l1_model = classifier.trainingSRD(l1_signal,Y_training) ## Training the classifier with best sparse state vector

    l1_signal = S_l1*graph_signal_ts
    l1_signal = np.tile(l1_signal,(graph_signal_ts.shape[0],1))
    test_score_l1,f1,mcc = classifier.inference(l1_signal,l1_model,graph_signal_ts,Y_test,ppvmax_mode) # inference performance of full state vector on the test dataset using pre-trained model
    print(10*'~')
    print('l1 norm:')
    print('F1:%.2f,MCC:%.2f,Score:%.2f' % (f1,mcc,test_score_l1))
    print(10*'~')
    mcc_f1_acc.append(['l1',np.round(f1,2),np.round(mcc,2),np.round(test_score_l1,2)])

    ## l2 norm pruning
    kept_kernel_count = int(np.sum(S_best))# number of active kernels srocket found; applying as l1 threshold
    sorted_index = np.argsort(kernels_norm1[1,:])[-kept_kernel_count:]
    S_l2 = np.ones((1,arguments.num_kernels))
    S_l2[0,sorted_index] = 0

    l2_signal = S_l2*graph_signal_tr
    # l2_signal = np.tile(l2_signal,(graph_signal_tr.shape[0],1))
    l2_model = classifier.trainingSRD(l2_signal,Y_training) ## Training the classifier with best sparse state vector

    l2_signal = S_l2*graph_signal_ts
    l2_signal = np.tile(l2_signal,(graph_signal_ts.shape[0],1))
    test_score_l2,f1,mcc = classifier.inference(l2_signal,l2_model,graph_signal_ts,Y_test,ppvmax_mode) # inference performance of full state vector on the test dataset using pre-trained model
    print(10*'~')
    print('l2 norm:')
    print('F1:%.2f,MCC:%.2f,Score:%.2f' % (f1,mcc,test_score_l2))
    print(10*'~')
    mcc_f1_acc.append(['l2',np.round(f1,2),np.round(mcc,2),np.round(test_score_l2,2)])
    mcc_f1_df = pd.DataFrame(mcc_f1_acc,columns=['Model','F1','MCC','Score'])
    mcc_f1_df.to_csv(file_path+'/results_MCC_F1.csv', sep=',', encoding='utf-8',index=False)

    ###### Saving Results #######

    print(ds_name)
    print(10*'=','With full state vector',10*'=')
    print('Avg. Test on Full premodel:',test_score_full_pre)
    print('Avg. Test on Full postmodel:',test_score_full_post)

    print(10*'=','With random state vector',10*'=')
    print('Avg. Test on random sparsity premodel:',test_score_random_pre)
    print('Avg. Test on random sparsity postmodel:',test_score_random_post)
    print('Avg. Test on random sparsity random model 0.5:',test_score_random_randmodel05,np.sum(S_random1)/arguments.num_kernels)
    print('Avg. Test on random sparsity random model len S:',test_score_random_randmodelD,np.sum(S_randomD1)/arguments.num_kernels)

    print(10*'=','With best cost vector',10*'=')
    print('Avg. Test on Sparse postmodel:',test_score_sparse_post,np.sum(S_best)/arguments.num_kernels)
    print('Avg. Test on Sparse premodel:',test_score_sparse_pre,np.sum(S_best)/arguments.num_kernels)
    
    print(10*'=','With best score vector',10*'=')
    print('Best score:',best_score_sofar[0],'Sparsity:',best_score_sofar[1],'Cost:',best_score_sofar[2])
    print('Avg. Test on Best score on premodel:',test_best_score_sparse_pre,np.sum(best_state_scorewise)/arguments.num_kernels)
    print('Avg. Test on Best score on postmodel:',test_best_score_sparse_post,np.sum(best_state_scorewise)/arguments.num_kernels)
    print('Avg. Test on trained with Best score:',score_bestscore,np.sum(best_state_scorewise)/arguments.num_kernels)

    print(10*'=','With best sparsity vector',10*'=')
    print('Score:',best_sparsity_sofar[0],'Best Sparsity:',best_sparsity_sofar[1],'Cost:',best_sparsity_sofar[2])
    print('Avg. Test on Best sparsity on premodel:',test_best_score_sparse_pre,np.sum(best_state_sparsitywise)/arguments.num_kernels)
    print('Avg. Test on Best sparsity on postmodel:',test_best_score_sparse_post,np.sum(best_state_sparsitywise)/arguments.num_kernels)
    print('Avg. Test on trained with Best sparsity:',score_bestsparse,np.sum(best_state_sparsitywise)/arguments.num_kernels)

    test_results_[cv_indx,0] = test_score_full_pre
    test_results_[cv_indx,1] = test_score_full_post
 
    test_results_[cv_indx,2] = test_score_random_pre
    test_results_[cv_indx,3] = test_score_random_post
    test_results_[cv_indx,4] = test_score_random_randmodel05
    test_results_[cv_indx,5] = test_score_random_randmodelD # for revision paper
    test_results_[cv_indx,6] = np.sum(S_randomD1)/arguments.num_kernels
 
    test_results_[cv_indx,7] = test_score_sparse_post
    test_results_[cv_indx,8] = np.sum(S_best)/arguments.num_kernels
    test_results_[cv_indx,9] = test_score_sparse_pre
 
    test_results_[cv_indx,10] = best_score_sofar[0]
    test_results_[cv_indx,11] = best_score_sofar[1]
    test_results_[cv_indx,12] = best_score_sofar[2]
    test_results_[cv_indx,13] = test_best_score_sparse_pre
    test_results_[cv_indx,14] = test_best_score_sparse_post
    test_results_[cv_indx,15] = score_bestscore
    test_results_[cv_indx,16] = np.sum(best_state_scorewise)/arguments.num_kernels

    test_results_[cv_indx,17] = best_sparsity_sofar[0]
    test_results_[cv_indx,18] = best_sparsity_sofar[1]
    test_results_[cv_indx,19] = best_sparsity_sofar[2]
    test_results_[cv_indx,20] = test_best_score_sparse_pre
    test_results_[cv_indx,21] = test_best_score_sparse_post
    test_results_[cv_indx,22] = score_bestsparse
    test_results_[cv_indx,23] = np.sum(best_state_sparsitywise)/arguments.num_kernels

    test_results_[cv_indx,24] = best_score_epoch
    test_results_[cv_indx,25] = best_sparsity_epoch


### Saving results
np.savetxt(file_path+'/results.txt',test_results_, fmt="%f",delimiter=',')
np.savetxt(file_path+'/scores_collection.txt',scores_collection, fmt="%f",delimiter=',')
np.savetxt(file_path+'/avg_scores_collection.txt',avg_scores_collection, fmt="%f",delimiter=',')
np.savetxt(file_path+'/sparsity_collection.txt',sparsity_collection, fmt="%f",delimiter=',')
np.savetxt(file_path+'/avg_sparsity_collection.txt',avg_sparsity_collection, fmt="%f",delimiter=',')
np.savetxt(file_path+'/cost_collection.txt',cost_collection, fmt="%f",delimiter=',')
np.savetxt(file_path+'/avg_cost_collection.txt',avg_cost_collection, fmt="%f",delimiter=',')

with open(file_path+'/commandline_args.txt', 'w') as f:
    json.dump(arguments.__dict__, f, indent=2)


