
from ast import main
from unicodedata import name
from matplotlib.pyplot import axes, axis
import numpy as np
from tqdm import tqdm
import optuna

from models import Logisticregression, KernelRidgeRegression, KernelSVM
from utils import change_label, embedded_data

from parser import *




def CrossValidatation(x_data,
                      y_data,
                      model_name=None,
                      lr=0.01,
                      kernel="rbf",
                      lambd=0.2,
                      C=3,
                      sigma=0.5,
                      k=5,
                      power=2,
                      batch_size=16,
                      decay=0.8,
                      epoch=100):
    
    if len(x_data)%k != 0:
        
        print('cannot vsplit',len(x_data),' by ',k)

        return

    x_splitted = np.vsplit(x_data,k)
    y_splitted = np.vsplit(y_data.reshape(-1,1),k)

    aggregate_result = []
    
    print("**"*10,"Cross Validation start", "**"*10)
    print(" \n")
    for i in tqdm(range(len(x_splitted))):

        items = [x for x in range(len(x_splitted)) if x !=i ]
        x_test = x_splitted[i]
        y_test = y_splitted[i]

        for item in items:
            if i == 0:
                x_train = x_splitted[item]
                y_train = y_splitted[item]
            else:
                x_train = np.concatenate((x_train, x_splitted[item]), axis=0)
                y_train = np.concatenate((y_train, y_splitted[item]), axis=0)
                # x_train = np.hstack([x_train, x_splitted[item]])
                # y_train = np.hstack([y_train, y_splitted[item]])


        if model_name == 'KernelRidgeRegression':
            model = KernelRidgeRegression(
                    kernel=kernel,
                    lambd=lambd,
                    sigma=sigma,
                    power=power
                ).fit(x_train, y_train)
            result =sum(np.sign(model.predict(x_test))==y_test)/len(y_test)

        elif model_name == 'KernelSVM':

            model = KernelSVM(C=C,
                              kernel=kernel,
                              lambd=lambd,
                              sigma=sigma,
                              power=power)
            model.fit(x_train, y_train.flatten())
            y_pred = model.predict(x_test)

            result = np.sum((y_pred.flatten()==y_test.flatten()))/len(y_test)

        elif model_name == 'Logisticregression':
            logistic =  Logisticregression( x_train,
                                            y_train,
                                            batch_size=batch_size,
                                            lamda=lambd,
                                            lr=lr,
                                            decay=decay,
                                            epoch=epoch,
                                            print_every=None)
            logistic.train()

            result = logistic.evaluate(x_test,y_test)

        else:
            print('wrong model name, Please choose amoung: Logisticregression, KernelSVM or KernelRidgeRegression ')
            return 0

        aggregate_result.append(result)

        accuracy = np.sum(aggregate_result)/len(aggregate_result)
        
    print(" \n")
    print("**"*10,"End of the cross Validation ", "**"*10)
    print(" \n")
    return accuracy




def K_fold(trial_test):

    kmer_size =  trial_test.suggest_int('kmer_size', 3,10)
    degree    =  trial_test.suggest_int('degree', 1,4)
    kernel    =  trial_test.suggest_categorical('kernel', ['linear','rbf','polynomial'])
    lambd     =  trial_test.suggest_float('lambd', 1e-10, 150)
    sigma     =  trial_test.suggest_float('sigma', 1e-5, 150)
    k         =  trial_test.suggest_categorical('k', [4,5,8,10])
    C         =  trial_test.suggest_float('C', 0.1,50)
    


    
    return CrossValidatation(embedded_data(kmer_size)[:2000,:],
                            change_label(label=1),
                            model_name='KernelSVM',
                            C=C,
                            kernel=kernel,
                            lambd=lambd,
                            k=k,
                            sigma=sigma,
                            power=degree)

if __name__ == "__main__":

    if optuma:

        sampler = optuna.samplers.TPESampler()

        study = optuna.create_study(sampler=sampler, direction='maximize')

        df = study.optimize(func=K_fold, 
                            n_trials= 1 if trial==None else trial,
                            show_progress_bar=True)

    if show:
        model="KernelSVM" if model_name==None else Models[model_name]
        accuracy = CrossValidatation(embedded_data(6)[:2000,:],
                                    change_label(0).reshape(-1,1),
                                    model_name=model)

        print(f" {accuracy} is the accuracy of the {model} Model after cross validation for a fixed parameters")

    