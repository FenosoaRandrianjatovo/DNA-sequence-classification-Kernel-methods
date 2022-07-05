import numpy as np
import pandas as pd
from tqdm import tqdm

def load_data():
    try :
        X_test = pd.read_csv('data/Xte.csv',sep=',',index_col=0)
        X_train = pd.read_csv('data/Xtr.csv',sep=',',index_col=0)
        X_test_vectors = pd.read_csv('data/Xte_vectors.csv',sep=' ',header=None).values
        X_train_vectors = pd.read_csv('data/Xtr_vectors.csv',sep=' ',header=None).values
        y_train = pd.read_csv('data/Ytr.csv',sep=',',index_col=0)
        
    except:
        print('No file found')
        

    return X_test, X_train, X_test_vectors, X_train_vectors, y_train


X_test, X_train, X_test_vectors, X_train_vectors, y_train = load_data()


def getKmers(sequences, size=3):
    return [sequences[x:x+size].lower() for x in range(len(sequences) - size + 1)]


def base2int(key):
    return {'a':0,'c':1,'g':2,'t':3}.get(key,0)

def index(kmer):
    base_idx = np.array([base2int(base) for base in kmer])
    multiplier = 4** np.arange(len(kmer))
    kmer_idx = multiplier.dot(base_idx)
    return kmer_idx

def spectral_embedding(sequences, kmer_size=3):

    kmers = getKmers(sequences, kmer_size)
    kmer_idxs = [index(kmer) for kmer in kmers]
    one_hot_vector = np.zeros(4**kmer_size)
    for kmer_idx in kmer_idxs:
        one_hot_vector[kmer_idx] += 1.0
    return one_hot_vector

def embedded_data(kmer_size):

    print(f"Creating of Embedding of Kmer size of {kmer_size}")
    print(" ")
    df = pd.DataFrame(pd.concat([X_train.Sequence,X_test.Sequence],axis=0))
    training_text = df.Sequence.values
    kmer_data = []
    for train in tqdm(training_text):
        kmer_data.append(spectral_embedding(train,kmer_size=kmer_size))
        
    print(" ")
    print(f"embedding of Kmer size of {kmer_size} Created")
    return np.array(kmer_data)

def change_label(label=0):
    '''
    label=0 corresponds to   0 or 1 
    label=1 corresponds to   -1 or 1 
    else corresponds to   0 or 1 
    '''

    if label != 1 or label ==0:
        return y_train.Covid.values
    else:
        y_train['Covid'] = y_train.Covid.apply(lambda x: 1 if x != 0 else -1)
        return y_train.Covid.values



def train_test_split(X,y,p=0.33, seed=102):

    n, _ = X.shape
    assert (n == len(y)) 
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    test_size =int(n*p)
    X_test= X[idx[:test_size]]
    X_train=X[idx[test_size:]]
    y_test=y[idx[:test_size]]
    y_train=y[idx[test_size:]]

    return X_train, X_test, y_train, y_test



def Saved(y_pred, name="prediction")-> None:

    sub = np.where(y_pred == -1,0,1)
    y_pred = np.vstack([1 + np.arange(len(sub)), sub ]).T
    
    np.savetxt(f'csv_file_saved/{name}'+'.csv', 
                y_pred,
                delimiter=',',
                header='Id,Covid', 
                fmt='%i', 
                comments='')  



    