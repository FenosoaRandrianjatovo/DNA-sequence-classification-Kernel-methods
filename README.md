# DNA-sequence-classification-Kernel-methods



## Goal

The goal of the data challenge is to learn how to implement machine learning algorithms, gain understanding about them and adapt them to structural data.
For this reason, we have chosen a sequence classification task: predicting whether a DNA sequence (or read) belongs to the SARS-CoV-2 (Covid-19).


## Data description

Both the training and evaluation data are sets of DNA sequencing reads: short DNA fragments (~100 to 300bp long), that come from sequencing experiments, or that were simulated from full genomes. Some of these fragments come from Covid-19 genomes, others from human or random bacteria.
The goal is to discriminate the Covid-19 fragments, hence the task is a binary classification task: the labels are either 1 if the fragment is identified as Covid-19, and 0 otherwise.

## Run

To generate a submission file, please start the script: main.py for the default value which gives us the same accuracy as in Kaggle leaderboard. Thereofor feel free to try different parameters 

For exmpale: 

```
python main.py -kmer 4 -po 2 -sg 19  -c 2 -la 40  -kr "rbf" 
```

These paramaters gave us a private score of 0.97600. However, the following code gave us the public score  of 0.99400.


```
python main.py -kmer 10 -po 1  -sg 144.63369545821408  -c  1.4035505083702005  -la  57.6206097294714  -kr "linear" 
```


## Description

The file kernels.py contains all the kernels that have been implemented (Linear, Polynomial, Rbf). In the folder csv_file_saved you will find different csv files for prediction. In models file you will find all the implimentation of  Logistic Regression, Kernel Ridge Regression, and Kernel SVM.

For the cross validation, the file named cross_validation contained all the things raleted to cross validatio. Feel free to cross validate over the choice of kernel, the other hyperparamaters such as batch_size, epoch, etc,...

To run the cross validation file 

```
 python cross_validation.py -b 16 -t 2 -lr 0.02 -d 0.8 -e 100 -m "KR"
```
N.B: This cross validation takes time. At least 30 min depending on the machine.



