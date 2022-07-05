import argparse


parser =argparse.ArgumentParser()

# Parser for cross Validation
parser.add_argument('-e',
                    '--epoch', 
                    help='This is the number of epochs',
                    type=int,
                    required=False)

parser.add_argument('-b',
                    '--batch_size', 
                    help='Batch size',
                    type=int,
                    required=False)


parser.add_argument('-t',
                    '--trial', 
                    help='Number of trials',
                    type=int,
                    required=False)

parser.add_argument('-lr',
                    '--learning_rate', 
                    help='Batch size',
                    type=float,
                    required=False)

parser.add_argument('-d',
                    '--decay', 
                    help='Weight decay',
                    type=float,
                    required=False)   

parser.add_argument('-k',
                    '--k', 
                    help='k',
                    type=int,
                    required=False)     

parser.add_argument('-l',
                    '--l', 
                    help='Lambda',
                    type=float,
                    required=False)                                

parser.add_argument('-s',
                    '--show', 
                    default=False,
                    type=bool,
                    help='To show the accuracy',
                    required=False)  


parser.add_argument('-o',
                    '--optuma', 
                    default=True,
                    type=bool,
                    help='To cross Validate using Optuma',
                    required=False) 

parser.add_argument('-m',
                    '--model_name', 
                    help='''
                            "SVM": "KernelSVM",
                            "KR": "KernelRidgeRegression",
                            "L":"Logisticregression"
                        ''',    
                    type=str,
                    required=False) 

parser.add_argument('-kr',
                    '--kernel', 
                    help='''
                            Linear Kernel, 
                            Radial Basis Function Kernel,
                            Polynomial Kernel
                        ''',    
                    type=str,
                    required=False) 



# Parser for Main
parser.add_argument('-kmer',
                    '--kmer_size', 
                    help=' kmer_size',
                    type=int,
                    required=False)   


parser.add_argument('-po',
                    '--power', 
                    help=' Power',
                    type=int,
                    required=False)   


parser.add_argument('-sg',
                    '--sigma', 
                    help='sigma',
                    type=float,
                    required=False)  

parser.add_argument('-c',
                    '--C', 
                    help='C',
                    type=float,
                    required=False)    


parser.add_argument('-la',
                    '--lambd', 
                    help='lambd',
                    type=float,
                    required=False)   


parser.add_argument('-ak',
                    '--aak', 
                    help=' k',
                    type=int,
                    required=False)   



mains_args=vars(parser.parse_args())


# Parser for Main
KERNEL=mains_args['kernel']
CC=mains_args['C']
KK=mains_args['aak']
LAMBDA=mains_args['lambd']
SIGMA=mains_args['sigma']
KMER_SIZE=mains_args['kmer_size']
POWER=mains_args['power']




# Parser for cross Validation
trial=mains_args['trial']
model_name =mains_args['model_name']
epoch= mains_args['epoch']
batch_size= mains_args['batch_size']
decay= mains_args['decay']
lr= mains_args['learning_rate']
k= mains_args['k']
lamda= mains_args['l']
show= mains_args['show']
optuma = mains_args['optuma']

Models = {"SVM": "KernelSVM",
          "KR": "KernelRidgeRegression",
          "L":"Logisticregression"}