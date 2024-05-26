
import math
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
from sklearn.model_selection import train_test_split

def Entropy(X):
    Entropies = np.ones((1,X.shape[1]))
    Entropies = list([entropy(np.unique(X[:,i], return_counts=True)[1],base=10) for i in range(0,X.shape[1])])
    Entropies = sorted(enumerate(Entropies),key = lambda x: x[1])
    selected_features = list(map(lambda x: x[0], [Entropies[i] for i in range(0,math.floor(len(Entropies) * .5))]))
    return (selected_features)


if __name__ == '__main__':

    base_dir = "/opt/ml/processing" # address to container image folder for saving the input and output data

    print('Starting Preprocessing Step')
    
    #_________Reading the input csv file at first______________
    
    Dataset = pd.read_csv(f"{base_dir}/input/Customers.csv" , header=0) # address to base data directory in container and read the data files
    X = Dataset.iloc[:,:-1]
    Gender = {"Female" : 1 , "Male" : 0}
    X['Gender'] = X['Gender'].map(Gender)
    Y = Dataset.iloc[:,-1].to_numpy()

    #_______________Data Smooting________________________________

    nan_df = X.isna().any(axis=1) # get access to nan values data
    nan_df = list(map(lambda element: element[0] , filter(lambda idx: idx[1]== True , enumerate(nan_df)))) # extract nan values

    for data in nan_df:
        nan_row = X.iloc[data , :].isna().any(axis=2) # extract coloumns
        nan_cols = list(map(lambda element: element[0] , filter(lambda idx: idx[1]== True , enumerate(nan_row)))) # extract nan values
        map(lambda pointer : X.iloc[data , pointer].interpolate() , nan_cols) # Estimating Nan Values Based on it's Neighbours (Mean and interpolation)

    X = X.to_numpy()
    
    #_________________Noise Detection and removal___________________

    dist = (np.average(list(map(lambda x: np.linalg.norm(x - X) , X))) / X.shape[0]) / (0.05 * X.shape[1])
    clustering = DBSCAN(eps=dist, min_samples=3).fit(X) # Output -1 Equals to Noise
    noise_indexes = np.array(list(map( lambda y: y[0] , filter(lambda x : x[1] == -1 , enumerate(clustering.labels_))))) # Select Noise Indexes
    clean_indexes = np.array(list(filter(lambda x: noise_indexes.__contains__(x) == False ,range(0,X.shape[0])))) # Selecting Clean Indexes

    X = X[clean_indexes , :] # Remove Noise Indexes from X
    Y = Y[clean_indexes] # Remove Noise Indexes from Y
    Entropy_features = Entropy(X) # The best Features Base on sample Entropy function

    #__________Save Preprocessed data in separate files_____________

    X = X[: , Entropy_features] # feature selection
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    pd.DataFrame(np.column_stack((X_train , y_train))).to_csv(f"{base_dir}/output/Train/Train.csv", header=False, index=False)
    pd.DataFrame(np.column_stack((X_test , y_test))).to_csv(f"{base_dir}/output/Test/Test.csv", header=False, index=False)

print('The first step is executed now!')
