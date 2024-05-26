
import json
import pathlib
import boto3
import sagemaker
import pickle
import tarfile
import os

import torch.optim
import torch.nn as nn
import torch
import joblib
import numpy as np
import pandas as pd


class Dense_Network(nn.Module):

    def __init__(self , features_length , number_of_clesses):
        super(Dense_Network , self).__init__()
        self.FC1 = nn.Sequential(nn.Linear(features_length,256) , nn.BatchNorm1d(256) , nn.ReLU())
        self.FC2 = nn.Sequential(nn.Linear(256,512) , nn.BatchNorm1d(512) ,  nn.ReLU())
        self.FC3 = nn.Sequential(nn.Linear(512,1024) , nn.BatchNorm1d(1024) ,  nn.ReLU())
        self.FC4 = nn.Sequential(nn.Linear(1024,number_of_clesses))

    def forward(self , x):
        out = self.FC1(x)
        out = self.FC2(out)
        out = self.FC3(out)
        out = self.FC4(out)
        return out

if __name__ == "__main__":

    #___________Get model from s3 bucket_____________________

    model_path = f"/opt/ml/processing/model/model.tar.gz" # local model path in container
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    model = pickle.load(open("model.sav", "rb"))
    #model = pickle.load(open(f"{model_path}/model.sav", "rb"))

    test_path = "/opt/ml/processing/test" # local test data path in container
    test_Data = pd.read_csv(f"{test_path}/Test.csv").to_numpy()
    X_test = test_Data[:,:-1] # get test csv file from processing step output
    y_test = test_Data[:,-1]
    
    X_test = torch.from_numpy(X_test).float() # convert to Tensor object
    y_test = torch.from_numpy(y_test).float() # convert to Tensor object

    #________________Test Phase______________________________

    acc = 0 # accuracy value
    #with torch.no_grad(): # stop weight updating
    outputs = model(X_test).squeeze(1) # test with 20% of data
    predicted = (outputs.data > 0.5).float() # predicted labels
    del outputs # free Ram Space
    true = (predicted == y_test).sum().item() # correct answers
    acc = (true / len(y_test))
    print('The Model Accuracy Is : ' , str(100 * (true / len(y_test))) + '%') # Pprint the final accuracy

    report_dict = {
        "Model Accuracy": {
            "acc": {"value": acc},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        
