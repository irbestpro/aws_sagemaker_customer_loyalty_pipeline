
#_________Import Libraries_______________________

import numpy as np
import pandas as pd
import os
import torch.optim
import torch.nn as nn
import torch
import boto3
import sagemaker
import pickle

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


if __name__ == '__main__':

    #______________setting the model hyperparameters_______________

    LEARNING_RATE = 0.01
    EPOCHS = 100
    BATCH_SIZE = 32

    base_dir = "/opt/ml/input/data" # base directory of local container input data (Training step container)

    train_Data = pd.read_csv(f"{base_dir}/train/Train.csv").to_numpy()
    X_train = train_Data[:,:-1] # get train csv file from processing step output
    y_train = train_Data[:,-1]

    #______________Create Dense_Network Instances_____________________

    model = Dense_Network( X_train.shape[1] , 1) # create Dense_Network Instance
    optimizer = torch.optim.Adam(model.parameters() , LEARNING_RATE) # using adam weight Optimizer
    Loss = nn.BCEWithLogitsLoss() # using binrary cross entropy loss function

    #_______________Train Phase__________________

    for epoch in range(EPOCHS): # running the model in N epochs
        temp_loss = 0 # temp loss in each epoch
        steps = 0 # counting number of batches
        temp_acc = 0
        for batch in range(0 , len(X_train) , BATCH_SIZE): # reading all coefficients in serveral batches (80% For Train)

            X = torch.from_numpy(X_train[batch : (batch + BATCH_SIZE) , :]).float() # reading current coefficients vectors
            targets = torch.from_numpy(y_train[batch : (batch + BATCH_SIZE)]).float() # reading datalabels
            output = model(X).squeeze(1) # calling the model and get generated outputs
            loss = Loss(output, targets) # calc loss function
            temp_loss += loss.item() # accumulative sum of loss values
            steps += 1

            model.zero_grad() # don't save gradient history
            loss.backward() # backPropagation process
            optimizer.step() # update Weights

        print('Train Phase - Epoch # ' , str(epoch + 1) , ', Loss : ' , str(temp_loss / steps))


    #____________Saving the model and upload to s3 default bucket___________________

    local_model_path = '/opt/ml/model' 
    pickle.dump(model, open(f"{local_model_path}/model.sav", 'wb')) # save model to a separate file
