import pandas as pd
from data_ingestion import image_data
from data_preprocessing import preprocess
from model_training import model_training_personal
from model_training import transfer_learning_model
from model_evaluation import model_evaluation



def pipeline(size):

    data_dir = "dataset/pixel_data_" + str(size[0])+"_" + str(size[1]) + ".csv"
    
    #data_Ingestion
    image_data(size=size)

    #data_preprocessing
    X_train,X_test,y_train,y_test=preprocess(data_dir=data_dir,size=size)

    
    #Model Training Personal    
    model,history=model_training_personal(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,size=size,epochs=5)


    #model Evalution
    model=model_evaluation(model,history,X_train,X_test,y_train,y_test)
    return model
