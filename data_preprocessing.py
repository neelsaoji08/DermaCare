import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical



def preprocess(data_dir,size):
    df=pd.read_csv(data_dir)

    Label = df["label"]
    Data = df.drop(columns=["label"])

    print("Balancing the Data")

    oversample = RandomOverSampler()
    Data, Label  = oversample.fit_resample(Data, Label)
    Data = np.array(Data).reshape(-1, size[0], size[1] , 3)
    Label = np.array(Label)
    print('Shape of Data :', Data.shape)

    print("Balancing the Data: Complete")

    X_train_full , X_test , y_train_full , y_test = train_test_split(Data , Label , test_size = 0.10 , random_state = 49)
    X_train,X_valid,y_train,y_valid = train_test_split(X_train_full , y_train_full , test_size = 0.20 , random_state = 49)

    X_train=X_train/255.0
    X_test=X_test/255.0
    X_valid=X_valid/255.0
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    y_test = to_categorical(y_test)

    print("Spliting the Data: Complete")

    return X_train,X_test,y_train,y_test,X_valid,y_valid

