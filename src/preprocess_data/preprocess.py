from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np

def _preprocess_data():
     data = pd.read_csv("dataset.csv")
     print(data)
     y = np.array(data["label"])
     data.drop(columns=["label"])
     X = np.array(data)
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
     X_train1, X_train2, y_train1, y_train2 = train_test_split(X, y, test_size=0.5)
     np.save('x_train1.npy', X_train1)
     np.save('x_test.npy', X_test)
     np.save('y_train1.npy', y_train1)
     np.save('y_test.npy', y_test)
     np.save('x_train2.npy', X_train2)
     np.save('y_train2.npy', y_train2)
     
if __name__ == '__main__':
     print('Preprocessing data...')
     _preprocess_data()