import argparse
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

def test_model(model1, model2, x_test, y_test):
    x_test_data = np.load(x_test)
    y_test_data = np.load(y_test)

    model1 = joblib.load(model1)
    model2 = joblib.load(model2)
    model2.estimators_ += model2.estimators_
    model2.n_estimators = 20
    print("This is Test of Pro Player")
    print("Accuracy of merged model: ", model2.score(x_test_data, y_test_data))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_test')
    parser.add_argument('--y_test')
    parser.add_argument('--model1')
    parser.add_argument('--model2')
    args = parser.parse_args()
    test_model(args.model1, args.model1, args.x_test, args.y_test)