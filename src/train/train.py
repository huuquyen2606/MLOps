import argparse
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model(x_train, y_train, x_test, y_test):
    x_train_data = np.load(x_train)
    y_train_data = np.load(y_train)
    y_test = np.load(y_test)
    x_test = np.load(x_test)
    model = RandomForestClassifier(n_estimators=100,max_depth=1, random_state=42)
    model.fit(x_train_data, y_train_data)
    print(model.score(x_test, y_test))
    #joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    parser.add_argument('--x_test')
    parser.add_argument('--y_test')
    args = parser.parse_args()
    train_model(args.x_train, args.y_train, args.x_test, args.y_test)