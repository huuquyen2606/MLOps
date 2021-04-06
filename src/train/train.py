import argparse
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def train_model(x_train, y_train, x_test, y_test):
    x_train_data = np.load(x_train)
    y_train_data = np.load(y_train)
    print("Total samples: ", y_train_data.shape[0])
    print("Attack samples: ", np.sum(y_train_data))
    y_test = np.load(y_test)
    x_test = np.load(x_test)
    model = RandomForestClassifier(n_estimators=10,max_depth=1, random_state=42)
    model.fit(x_train_data, y_train_data)
    print("Accuracy: ", model.score(x_test, y_test))
    #joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    parser.add_argument('--x_test')
    parser.add_argument('--y_test')
    args = parser.parse_args()
    train_model(args.x_train, args.y_train, args.x_test, args.y_test)