from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
import numpy as np
def model_pred(data_file):
    rep_data = {}
    with open(data_file, 'r') as f:
        rep_data = json.loads(f.read())

    arrays = [item['rep'] for item in rep_data]


    X = np.stack(arrays)
    y = [item['seen'] for item in rep_data]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 69)

    model = LogisticRegression().fit(X_train,y_train)


    print(model.score(X_test,y_test))