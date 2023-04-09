from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle

def create_mlp(Xpca, y_train):

    # Model creation
    mlp = MLPClassifier(random_state=0, max_iter=300, activation='logistic')

    # Hyperparameters tunning with grid search.
    # Some possible values are written into param_grid dictionary
    # and the grid search algorithm automatically selects the best combination
    # using cross validation with 5 folds.
    param_grid = {
        'solver': ["lbfgs", "sgd", "adam"],
        'hidden_layer_sizes': [(50,),(100,),(150,)],
        'max_iter': [50,100,200],
        'learning_rate_init': [0.01,0.001,0.0001]
    }

    CV_rfc = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5)
    CV_rfc.fit(Xpca, y_train)

    # Once obtained the best hyperparameters, the data is fitted in the model
    model = MLPClassifier(
        random_state = 0, 
        max_iter = CV_rfc.best_params_['max_iter'],
        hidden_layer_sizes = CV_rfc.best_params_['hidden_layer_sizes'],
        learning_rate_init = CV_rfc.best_params_['learning_rate_init'],
        solver = CV_rfc.best_params_['solver'],
        activation = 'logistic'
    )

    model.fit(Xpca, y_train)

    # Finally, the model is saved to use it latter
    filename = '../Models/multi_layer_perceptron.sav'
    pickle.dump(model, open(filename, 'wb'))

    print('Multi layer perceptron model created')
    return True


def create_logistic_reg(Xpca, y_train):
    # Model creation
    model = LogisticRegression(random_state=0)
    model.fit(Xpca, y_train)

    # Model saving to use it later.
    filename = '../Models/logistic_regression.sav'
    pickle.dump(model, open(filename, 'wb'))

    print('Logistic regression model created')
    return True