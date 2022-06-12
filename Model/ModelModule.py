
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
# from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

import pandas as pd

import time



class DSWorkshopModel:
    def __init__(self, data) -> None:
        # Saving the original data
        self.og_data = data

        ## Models Parameters ## 

        # Logisitic Regression Parameters:
        # self.logisitic_regression_iter_num = 100

        # Voting Model Parameters:
        # Preparing a list of estimators for Voting Classifier. 
        self.voting_model_estimators = [
            ('lr',LogisticRegression()),
            ('dtc',DecisionTreeClassifier()),    
            ('rfc',RandomForestClassifier()),
            ('knc',KNeighborsClassifier())
        ]

        ## Classifiers Initialization ##
        self.classifiers_list = [
            RandomForestClassifier(),
            ExtraTreesClassifier(),
            # GradientBoostingClassifier(),
            VotingClassifier(estimators=self.voting_model_estimators),
        ]

        # Labels of classifiers.
        # MUST BE IN CORRECT ORDER TO LIST INITIALIZATION #
        self.labels = [
            "RandomForest",
            "ExtraTrees",
            # "GradientBoosting",
            "Votingr",
        ]

        # Flag if the model is ready for action.
        self.is_ready = True

        # Array to save the fit running times of the model.
        self.train_running_times = []



    # A method for setting up the split of the data.
    def set_split(self, x_train, y_train, x_test, y_test):
        # Training data.
        self.train_data = x_train
        # True values of the training data.
        self.train_data_values = y_train
        # Testing data.
        self.test_data = x_test
        # True values of testing data.
        self.test_data_values = y_test

        # Flag if the model is ready for action.
        self.is_ready = True



    # A method that starts the training process of the model.
    # A call more than once will results in learning the data again on top of the 
    # previous trains. 
    def train(self):
        if not self.is_ready:
            print("Something went wrong! The model isn't ready for training")
            return
        # Fit Loop
        i = 0
        for classifier in self.classifiers_list:
            i = i + 1
            start_time = time.time()
            classifier.fit(self.train_data, self.train_data_values)
            end_time = time.time()

            self.train_running_times.append(f'{round(end_time-start_time,2)}s')

    

    # A method to test the model.
    # MUST CALL AFTER YOU TRAINED!
    def test(self):
        if not self.is_ready:
            print("Something went wrong! The model isn't ready for testing")
            return

        # Arrays to keep evaluations form the metrics we use.
        precision_score = []
        recall_score = []
        f1_score = []
        accuracy_score = []
        predictions = []
        names = []

        # Prediction loop

        i = 0
        for classifier in self.classifiers_list:

            i = i + 1

            model_prediction = classifier.predict(self.test_data)
            predictions.append(model_prediction)
            
            precision_score.append(metrics.precision_score(self.test_data_values, model_prediction))
            recall_score.append(metrics.recall_score(self.test_data_values, model_prediction))
            f1_score.append( metrics.f1_score(self.test_data_values, model_prediction))
            accuracy_score.append(metrics.accuracy_score(self.test_data_values, model_prediction))
            names.append(self.labels[i-1])
            # oversampling.append(f'{over}')

        results_dataFrame = {
            'precision_score': precision_score, 
            'recall_score': recall_score, 
            'f1_score': f1_score,
            'accuracy_score' : accuracy_score,
            # 'oversampling': oversampling,
            'time ': self.train_running_times
        }

        results_dataFrame = pd.DataFrame(data=results_dataFrame)
        results_dataFrame.insert(loc=0, column='Method', value=names)

        return predictions, results_dataFrame


    def print_details(self):
        print ("Data Shape: " + str(self.og_data.shape))
        print("Data preview:")
        return self.og_data.head()
