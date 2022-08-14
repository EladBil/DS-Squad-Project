
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
from xgboost import XGBClassifier


from sklearn.metrics import balanced_accuracy_score, log_loss, recall_score, precision_score, f1_score, accuracy_score

import pandas as pd

import time




"""
Models Used:
    1. Random Forests (in code: RandomForestClassifier):
        In random forests, each tree in the ensemble is built from a sample drawn with replacement from the
        training set. Furthermore, when splitting each node during the construction of a tree,
        the best split is found either from all input features or a random subset of size max_features.
        The purpose of these two sources of randomness is to decrease the variance of the forest estimator.
        Indeed, individual decision trees typically exhibit high variance and tend to overfit. The injected randomness
        in forests yield decision trees with somewhat decoupled prediction errors. By taking an average of those
        predictions, some errors can cancel out.

        Random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight
        increase in bias. In practice the variance reduction is often significant hence yielding an overall better
        model.
    

    2. Extremely Randomized Trees (in code: ExtraTreesClassifier):
        In extremely randomized trees, randomness goes one step further in the way splits are computed.
        
        As in random forests, a random subset of candidate features is used, but instead of looking for the most
        discriminative thresholds, thresholds are drawn at random for each candidate feature and the best of these
        randomly-generated thresholds is picked as the splitting rule. 
        This allows to reduce the variance of the model a bit more, at the expense of slightly greater increase in bias.


    3. Extreme Gradient Boosting (in code: XGBClassifier):
        XGBoost is an optimized distributed gradient boosting library designed to be highly efficient,
        flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.
        XGBoost is a scalable and highly accurate implementation of gradient boosting that pushes the limits of
        computing power for boosted tree algorithms, being built largely for energizing machine learning model
        performance and computational speed. With XGBoost, trees are built in parallel, instead of sequentially like
        GBDT. It follows a level-wise strategy, scanning across gradient values and using these partial sums to
        evaluate the quality of splits at every possible split in the training set.

    For more info and models:
    https://scikit-learn.org/stable/modules/ensemble.html


Evaluation Metrics Used:
    1. Balanced Accuarcy Score:
        The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets.
        It is defined as the average of recall obtained on each class.
        The best value is 1 and the worst value is 0.

    2. Recall:
        The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false
        negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    3. Precision:
        The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of
        false positives.
        The precision is intuitively the ability of the classifier not to label as positive
        a sample that is negative.
        The best value is 1 and the worst value is 0.

    4. F1 Score:
        The F1 score can be interpreted as a harmonic mean of the precision and recall, 
        where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and
        recall to the F1 score are equal.

    For more info and metrics:
    https://scikit-learn.org/stable/modules/model_evaluation.html
"""

class FinalModel:
    def __init__(self, data, number_of_trees=100) -> None:
        # Saving the original data
        self.og_data = data

        ## Models Parameters ## 

        # Logisitic Regression Parameters:
        # self.logisitic_regression_iter_num = 100

        # Voting Model Parameters:
        # Preparing a list of estimators for Voting Classifier. 
        # self.voting_model_estimators = [
        #     ('lr',LogisticRegression()),
        #     ('dtc',DecisionTreeClassifier()),    
        #     ('rfc',RandomForestClassifier()),
        #     ('knc',KNeighborsClassifier())
        # ]

        # Number of processes given to computations:
        NUMBER_OF_JOBS = 3
        self.number_of_trees = number_of_trees # The default of the classifier is 100

        ## Classifiers Initialization ##
        self.classifiers_list = [
            # RandomForestClassifier(n_jobs=NUMBER_OF_JOBS, n_estimators=self.number_of_trees),
            # ExtraTreesClassifier(n_jobs=NUMBER_OF_JOBS, n_estimators=self.number_of_trees),
            XGBClassifier(objective="binary:logistic", n_estimators=self.number_of_trees, eval_metric="aucpr", scale_pos_weight="6")
            # GradientBoostingClassifier(),
            # VotingClassifier(estimators=self.voting_model_estimators),
        ]

        # Labels of classifiers.
        # MUST BE IN CORRECT ORDER TO LIST INITIALIZATION #
        self.labels = [
            # "RandomForest",
            # "ExtraTrees",
            "XGBClassifier",
        ]

        # Flag if the model is ready for action.
        self.is_ready = True

        # Array to save the fit running times of the model.
        self.train_running_times = []


    def get_models(self):
        return self.classifiers_list


    """
    A custom accuarcy metric that return the number of successes in predictions for negatives and positives.
    The best value is 1 and the worst value is 0.
    """
    def detaild_accuarcy_metric(self, true_values, pred_values):
        
        NEGATIVE_CLASS = 0
        POSITIVE_CLASS = 1

        # Array to split the data into classes and count each successful prediction of class.
        pred_succ_split = [0,0] 
        # Array to split the data into classes and count each instance of class.
        class_split = [0, 0]

        for index, value in enumerate(true_values):
            class_split[value] = class_split[value] + 1
            # In case the prediction is correct, add a point to the correct counter of the class.
            if pred_values[index] == value:
                pred_succ_split[value] = pred_succ_split[value] + 1
        

        positive_succ = round(pred_succ_split[POSITIVE_CLASS] / class_split[POSITIVE_CLASS], 5)
        negative_succ = round(pred_succ_split[NEGATIVE_CLASS] / class_split[NEGATIVE_CLASS], 5)
        # print("Accuarcy of positive predictions: ", positive_succ)
        # print("Accuarcy of negative predictions: ", negative_succ)

        return positive_succ, negative_succ


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
        arr_precision_score = []
        arr_recall_score = []
        arr_f1_score = []
        arr_unbalanced_accuracy_score = []
        arr_balanced_accuracy_score = []
        arr_positive_accuracy = []
        arr_negative_accuracy = []
        predictions = []
        names = []

        # Prediction loop
        i = 0
        for classifier in self.classifiers_list:

            i = i + 1

            model_prediction = classifier.predict(self.test_data)
            predictions.append(model_prediction)

            arr_precision_score.append(precision_score(self.test_data_values, model_prediction))
            
            arr_recall_score.append(recall_score(self.test_data_values, model_prediction))
            
            arr_f1_score.append(f1_score(self.test_data_values, model_prediction))

            arr_unbalanced_accuracy_score.append(accuracy_score(self.test_data_values, model_prediction))
            
            arr_balanced_accuracy_score.append(balanced_accuracy_score(self.test_data_values, model_prediction))
            
            pos_score, neg_score = self.detaild_accuarcy_metric(self.test_data_values, model_prediction)
            arr_positive_accuracy.append(pos_score)
            arr_negative_accuracy.append(neg_score)
            
            names.append(self.labels[i-1])


        # DataFrame table titles.
        results_dataFrame = {
            'Precision Score': arr_precision_score,
            'Recall Score': arr_recall_score, 
            'F1 Score': arr_f1_score,
            'Unbalanced Accuracy Score' : arr_unbalanced_accuracy_score,
            'Balanced Accuracy Score' : arr_balanced_accuracy_score,
            'Positive Accuracy Score' : arr_positive_accuracy,
            'Negative Accuracy Score': arr_negative_accuracy,
            'Time Needed for Training' : self.train_running_times
        }


        results_dataFrame = pd.DataFrame(data=results_dataFrame)
        results_dataFrame.insert(loc=0, column='Method', value=names)

        return predictions, results_dataFrame


    def print_details(self):
        print ("Data Shape: " + str(self.og_data.shape))
        print("Data preview:")
        return self.og_data.head()
