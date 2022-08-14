import pandas as pd

categorial_features = ["hospital_id", "ethnicity", "gender", "icu_admit_source",
                       "apache_3j_bodysystem", "apache_2_bodysystem", "icu_stay_type", "icu_type"]

def getBasicDataset(): #return dataset without empty column, shuffled and with onehot encoding
    first_dest_file_path = "dataset/dataset_part1.csv"
    second_dest_file_path = "dataset/dataset_part2.csv"
    df1 = pd.read_csv(first_dest_file_path)
    df2 = pd.read_csv(second_dest_file_path)
    df = pd.concat([df1, df2])
    df.drop(df.columns[[0,1,83]], axis=1, inplace=True) #delete feature 83 which is empty column
    df = df.sample(frac=1) #shuffle
    df = pd.get_dummies(df,columns=categorial_features) #one hot encoding for categories
    return df

# Filling missing NUMERICAL values with mean
def fill_missing_num_values_with_mean(complete_data):
    for feature in numerical_features:
        if feature in complete_data.columns:
            mean_value = complete_data[feature].mean()
            complete_data[feature].fillna(value=mean_value, inplace=True)
    return complete_data

def fill_test_missing_values_binary(x_test,x_train):
    #add column for binary missing values
    binary_features_with_missing_values = x_test[binary_features].columns[x_test[binary_features].isnull().any()]
    for f in binary_features_with_missing_values:
        name = "missing " + f
        x_test[name] = (x_test[f].isnull()).astype(int) # add feature which tell when value is missing
    # change missing values to 0 in the original feature
    for f in binary_features:
        x_test[f] = x_test[f].fillna(0)
    return x_test

# Filling missing NUMERICAL values with mean
def fill_test_missing_num_values_with_mean(x_test,x_train):
    for feature in numerical_features:
        if feature in x_test.columns:
            mean_value = x_train[feature].mean()
            x_test[feature].fillna(value=mean_value, inplace=True)
    return x_test

def fill_missing_values_binary(df):
    #add column for binary missing values
    binary_features_with_missing_values = df[binary_features].columns[df[binary_features].isnull().any()]
    for f in binary_features_with_missing_values:
        name = "missing " + f
        df[name] = (df[f].isnull()).astype(int) # add feature which tell when value is missing
    # change missing values to 0 in the original feature
    for f in binary_features:
        df[f] = df[f].fillna(0)
    return df


def selectSampleForSHAP(y_true, y_pred):
    y_true.to_numpy()
    false_pos = []
    true_pos = []
    false_neg = []
    true_neg = []
    for i in range(len(y_true)):
        print(y_pred[i].type())
        if y_pred[i] == 0 and y_true[i] == 0:
            true_neg.append(i)
            if y_pred[i]==0 and y_true[i] ==1 :
                false_neg.append(i)
            if y_pred[i]==1 and y_true[i] ==1 :
                true_pos.append(i)
            if y_pred[i]== 1 and y_true[i] ==0 :
                true_pos.append(i)

    for x in range(len(false_pos)):
        print(false_pos[x])
    for x in range(len(false_pos)):
        print(true_pos[x])
    for x in range(len(false_pos)):
        print(false_neg[x])
    for x in range(len(false_pos)):
        print(true_neg[x])