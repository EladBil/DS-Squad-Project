import pandas as pd

categorial_features = ["hospital_id", "ethnicity", "gender", "icu_admit_source",
                       "apache_3j_bodysystem", "apache_2_bodysystem", "icu_stay_type", "icu_type"]


numerical_features = ["age", "bmi","height", "weight",
                      "pre_icu_los_days", "gcs_eyes_apache","apache_2_diagnosis",
                      "gcs_motor_apache", "gcs_verbal_apache", "heart_rate_apache",
                     "map_apache", "resprate_apache", "temp_apache", "d1_diasbp_max",
                      "d1_diasbp_min","d1_diasbp_noninvasive_max", "d1_diasbp_noninvasive_min",
                      "d1_heartrate_max", "d1_heartrate_min", "d1_mbp_max", "d1_mbp_min",
                      "d1_mbp_noninvasive_max", "d1_mbp_noninvasive_min", "d1_resprate_max",
                      "d1_resprate_min","d1_spo2_max", "d1_spo2_min", "d1_sysbp_max", "d1_sysbp_min",
                      "d1_sysbp_noninvasive_max", "d1_sysbp_noninvasive_min", "d1_temp_max",
                      "d1_temp_min","h1_diasbp_max", "h1_diasbp_min", "h1_diasbp_noninvasive_max",
                      "h1_diasbp_noninvasive_min","h1_heartrate_max", "h1_heartrate_min",
                      "h1_mbp_max", "h1_mbp_min","h1_mbp_noninvasive_max", "h1_mbp_noninvasive_min",
                      "h1_resprate_max", "h1_resprate_min","h1_spo2_max", "h1_spo2_min",
                      "h1_sysbp_max", "h1_sysbp_min","h1_sysbp_noninvasive_max",
                      "h1_sysbp_noninvasive_min", "d1_glucose_max", "d1_glucose_min",
                      "d1_potassium_max", "d1_potassium_min", "apache_4a_hospital_death_prob",
                      "apache_4a_icu_death_prob","apache_3j_diagnosis"]


binary_features = ["arf_apache", "gcs_unable_apache", "intubated_apache",
                   "ventilated_apache", "elective_surgery", "apache_post_operative",
                   "aids", "cirrhosis", "diabetes_mellitus", "hepatic_failure", "immunosuppression",
                   "leukemia", "lymphoma", "solid_tumor_with_metastasis"]

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
