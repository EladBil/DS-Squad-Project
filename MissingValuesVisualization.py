# Quantity of missing values for each type of features: binary, numerical and categorical
import matplotlib.pylab as plt
import numpy as np

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

def NaN_info(df):
    global null_view
    try:
        null_view = df[[col for col in df.columns if df[col].isna().sum() > 0]].isna().sum()
    except:
        return null_view
    return null_view.sum()

def visualizeMissingValues(df):

    labels = ["With Value", "NaN"]
    plt.figure(figsize=(20,20))
    plt.rcParams.update({'font.size': 20})

    # Numerical
    nan_values = NaN_info(df[numerical_features])
    total_values = df[numerical_features].shape[0] * df[numerical_features].shape[1]
    missing_percent = np.array([total_values - nan_values, nan_values])

    plt.subplot(1, 5, 1)
    plt.title("Numerical")
    plt.pie(missing_percent, labels=labels)

    # Categorial
    nan_values = NaN_info(df[categorial_features])
    total_values = df[categorial_features].shape[0] * df[categorial_features].shape[1]
    missing_percent = np.array([total_values - nan_values, nan_values])

    plt.subplot(1, 5, 3)
    plt.title("Categorial")
    plt.pie(missing_percent, labels=labels)

    # Binary
    nan_values = NaN_info(df[binary_features])
    total_values = df[binary_features].shape[0] * df[binary_features].shape[1]
    missing_percent = np.array([total_values - nan_values, nan_values])

    plt.subplot(1, 5, 5)
    plt.title("Binary")
    plt.pie(missing_percent, labels=labels)
    plt.show()

def missingValuesDistribution(df):
    # Distribution of missing values by type of features
    plt.rcParams.update({'font.size': 15})
    missing_values_in_data = np.array([NaN_info(df[numerical_features]),
                                       NaN_info(df[categorial_features]),
                                       NaN_info(df[binary_features])])
    labels = ["Numerical", "Categorial", "Binary"]
    plt.figure(figsize=(8, 8))
    plt.pie(missing_values_in_data, labels=labels, autopct='%1.1f%%')
    plt.title("Missing values represented in the dataset")
    plt.show()