import streamlit as st
from streamlit_shap import st_shap
import shap
from sklearn.model_selection import train_test_split
import xgboost
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, make_scorer
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib


# Import DataSet from GitHub Respository and define which Excelsheets are to be transferred into pandas dataframes
path ="https://github.com/KingKai69/VisualAnalytics/raw/main/Automobile_data.csv"
df = pd.read_csv(path)
df.head()

# "?"-Symbole durch NaN-Werte ersetzen
df.replace('?', float('nan'), inplace=True)

# Imputing 
df_imputed = df.fillna(df.mode().iloc[0])

# Erstellen Sie einen LabelEncoder-Objekt
le = LabelEncoder()

# Wenden Sie den LabelEncoder auf mehrere Spalten in Ihrem DataFrame an
columns_to_encode = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location','engine-type','num-of-cylinders','fuel-system']  # Liste mit den Spaltennamen
for column in columns_to_encode:
    df_imputed[column] = le.fit_transform(df_imputed[column])

# Datentyp der "Alter"-Spalte ändern
df_imputed['horsepower'] = df_imputed['horsepower'].astype('int64')

# Fixing the SEED
SEED = 27

# Define features
features = [
    'horsepower',
    'engine-size',
    'body-style',
    'num-of-cylinders'
]

#Define X and Y Data
x_data = df_imputed[features]
y_data = df_imputed['highway-mpg']


# Modelling
model = RandomForestRegressor()
X, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=SEED)
model.fit(X, y_train)
y_pred = model.predict(X_test)

#######################################################
################ MACHINE LEARNING #####################
#######################################################


# Import DataSet from GitHub Respository and define which Excelsheets are to be transferred into pandas dataframes
path ="https://github.com/KingKai69/VisualAnalytics/raw/main/Automobile_data.csv"
df = pd.read_csv(path)
df.head()

###### Data Preprocessing ######

# Imputing
df.replace('?', float('nan'), inplace=True)
df_imp = df.fillna(df.mode().iloc[0])

# Label Encoding
le = LabelEncoder()
df_enc = df_imp.copy()
columns_to_encode = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location','engine-type','num-of-cylinders','fuel-system']  # Liste mit den Spaltennamen
for column in columns_to_encode:
    df_enc[column] = le.fit_transform(df_enc[column])

# Convert Object datatyp to int64 respectively float64
for col in df_enc.columns:
    if df_enc[col].dtype == 'object':
        try:
            df_enc[col] = df_enc[col].astype('int64')
        except ValueError:
            try:
                df_enc[col] = df_enc[col].astype('float64')
            except ValueError:
                pass

# Create 3 Classification labels
df_enc_3 = df_enc.copy()
df_enc_3['symboling'] = df_enc_3['symboling'].replace(3, 'high-risk')
df_enc_3['symboling'] = df_enc_3['symboling'].replace(2, 'high-risk')
df_enc_3['symboling'] = df_enc_3['symboling'].replace(1, 'medium-risk')
df_enc_3['symboling'] = df_enc_3['symboling'].replace(0, 'medium-risk')
df_enc_3['symboling'] = df_enc_3['symboling'].replace(-1, 'low-risk')
df_enc_3['symboling'] = df_enc_3['symboling'].replace(-2, 'low-risk')

# Define features nonCor
features_nonCor = [
    'make',
    'fuel-type',
    'aspiration',
    'num-of-doors',
    'body-style',
    'drive-wheels',
    'engine-location',
    'wheel-base',
    'length',
    'width',
    'height',
    'engine-type',
    'num-of-cylinders',
    'fuel-system',
    'bore',
    'stroke',
    'compression-ratio',
    'peak-rpm',
    'price',
]

###### Modelling ######

# Fixing the SEED
SEED = 27

# Define X and Y Data
x_data = df_enc_3[features_nonCor]
y_data = df_enc_3['symboling']

# Train Test Split
kf_cv = KFold(n_splits = 5, shuffle = True, random_state=SEED)

# Modelling
clf_final = ExtraTreesClassifier(n_estimators=117, criterion="entropy", max_depth=19 , class_weight='balanced', random_state = SEED)
A_train, A_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=SEED)
clf_final.fit(A_train, y_train)
y_pred = clf_final.predict(A_test)

# compute SHAP values
#explainer = shap.Explainer(model, X)
#shap_values = explainer(X)

#st_shap(shap.plots.waterfall(shap_values[0]), height=300)
#st_shap(shap.plots.beeswarm(shap_values), height=300)

explainer = shap.TreeExplainer(model, X)
shap_values = explainer.shap_values(X_test)

explainer_2 = shap.TreeExplainer(clf_final, A_train)
shap_values_2 = explainer.shap_values(A_train)

st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]), height=200, width=1000)
st_shap(shap.force_plot(explainer.expected_value, shap_values, X), height=400, width=1000)