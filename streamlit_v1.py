import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# For our modelling we will use the two models we learned today
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Import a set of standard classification metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, make_scorer

# We also import the functions for the cross-validation
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate

import shap

#@title Import libraries and functions

# TQDM provides progressbars
from tqdm import tqdm

# We some of the standard classifiers.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# We also import some preprocessing models. These will be used later
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

# Import a set of standard classification metrics
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, make_scorer

# We also import the functions for the cross-validation
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_validate
from sklearn.model_selection import TimeSeriesSplit, cross_validate

# Import Libraries for Resampling
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
reg = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=SEED)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
np.bool = np.bool_

explainer = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(X_train)

# Define SHAP Summary plot
fig_sum = shap.summary_plot(shap_values, features=X_train, feature_names=X_train.columns)

# Define SHAP Force plot
i = 127
fig_force = shap.force_plot(explainer.expected_value, shap_values[i], features=X_train.loc[i], feature_names=X_train.columns, matplotlib=True)
#fig_force = go.Figure(force_plot.data[0])

# Define SHAP Dependence plot
fig_dep = shap.dependence_plot("horsepower", shap_values, X_train)

##################################################
########## Streamlite Dashboard ##################
##################################################

import streamlit.components.v1 as components

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Some Basic settings
st.set_option('deprecation.showPyplotGlobalUse', False)
# Set dashboard width to entire width of screen
st.set_page_config(layout="wide")

st.title('Explainable AI with automobile dataset')

col1, col2, col3 = st.columns(3)

with col1:
   st.pyplot(fig_sum, clear_figure=False)

with col2:
   option = st.selectbox(
    'Which data record do you want to analyze?',
    (1, 2, 3))
   st.write('You selected:', option)
   st.pyplot(fig_force, clear_figure=False)

with col3:
   st.pyplot(fig_dep, clear_figure=False)


# visualize the training set predictions
st_shap(shap.force_plot(explainer.expected_value, shap_values[i], features=X_train.loc[i], feature_names=X_train.columns), 400)










