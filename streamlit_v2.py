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
matplotlib.use('Agg')

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

# Create Metric
mse = mean_squared_error(y_test, y_pred)**(0.5)
np.bool = np.bool_

explainer = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(X_train)

# Define SHAP Summary plot
# fig_sum = shap.summary_plot(shap_values, features=X_train, feature_names=X_train.columns)

# Define SHAP Force plot
i = 127
#fig_force = shap.force_plot(explainer.expected_value, shap_values[i], features=X_train.loc[i], feature_names=X_train.columns)
#fig_force = go.Figure(force_plot.data[0])

#fig_force_2 = shap.force_plot(explainer.expected_value, shap_values, X_train)

# Define SHAP Dependence plot
#fig_dep = shap.dependence_plot("horsepower", shap_values, X_train)

##################################################
########## Streamlite Dashboard ##################
##################################################

#import streamlit.components.v1 as components
from streamlit_shap import st_shap
shap.initjs()

#def st_shap(plot, height=None):
#    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#    components.html(shap_html, height=height)

# Some Basic settings
st.set_option('deprecation.showPyplotGlobalUse', False)
# Set dashboard width to entire width of screen
st.set_page_config(layout="wide")

with st.container():
   st.title('Explainable AI with automobile dataset')

st.divider()

tab1, tab2, tab3 = st.tabs(["Model", "XAI", "3"])

with tab1:
   st.write('Below is a snapshot of the original dataframe')
   st.write(df.head(5))
   st.divider()
   st.write('Below is a snapshot of the dataframe, with labeled encoded columns')
   st.write(df_imputed.head(5))
   #st.table(x_data.columns)
   st.write('The machine learning model was trained with the following features to predict the target variable')
   for feature in features:
    st.markdown(f"- {feature}")
   st.write('Prediction target:')
   st.markdown(f"- Highway-mpg")
   st.divider()
   st.write('The models mean squared error is:')
   st.metric(label="MSE", value=mse, delta="1.2 °F")

with tab2:
    #col1, col2, col3 = st.columns(3, gap="medium")

    #with col1:
    st.write("This is col 1") 
    #st.pyplot(fig_sum, clear_figure=False)
    #summary_plot = shap.summary_plot(shap_values, features=X_train, feature_names=X_train.columns)
    #st.pyplot(summary_plot, bbox_inches='tight', clear_figure=False)
    st_shap(shap.summary_plot(shap_values, features=X_train, feature_names=X_train.columns))

    #with col2:
    st.write("This is col 2")  
    option = st.selectbox(
    'Which data record do you want to analyze?',
    (1, 2, 3))
    st.write('You selected:', option)
    st_shap(shap.dependence_plot("horsepower", shap_values, X_train))

    #with col3:
    st.write("This is col 3")
    shap.initjs()  
    st_shap(shap.force_plot(explainer.expected_value, shap_values[i], features=X_train.loc[i], feature_names=X_train.columns))
    #st_shap(shap.force_plot(explainer.expected_value, shap_values, X_train))
    #shap.force_plot(explainer.expected_value, shap_values, X_train)
    #st.pyplot(shap.force_plot(explainer.expected_value, shap_values, X_train))

with tab3:
    shap.initjs()
    option2 = st.selectbox(
    'Which data record do you want to analyze?',
    (5, 4, 6))
    st.write('You selected:', option2)
    st.divider()
    #force = shap.force_plot(explainer.expected_value, shap_values, X_train)
    #st.pyplot(shap.force_plot(explainer.expected_value, shap_values, X_train))
    #x = shap.force_plot(explainer.expected_value, shap_values, X_train)
    #st.pyplot(x)










