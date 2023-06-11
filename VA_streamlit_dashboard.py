import streamlit as st
from streamlit_shap import st_shap
import shap

# Import of basic libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Sklearn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, cross_validate, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, make_scorer, confusion_matrix

# Numpy & matplotlib definitions
matplotlib.use('Agg')
np.bool = np.bool_

#######################################################
################ MACHINE LEARNING #####################
#######################################################

# Import dataset
path ="https://github.com/KingKai69/VisualAnalytics/raw/main/Automobile_data.csv"
df = pd.read_csv(path)
df.rename(columns={'symboling': 'risk-score'}, inplace=True)

###### Data Preprocessing ######

# Imputing
df.replace('?', float('nan'), inplace=True)
df_imp = df.copy()
df_imp['normalized-losses'] = df_imp['normalized-losses'].astype('float32')
df_imp['normalized-losses'] = df_imp[['normalized-losses']].interpolate(method="pad")
df_imp = df_imp.fillna(df_imp.mode().iloc[0])

# Label Encoding
le = LabelEncoder()
df_enc = df_imp.copy()
columns_to_encode = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location','engine-type','num-of-cylinders','fuel-system']  # Liste mit den Spaltennamen
encoding_mapping = {}
for column in columns_to_encode:
    df_enc[column] = le.fit_transform(df_enc[column])
    encoding_mapping[column] = dict(zip(le.classes_, le.transform(le.classes_)))
for feature, mapping in encoding_mapping.items():
    df_data = []
    for category, encoded_value in mapping.items():
        df_data.append({'Origin label': category, 'Encoded label': encoded_value})
    
    df_lab = pd.DataFrame(df_data)
    df_name = f"{feature}_df"
    globals()[df_name] = df_lab

# Convert datatyp object to int64 respectively float64
for col in df_enc.columns:
    if df_enc[col].dtype == 'object':
        try:
            df_enc[col] = df_enc[col].astype('int64')
        except ValueError:
            try:
                df_enc[col] = df_enc[col].astype('float64')
            except ValueError:
                pass

# Create 3 classification labels
df_enc_3 = df_enc.copy()
df_enc_3['risk-score'] = df_enc_3['risk-score'].replace(3, 'high-risk')
df_enc_3['risk-score'] = df_enc_3['risk-score'].replace(2, 'high-risk')
df_enc_3['risk-score'] = df_enc_3['risk-score'].replace(1, 'medium-risk')
df_enc_3['risk-score'] = df_enc_3['risk-score'].replace(0, 'medium-risk')
df_enc_3['risk-score'] = df_enc_3['risk-score'].replace(-1, 'low-risk')
df_enc_3['risk-score'] = df_enc_3['risk-score'].replace(-2, 'low-risk')

# Define list of all possible features 
features = [
    'normalized-losses',
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
    'curb-weight',
    'engine-type',
    'num-of-cylinders',
    'engine-size',
    'fuel-system',
    'bore',
    'stroke',
    'compression-ratio',
    'horsepower',
    'peak-rpm',
    'city-mpg',
    'highway-mpg',
    'price',
]

# Create correlation matrix
Val = df_enc_3[features]
corrmat = Val.corr()

# Create dataframe with features with hight correlation
corrmat_masked = corrmat.mask(abs(corrmat) >= 1)
high_corr = corrmat_masked[(corrmat_masked > 0.6) | (corrmat_masked < -0.6)].stack().reset_index()
high_corr.columns = ['Feature_A', 'Feature_B', 'Correlation']
high_corr = high_corr[high_corr['Feature_A'] < high_corr['Feature_B']]
high_corr.head(n=120)

# Define features nonCor
features_nonCor = [
    'normalized-losses',
    'make',
    'fuel-type',
    'aspiration',
    'num-of-doors',
    'body-style',
    'drive-wheels',
    'engine-location',
    'height',
    'curb-weight',
    'engine-type',
    'num-of-cylinders',
    'fuel-system',
    'bore',
    'stroke',
    'horsepower',
    'peak-rpm',
    'city-mpg',
]

# Create df that shows which feature is used
col_names = list(df.columns)
df_feature = pd.DataFrame({'feature': col_names,
                           'is_feature': [True if value in features_nonCor else False for value in col_names],
                           'is_target_feature': [True if value == 'risk-score' else False for value in col_names]})

###### Modelling ######

# Fixing the SEED
SEED = 27

# Define X and Y Data
x_data = df_enc_3[features_nonCor]
y_data = df_enc_3['risk-score']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=SEED, stratify=y_data)

# Init KFold for cross validation
kf_cv = KFold(n_splits = 5, shuffle = True, random_state=SEED)

# Evaluation cross validation
clf_final = ExtraTreesClassifier(n_estimators=301, criterion="gini", max_depth=12 , max_features='sqrt', min_samples_split=5, min_samples_leaf=1, bootstrap=False, class_weight='balanced', random_state = SEED)
scoring = {'f1_score': make_scorer(f1_score, average='weighted', zero_division=1)}
cv_final = cross_validate(estimator = clf_final, X = X_train, y = y_train, cv = kf_cv, scoring = scoring)
f1_scores_cv = cv_final['test_f1_score']
mean_f1_score_cv = np.mean(f1_scores_cv)
std_f1_score_cv = np.std(f1_scores_cv)
y_pred_cv = cross_val_predict(clf_final, X_train, y_train, cv=kf_cv)
cm_cv = confusion_matrix(y_train, y_pred_cv)

# Evaluation test_data
clf_final.fit(X_train, y_train)
y_pred_final = clf_final.predict(X_test)
f1_final = f1_score(y_test, y_pred_final, average='weighted', zero_division=1)
cm_final = confusion_matrix(y_test, y_pred_final)

# df result
df_results = pd.DataFrame({'Test Labels': y_test, 'Predicted Labels': y_pred_final})
df_exp = pd.concat([X_test, df_results], axis=1)
df_exp_false = df_exp[df_exp['Test Labels'] != df_exp['Predicted Labels']]
df_exp_false.loc[:, 'iloc_xtest'] = [21, 29, 31, 38, 45, 50]
new_order = ['Test Labels', 'Predicted Labels', 'iloc_xtest'] + list(df_exp_false.columns[:18])
df_exp_false = df_exp_false[new_order]

##
df_exp_corr = df_exp[df_exp['Test Labels'] == df_exp['Predicted Labels']]
df_exp_corr_hr = df_exp[(df_exp['Test Labels'] == 'high-risk') & (df_exp['Predicted Labels'] == 'high-risk')]
df_exp_corr_mr = df_exp[(df_exp['Test Labels'] == 'medium-risk') & (df_exp['Predicted Labels'] == 'medium-risk')]


##### Compute SHAP values #####
explainer_tree = shap.TreeExplainer(clf_final)
shap_values_tree = explainer_tree.shap_values(X_test)

#######################################################
################ Explainable AI #######################
#######################################################

#import streamlit.components.v1 as components
#shap.initjs()

# Some Basic settings
st.set_option('deprecation.showPyplotGlobalUse', False)
# Set dashboard width to entire width of screen
st.set_page_config(layout="wide")


#### title ####
with st.container():
   st.title('Explainable AI with automobile data')

tab1, tab2 = st.tabs(["Main", "Expand detailed explanation"])

with tab1:
   
    col1, col2, col3 = st.columns([1,1,1], gap="medium")

    with col1:
        st.subheader('Classification Model')
        col7, col8 = st.columns([1,1], gap="small")
        with col7:
            st.write('Dataframe with origin data:') 
            st.dataframe(df, width=400, height=200)
        with col8:
            st.write('Dataframe with preprocessed data:')
            #st.write('Below is a snapshot of the preproceddes dataframe, with removed NaN values, labeled encoded columns and scaled features')
            st.dataframe(df_enc_3, width=400, height=200)
        st.divider()
        st.write('**Feature Selection with correlation matrix**')    
    
        #st.subheader('Correlation Analysis')
        #st.write('After some preprocessing steps a correlations analysis was performed to identify pairs of features that have a high correlation. The goal is to remove features so that in the end no features have a high correlation. ')
        col9, col10 = st.columns([1,1], gap="small")
        st.divider()
        st.write('**Evaluation**')

        with col9:
            st.write('Correlation matrix:')
            fig1, ax1 = plt.subplots()
            sns.heatmap(corrmat, vmax=.8, square=True, ax=ax1)
            st.pyplot(fig1, clear_figure=True)
        
        with col10:
            st.write('Selected features:')
            st.experimental_data_editor(df_feature, width=400, height=160, disabled=True)


        col11, col12 = st.columns([1,1], gap="large")
        with col11:
            st.metric(label="F1-Score (Cross validation)", value="0.86", help=" Standard deviation of the 5 fold cross validation is +/- 0.09", delta_color="off")
            st.metric(label="F1-Score (Unseen test data)", value="0.89")
        with col12:
            st.write('Confusion matrix for unseen test data:')
            fig, ax = plt.subplots()
            sns.heatmap(cm_final, annot=True, cmap='Blues', fmt='d', xticklabels=clf_final.classes_, yticklabels=clf_final.classes_)
            plt.figure()
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig, clear_figure=True)
            
            

        #st.write('The machine learning model was trained with the following features to predict the target variable')
        #for feature in features:
            #st.markdown(f"- {feature}")
        #st.write('Prediction target:')
        
        
        #st.markdown(f"- Highway-mpg")
        #st.title('Performance')
        #st.metric(label="F1-score", value="89,2 %", delta="XX")


    with col2:
        #st.title('Shap Force Plot')
        st.subheader('Summary explanation')
        st.write('Shap Summary Plot')
        fig_summary=shap.summary_plot(shap_values_tree, X_test, plot_type="bar", class_inds="original", class_names=clf_final.classes_)
        st.pyplot(fig_summary)

        col4, col5, col6 = st.columns([1,1,1], gap="small")

        with col4:
            #Summary Plot der Klasse 0
            st.write('Summary plot of class high-risk')
            summaryplot0=shap.summary_plot(shap_values_tree[0], X_test)
            st.pyplot(summaryplot0)

        with col5:
            #Summary Plot der Klasse 1
            st.write('Summary plot of class low-risk')
            summaryplot1=(shap.summary_plot(shap_values_tree[1], X_test))
            st.pyplot(summaryplot1)
            
        with col6:
            #Summary Plot der Klasse 2
            st.write("Summary plot of class med-risk")
            summaryplot2=(shap.summary_plot(shap_values_tree[2], X_test))
            st.pyplot(summaryplot2)

        

    with col3:
        st.subheader('Detail explanation')
        st.write('Overview False Predictions')
        st.dataframe(df_exp_false, width=800, height=180)
        
        st.divider()
        option = [21, 29, 31, 38, 45, 50]
        iloc = st.selectbox('Choose a single data instance for analysis:', option, key=2)
        st.write('The plots show explanation for data instance with iloc:', iloc)

        # Explain Single prediction from test set from Class 0-High risk
        st.write("Force plot high-risk")
        st_shap(shap.force_plot(explainer_tree.expected_value[0], shap_values_tree[0][iloc], X_test.iloc[iloc,:]))
        # Explain Single prediction from test set from Class 1-Low risk
        st.write("Force plot low risk")
        st_shap(shap.force_plot(explainer_tree.expected_value[1], shap_values_tree[1][iloc], X_test.iloc[iloc,:]))
        # Explain Single prediction from test set from Class 2-Medium risk
        st.write("Force plot medium-risk")
        st_shap(shap.force_plot(explainer_tree.expected_value[2], shap_values_tree[2][iloc], X_test.iloc[iloc,:]))


with tab2:
    col21, col22, col23 = st.columns([2,1,1], gap="medium")
    with col21:
        st.subheader('Expanded detail explanation')
        st.write('Overview False Predictions')
        st.dataframe(df_exp_false, width=800, height=180)
        
        st.divider()
        iloc2 = st.selectbox('Choose a single data instance for analysis:', option, key=1)
        st.write('The plots show explanation for data instance with iloc:', iloc2)
       

        # Explain Single prediction from test set from Class 0-High risk
        st.write("Force plot high-risk")
        st_shap(shap.force_plot(explainer_tree.expected_value[0], shap_values_tree[0][iloc2], X_test.iloc[iloc2,:]))
        # Explain Single prediction from test set from Class 1-Low risk
        st.write("Force plot low risk")
        st_shap(shap.force_plot(explainer_tree.expected_value[1], shap_values_tree[1][iloc2], X_test.iloc[iloc2,:]))
        # Explain Single prediction from test set from Class 2-Medium risk
        st.write("Force plot medium-risk")
        st_shap(shap.force_plot(explainer_tree.expected_value[2], shap_values_tree[2][iloc2], X_test.iloc[iloc2,:]))
        
    with col22:
        st.subheader('Encoded feature analysis')
        enc_feature = st.selectbox('Choose a feature:', columns_to_encode)
        st.write('Selected feature:', enc_feature)
        show_df = enc_feature +"_df"
        df_var_enc = locals()[show_df]
        st.dataframe(df_var_enc, width=800, height=180)


        #feature_analysis = st.selectbox('Choose a feature', features_nonCor)
        #st.write('The plots show explanation for data instance with iloc:', feature_analysis)

        counts_mr = df_exp_corr_mr[enc_feature].value_counts()
        fig3, ax3 = plt.subplots()
        plt.bar(counts_mr.index, counts_mr.values)
        plt.xlabel(enc_feature)
        plt.ylabel('Häufigkeit')
        plt.title('Histogramm Klasse medium-risk')
        plt.xticks(counts_mr.index)
        st.pyplot(fig3, clear_figure=True)
        st.divider()
        
        counts_hr = df_exp_corr_hr[enc_feature].value_counts()
        fig4, ax4 = plt.subplots()
        plt.bar(counts_hr.index, counts_hr.values)
        plt.xlabel(enc_feature)
        plt.ylabel('Häufigkeit')
        plt.title('Histogramm Klasse high-risk')
        plt.xticks(counts_hr.index)
        st.pyplot(fig4, clear_figure=True)
    
    with col23:
        st.subheader('Numerous feature analysis')
        features_non_enc = ['normalized-losses', 'height', 'curb-weight', 
                            'bore', 'stroke', 'horsepower', 'peak-rpm', 'city-mpg',]
        fne = st.selectbox('Choose a feature:', features_non_enc)
        st.write('Selected feature:', fne)
        mean_val_corr_mr = round(df_exp_corr_mr[fne].mean(), 2)
        mean_val_corr_hr = round(df_exp_corr_hr[fne].mean(), 2)
        st.metric(label="Mean value for class medium-risk", value=mean_val_corr_mr)
        st.metric(label="Mean value for class high-risk", value=mean_val_corr_hr)
   



# 0 high
# 1 low
# 2 medium








#tab1, tab2, tab3 = st.tabs(["Classification model", "XAI Summary", "XAI Detail"])

#### Classification model ####
#with tab1:
#   st.write('Below is a snapshot of the original dataframe without any preprocessing steps:') 
#   st.write(df.head(5))
#   st.divider()
#   st.write('Below is a snapshot of the dataframe, with removed NaN values, labeled encoded columns and scaled features')
#   st.write(df_imp.head(5))
#   st.divider()
#   st.subheader('Correlation Analysis')
#   st.write('After some preprocessing steps a correlations analysis was performed to identify pairs of features that have a high correlation. The goal is to remove features so that in the end no features have a high correlation. ')
   
#   fig1, ax1 = plt.subplots(figsize=(4, 4))
#   sns.heatmap(corrmat, vmax=.8, square=True, ax=ax1)
#   st.pyplot(fig1, clear_figure=True)

#   st.divider()
#   st.write('Below all feature pairs with a correlation above 0.6 / below -0.6 are displayed. Accordingly for each feature pair, one feature is removed.')
#   st.dataframe(high_corr)
#   st.divider()
#   st.write('In the end the following features are used for modelling')
#   st.experimental_data_editor(df_feature)
   #st.table(x_data.columns)
#   st.write('The machine learning model was trained with the following features to predict the target variable')
#   for feature in features:
#    st.markdown(f"- {feature}")
#   st.write('Prediction target:')
#   st.markdown(f"- Highway-mpg")
#   st.divider()
#   st.title('Performance')
#   st.metric(label="F1-score", value="89,2 %", delta="XX")
#   fig, ax = plt.subplots()
#   sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=clf_final.classes_, yticklabels=clf_final.classes_)
#   plt.figure(figsize=(4, 4))
#   plt.xlabel('Predicted')
#   plt.ylabel('Actual')
#   st.pyplot(fig, clear_figure=True)

#### tab2 ####
#with tab2:
   #st.title('Shap Force Plot')
   #st.write('Explanation')
   #st_shap(shap.force_plot(explainer_tree.expected_value[0], shap_values_tree[0], X_test))
   #st.title('Shap Summary Plot')
   #fig_summary=shap.summary_plot(shap_values_tree, X_train, plot_type="bar")
   #st.pyplot(fig_summary)
   
   #height=800, width=600

   #st_shap(shap.summary_plot(shap_values_tree[2], X_test))
   #st_shap(shap.summary_plot(shap_values_tree[1], X_test))

   #col1, col2, col3 = st.columns([1,1,1], gap="large")
   #with col1:
      # Summary Plot der Klasse 0
      #st.write('Summary Plot der Klasse 0')
      #summaryplot0=shap.summary_plot(shap_values_tree[0], X_test)
      #st.pyplot(summaryplot0)
      
   #with col2:
      # Summary Plot der Klasse 1
      #st.write('Summary Plot der Klasse 1')
      #summaryplot1=(shap.summary_plot(shap_values_tree[1], X_test))
      #st.pyplot(summaryplot1)
    
   #with col3:
      # Summary Plot der Klasse 2
      #st.write("Summary Plot der Klasse 2")
      #summaryplot2=(shap.summary_plot(shap_values_tree[2], X_test))
      #st.pyplot(summaryplot2)
      
    
   
   # Summary Plot der Klasse 1

   #st_shap(shap.summary_plot(shap_values_tree[2], X_test))
    
#### tab3 ####
#with tab3:
    #st.subheader('Overview false predictions')
    #st.dataframe(def_pred_res_fil)
    #iloc = 31

    # Explain Single prediction from test set from Class 0-High risk
    #st_shap(shap.force_plot(explainer_tree.expected_value[0], shap_values_tree[0][iloc], X_test.iloc[iloc,:]))
    # Explain Single prediction from test set from Class 1-Low risk
    #st_shap(shap.force_plot(explainer_tree.expected_value[1], shap_values_tree[1][iloc], X_test.iloc[iloc,:]))
    # Explain Single prediction from test set from Class 2-Medium risk
    #st_shap(shap.force_plot(explainer_tree.expected_value[2], shap_values_tree[2][iloc], X_test.iloc[iloc,:]))


    #st_shap(shap.force_plot(explainer_tree.expected_value[0], shap_values_tree[0]))







 #st_shap(shap.force_plot(explainer_tree.expected_value, shap_values_tree[0,:], X_train.iloc[0,:]), height=200, width=1000)
   #st_shap(shap.force_plot(explainer_tree.expected_value, shap_values_tree, X_train), height=400, width=1000)
   #st_shap(shap.force_plot(explainer_tree.expected_value, shap_values_tree[0], X_train.iloc[0]), height=200, width=1000)
   #st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0]))
   #st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :], X_train.iloc[0, :]))
   #st_shap(shap.force_plot(explainer_tree.expected_value, shap_values_tree, X_train), height=400, width=1000)

#def st_shap(plot, height=None):
#    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#    components.html(shap_html, height=height)

#st_shap(shap.plots.waterfall(shap_values[0]), height=300)
#st_shap(shap.plots.beeswarm(shap_values), height=300)



#st_shap(force_plot = shap.force_plot(explainer_tree.expected_value, shap_values_tree, X_test))
