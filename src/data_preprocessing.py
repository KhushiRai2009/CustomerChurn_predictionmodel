from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA,TruncatedSVD #faster computation than pca
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,LabelEncoder
import numpy as np
'''
WorkFlow Design:
1.Clean the unwanted columns from the dataset
2.split the data into X and y
3.split the data into Train and Test
4.split the data into numerical and categorical columns
5.use Pipeline for numerical and categorical columns
6.use columns Transformer to fit our model
7.use SMOTE Technique and then PCA (for dimension Reductionality)
8.Return X_train,X_test,y_train,y_test

'''
def data_preprocessing(df):

    #1.Clean the unwanted columns from the dataset
    df=df.drop_duplicates()

    #mappping
    #df['Churn']=df['Churn'].map({'Yes':1,'No':0})

    # Encode target to 0 and 1
    df["Churn"]=df["Churn"].apply(lambda x:1 if x=="Yes" else 0)
    
    #2.split the data into X and y
    X=df.drop(columns=["customerID","Churn"],errors="ignore")
    y=df["Churn"]

    #3.split the data into Train and Test
    X_train,X_test,y_train,y_test=train_test_split(X,y,
                                                   test_size=0.3,
                                                   random_state=1)
    
    #4.split the data into numerical and categorical columns 
    numerical_col=X_train.select_dtypes(exclude="object").columns
    categorical_col=X_train.select_dtypes(include="object").columns

    #5.using Pipeline for numerical and categorical columns
    Numerical_Pipeline=Pipeline(steps=[
        ("Imputer",SimpleImputer(strategy="median")),
        ("Scaling",MinMaxScaler())
    ])

    Categorical_Pipeline=Pipeline(steps=[
        ("Imputer",SimpleImputer(strategy="most_frequent")),
        ("Encoder",OneHotEncoder(handle_unknown="ignore"))
    ])

    #6.use columns Transformer to fit our model
    preprocessor=ColumnTransformer(transformers=[
        ("Numerical_pipe",Numerical_Pipeline,numerical_col),
        ("Categorical_pipe",Categorical_Pipeline,categorical_col)
    ])

    X_train=preprocessor.fit_transform(X_train)
    X_test=preprocessor.transform(X_test)

    # Use SMOTE Technique
    sm=SMOTE()

    X_train,y_train=sm.fit_resample(X_train,y_train)

    # Use PCA (Principal Component Analysis:Dimension Reductionality Technique)
    '''
    pca=PCA()

    X_train=pca.fit_transform(X_train)
    X_test=pca.transform(X_test)

    return X_train,X_test,y_train,y_test
    '''

    # Step 1:Fit SVD with higher components
    svd_temp=TruncatedSVD(n_components=100,random_state=1)
    X_train_temp=svd_temp.fit_transform(X_train)

    # Step 2:Calculate cumulative variance
    cumulative_variance=np.cumsum(svd_temp.explained_variance_ratio_)

    # Step 3:Find components for 95% variance
    n_components_95=np.argmax(cumulative_variance>=0.95)+1

    # Step 4:Final SVD
    svd=TruncatedSVD(n_components=n_components_95,random_state=1)
    X_train=svd.fit_transform(X_train)
    X_test=svd.transform(X_test)

    print(f"Selected Components for 95% variance:{n_components_95}")

    return X_train,X_test,y_train,y_test,preprocessor,svd



