from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,LabelEncoder

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

    # Encode target to 0 and 1
    df["Churn"]=df["Churn"].apply(lambda x:1 if x=="Yes" else 0)
    
    #2.split the data into X and y
    X=df.drop(columns=["customerID","Churn"])
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
    Preprocessor=ColumnTransformer(transformers=[
        ("Numerical_pipe",Numerical_Pipeline,numerical_col),
        ("Categorical_pipe",Categorical_Pipeline,categorical_col)
    ])

    X_train=Preprocessor.fit_transform(X_train)
    X_test=Preprocessor.transform(X_test)

    # Use SMOTE Technique
    sm=SMOTE()

    X_train,y_train=sm.fit_resample(X_train,y_train)

    # Use PCA (Principal Component Analysis:Dimension Reductionality Technique)
    pca=PCA()

    X_train=pca.fit_transform(X_train)
    X_test=pca.transform(X_test)

    return X_train,X_test,y_train,y_test

