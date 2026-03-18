from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing

def main():

    # step 1: Data Ingestion
    df=data_ingestion()
    print(df.shape)
    X_train,X_test,y_train,y_test=data_preprocessing(df)
    print(X_train,X_test,y_train,y_test)


main()
