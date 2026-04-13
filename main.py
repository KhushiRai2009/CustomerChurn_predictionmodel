from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_building import model_building

def main():

    # step 1: Data Ingestion
    df=data_ingestion()
    print(df.shape)
    X_train,X_test,y_train,y_test,Preprocessor,svd=data_preprocessing(df)
    print(X_train.shape)
    print(X_test.shape)
    model=model_building(X_train,X_test,y_train,y_test,Preprocessor,svd)
    print(model)


main()
