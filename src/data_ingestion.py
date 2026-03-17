import os
import pandas as pd


#path configuration
data_path=os.path.join("data","churn.csv")
model_dir=os.path.join("models")
os.makedirs(model_dir,exist_ok=True)
pickle_path=os.path.join(model_dir,"churn_model.pkl")

#data loading
def data_ingestion():

    # dataloading and it will retutn dataframe
    df=pd.read_csv(data_path)
    return df

