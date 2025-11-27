import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path

from src.utils.paths import get_data_path, get_config_path, get_model_path
from src.utils.config_loader import load_config


class DataTransformation:
    """Handles feature engineering, scaling, and enconding."""
    def __init__(self):
        self.config = load_config(get_config_path('paths_config.yaml'))
        c = self.config

        # Define artifact path
        self.preprocessor_path = get_model_path() / c['PREPROCESSOR_FILENAME']
        self.target_column = c['TARGET_COLUMN']  # Define target column

    def get_data_transformer_object(self):
        """ Creates the preprocessing ColumnTransformer."""
        # Define features based on schema
        numerical_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary',]
        categorical_features = ['Geography','Gender',]
        pre_encoded_features = ['NumOfProducts','HasCrCard','IsActiveMember','Tenure',] # New Feature Type!

        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        pre_encoded_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])

        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features),
                ("pre_encoded_pipeline", pre_encoded_pipeline, pre_encoded_features)
            ],
            # Optionally, use remainder="passthrough" to keep columns not specified
        )
        return preprocessor
    
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """Loads data, applies transformation, and saves the preprocessor."""
        print("----- Starting Data Transfomration -----")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]

            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save the Preprocessor Object
            self.preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            print(f"Preprocessor saved at: {self.preprocessor_path}")
            return (
                X_train_transformed,
                y_train.values,
                X_test_transformed,
                y_test.values,
                str(self.preprocessor_path)
            )
        
        except Exception as e:
            print(f"Error during data transformation: {e}")
            raise e
        