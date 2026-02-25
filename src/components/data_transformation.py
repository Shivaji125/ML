
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils.paths import get_data_path, get_config_path, get_model_path
from src.utils.config_loader import load_config



class DataTransformation:
    """Handles feature engineering, scaling, encoding."""
    def __init__(self):
        self.config = load_config(get_config_path("paths_config.yaml"))

        # Define artifact path
        self.preprocessor_path = get_model_path() / self.config["PREPROCESSOR_FILENAME"]
        self.target_column = self.config["TARGET_COLUMN"]

    def get_data_transformer_object(self):
        """Creates the preprocessig ColumnTransformer."""
        # Define features based on schema
        numerical_features = self.config["NUMERICAL_FEATURES"]
        categorical_features = self.config["CATEGORICAL_FEATURES"]
        pre_encoded_features = self.config["PRE_ENCODED_FEATURES"]

        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        pre_encoded_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ])

        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features),
                ("pre_encoded_pipeline", pre_encoded_pipeline, pre_encoded_features)
            ],
        )

        return preprocessor

    def initiate_data_transformation(self, train_path:str, test_path:str):
        """Loads data, applies transformation, and saves the preprocessor."""
        print("--- Starting Data Transformation ---")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.drop(columns=[self.target_column])
            y_train = train_df[self.target_column]

            X_test = test_df.drop(columns=[self.target_column])
            y_test = test_df[self.target_column]

            preprocessor = self.get_data_transformer_object()

            # X_train_transformed = preprocessor.fit_transform(X_train)
            X_train_transformed = preprocessor.fit(X_train)

            # X_test_transformed = preprocessor.transform(X_test)

            # Save the Preprocessor Object
            self.preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(preprocessor, self.preprocessor_path)
            print(f"Preprocessor saved at: {self.preprocessor_path}")
           
            # # Save schema(original column order) so inference can reorder new data correctly
            # schema = {"columns": list(X_train.columns)}
            # schema_path = self.preprocessor_path.parent / f"schema_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
            # schema_active = self.preprocessor_path.parent / "schema_active.json"
            # schema_path.write_text(json.dumps(schema, indent=4))
            # schema_active.write_text(json.dumps(schema, indent=2))
            # print(f"Schema saved at: {schema_path} and promoted as {schema_active}")
            

            return (
                X_train, y_train.values,
                X_test, y_test.values,
                str(self.preprocessor_path)
            )
        
        except Exception as e:
            print(f"Error during data transformation {e}")
            raise e
