import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import joblib


def main():
    # Load dataset
    df = pd.read_csv('heart_disease.csv')
    X = df.drop('Heart Disease Status', axis=1)
    y = df['Heart Disease Status']

    # Identify categorical and numeric columns
    categorical_features = X.select_dtypes(include=['object']).columns
    numeric_features = X.select_dtypes(exclude=['object']).columns

    # Preprocess: One-hot encode categorical variables
    preprocess = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ]
    )

    # LGBM model
    model = LGBMClassifier()

    # Combine preprocessing and model into a pipeline
    clf = Pipeline(steps=[('preprocess', preprocess), ('model', model)])

    # Train model
    clf.fit(X, y)

    # Save the trained pipeline
    joblib.dump(clf, 'lgbm_model.pkl')


if __name__ == '__main__':
    main()
