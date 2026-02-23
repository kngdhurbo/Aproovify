import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def main():
    print("Loading dataset...")
    df = pd.read_csv('loan_approval_dataset.csv')
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
        
    print("Dataset loaded. Shape:", df.shape)

    # Drop project/loan ID as it's not a predictor
    if 'loan_id' in df.columns:
        df = df.drop('loan_id', axis=1)

    X = df.drop('loan_status', axis=1)
    y = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print("Numerical Columns:", numerical_cols)
    print("Categorical Columns:", categorical_cols)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)
    print(f"Model trained successfully. Test Accuracy: {score:.4f}")

    print("Saving model pipeline...")
    joblib.dump(pipeline, 'model.pkl')
    print("Model saved to model.pkl")

if __name__ == "__main__":
    main()
