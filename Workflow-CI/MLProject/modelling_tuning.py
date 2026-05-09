import pandas as pd
import os
import mlflow
import dagshub
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

def train_and_tune():
    dagshub.init(repo_owner='AlifFauzan21', repo_name='SMSML_Alif-Fauzan', mlflow=True)
    mlflow.set_experiment("Breast_Cancer_Tuning_Experiment")
    
    # Path disesuaikan karena sekarang jalan di dalam folder MLProject
    train_path = 'breast_cancer_preprocessing/train.csv'
    test_path = 'breast_cancer_preprocessing/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    with mlflow.start_run(run_name="RF_GridSearchCV_CI"):
        rf = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [50], 'max_depth': [5], 'min_samples_split': [2]}
        
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        
        mlflow.sklearn.log_model(best_model, "best_random_forest_model")
        
        # Simpan model lokal untuk keperluan build-docker CI
        if os.path.exists("local_model"):
            shutil.rmtree("local_model")
        mlflow.sklearn.save_model(best_model, "local_model")
        print("✅ Training CI selesai! Model lokal disiapkan untuk Docker.")

if __name__ == '__main__':
    train_and_tune()
