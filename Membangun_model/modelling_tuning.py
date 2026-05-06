import pandas as pd
import os
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train_and_tune():
    # 1. Inisiasi DagsHub
    dagshub.init(repo_owner='AlifFauzan21', repo_name='SMSML_Alif-Fauzan', mlflow=True)
    
    # 2. Setup Eksperimen di MLflow
    mlflow.set_experiment("Breast_Cancer_Tuning_Experiment")
    
    # 3. Load Data hasil preprocessing
    train_path = '../preprocessing/breast_cancer_preprocessing/train.csv'
    test_path = '../preprocessing/breast_cancer_preprocessing/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    with mlflow.start_run(run_name="RF_GridSearchCV"):
        # 4. Hyperparameter Tuning setup (Random Forest)
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        
        # 5. Proses Training
        print("Mulai training dan tuning...")
        grid_search.fit(X_train, y_train)
        
        # Ambil model terbaik hasil tuning
        best_model = grid_search.best_estimator_
        
        # 6. Prediksi & Evaluasi
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 7. LOGGING MANUAL (DILARANG AUTOLOG)
        # Log Parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # Log Model
        mlflow.sklearn.log_model(best_model, "best_random_forest_model")
        
        # 8. Artefak Tambahan (Minimal 2 sesuai syarat Bintang 4)
        os.makedirs("artifacts", exist_ok=True)
        
        # Artefak 1: Gambar Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        
        # Artefak 2: Gambar Feature Importances
        importances = best_model.feature_importances_
        plt.figure(figsize=(10,6))
        pd.Series(importances, index=X_train.columns).nlargest(10).plot(kind='barh')
        plt.title('Top 10 Feature Importances')
        feat_path = "artifacts/feature_importances.png"
        plt.savefig(feat_path)
        plt.close()
        mlflow.log_artifact(feat_path)
        
        print(f"✅ Training selesai! Best Params: {grid_search.best_params_}")
        print(f"✅ Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
        print("✅ Metrik, parameter, dan artefak sudah berhasil dikirim ke DagsHub!")

if __name__ == '__main__':
    train_and_tune()
