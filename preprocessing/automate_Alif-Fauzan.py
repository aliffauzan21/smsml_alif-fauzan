import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # 1. Load Data mentah
    raw_path = '../breast_cancer_raw/dataset.csv'
    df = pd.read_csv(raw_path)

    # 2. Pisahkan fitur dan target
    X = df.drop('target', axis=1)
    y = df['target']

    # 3. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Gabungkan kembali ke DataFrame biar rapi
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['target'] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['target'] = y_test.values

    # 6. Simpan hasil preprocessing
    output_dir = 'breast_cancer_preprocessing'
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print("✅ Preprocessing selesai! Data bersih disimpan di folder:", output_dir)

if __name__ == '__main__':
    preprocess_data()
