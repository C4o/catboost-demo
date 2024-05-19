import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data(json_data, label_encoders=None):
    df = pd.DataFrame(json_data)

    # 对字符串特征进行编码
    if label_encoders:
        for column in df.columns:
            if column in label_encoders:
                if df[column].dtype == object:
                    # 处理训练期间未见过的类别
                    df[column + '_encoded'] = df[column].apply(lambda x: label_encoders[column].transform([x])[0] if x in label_encoders[column].classes_ else -1)
                    df = df.drop(column, axis=1)
    else:
        label_encoders = {}
        for column in df.columns:
            if df[column].dtype == object:
                label_encoders[column] = LabelEncoder()
                df[column + '_encoded'] = label_encoders[column].fit_transform(df[column])
                df = df.drop(column, axis=1)

    return df, label_encoders

def save_label_encoders(label_encoders, filepath):
    joblib.dump(label_encoders, filepath)

def load_label_encoders(filepath):
    return joblib.load(filepath)
