import json
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from prepare import preprocess_data, save_label_encoders

# 加载 JSON 数据
with open('data.json', 'r') as f:
    json_data = json.load(f)

# 预处理数据
df, label_encoders = preprocess_data(json_data)

# 划分特征和目标变量
X = df.drop('target_variable_encoded', axis=1)
y = df['target_variable_encoded']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练 CatBoost 模型
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, early_stopping_rounds=10)
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=10)

# 保存模型
model.save_model("catboost_model.cbm")

# 保存编码器
save_label_encoders(label_encoders, "label_encoders.pkl")

# 在测试集上评估模型
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy}")
print(f"Predicted: {y_pred}")
print(f"Actual: {y_test.values}")
