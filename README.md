# catboost
## 环境安装
```shell
// 安装基础库
pip install -q numpy pandas catboost

// 验证安装结果
$ cat cb.py 
import catboost as cb
import catboost.datasets as cbd
import numpy as np
import pandas as pd

# print module versions for reproducibility
print('CatBoost version {}'.format(cb.__version__))
print('NumPy version {}'.format(np.__version__))
print('Pandas version {}'.format(pd.__version__))

$ python cb.py
CatBoost version 0.14.2
NumPy version 1.16.3
Pandas version 0.24.2
```

## 准备和预处理数据
```python
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
```


## 训练和生产模型
`security.cbm`是生产出的模型文件
```python
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
model.save_model("security.cbm")

# 保存编码器
save_label_encoders(label_encoders, "security_label_encoders.pkl")

# 在测试集上评估模型
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy}")
print(f"Predicted: {y_pred}")
print(f"Actual: {y_test.values}")
```
### output
```shell
$ python cb_train.py  
0:      learn: 0.6822075        test: 0.6789635 best: 0.6789635 (0)     total: 137ms    remaining: 13.6s
10:     learn: 0.6044272        test: 0.6087474 best: 0.6087474 (10)    total: 140ms    remaining: 1.13s
20:     learn: 0.5385005        test: 0.5483047 best: 0.5480958 (19)    total: 141ms    remaining: 532ms
30:     learn: 0.4829111        test: 0.5150429 best: 0.5150429 (30)    total: 143ms    remaining: 319ms
40:     learn: 0.4405540        test: 0.4843827 best: 0.4843827 (40)    total: 145ms    remaining: 209ms
50:     learn: 0.4022986        test: 0.4764177 best: 0.4737219 (48)    total: 148ms    remaining: 142ms
60:     learn: 0.3703142        test: 0.4437111 best: 0.4437111 (60)    total: 150ms    remaining: 96.1ms
70:     learn: 0.3422533        test: 0.4113572 best: 0.4113572 (70)    total: 153ms    remaining: 62.3ms
80:     learn: 0.3175111        test: 0.3858701 best: 0.3858701 (80)    total: 155ms    remaining: 36.3ms
90:     learn: 0.2956228        test: 0.3603579 best: 0.3603579 (90)    total: 157ms    remaining: 15.5ms
99:     learn: 0.2780358        test: 0.3476682 best: 0.3473103 (96)    total: 159ms    remaining: 0us

bestTest = 0.3473103455
bestIteration = 96

Test Accuracy: 1.0
Predicted: [0 0]
Actual: [0 0]
```

## 调用模型
```python
import json
import pandas as pd
from catboost import CatBoostClassifier
from prepare import preprocess_data, load_label_encoders

# 加载模型
model = CatBoostClassifier()
model.load_model("security.cbm")

# 加载编码器
label_encoders = load_label_encoders("security_label_encoders.pkl")

# 示例数据，包括一条新的 unsafe 验证数据
example_data = '''
[
    {"uri": "/login", "host": "example.com", "referer": "https://www.google.com", "cookie": "session_id=12345;user_id=67890", "method": "GET", "body": "{\\"username\\": \\"user1\\", \\"password\\": \\"pass123\\"}", "headers": "Accept-Language: en-US"},
    {"uri": "/home", "host": "example.com", "referer": "https://www.google.com", "cookie": "session_id=54321;user_id=09876", "method": "POST", "body": "{\\"search_query\\": \\"product_name\\"}", "headers": "Content-Type: application/json"},
    {"uri": "/admin", "host": "example.com", "referer": "", "cookie": "", "method": "POST", "body": "{\\"admin\\": true}", "headers": "User-Agent: chaitin-bypass"}
]
'''

example_data = json.loads(example_data)

# 预处理示例数据
X_example, _ = preprocess_data(example_data, label_encoders)

# 使用模型进行预测
predictions = model.predict(X_example)

# 打印预测结果
for i, prediction in enumerate(predictions):
    print(f"Example {i+1}: Predicted result is {prediction}")
```

### output 
```shell
$ python cb_predict.py
Example 1: Predicted result is 0
Example 2: Predicted result is 0
Example 3: Predicted result is 1
```

# WAF调用
