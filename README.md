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

```


## 训练和生产模型
`security.cbm`是生产出的模型文件
```python
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

# 假设你有一个包含 HTTP 请求数据的 DataFrame
data = {
    'uri': ['/login', '/home', '/products', '/login', '/about', '/admin'],
    'host': ['example.com', 'example.com', 'example.com', 'example.com', 'example.com', 'example.com'],
    'referer': ['https://www.google.com', 'https://www.google.com', 'https://www.bing.com', 'https://www.yahoo.com', '', ''],
    'cookie': ['session_id=12345;user_id=67890', 'session_id=54321;user_id=09876', 'session_id=98765;user_id=43210', 'session_id=24680;user_id=13579', '', ''],
    'method': ['GET', 'POST', 'GET', 'POST', 'GET', 'POST'],
    'body': ['{"username": "user1", "password": "pass123"}', '{"search_query": "product_name"}', '', '{"username": "user2", "password": "pass456"}', '', '{"admin": true}'],
    'headers': ['Accept-Language: en-US', 'Content-Type: application/json', 'Accept-Encoding: gzip, deflate', 'User-Agent: Mozilla/5.0', '', ''],
    'target_variable': ['safe', 'safe', 'unsafe', 'unsafe', 'safe', 'unsafe']  # 示例的目标变量
}

df = pd.DataFrame(data)

# 对字符串特征进行编码并保存编码器对象
label_encoders = {}
for column in ['uri', 'host', 'referer', 'cookie', 'method', 'body', 'headers']:
    label_encoders[column] = LabelEncoder()
    df[column + '_encoded'] = label_encoders[column].fit_transform(df[column])
    df = df.drop(column, axis=1)

# 划分特征和目标变量
X = df.drop('target_variable', axis=1)
y = df['target_variable']

# 创建并训练 CatBoost 模型
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6)
model.fit(X, y, verbose=10)

# 保存模型
model.save_model("security.cbm")
```

## 调用模型
```python
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

# 加载模型
model = CatBoostClassifier()
model.load_model("security.cbm")

# 刚才示例数据
example_data = [
    {'uri': '/login', 'host': 'example.com', 'referer': 'https://www.google.com', 'cookie': 'session_id=12345;user_id=67890', 'method': 'GET', 'body': '{"username": "user1", "password": "pass123"}', 'headers': 'Accept-Language: en-US'},
    {'uri': '/home', 'host': 'example.com', 'referer': 'https://www.google.com', 'cookie': 'session_id=54321;user_id=09876', 'method': 'POST', 'body': '{"search_query": "product_name"}', 'headers': 'Content-Type: application/json'},
    {'uri': '/admin', 'host': 'example.com', 'referer': '', 'cookie': '', 'method': 'POST', 'body': '{"admin": true}', 'headers': ''}
]

# 对字符串特征进行编码并保存编码器对象
label_encoders = {}
for column in ['uri', 'host', 'referer', 'cookie', 'method', 'body', 'headers']:
    label_encoders[column] = LabelEncoder()
    df[column + '_encoded'] = label_encoders[column].fit_transform(df[column])
    df = df.drop(column, axis=1)

# 对示例数据进行编码
for example in example_data:
    for column in ['uri', 'host', 'referer', 'cookie', 'method', 'body', 'headers']:
        example[column + '_encoded'] = label_encoders[column].transform([example[column]])[0]

# 提取特征
X_example = pd.DataFrame(example_data).drop(['uri', 'host', 'referer', 'cookie', 'method', 'body', 'headers'], axis=1)

# 使用模型进行预测
predictions = model.predict(X_example)

# 打印预测结果
for i, prediction in enumerate(predictions):
    print(f"Example {i+1}: Predicted result is {prediction}")
```

# WAF调用
