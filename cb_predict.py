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
