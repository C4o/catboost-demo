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
CatBoost version 1.2.5
NumPy version 1.23.5
Pandas version 1.5.3
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
## HTTP
### openresty client
nginx.conf
```nginx
...
http{
    ...
    server {
        listen       80;
        server_name  _;

        #charset koi8-r;

        #access_log  logs/host.access.log  main;

        rewrite_by_lua_file /usr/local/openresty/lualib/cb_client.lua;
    }
    ...
}
```
cb_client.lua
```lua
local http = require "resty.http"
local cjson = require "cjson"

-- 读取客户端请求的body
ngx.req.read_body()
local body_data = ngx.req.get_body_data()

-- 获取请求的各个特征
local uri = ngx.var.uri
local host = ngx.var.host
local referer = ngx.req.get_headers()["referer"] or ""
local cookie = ngx.var.http_cookie or ""
local method = ngx.req.get_method()
local body = body_data or ""
local ua = ngx.req.get_headers()["user-agent"] or ""
local headers = ngx.req.get_headers()

-- 拼接成JSON格式
local request_data = {
    uri = uri,
    host = host,
    referer = referer,
    cookie = cookie,
    method = method,
    body = body,
    -- headers = cjson.encode(headers)
    headers = "User-Agent: "..ua
}

-- 创建HTTP客户端实例
local httpc = http.new()

-- 发送请求到本地的Python服务器
local res, err = httpc:request_uri("http://127.0.0.1:5000/predict", {
    method = "POST",
    body = cjson.encode(request_data),
    headers = {
        ["Content-Type"] = "application/json",
    }
})

if not res then
    ngx.status = ngx.HTTP_INTERNAL_SERVER_ERROR
    ngx.say("Failed to request: ", err)
    return
end

-- 解析响应
local prediction = res.headers["X-Prediction"]

if not prediction then
    ngx.status = ngx.HTTP_INTERNAL_SERVER_ERROR
    ngx.say("Prediction not found in response headers")
    return
end

-- 将预测结果返回给客户端
ngx.status = res.status
if predict == 1 then
    ngx.status = 403
    ngx.say("Prediction is DENY")
end
```

### python server
cb_server.py
```python
from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier
import joblib

app = Flask(__name__)

# 加载模型和编码器
model = CatBoostClassifier()
model.load_model("security.cbm")
label_encoders = joblib.load("security_label_encoders.pkl")

def preprocess_data(json_data):
    if isinstance(json_data, dict):
        json_data = [json_data]
    
    df = pd.DataFrame(json_data)
    encoded_data = {}
    for column in df.columns:
        if column in label_encoders:
            if df[column].dtype == object:
                df[column + '_encoded'] = df[column].apply(lambda x: label_encoders[column].transform([x])[0] if x in label_encoders[column].classes_ else -1)
                encoded_data[column + '_encoded'] = df[column + '_encoded'].tolist()
            else:
                encoded_data[column] = df[column].tolist()
    return pd.DataFrame(encoded_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_example = preprocess_data(data)
    predictions = model.predict(X_example)
    return '', 200, {'X-Prediction': str(predictions[0])}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```
### 手动请求测试
```shell
$ cat cb_body.txt
{"uri": "/admin", "host": "example.com", "referer": "", "cookie": "", "method": "POST", "body": "{\"admin\": true}", "headers": "User-Agent: chaitin-bypass"}

$ curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @cb_body.txt -v
Note: Unnecessary use of -X or --request, POST is already inferred.
*   Trying [::1]:5000...
*   Trying 127.0.0.1:5000...
* Connected to localhost (127.0.0.1) port 5000
> POST /predict HTTP/1.1
> Host: localhost:5000
> User-Agent: curl/8.4.0
> Accept: */*
> Content-Type: application/json
> Content-Length: 157
>
< HTTP/1.1 200 OK
< Server: Werkzeug/2.2.3 Python/3.10.10
< Date: Tue, 21 May 2024 06:21:50 GMT
< X-Prediction: 1
< Content-Type: text/html; charset=utf-8
< Content-Length: 0
< Connection: close
<
* Closing connection
```
## FFI - TODO
