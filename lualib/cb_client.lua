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
