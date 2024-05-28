cat client.lua
local http = require "resty.http"
local cjson = require "cjson"
local t1k = require "resty.t1k"

local chaitin_server_addr = "10.18.47.200"
local modsecurity_server_addr = "10.18.47.199"

ngx.req.read_body()
local body_data = ngx.req.get_body_data()

local uri = ngx.var.uri
local host = ngx.var.host
local referer = ngx.req.get_headers()["referer"] or ""
local cookie = ngx.var.http_cookie or ""
local method = ngx.req.get_method()
local body = body_data or ""
local ua = ngx.req.get_headers()["user-agent"] or ""
local headers = ngx.req.get_headers()

-- modsecurity
local httpc = http.new()
local res, err = httpc:request_uri("http://".. modsecurity_server_addr .. ngx.var.request_uri, {
    method = ngx.req.get_method(),
    headers = ngx.req.get_headers(),
    body = body,

})

if not res then
    ngx.say("Failed to request to modsecurity: ", err)
end
if res and res.status == ngx.HTTP_FORBIDDEN then
    ngx.status = ngx.HTTP_FORBIDDEN
    ngx.say("denied by modsecurity")
end


-- chaitin
local t = {
    mode = "block",
    host = chaitin_server_addr,
    port = 8000,
    connect_timeout = 1000,
    send_timeout = 1000,
    read_timeout = 1000,
    req_body_size = 1024,
    keepalive_size = 256,
    keepalive_timeout = 60000,
    remote_addr = "http_x_forwarded_for: " .. ngx.var.remote_addr,
}
local ok, err, _ = t1k.do_access(t, true)
if not ok then
    ngx.log(ngx.ERR, err)
end


-- catboost prediction
local request_data = {
    uri = uri,
    host = host,
    referer = referer,
    cookie = cookie,
    method = method,
    body = body,
    headers = "User-Agent: "..ua
}

local encoder = cjson.new()
encoder.encode_sparse_array(true, 1, 0)
encoder.encode_escape_forward_slash(false)

--init httpc
local httpc = http.new()

local res, err = httpc:request_uri("http://127.0.0.1:5000/predict", {
    method = "POST",
    body = encoder.encode(request_data),
    headers = {
        ["Content-Type"] = "application/json",
    }
})

if not res then
    ngx.status = ngx.HTTP_INTERNAL_SERVER_ERROR
    ngx.say("Failed to request: ", err)
    return
end

local prediction = res.headers["X-Prediction"]

if not prediction then
    ngx.status = ngx.HTTP_INTERNAL_SERVER_ERROR
    ngx.say("Prediction not found in response headers")
    return
end

if prediction == "1" then
    ngx.status = ngx.HTTP_FORBIDDEN
    ngx.say("Prediction is DENY")
end