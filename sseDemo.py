# 导入所需的模块
import json
import time
import datetime
from flask import Flask, request, Response, render_template
from transformers import AutoModel, AutoTokenizer
import torch
from peft import PeftModel

app = Flask(__name__)

torch.set_default_dtype(torch.float16)
model = AutoModel.from_pretrained(
    "/home/life/.cache/huggingface/hub/models--THUDM--chatglm2-6B/snapshots/7fabe56db91e085c9c027f56f1c654d137bdba40",
    trust_remote_code=True, device_map='auto')
model = PeftModel.from_pretrained(model, "./output")
torch.set_default_dtype(torch.float16)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/life/.cache/huggingface/hub/models--THUDM--chatglm2-6B/snapshots/7fabe56db91e085c9c027f56f1c654d137bdba40",
    trust_remote_code=True)
device = "cpu"


# 解决跨域问题
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


# 获取当前时间，并转换为 JSON 格式
def get_time_json():
    dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    return json.dumps({'time': dt_ms}, ensure_ascii=False)


# 设置路由，返回 SSE 流
@app.route('/')
def hello_world():
    return render_template('sse.html')


@app.route('/sse', methods=['POST'])
def stream():
    jsonData = request.get_json() # 可选，用于区分不同用户的连接
    input_text = jsonData['content']
    print(input_text)
    def eventStream():
        with torch.no_grad():
            # ids = tokenizer.encode(input_text)
            # input_ids = torch.LongTensor([ids]).to(device)
            current_length = 0
            past_key_values, history = None, []
            for response, history, past_key_values in model.stream_chat(tokenizer, input_text, history=history,
                                                                        past_key_values=past_key_values,
                                                                        return_past_key_values=True):
                print(response[current_length:], end="", flush=True)
                event_name = 'time_reading'
                str_out = f'id: {id}\nevent: {event_name}\ndata: {response[current_length:]}\n\n'
                print(str_out)  # 在服务器端打印发送的数据
                yield str_out
              
                current_length = len(response)

             
    return Response(eventStream(), mimetype="text/event-stream")




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
