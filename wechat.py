from wxpy import *
import random
from http import HTTPStatus
from dashscope import Generation
from ollama import Client
from collections import defaultdict

cur_model_name = "qwen"


# 初始化机器人，扫码登录
bot = Bot()

client = Client(host='http://localhost:11434')

# 使用 defaultdict 简化字典初始化
chatMapping = defaultdict(lambda: [{"role": "system", "content": "你来参与聊天."}])

def gen_response_qwen(msg):
    obj = msg.chat.name
    cur_message = {'role': 'user', 'content': msg.text}
    
    # 维护消息历史记录并确保长度不超过 5 条
    chatMapping[obj].append(cur_message)
    if len(chatMapping[obj]) > 5:
        chatMapping[obj].pop(1)
    
    try:
        response = Generation.call(model="qwen-turbo",
                                   messages=chatMapping[obj],
                                   seed=random.randint(1, 10000),
                                   result_format='message')
        if response.status_code == HTTPStatus.OK:
            assistant_output = response['output']['choices'][0]['message']
            chatMapping[obj].append(assistant_output)
            return assistant_output['content']
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message))
            return "对不起，发生了一些错误。"
    except Exception as e:
        print(f"Error generating response from Qwen-Turbo: {e}")
        return "对不起，生成回复时发生了错误。"

def gen_response_llama3(msg):
    try:
        response = client.chat(model='llama3:8b', messages=[
            {
                'role': 'user',
                'content': '你是xxx，你替他回复以下消息：' + msg.text,
            },
        ])['message']['content']
        return response
    except Exception as e:
        print(f"Error generating response from LLaMA 3: {e}")
        return "对不起，生成回复时发生了错误。"

def generate_response(msg):
    global cur_model_name

    # 切换模型的逻辑
    model_switch = {
        'qwen': 'Qwen-Turbo 模型',
        'llama': 'LLaMA 3 模型'
    }
    
    model_key = msg.text.lower()
    if model_key in model_switch:
        cur_model_name = model_key
        return f"已切换到 {model_switch[model_key]}。"
    
    if cur_model_name == "qwen":
        return gen_response_qwen(msg)
    else:
        return gen_response_llama3(msg)

myself = bot.self

@bot.register(Group, [TEXT])
def auto_reply_group(msg):
    if msg.chat.name == 'xxxx':
        reply = generate_response(msg)
        msg.reply(reply)

@bot.register(Friend, [TEXT])
def auto_reply_friend(msg):
    reply = generate_response(msg)
    msg.reply(reply)

bot.join()
