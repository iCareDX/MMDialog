import requests
import json
import gradio as gr

talk_log_list=[[]]

# FastAPIエンドポイントのURL
url = 'http://0.0.0.0:8005/generate/'  # FastAPIサーバーのURLに合わせて変更してください

def  genereate(sys_msg, user_query,user,max_token,get_temperature , talk_log_list,log_f,log_len, repeat_penalty, top_k , top_p, frequency_penalty):
    #  POSTリクエスト・ボディー
    data = {"sys_msg" : sys_msg,
                    "user_query":user_query,
                    "user":user,
                    "max_token":max_token,
                    "temperature":get_temperature,
                    "talk_log_list":talk_log_list,
                    "log_f":log_f,
                    "log_len":log_len,
                    "repeat_penalty":repeat_penalty,
                    "top_k":top_k,
                    "top_p":top_p,
                    "frequency_penalty":frequency_penalty,
                }

    # POSTリクエストを送信
    response = requests.post(url, json=data)
    # 返信を評価
    if response.status_code == 200:
        result = response.json()
        log_list=result.get("log_list"),
        all_out=result.get("all_out"),
        prompt=result.get("prompt"),
        talk_log_list=result.get("talk_log_list"),
        return result.get("out"), all_out, prompt, talk_log_list
    else:
        return response.status_code

# Gradioからアクセスするときの関数、talk_log_listを保持したりクリアするため
def  gradio_genereate(sys_msg, user_query,user,max_token,get_temperature, log_f, log_len, repeat_penalty, top_k , top_p, frequency_penalty ):
    global talk_log_list
    out, all_out, prompt ,talk_log_list=genereate(sys_msg, user_query,user,max_token,get_temperature, talk_log_list,log_f,log_len, repeat_penalty, top_k , top_p, frequency_penalty)
    return  out, all_out, prompt,talk_log_list
def gradio_clr():
    global talk_log_list
    talk_log_list=[[]]

# GradioのUIを定義します
with gr.Blocks() as webui:
    gr.Markdown("japanese-stablelm-instruct-alpha-7b-v2 prompt test")
    with gr.Row():
          with gr.Column():
            sys_msg           = gr.Textbox(label="sys_msg", placeholder=" システムプロンプト")
            user_query    =gr.Textbox(label="user_query", placeholder="命令を入力してください")
            user                    =gr.Textbox(label="入力", placeholder="ユーザーの会話を入力してください")
            with gr.Row():
                log_len              =gr.Number(5, label="履歴ターン数")
                log_f                   =gr.Checkbox(True, label="履歴有効・無効")
            with gr.Row():
                max_token        = gr.Number(400, label="max out token:int")
                temperature     = gr.Number(0.8, label="temperature:float")
                repeat_penalty= gr.Number(1.1, label="repeat_penalty:float")
                top_k                     = gr.Number(40, label="top_k:int")
                top_p                    = gr.Number(0.95, label="top_p:float")
                frequency_penalty=gr.Number(0.0, label=" frequency_penalty:float")
            with gr.Row():
                prompt_input   = gr.Button("Submit prompt",variant="primary")
                log_clr   = gr.Button("ログクリア",variant="secondary")
          with gr.Column():
             out_data=[gr.Textbox(label="システム"),
                                 gr.Textbox(label="tokenizer全文"),
                                 gr.Textbox(label="プロンプト"),
                                 gr.Textbox(label="会話ログリスト")]
    prompt_input.click(gradio_genereate, inputs=[sys_msg, user_query,  user, max_token, temperature,log_f,log_len,repeat_penalty,top_k ,top_p, frequency_penalty], outputs=out_data )
    log_clr  .click(gradio_clr)
# Gradioアプリケーションを起動します
webui.launch()