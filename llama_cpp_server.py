from llama_cpp import Llama
from fastapi import FastAPI,Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# LLMの準備
"""Load a llama.cpp model from `model_path`.
            model_path: Path to the model.
            seed: Random seed. -1 for random.
            n_ctx: Maximum context size.
            n_batch: Maximum number of prompt tokens to batch together when calling llama_eval.
            n_gpu_layers: Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
            main_gpu: Main GPU to use.
            tensor_split: Optional list of floats to split the model across multiple GPUs. If None, the model is not split.
            rope_freq_base: Base frequency for rope sampling.
            rope_freq_scale: Scale factor for rope sampling.
            low_vram: Use low VRAM mode.
            mul_mat_q: if true, use experimental mul_mat_q kernels
            f16_kv: Use half-precision for key/value cache.
            logits_all: Return logits for all tokens, not just the last token.
            vocab_only: Only load the vocabulary no weights.
            use_mmap: Use mmap if possible.
            use_mlock: Force the system to keep the model in RAM.
            embedding: Embedding mode only.
            n_threads: Number of threads to use. If None, the number of threads is automatically determined.
            last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.
            lora_base: Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.
            lora_path: Path to a LoRA file to apply to the model.
            numa: Enable NUMA support. (NOTE: The initial value of this parameter is used for the remainder of the program as this value is set in llama_backend_init)
            verbose: Print verbose output to stderr.
            kwargs: Unused keyword arguments (for additional backwards compatibility).
"""
sys_msg_HDSR = """
以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。
リクエストを適切に完了するための回答を記述してください。
"""
user_msg_HDSR = """
AIは以下の条件を元にstep-by-stepでユーザスロットを埋めていきます。
        
        1. ユーザの発話に「診断して」または「認知症診断をして」という要求があったとき、質問を開始する。
        2. AIはユーザに対して、1問ずつ会話形式で質問をする。(対話履歴を参考にする)
        3. 2の質問で得られた回答を元にユーザスロットを一つずつ順番に埋める。
        4. 会話で得られた情報を元にユーザスロットに埋めていく。
        5. ユーザの発話が聞き返しやスキップなどに該当する場合、それに応じた遷移をする。
        6. 分からないと言われた場合、ヒントを教えるか、次の問題に移る。
        7. 最後のスロットが埋まったら診断を終える。
        
    
## ユーザスロット:
        1. 年齢
        2. 今日の年,月,日,曜日
        3. ユーザが現在居る場所
        4. 三つの言葉の復唱(桜、猫、電車)
        5. 100引く7の計算
        6. 93引く7の計算
        7. 次の数字を逆唱(2、8、6)
        8. 次の数字を逆唱(3、5、2、9)
        9. 復唱した三つの言葉を覚えているか
        10. 知っている野菜をできるだけ多く答えてもらう(10個を超えたら終了)
        
## 条件:
        ユーザスロット6と8は、それぞれユーザスロット5と7が正解だったときのみ埋める必要がある。
            
対話例
###　入力:
診断してください。

###　応答:
わかりました。まずあなたのお名前を教えてください。

###　入力:
沼尾雅之です

###　応答:
沼尾さん、あなたの年齢を教えて下さい。

###　入力:
65歳です

###　応答:
今日は何年何月何日何曜日ですか。

###　入力:
今日は2024年3月10日です

###　応答:
今日は何曜日ですか。

###　入力:
日曜日

###　応答:
今いる場所を教えてください。

###　入力:
今は電気通信大学の研究室にいます

###　応答:
これから言う3つの言葉を繰り返してください。桜、猫、電車。

###　入力:
桜、猫、電車

###　応答:
計算の問題です。100引く7はいくつですか。

###　入力: 
93です

###　応答:
そこから、さらに7を引くといくつですか。

###　入力: 
うーんと、86です

###　応答:
これから言う3つの数字を逆から言ってください。2、8、6。

###　入力: 
6、8、2

###　応答:
今度は4つの数字を逆から言ってください。3、5、2、9。

###　入力: 
9、2、3、5

###　応答:
さきほど覚えた3つの言葉を教えてください。

###　入力:
えーと、桜、猫、電車だったかな

###　応答:
最後の質問です。知っている野菜の名前をすべて言ってください。

###　入力: 
大根、きゅうり、ごぼう、人参

###　応答: 
まだありますか。

###　入力: 
じゃがいも、ほうれん草、かぼちゃ

###　応答: 
まだありますか。

###　入力: 
ええと，もう思い出せない

###　応答: 
それでは、これで診断を終わります。

"""
llm = Llama(model_path=".model/Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
               n_gpu_layers=35,
               n_ctx=2048
                 )

app = FastAPI()

class AnswerRequest(BaseModel):
     sys_msg : str
     user_query:str
     user:str
     talk_log_list:list =[[]]
     log_f:bool = False
     log_len :int = 0
     max_token:int = 256
     temperature:float = 0.8
     repeat_penalty:float =  1.1
     top_k:int  = 40
     top_p:float = 0.95
     frequency_penalty:float = 0.0

@app.post("/generate/")
def  genereate(gen_request: AnswerRequest):
    sys_msg         =gen_request.sys_msg
    user_query  =gen_request.user_query
    user                  =gen_request.user
    talk_log_list=gen_request.talk_log_list
    log_f                =gen_request.log_f
    log_len           =gen_request.log_len
    max_token =gen_request.max_token
    top_k              =gen_request.top_k
    top_p              =gen_request.top_p
    get_temperature     =gen_request.temperature
    repeat_penalty         =gen_request.repeat_penalty
    frequency_penalty =gen_request.frequency_penalty
    print("top_k:",top_k,"top_p:",top_p,"get_temperature :",get_temperature ,"repeat_penalty:",repeat_penalty,"frequency_penalty:",frequency_penalty)

    talk_log_list= talk_log_list[0]
     
    prompt = sys_msg+"\n\n" + "### 指示: "+"\n" + user_query + "\n\n"  +  "### 入力:" +"\n"+ user + "\n\n"  +  "### 応答:"
    print("-------------------talk_log_list-----------------------------------------------------")
    print("talk_log_list",talk_log_list)  

    #会話ヒストリ作成。プロンプトに追加する。
    log_len = int(log_len)
    if  log_f==True and log_len >0: # 履歴がTrueでログ数がゼロでなければtalk_log_listを作成
        sys_prompt=prompt.split("### 入力:")[0]
        talk_log_list.append( " \n\n"+ "### 入力:"+ " \n" + user+ " \n" )
        new_prompt=""
        for n in range(len(talk_log_list)):
            new_prompt=new_prompt + talk_log_list[n]
        prompt= sys_prompt + new_prompt+" \n \n"+ "### 応答:"+" \n"
    # 推論の実行
        """Sample a token from the model.
            top_k: The top-k sampling parameter.
            top_p: The top-p sampling parameter.
            temp: The temperature parameter.
            repeat_penalty: The repeat penalty parameter.
        Returns:
            The sampled token.
              # デフォルトパラメータ
               top_k: int = 40,
               top_p: float = 0.95,
               temp: float = 0.80,
               repeat_penalty: float = 1.1,
               frequency_penalty: float = 0.0,
               presence_penalty: float = 0.0,
               tfs_z: float = 1.0,
               mirostat_mode: int = 0,
               mirostat_eta: float = 0.1,
               mirostat_tau: float = 5.0,
               penalize_nl: bool = True,
        """
    print("-----------------prompt---------------------------------------------------------")
    print(prompt)
    output = llm(
        prompt,
        stop=["### 入力","\n\n### 指示"],
        max_tokens=max_token,
        top_k = top_k ,
        top_p = top_p,
        temperature=get_temperature,
        repeat_penalty=repeat_penalty,
        frequency_penalty  =frequency_penalty,
        echo=True,
        )
    print('------------------output["choices"][0]-------------------------------------------------')
    print(output["choices"][0])
    #output の"### 応答:"のあとに、"###"がない場合もあるので、ない場合は最初の"### 応答:"を選択
    try:
             ans = ans=output["choices"][0]["text"].split("### 応答:")[1].split("###")[0]
    except:
             ans = output["choices"][0]["text"].split("### 応答:")[1]
    print("-----------------final ans  ----------------------------------------------------------")
    print(ans)
    if len(talk_log_list)>log_len:
        talk_log_list=talk_log_list[2:] #ヒストリターンが指定回数を超えたら先頭(=一番古い）の会話（入力と応答）を削除
    talk_log_list.append("\n" +"###"+  "応答:"+"\n" + ans .replace("\n" ,""))
    result=200
    return {'message':result, "out":ans,"all_out":output,"prompt":prompt,"talk_log_list":talk_log_list }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
