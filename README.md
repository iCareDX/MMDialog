# MMDialog
Multi modal dialog engine by LLM
Webサーバ型の複数のLLMプログラムを試しています．

1．Streamlitによるサーバ
app.py
startapp.shでたちあげます．

2．2つのサーバの連携による長谷川式プロンプトテスト
llama_cpp_server.py
llama_cpp_client.py
2つを立ち上げて，localhost:7860でアクセスします．

3．OpenAI互換サーバ
startserver.sh