# Reference Implementations

このディレクトリには、開発時の参考実装を格納しています。

## ファイル一覧

### llm_client.py

Azure OpenAI / OpenAI の自動検出クライアントの汎用実装です。

**本番コードでの使用箇所**: `src/seasonality/debug/llm_client.py`

このファイルは、別プロジェクト（Business-flow-maker）から移植した汎用LLMクライアントです。
季節性分析パイプラインでは、デバッグ診断機能に特化した形で`debug/llm_client.py`に実装されています。

#### 主な機能

- Azure OpenAI と OpenAI の自動検出
- 環境変数の検証（ダミー値の検出）
- JSON Schema に準拠した構造化出力

#### 使用方法（参考）

```python
from llm_client import create_llm_client

# 自動検出でクライアント生成
client = create_llm_client()

# 構造化出力のリクエスト
result = client.structured_flow(
    prompt="Analyze this data...",
    schema={"type": "object", "properties": {...}},
    model="gpt-4o"
)
```

## 注意事項

- このディレクトリのコードは**直接使用しないでください**
- 本番コードは`src/seasonality/`内にあります
- 参考実装として残しているだけです
