# Pipeline Phase - 本番運用パイプライン

このディレクトリは、季節性検出の**本番運用向けCLIツール**を提供します。

## 目的

- CSVデータから季節性を自動検出
- 複数センサーの一括処理とランキング生成
- 設定ファイルによる柔軟なパラメータ管理
- ログ出力とデバッグ支援機能
- LLMによる診断サポート（オプション）

## ディレクトリ構成

```
pipeline/
├── README.md                    # このファイル
├── pyproject.toml               # パッケージ設定
├── setup.py                     # インストール用
├── requirements.txt             # 依存関係
├── config/                      # 設定ファイル
│   ├── default.yaml             # 標準設定
│   ├── strict_mode.yaml         # 厳格モード
│   └── exploratory_mode.yaml    # 探索モード
├── src/seasonality/             # ソースコード
│   ├── __init__.py
│   ├── cli.py                   # CLIエントリーポイント
│   ├── config.py                # Pydantic設定モデル
│   ├── io/                      # 入出力
│   ├── preprocessing/           # 前処理
│   ├── detection/               # 季節性検出
│   ├── scoring/                 # スコアリング
│   ├── visualization/           # 可視化
│   ├── debug/                   # デバッグ・LLM
│   └── utils/                   # ユーティリティ
├── tests/                       # テストコード
├── reference/                   # 参考実装
└── results/                     # 出力結果（自動生成）
```

## セットアップ

### 1. 仮想環境の作成（推奨）

```bash
# プロジェクトルートから
cd pipeline
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. パッケージのインストール

```bash
# 開発モードでインストール
pip install -e .

# または依存関係のみ
pip install -r requirements.txt
```

### 3. LLM機能を使用する場合（オプション）

```bash
pip install -e ".[llm]"

# 環境変数の設定（../.env.exampleを参照）
cp ../.env.example ../.env
# .envファイルを編集してAPIキーを設定
```

## 基本的な使い方

### CLIコマンド一覧

```bash
# ヘルプを表示
seasonality --help

# バージョン確認
seasonality --version
```

### `seasonality run` - 季節性分析の実行

```bash
# 基本的な実行
seasonality run --data ../data/sample_sensor_data.csv

# 設定ファイルを指定
seasonality run --config config/strict_mode.yaml --data ../data/sensor_data.csv

# 出力先を指定
seasonality run --data ../data/sensor_data.csv --output results/run_001

# 特定のセンサーのみ分析
seasonality run --data ../data/sensor_data.csv --sensors sensor_1,sensor_2,sensor_3

# 図表生成を無効化（高速化）
seasonality run --data ../data/sensor_data.csv --no-figures

# ログレベルを指定
seasonality run --data ../data/sensor_data.csv --log-level DEBUG
```

### `seasonality init` - 設定ファイルの生成

```bash
# カレントディレクトリに設定ファイルを生成
seasonality init

# 出力先を指定
seasonality init --output my_config/
```

生成されるファイル：
- `default.yaml` - 標準設定
- `strict_mode.yaml` - 厳格モード
- `exploratory_mode.yaml` - 探索モード

### `seasonality debug` - デバッグバンドルの作成

```bash
# ログファイルからデバッグバンドルを作成
seasonality debug --log results/logs/run.log

# 出力先を指定
seasonality debug --log results/logs/run.log --bundle debug_bundles/
```

### `seasonality diagnose` - LLMによる診断

```bash
# デバッグバンドルをLLMで診断
seasonality diagnose --bundle debug_bundles/bundle_001.json

# 匿名化して診断
seasonality diagnose --bundle debug_bundles/bundle_001.json --anonymize
```

## 設定ファイル

### default.yaml（標準設定）

```yaml
data:
  date_column: "date"
  time_column: "time"
  sensor_columns: null  # 自動検出

preprocessing:
  resample_freq: "D"
  resample_method: "mean"
  interpolation_method: "linear"
  outlier_method: "iqr"
  outlier_threshold: 1.5

detection:
  methods:
    - stl
    - acf
    - fourier
  periods:
    - 7
    - 30
    - 365

scoring:
  mode: "strict"
  weights:
    stl: 0.3
    acf: 0.3
    fourier: 0.4
```

### モード別閾値

| モード | STL F_season | ACF peak | Fourier ΔR² |
|--------|--------------|----------|-------------|
| strict | > 0.5 | > 0.3 | > 0.1 |
| exploratory | > 0.3 | > 0.2 | > 0.05 |

## 出力ファイル

`seasonality run`実行後、以下のファイルが生成されます：

```
results/
├── seasonality_scores.csv       # 全センサーのスコア
├── seasonality_ranking.csv      # ランキング
├── summary_report.html          # HTMLサマリーレポート
├── figures/                     # 可視化結果
│   ├── stl_decomposition_*.png
│   ├── acf_analysis_*.png
│   ├── periodogram_*.png
│   └── thumbnail_grid.png
└── logs/
    └── run_YYYYMMDD_HHMMSS.log  # 実行ログ
```

## テストの実行

```bash
# 全テストを実行
pytest tests/ -v

# カバレッジレポート付き
pytest tests/ -v --cov=src/seasonality --cov-report=term-missing

# 特定のテストのみ
pytest tests/test_detection.py -v
```

## プログラムからの利用

```python
from seasonality.detection.ensemble import EnsembleDetector
from seasonality.io.loader import load_sensor_data
from seasonality.scoring.ranking import create_scores_dataframe, generate_ranking

# データ読み込み
df = load_sensor_data("../data/sample_sensor_data.csv")

# 検出器の初期化
detector = EnsembleDetector(methods=["stl", "acf", "fourier"])

# 一括検出
results = detector.detect_batch(
    df,
    sensor_cols=["sensor_1", "sensor_2", "sensor_3"],
    periods=[7, 30, 365]
)

# スコアリングとランキング
scores_df = create_scores_dataframe(results)
ranking = generate_ranking(scores_df)
print(ranking)
```

## 処理フロー

```
1. 設定読み込み（YAML + CLI引数）
       ↓
2. データ読み込み → ヘルスチェック①
       ↓
3. 前処理パイプライン
   - リサンプリング（日次）
   - 補間（欠測値処理）
   - 外れ値検出・除去
   - 変化点検出（オプション）
       ↓
4. 季節性検出ループ（センサー単位）
   - STL分解
   - ACF分析
   - Fourier回帰
       ↓
5. スコアリング・ランキング
       ↓
6. 可視化・レポート生成
       ↓
7. 結果出力 → ヘルスチェック②
```

## LLM診断機能

### 環境変数の設定

```bash
# Azure OpenAI（優先）
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# または OpenAI
OPENAI_API_KEY=your-api-key
```

### 診断の流れ

1. `seasonality run`でエラーが発生
2. `seasonality debug --log logs/run.log`でバンドル作成
3. `seasonality diagnose --bundle bundle.json`でLLM診断
4. 診断結果に基づいて設定を調整

## トラブルシューティング

### インストールエラー

```bash
# 依存関係の競合を解決
pip install --upgrade pip
pip install -e . --no-cache-dir
```

### メモリ不足

```yaml
# config/default.yaml
output:
  max_sensors_per_batch: 10  # バッチサイズを小さく
```

### 検出結果が全て「None」

1. データ期間が短すぎる（最低2周期分必要）
2. 欠損率が高すぎる（>50%）
3. 閾値が厳しすぎる → `exploratory_mode.yaml`を使用

## 関連リンク

- [探索フェーズ](../exploration/README.md) - 手法選定・可視化
- [プロジェクト全体](../README.md) - プロジェクト概要
