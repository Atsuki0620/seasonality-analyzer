# パイプライン利用ガイド

Seasonality Analyzerのパイプライン（CLI）を使用した季節性分析の詳細ガイドです。

## 目次

1. [概要](#概要)
2. [セットアップ](#セットアップ)
3. [基本的な使い方](#基本的な使い方)
4. [実行例：ステップバイステップ](#実行例ステップバイステップ)
5. [出力ファイルの説明](#出力ファイルの説明)
6. [設定ファイルのカスタマイズ](#設定ファイルのカスタマイズ)
7. [LLM診断機能の使い方](#llm診断機能の使い方)
8. [トラブルシューティング](#トラブルシューティング)
9. [よくある質問](#よくある質問)

---

## 概要

Seasonality Analyzerのパイプラインは、時系列データから季節性パターンを自動検出するCLIツールです。

### 主な機能

- **複数センサーの一括処理**: 複数の時系列データを一度に分析
- **自動季節性検出**: STL分解、ACF分析、Fourier回帰の3つの手法を使用
- **スコアリングとランキング**: 季節性の強度を数値化し、センサーをランキング
- **可視化とレポート生成**: グラフとHTMLレポートを自動生成
- **デバッグ支援**: 実行ログとデバッグバンドルで問題を追跡
- **LLM診断**: オプションでAI診断による詳細な分析

### パイプラインの処理フロー

```
1. データ読み込み
   ↓
2. データヘルスチェック（欠損率、期間、レコード数）
   ↓
3. 前処理
   - リサンプリング（デフォルト: 日次）
   - 欠損値補間
   - 外れ値検出・除去
   ↓
4. 季節性検出（センサーごと）
   - STL分解による季節成分の抽出
   - ACF分析による周期性の検出
   - Fourier回帰による周期の評価
   ↓
5. スコアリングとランキング
   - 各手法のスコアを統合
   - 信頼度レベルの判定（High/Medium/Low）
   ↓
6. 可視化とレポート生成
   - 各センサーの詳細プロット
   - 信頼度分布グラフ
   - HTMLサマリーレポート
   ↓
7. 結果の保存
   - CSV（スコア、ランキング）
   - JSON（サマリー、設定）
   - PNG（グラフ）
   - ログとデバッグバンドル
```

---

## セットアップ

### 前提条件

- Python 3.10以上
- pip（パッケージ管理ツール）

### インストール手順

#### 1. pipelineディレクトリに移動

```bash
cd /path/to/seasonality-analyzer/pipeline
```

#### 2. パッケージのインストール

```bash
# 基本機能のみ（推奨）
pip install -e .

# LLM診断機能を含める場合
pip install -e ".[llm]"

# 開発者向け（テスト機能を含む）
pip install -e ".[dev]"
```

#### 3. インストールの確認

```bash
seasonality --version
# 出力: seasonality, version 0.1.0

seasonality --help
# CLIコマンド一覧が表示されます
```

---

## 基本的な使い方

### 利用可能なコマンド

Seasonality Analyzerには4つの主要コマンドがあります：

1. **`seasonality run`** - 季節性分析の実行（メインコマンド）
2. **`seasonality init`** - 設定ファイルのひな形を生成
3. **`seasonality debug`** - ログからデバッグバンドルを作成
4. **`seasonality diagnose`** - LLMによる診断分析

---

### 1. `seasonality run` - 季節性分析の実行

最も重要なコマンドです。データを読み込み、分析し、結果を出力します。

#### 基本構文

```bash
seasonality run --data <データファイルのパス> [オプション]
```

#### 必須オプション

- `--data, -d`: 入力CSVファイルのパス（必須）

#### 主なオプション

| オプション | 説明 | デフォルト値 |
|-----------|------|------------|
| `--config, -c` | 設定YAMLファイルのパス | デフォルト設定 |
| `--output, -o` | 出力ディレクトリ | `outputs` |
| `--sensors` | 分析対象のセンサー（カンマ区切り） | 全センサー |
| `--log-level` | ログレベル（DEBUG/INFO/WARNING/ERROR） | `INFO` |
| `--no-figures` | 図の生成をスキップ（高速化） | False |

#### 使用例

```bash
# 基本的な実行
seasonality run --data ../data/sample_sensor_data.csv

# 出力先を指定（日付時刻つきフォルダに保存）
OUTPUT_DIR="results/run_$(date +%Y%m%d_%H%M%S)"
seasonality run --data ../data/sample_sensor_data.csv --output "$OUTPUT_DIR"

# 特定のセンサーのみ分析
seasonality run --data ../data/sample_sensor_data.csv --sensors sensor_1,sensor_2,sensor_3

# 設定ファイルを指定（厳格モード）
seasonality run --data ../data/sample_sensor_data.csv --config config/strict_mode.yaml

# 詳細ログを出力
seasonality run --data ../data/sample_sensor_data.csv --log-level DEBUG

# 図の生成をスキップして高速化
seasonality run --data ../data/sample_sensor_data.csv --no-figures
```

---

### 2. `seasonality init` - 設定ファイルの生成

設定ファイルのひな形を生成します。3種類の設定ファイルが作成されます。

#### 基本構文

```bash
seasonality init [--output <出力ディレクトリ>]
```

#### 使用例

```bash
# カレントディレクトリに生成
seasonality init

# 指定したディレクトリに生成
seasonality init --output my_configs/
```

#### 生成されるファイル

1. `default.yaml` - 標準設定
2. `strict_mode.yaml` - 厳格モード（高い閾値）
3. `exploratory_mode.yaml` - 探索モード（低い閾値）

---

### 3. `seasonality debug` - デバッグバンドルの作成

実行ログからデバッグバンドル（JSON）を作成します。エラー解析に役立ちます。

#### 基本構文

```bash
seasonality debug --log <ログファイルのパス> [--output <出力ディレクトリ>]
```

#### 使用例

```bash
# ログファイルからデバッグバンドルを作成
seasonality debug --log results/run_20260128_234332/logs/run_20260128_234337.log

# 出力先を指定
seasonality debug --log results/logs/run.log --output debug_bundles/
```

---

### 4. `seasonality diagnose` - LLM診断

デバッグバンドルをLLMで分析し、問題の診断と推奨事項を提供します。

#### 基本構文

```bash
seasonality diagnose --bundle <デバッグバンドルのパス>
```

#### 使用例

```bash
# デバッグバンドルを診断
seasonality diagnose --bundle results/debug_bundles/bundle_20260128_234337.json
```

**注意**: LLM診断を使用するには、APIキーの設定が必要です（[LLM診断機能の使い方](#llm診断機能の使い方)を参照）。

---

## 実行例：ステップバイステップ

実際の分析フローを順を追って説明します。

### ステップ1: データの準備

分析対象のCSVファイルを準備します。

#### データフォーマット要件

- **必須列**: `date`, `time`, `sensor_*` 列
- **日付形式**: YYYY-MM-DD（例: 2022-01-01）
- **時刻形式**: HH:MM:SS（例: 11:12:57）
- **センサー列**: `sensor_1`, `sensor_2`, ... のような名前

#### サンプルデータ構造

```csv
date,time,sensor_1,sensor_2,sensor_3
2022-01-01,11:12:57,23.5,45.2,67.8
2022-01-02,11:13:00,24.1,44.9,68.2
...
```

### ステップ2: 初回実行（探索モード）

まず、探索モードで実行して、データの特性を把握します。

```bash
# 出力ディレクトリを日付時刻つきで作成
OUTPUT_DIR="results/run_$(date +%Y%m%d_%H%M%S)"

# 探索モードで実行
seasonality run \
  --data ../data/sample_sensor_data.csv \
  --config config/exploratory_mode.yaml \
  --output "$OUTPUT_DIR" \
  --log-level INFO
```

### ステップ3: 結果の確認

実行が完了すると、以下のような出力が表示されます：

```
=== Analysis Complete ===
Total sensors analyzed: 40
Confidence distribution (strict mode):
  High: 3
  Medium: 5
  Low: 2
  None: 30

Results saved to: results/run_20260128_234332/results
Figures saved to: results/run_20260128_234332/figures
Logs saved to: results/run_20260128_234332/logs/run_20260128_234337.log
```

### ステップ4: 結果ファイルの閲覧

#### ランキングCSVを確認

```bash
# ランキングファイルを表示
cat results/run_20260128_234332/results/seasonality_ranking_*.csv
```

出力例:
```csv
sensor,rank,best_period,max_combined_score,composite_score,confidence_strict,confidence_exploratory
sensor_3,1,365,0.789,0.918,High,High
sensor_1,2,30,0.812,0.893,High,High
sensor_2,3,30,0.695,0.850,High,High
```

#### HTMLレポートを開く

```bash
# ブラウザでHTMLレポートを開く（Linux）
xdg-open results/run_20260128_234332/results/report.html

# macOS
open results/run_20260128_234332/results/report.html

# Windows
start results/run_20260128_234332/results/report.html
```

### ステップ5: 特定センサーの詳細分析

上位センサーに絞って再分析します。

```bash
# 上位3センサーのみを分析
OUTPUT_DIR="results/run_$(date +%Y%m%d_%H%M%S)"
seasonality run \
  --data ../data/sample_sensor_data.csv \
  --config config/strict_mode.yaml \
  --sensors sensor_1,sensor_2,sensor_3 \
  --output "$OUTPUT_DIR" \
  --log-level DEBUG
```

### ステップ6: ログの確認（エラーがあった場合）

エラーが発生した場合、ログファイルを確認します。

```bash
# ログファイルの内容を表示
cat results/run_20260128_234332/logs/run_20260128_234337.log
```

### ステップ7: デバッグバンドルの作成（必要に応じて）

エラーの詳細を把握するため、デバッグバンドルを作成します。

```bash
seasonality debug \
  --log results/run_20260128_234332/logs/run_20260128_234337.log \
  --output results/run_20260128_234332/debug_bundles_manual
```

---

## 出力ファイルの説明

パイプライン実行後、以下のようなディレクトリ構造で結果が保存されます：

```
results/run_YYYYMMDD_HHMMSS/
├── results/                           # 分析結果
│   ├── seasonality_scores_*.csv       # 全センサーのスコア詳細
│   ├── seasonality_ranking_*.csv      # ランキング（推奨）
│   ├── seasonality_summary_*.json     # サマリー統計
│   ├── seasonality_config_*.json      # 使用した設定
│   └── report.html                    # HTMLレポート
├── figures/                           # 可視化結果
│   ├── thumbnail_grid.png             # 全センサーのサムネイル
│   ├── confidence_distribution.png    # 信頼度分布
│   └── detailed_sensor_*.png          # 各センサーの詳細プロット
├── logs/                              # ログファイル
│   └── run_YYYYMMDD_HHMMSS.log        # 実行ログ
└── debug_bundles/                     # デバッグ情報
    └── bundle_YYYYMMDD_HHMMSS.json    # デバッグバンドル
```

### 主要ファイルの詳細

#### 1. `seasonality_ranking_*.csv` ⭐ 最重要

センサーのランキングと主要メトリクスを含むCSVファイル。

**列の説明**:
- `sensor`: センサー名
- `rank`: ランキング（1位が最も強い季節性）
- `best_period`: 最も強い季節性の周期（日数）
- `max_combined_score`: 統合スコアの最大値（0〜1）
- `composite_score`: 総合評価スコア（0〜1）
- `confidence_strict`: 厳格モードでの信頼度（High/Medium/Low/None）
- `confidence_exploratory`: 探索モードでの信頼度

#### 2. `seasonality_scores_*.csv`

各センサーの詳細スコア（全手法、全周期）を含むCSVファイル。

**主要な列**:
- `stl_7`, `stl_30`, `stl_365`: STL分解スコア（周期別）
- `acf_7`, `acf_30`, `acf_365`: ACF分析スコア
- `fourier_7`, `fourier_30`, `fourier_365`: Fourier回帰スコア
- `combined_7`, `combined_30`, `combined_365`: 統合スコア

#### 3. `report.html`

対話的なHTMLレポート。ブラウザで開いて閲覧します。

**内容**:
- サマリー統計（分析数、信頼度分布）
- ランキングテーブル
- グラフの埋め込み
- 設定情報

#### 4. `run_YYYYMMDD_HHMMSS.log`

実行ログ。エラー発生時の診断に使用します。

**含まれる情報**:
- データ読み込みの詳細
- 前処理の進捗
- 各センサーの検出結果
- エラーメッセージとスタックトレース

#### 5. `bundle_YYYYMMDD_HHMMSS.json`

デバッグバンドル。LLM診断やサポートに使用します。

**含まれる情報**:
- 設定の詳細
- データサマリー
- ヘルスチェック結果
- 処理ステップのタイミング
- エラーと警告

---

## 設定ファイルのカスタマイズ

設定ファイル（YAML）をカスタマイズして、分析をチューニングできます。

### 設定ファイルの構造

```yaml
# data: データ読み込み設定
data:
  date_column: "date"
  time_column: "time"
  sensor_columns: null  # nullで自動検出
  sensor_prefix: "sensor_"

# preprocessing: 前処理設定
preprocessing:
  resample_freq: "D"  # リサンプリング頻度（D=日次, W=週次, M=月次）
  resample_method: "mean"  # 集約方法（mean, median, sum）
  outlier_method: "iqr"  # 外れ値検出手法（iqr, zscore, none）
  outlier_threshold: 1.5  # 外れ値の閾値
  impute_method: "linear"  # 補間方法（linear, polynomial, spline, none）

# detection: 季節性検出設定
detection:
  methods:
    - stl
    - acf
    - fourier
  periods:  # 検出対象の周期（日数）
    - 7     # 週次
    - 30    # 月次
    - 365   # 年次
  stl_seasonal: 7  # STLの季節成分の窓サイズ

# scoring: スコアリング設定
scoring:
  mode: "strict"  # strict または exploratory
  weights:  # 各手法の重み
    stl: 0.3
    acf: 0.3
    fourier: 0.4
  thresholds:
    strict:
      stl_min: 0.5
      acf_min: 0.3
      fourier_min: 0.1
    exploratory:
      stl_min: 0.3
      acf_min: 0.2
      fourier_min: 0.05

# output: 出力設定
output:
  base_dir: "results"
  results_dir: "results"
  figures_dir: "figures"
  logs_dir: "logs"
  debug_bundles_dir: "debug_bundles"
  figure_dpi: 100  # 図の解像度
```

### カスタマイズ例

#### 例1: 週次季節性のみを検出

```yaml
detection:
  methods:
    - stl
    - acf
    - fourier
  periods:
    - 7  # 週次のみ
```

#### 例2: より緩い閾値で検出

```yaml
scoring:
  mode: "exploratory"
  thresholds:
    exploratory:
      stl_min: 0.2  # デフォルトより低い
      acf_min: 0.15
      fourier_min: 0.03
```

#### 例3: 時間単位でリサンプリング

```yaml
preprocessing:
  resample_freq: "H"  # 時間単位
  resample_method: "mean"
```

---

## LLM診断機能の使い方

LLM（Large Language Model）を使った自動診断機能の設定と使用方法です。

### 前提条件

LLM診断機能を使用するには、以下のいずれかのAPIキーが必要です：

1. **OpenAI API** (推奨)
2. **Azure OpenAI**

### セットアップ手順

#### 1. LLM依存関係のインストール

```bash
pip install -e ".[llm]"
```

#### 2. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成します。

```bash
# プロジェクトルートに移動
cd /path/to/seasonality-analyzer

# .envファイルを作成
cp .env.example .env

# .envファイルを編集
nano .env  # または vi, emacs など
```

#### 3. APIキーの設定

`.env` ファイルに以下を記述します：

##### OpenAIの場合

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

##### Azure OpenAIの場合

```bash
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

### 使用方法

#### 基本的な診断フロー

```bash
# 1. パイプラインを実行（エラーが発生した場合）
seasonality run --data ../data/sample_sensor_data.csv --output results/test_run

# 2. ログファイルからデバッグバンドルを作成
seasonality debug --log results/test_run/logs/run_*.log

# 3. LLM診断を実行
seasonality diagnose --bundle results/test_run/debug_bundles/bundle_*.json
```

#### 診断結果の例

```
=== Diagnostic Results ===
Confidence: high

Summary:
分析は正常に完了しましたが、データ期間が短いため年次季節性の検出精度が低い可能性があります。

Issues Found (1):
  [WARNING] (data_quality) Date range is only 8 days, which is insufficient for annual seasonality detection (365 days required).

Recommendations:
  - より長い期間のデータ（最低2年分）を使用してください
  - 週次・月次季節性に焦点を当てることを検討してください
  - exploratory_mode.yaml を使用して、より柔軟な検出を試してください
```

### LLM診断なしでも使用可能

APIキーが設定されていない場合、基本的な分析が表示されます：

```
Basic analysis (without LLM):
  Bundle ID: 20260128_234337
  Created: 2026-01-28T23:43:51
  Errors: 0
  Warnings: 0
```

---

## トラブルシューティング

### よくある問題と解決策

#### 1. "No valid sensors to analyze"

**原因**: データ量が不足している（センサーあたり100レコード未満）

**解決策**:
- より長い期間のデータを使用する
- テストデータではなく、本番データを使用する
- `--sensors` オプションで特定のセンサーのみを指定する

```bash
# データの行数を確認
wc -l ../data/sample_sensor_data.csv

# データ期間を確認
head -2 ../data/sample_sensor_data.csv
tail -1 ../data/sample_sensor_data.csv
```

#### 2. "TypeError: unsupported operand type(s) for /: 'str' and 'str'"

**原因**: 設定ファイルのパスの型エラー（修正済み）

**解決策**: 最新版のコードを使用してください。

#### 3. インストールエラー（antlr4-python3-runtime）

**原因**: pipやsetuptoolsのバージョンが古い

**解決策**:
```bash
pip install --upgrade pip setuptools
pip install -e .
```

#### 4. メモリ不足エラー

**原因**: 大量のセンサーを一度に処理している

**解決策**:
```bash
# センサーを分割して処理
seasonality run --data data.csv --sensors sensor_1,sensor_2,sensor_3
seasonality run --data data.csv --sensors sensor_4,sensor_5,sensor_6

# または、図の生成をスキップ
seasonality run --data data.csv --no-figures
```

#### 5. 検出結果が全て "None"

**原因**:
- データ期間が短すぎる（周期の2倍未満）
- 欠損率が高すぎる（>50%）
- 閾値が厳しすぎる

**解決策**:
```bash
# 探索モードで実行
seasonality run --data data.csv --config config/exploratory_mode.yaml

# 週次季節性のみを検出
# config.yamlを編集して periods: [7] にする

# ログレベルをDEBUGにして詳細を確認
seasonality run --data data.csv --log-level DEBUG
```

#### 6. 図が生成されない

**原因**: matplotlibのバックエンドエラー

**解決策**:
```bash
# 環境変数を設定
export MPLBACKEND=Agg

# または、Pythonコード内で設定（cli.pyで既に設定済み）
import matplotlib
matplotlib.use('Agg')
```

---

## よくある質問

### Q1: データファイルのフォーマット要件は？

**A**: 以下の列が必要です：
- `date`: 日付（YYYY-MM-DD形式）
- `time`: 時刻（HH:MM:SS形式）
- `sensor_*`: センサー列（数値）

### Q2: 最小限のデータ量は？

**A**:
- センサーあたり最低100レコード
- 検出したい周期の2倍以上の期間
  - 週次（7日）: 最低14日分
  - 月次（30日）: 最低60日分
  - 年次（365日）: 最低730日分（2年）

### Q3: strictとexploratoryモードの違いは？

**A**:
- **strict**: 高い閾値で明確な季節性のみを検出（偽陽性を避ける）
- **exploratory**: 低い閾値で幅広く季節性を検出（探索的分析向け）

### Q4: センサーが多すぎて処理が遅い場合は？

**A**:
```bash
# 上位Nセンサーに絞る
# まず全センサーで実行
seasonality run --data data.csv --output results/all --no-figures

# ランキングを確認
head -10 results/all/results/seasonality_ranking_*.csv

# 上位10センサーのみを再分析
seasonality run --data data.csv --sensors sensor_1,sensor_3,sensor_7,... --output results/top10
```

### Q5: カスタム周期（例: 14日）を検出できますか？

**A**: はい、設定ファイルを編集してください：

```yaml
detection:
  periods:
    - 14  # 2週間
    - 21  # 3週間
    - 90  # 四半期
```

### Q6: 結果をExcelで開きたい

**A**: CSVファイルはExcelで直接開けます：
```bash
# Windows
start excel results/run_*/results/seasonality_ranking_*.csv

# macOS
open -a "Microsoft Excel" results/run_*/results/seasonality_ranking_*.csv
```

### Q7: 複数の実行結果を比較したい

**A**: 各実行を日付時刻つきフォルダに保存し、ランキングCSVを比較します：
```bash
# 実行1（デフォルト設定）
seasonality run --data data.csv --output results/run_default

# 実行2（厳格モード）
seasonality run --data data.csv --config config/strict_mode.yaml --output results/run_strict

# 結果を比較
diff results/run_default/results/seasonality_ranking_*.csv \
     results/run_strict/results/seasonality_ranking_*.csv
```

### Q8: Pythonスクリプトから使用できますか？

**A**: はい、パッケージをインポートして使用できます：

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
    sensor_cols=["sensor_1", "sensor_2"],
    periods=[7, 30, 365]
)

# ランキング生成
scores_df = create_scores_dataframe(results)
ranking = generate_ranking(scores_df)
print(ranking)
```

---

## まとめ

このガイドでは、Seasonality Analyzerパイプラインの使い方を詳しく説明しました。

### 基本的なワークフロー

1. **データ準備**: CSV形式で日付、時刻、センサー列を用意
2. **初回実行**: 探索モードで全体像を把握
3. **結果確認**: ランキングCSVとHTMLレポートを確認
4. **詳細分析**: 上位センサーに絞って厳格モードで再分析
5. **設定調整**: 必要に応じて設定ファイルをカスタマイズ
6. **エラー対応**: ログとデバッグバンドルで問題を解決

### 推奨する使い方

- **探索フェーズ**: `exploratory_mode.yaml` で幅広く検出
- **本番フェーズ**: `strict_mode.yaml` で高信頼度の季節性のみを抽出
- **定期実行**: 日付時刻つきフォルダで履歴を管理
- **エラー発生時**: デバッグバンドルとLLM診断を活用

### サポート

問題が解決しない場合は、以下の情報を添えてお問い合わせください：
- 実行コマンド
- ログファイル（`logs/run_*.log`）
- デバッグバンドル（`debug_bundles/bundle_*.json`）
- データの概要（レコード数、期間、センサー数）

---

**Happy Analyzing!** 🎉
