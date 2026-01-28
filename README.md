# Seasonality Analyzer

マルチセンサーの時系列データから季節性を検出するためのリポジトリです。素早い試行用のJupyterノートブック（exploration）と、再現性のあるバッチ処理・レポート生成を行うCLIパイプライン（pipeline）の2フェーズ構成です。

## リポジトリ構成
- data/sample_sensor_data.csv : サンプルデータ。
- exploration/ : 解析ノートブック exploration.ipynb と最小限の依存関係。
- pipeline/ : Pythonパッケージ/CLI seasonality。設定YAML、テスト、リファレンス出力、結果格納フォルダを含みます。
- .env.example : LLM診断を使う際のAPIキー用テンプレート。

## 使い方（CLIパイプライン）
`ash
cd pipeline
python -m venv .venv
# Windows
./.venv/Scripts/Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -e .               # コア
pip install -e ".[llm]"        # 任意: LLM診断
pip install -e ".[dev]"        # 任意: 開発/テスト用

seasonality --help
seasonality run --data ../data/sample_sensor_data.csv --output results/demo_run
# よく使うオプション例
# --config config/strict_mode.yaml     # あるいは config/exploratory_mode.yaml
# --sensors sensor_1,sensor_2          # 対象カラムを指定
# --no-figures                         # 図の生成をスキップ
# --log-level DEBUG                    # 詳細ログ
`
出力は pipeline/results/<run_id>/ にスコアCSV、ランキングCSV、HTMLレポート、図、ログとして保存されます。

## 設定
pipeline/config/ にプリセットYAML（default.yaml, strict_mode.yaml, exploratory_mode.yaml）があります。リサンプリング頻度、補間方法、外れ値処理、検出手法（STL / ACF / Fourier）、スコア重みなどを調整できます。

## ノートブック（exploration）
`ash
cd exploration
python -m venv .venv && ./.venv/Scripts/Activate.ps1  # または source .venv/bin/activate
pip install -r requirements.txt
jupyter lab  # または jupyter notebook
`
exploration.ipynb ではデータ点検、前処理の比較、検出手法の試行、可視化を行い、パイプライン設定を固めるための知見を得ます。

## データ要件
date, 	ime, sensor_* 列を持つCSV（YYYY-MM-DD と HH:MM:SS が解釈できる形式）。data/sample_sensor_data.csv をひな形にしてください。

## 主なCLIコマンド
- seasonality run : データ読み込み、スコア算出、レポート生成。
- seasonality init : 設定ファイルのひな形を出力。
- seasonality debug : 実行ログを可視化。
- seasonality diagnose : LLMを用いた異常診断（要APIキー、.env 設定）。

## テスト
cd pipeline 後に pytest または pytest --cov=src/seasonality --cov-report=term-missing。

## バージョンとライセンス
パッケージバージョン: 0.1.0 / ライセンス: MIT
