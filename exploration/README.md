# Exploration Phase - 探索的分析

このディレクトリは、季節性検出手法の**探索・可視化・手法選定**を行うためのJupyter Notebook環境です。

## 目的

- 時系列データの可視化と理解
- 複数の季節性検出手法の比較検証
- 最適な前処理パイプラインの設計
- 閾値パラメータのチューニング

## ディレクトリ構成

```
exploration/
├── README.md              # このファイル
├── requirements.txt       # 探索フェーズ用の依存関係
└── exploration.ipynb      # メインの探索ノートブック
```

## セットアップ

### 1. 仮想環境の作成（推奨）

```bash
# プロジェクトルートから
cd exploration
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 3. Jupyter Notebookの起動

```bash
jupyter notebook exploration.ipynb
# または
jupyter lab
```

## ノートブック構成

`exploration.ipynb`は以下のセクションで構成されています：

### セクション1: データ読み込みと基礎統計
- CSVロード、timestamp生成
- 欠損パターンの可視化（ヒートマップ）
- 各センサーの記述統計（平均、標準偏差、歪度、尖度）
- 時系列プロット（全センサー）

### セクション2: 前処理の比較検証
- **リサンプリング手法**: mean, median, ffill
- **補間手法**: linear, spline, None
- **外れ値検出**: IQR法, Z-score法, Isolation Forest
- **変化点検出**: PELT法, Window法

### セクション3: 季節性検出手法の実装と比較
- **STL分解**: Seasonal-Trend decomposition using Loess
- **ACF/PACF**: 自己相関関数による周期検出
- **Lomb-Scargle Periodogram**: 不等間隔データ対応のスペクトル解析
- **Fourier回帰**: 周期フィッティングとΔR²スコア
- **Prophet分解**（オプション）: Meta製の時系列分析ライブラリ

### セクション4: 複数メトリクスの統合と2モード分析
- スコア統合ロジックの設計
- 厳格モード（Strict）と探索モード（Exploratory）の閾値設定
- 全センサー一括処理

### セクション5: 可視化とレポート雛形
- トップ10センサーの詳細レポート
- 全センサーのサムネイルグリッド
- インタラクティブ探索（Plotly）

### セクション6: 探索結果のまとめと設計方針
- 推奨手法の選定理由
- パイプライン化に向けた設計方針

## データ形式

入力データは`../data/`ディレクトリに配置します。

### 期待されるCSV形式

```csv
date,time,sensor_1,sensor_2,sensor_3,...
2024-01-01,08:30:00,100.5,200.3,150.2,...
2024-01-01,14:45:00,102.1,198.7,151.8,...
```

- `date`: 日付（YYYY-MM-DD形式）
- `time`: 時刻（HH:MM:SS形式、オプション）
- `sensor_*`: センサー値（数値）

## 出力

分析結果は`results/exploration/`ディレクトリに保存されます：

```
results/exploration/
├── missing_pattern.png              # 欠損パターンの可視化
├── descriptive_statistics.csv       # 記述統計
├── time_series_all_sensors.png      # 全センサー時系列
├── resampling_comparison.png        # リサンプリング手法比較
├── outlier_detection_comparison.png # 外れ値検出比較
├── changepoint_detection.png        # 変化点検出結果
├── stl_decomposition_*.png          # STL分解結果
├── acf_pacf_analysis.png            # ACF/PACF分析
├── lomb_scargle_periodogram.png     # Periodogram
├── fourier_regression_*.png         # Fourier回帰結果
├── seasonality_scores_ranking.csv   # スコアランキング
├── detailed_report_*.png            # 詳細レポート
└── thumbnail_grid_all_sensors.png   # サムネイルグリッド
```

## 推奨ワークフロー

1. **データ確認**: セクション1でデータの品質と特性を把握
2. **前処理選定**: セクション2で最適な前処理パイプラインを決定
3. **手法比較**: セクション3で各検出手法の結果を比較
4. **閾値調整**: セクション4で厳格/探索モードの閾値を調整
5. **結果確認**: セクション5でトップセンサーの詳細を確認
6. **パイプライン移行**: 確定した設定を`../pipeline/`のYAML設定に反映

## トラブルシューティング

### Prophet がインストールできない

ProphetはC++コンパイラが必要です。以下を試してください：

```bash
# conda環境の場合（推奨）
conda install -c conda-forge prophet

# pipの場合（事前にC++コンパイラが必要）
pip install prophet
```

### メモリ不足エラー

大量のセンサーを処理する場合：

1. センサーを分割して処理
2. `gc.collect()`でメモリ解放
3. 中間結果をファイルに保存

### Lomb-Scargle計算が遅い

```python
# 周波数グリッドを粗くする
frequency, power = ls.autopower(
    minimum_frequency=1/500,
    maximum_frequency=1/2,
    samples_per_peak=5  # デフォルト5、小さくすると高速化
)
```

## 関連リンク

- [パイプラインフェーズ](../pipeline/README.md) - 本番運用向けCLIツール
- [プロジェクト全体](../README.md) - プロジェクト概要
