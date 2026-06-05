# Quantitative Mutual Fund Analytics & Forecasting Pipeline

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit)](https://quant-sip-engine.streamlit.app/)

## Description

An automated, end-to-end data pipeline and quantitative screening engine for the Indian Mutual Fund market. This system aggregates historical NAV data, performs rigorous data validation, and calculates risk-adjusted performance metrics to drive systematic portfolio allocation.

---

# Architecture & Technological Infrastructure

The project is structured for reliability and automated reporting, heavily emphasizing data integrity before any statistical modeling occurs.

### 1. Data Ingestion & Pipeline

Multi-threaded historical data extraction from the AMFI API, coupled with a resilient daily update mechanism (exponential backoff, retry logic).

### 2. Data Validation

Strict schema enforcement using Pandera. Erroneous data (e.g., negative NAVs, missing scheme codes) is filtered before database insertion.

### 3. Storage Layer

MySQL database managed via SQLAlchemy ORM, utilizing connection pooling for high-throughput I/O operations.

### 4. Analytical Engine

Calculates annualized volatility, CAGR across multiple horizons, and the Sharpe ratio:

```math
Sharpe = \frac{R_p - R_f}{\sigma_p}
```

for 5,000+ open-ended funds.

<img width="787" height="742" alt="image" src="https://github.com/user-attachments/assets/c0a9cd36-4725-4cd5-bbfe-8d053e4f6fd1" />

### 5. Interactive Frontend

A deployed Streamlit application that provides millisecond-latency portfolio recommendations via pre-computed matrix caching.

<img width="1916" height="729" alt="image" src="https://github.com/user-attachments/assets/3ee12eae-86d5-4384-9807-a74bd84e6ca2" />

---

## Workflow

```mermaid
flowchart TD
    A[AMFI API Source] -->|Multi-threaded Fetch| B(Raw Data Extraction)
    B --> C{Pandera Validation}

    C -->|Fails Schema| D[Error Logging / Reject]
    C -->|Passes Schema| E[(MySQL DB)]

    E --> F[Quant Engine Pre-computation]
    F --> G[Calculate Base Metrics]
    G --> H[Prophet & Holt-Winters Forecasts]

    H -->|Compression| I[(clean_nav_data.parquet)]

    I -->|Deploys to| J[Streamlit Cloud]
    J --> K[User Inputs: Horizon & Risk]
    K --> L[Dynamic Suitability Scoring]
    L --> M[Clustering Filter]
    M --> N(((Final Portfolio)))

    style E fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    style I fill:#bbf,stroke:#333,stroke-width:2px,color:#000
    style N fill:#bfb,stroke:#333,stroke-width:3px,color:#000
```

---

# Quantitative Backtesting Performance

The analytical engine implements a rigorous, systematic backtester that evaluates a dual-factor strategy combining Trend-Following and Momentum. Rather than relying on static buy-and-hold metrics, the engine simulates daily trading decisions over a 1200-day synthetic bull market (assuming a 0.08% daily return drift).

## 1. Signal Generation (The Dual-Factor Model)

The strategy only triggers a **Buy** signal when two independent technical conditions are met simultaneously.

### Trend Component (EMA Crossover)

Uses a Fast (20-day) and Slow (50-day) Exponential Moving Average. A bullish environment is identified when the Fast EMA crosses above the Slow EMA.

```math
EMA_t = \left( V_t \times \frac{2}{s+1} \right) + EMA_{t-1} \times \left(1 - \frac{2}{s+1}\right)
```

Where:

* (V_t) = NAV at time (t)
* (s) = EMA span

### Momentum / Conviction Filter (Prophet Validation)

Even if the EMA signals a buy, the trade is rejected unless the Prophet machine learning model forecasts the next day's NAV ((\hat{y}_{t+1})) to be greater than a 98% threshold of the current NAV. This prevents excessive whipsawing during volatile sideways markets.

---

## 2. Risk & Performance Evaluation

The engine calculates cumulative returns for both the strategy and a baseline buy-and-hold benchmark.

### Maximum Drawdown (MDD)

Measures the largest decline from peak to trough in portfolio value.

```math
MDD = \min\left(\frac{V_t - P_t}{P_t}\right)
```

Where:

* (V_t) = Portfolio value
* (P_t) = Peak portfolio value before time (t)

### Annualized Sharpe Ratio

Evaluates risk-adjusted performance by penalizing volatility.

```math
Sharpe = \frac{R_p - R_f}{\sigma_p} \times \sqrt{252}
```

Where:

* (R_p) = Portfolio return
* (R_f) = Risk-free rate
* (\sigma_p) = Standard deviation of daily returns

---

### Performance Metrics

| Metric                  | Strategy Performance |
| ----------------------- | -------------------- |
| Directional Accuracy    | 52.3%                |
| Total Strategy Return   | 185.4%               |
| Benchmark Return        | 162.1%               |
| Max Drawdown            | -12.4%               |
| Annualized Sharpe Ratio | 1.85                 |

---

# Forecasting Methodology

Forward-looking projections are handled by a dual-model ensemble approach to balance trend responsiveness with seasonal stability.

### Facebook Prophet

Handles:

* Daily seasonality
* Yearly seasonality
* Automatic changepoint detection

### Holt-Winters Exponential Smoothing

Captures:

* Additive trend
* Seasonal components
* 365-day periodic patterns

The final projected NAV is generated using a weighted ensemble of both models, reducing forecast variance and improving stability.

---

# Portfolio Construction Logic

The recommendation engine uses Agglomerative Hierarchical Clustering (Ward's Method) and a daily return correlation matrix.

Portfolio generation process:

1. Calculate suitability scores using risk-adjusted metrics.
2. Select the highest-scoring fund as the **Anchor Fund**.
3. Iteratively add additional funds.
4. Reject funds with correlation coefficients greater than **0.85** relative to existing portfolio constituents.
5. Produce a diversified portfolio with minimized concentration risk.

---

# Reproducibility & Setup

## Prerequisites

* Python 3.8+
* Running MySQL Server Instance

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/mf-quant-pipeline.git
cd mf-quant-pipeline
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Environment Configuration

Create a `.env` file in the project root:

```env
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=mutual_funds
```

---

## Execution

### Option 1: Full System Initialization (First Run)

Builds the historical database from scratch, executes forecasting models, and generates reports.

```bash
python scripts/run_local_pipeline.py --full
```

### Option 2: Daily Update Pipeline

Fetches latest T+1 NAV data, updates the database, and regenerates reports.

```bash
python scripts/run_local_pipeline.py
```

### Option 3: Launch Local Dashboard

Starts the Streamlit application locally.

```bash
streamlit run app/app.py
```

---

# Disclaimer

This project is intended solely for educational, research, and portfolio demonstration purposes.

* Not financial advice.
* Past performance does not guarantee future results.
* All investment decisions should be made independently after appropriate due diligence.
