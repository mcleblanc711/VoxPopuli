export interface BacktestMetrics {
  total_return: number
  cagr: number
  sharpe_ratio: number
  sortino_ratio: number
  max_drawdown: number
  max_drawdown_duration: number
  win_rate: number
  profit_factor: number
  avg_win: number
  avg_loss: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  avg_holding_period: number
  exposure_time: number
  volatility: number
  calmar_ratio: number
  benchmark_return: number | null
  alpha: number | null
  beta: number | null
}

export interface Trade {
  ticker: string
  direction: string
  entry_date: string
  exit_date: string
  entry_price: number
  exit_price: number
  shares: number
  pnl: number
  return_pct: number
  entry_sentiment: number | null
  exit_reason: string
}

export interface EquityCurvePoint {
  date: string
  value: number
}

export interface MonthlyReturn {
  year: number
  month: number
  return: number
}

export interface BacktestResult {
  metadata: {
    strategy: string
    date_range: [string, string]
    generated_at: string
    config: Record<string, unknown>
  }
  summary: BacktestMetrics
  equity_curve: EquityCurvePoint[]
  trades: Trade[]
  monthly_returns: MonthlyReturn[]
  monthly_heatmap: Record<string, Record<string, number>>
}
