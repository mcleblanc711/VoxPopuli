import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { BacktestResult } from '../types'
import MetricsCard from './MetricsCard'
import MonthlyHeatmap from './MonthlyHeatmap'

interface Props {
  results: BacktestResult
}

function SummaryDashboard({ results }: Props) {
  const { summary, equity_curve, monthly_heatmap, metadata } = results

  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`
  const formatNumber = (value: number) => value.toFixed(2)
  const formatCurrency = (value: number) => `$${value.toLocaleString(undefined, { minimumFractionDigits: 2 })}`

  return (
    <div className="space-y-6">
      {/* Strategy Info */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-2">
          {metadata.strategy.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
        </h2>
        <p className="text-gray-500">
          {metadata.date_range[0]} to {metadata.date_range[1]}
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricsCard
          label="Total Return"
          value={formatPercent(summary.total_return)}
          positive={summary.total_return > 0}
        />
        <MetricsCard
          label="CAGR"
          value={formatPercent(summary.cagr)}
          positive={summary.cagr > 0}
        />
        <MetricsCard
          label="Sharpe Ratio"
          value={formatNumber(summary.sharpe_ratio)}
          positive={summary.sharpe_ratio > 1}
        />
        <MetricsCard
          label="Max Drawdown"
          value={formatPercent(summary.max_drawdown)}
          positive={false}
        />
      </div>

      {/* Secondary Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <MetricsCard
          label="Win Rate"
          value={formatPercent(summary.win_rate)}
          positive={summary.win_rate > 0.5}
          small
        />
        <MetricsCard
          label="Profit Factor"
          value={formatNumber(summary.profit_factor)}
          positive={summary.profit_factor > 1}
          small
        />
        <MetricsCard
          label="Total Trades"
          value={summary.total_trades.toString()}
          small
        />
        <MetricsCard
          label="Sortino Ratio"
          value={formatNumber(summary.sortino_ratio)}
          positive={summary.sortino_ratio > 1}
          small
        />
        <MetricsCard
          label="Calmar Ratio"
          value={formatNumber(summary.calmar_ratio)}
          positive={summary.calmar_ratio > 1}
          small
        />
      </div>

      {/* Equity Curve */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Equity Curve</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={equity_curve}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })}
                minTickGap={50}
              />
              <YAxis
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                domain={['auto', 'auto']}
              />
              <Tooltip
                formatter={(value: number) => [formatCurrency(value), 'Portfolio Value']}
                labelFormatter={(date) => new Date(date).toLocaleDateString()}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#4f46e5"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Monthly Returns Heatmap */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Monthly Returns</h3>
        <MonthlyHeatmap data={monthly_heatmap} />
      </div>

      {/* Trade Statistics */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Trade Statistics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div>
            <p className="text-sm text-gray-500">Winning Trades</p>
            <p className="text-2xl font-semibold text-green-600">{summary.winning_trades}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Losing Trades</p>
            <p className="text-2xl font-semibold text-red-600">{summary.losing_trades}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Avg Win</p>
            <p className="text-2xl font-semibold text-green-600">{formatCurrency(summary.avg_win)}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Avg Loss</p>
            <p className="text-2xl font-semibold text-red-600">{formatCurrency(summary.avg_loss)}</p>
          </div>
        </div>
      </div>

      {/* Benchmark Comparison */}
      {summary.benchmark_return !== null && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Benchmark Comparison</h3>
          <div className="grid grid-cols-3 gap-6">
            <div>
              <p className="text-sm text-gray-500">Benchmark Return</p>
              <p className="text-2xl font-semibold">{formatPercent(summary.benchmark_return)}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Alpha</p>
              <p className={`text-2xl font-semibold ${summary.alpha! > 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPercent(summary.alpha!)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Beta</p>
              <p className="text-2xl font-semibold">{formatNumber(summary.beta!)}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default SummaryDashboard
