import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { BacktestResult } from '../types'

const COLORS = ['#4f46e5', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

function StrategyComparison() {
  const [strategies, setStrategies] = useState<BacktestResult[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadStrategies()
  }, [])

  const loadStrategies = async () => {
    const strategyNames = [
      'attention_momentum',
      'sentiment_divergence',
      'contrarian',
      'cross_subreddit',
      'velocity_sentiment',
    ]

    const loaded: BacktestResult[] = []

    for (const name of strategyNames) {
      try {
        const response = await fetch(`./data/${name}_results.json`)
        if (response.ok) {
          const data = await response.json()
          loaded.push(data)
        }
      } catch {
        // Skip missing strategies
      }
    }

    setStrategies(loaded)
    setLoading(false)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading strategies...</div>
      </div>
    )
  }

  if (strategies.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <p className="text-gray-500">
          No strategy results found. Run backtests for multiple strategies to compare them.
        </p>
      </div>
    )
  }

  // Normalize equity curves to percentage returns for comparison
  const normalizedData = () => {
    if (strategies.length === 0) return []

    // Get all unique dates
    const allDates = new Set<string>()
    strategies.forEach((s) => {
      s.equity_curve.forEach((point) => allDates.add(point.date))
    })

    const sortedDates = [...allDates].sort()

    return sortedDates.map((date) => {
      const point: Record<string, number | string> = { date }

      strategies.forEach((strategy) => {
        const curvePoint = strategy.equity_curve.find((p) => p.date === date)
        const initialValue = strategy.equity_curve[0]?.value || 1

        if (curvePoint) {
          point[strategy.metadata.strategy] = ((curvePoint.value / initialValue) - 1) * 100
        }
      })

      return point
    })
  }

  const formatPercent = (value: number) => `${value.toFixed(2)}%`

  return (
    <div className="space-y-6">
      {/* Comparison Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Equity Curve Comparison (Normalized)</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={normalizedData()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })}
                minTickGap={50}
              />
              <YAxis
                tickFormatter={(value) => `${value.toFixed(0)}%`}
              />
              <Tooltip
                formatter={(value: number) => [formatPercent(value), '']}
                labelFormatter={(date) => new Date(date).toLocaleDateString()}
              />
              <Legend />
              {strategies.map((strategy, idx) => (
                <Line
                  key={strategy.metadata.strategy}
                  type="monotone"
                  dataKey={strategy.metadata.strategy}
                  name={strategy.metadata.strategy.replace(/_/g, ' ')}
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Metrics Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <h3 className="text-lg font-semibold p-6 pb-4">Strategy Metrics Comparison</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Strategy</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Total Return</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">CAGR</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Sharpe</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Sortino</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Max DD</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Win Rate</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Trades</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {strategies.map((strategy, idx) => (
                <tr key={strategy.metadata.strategy} className="hover:bg-gray-50">
                  <td className="px-4 py-3 whitespace-nowrap text-sm font-medium">
                    <span className="flex items-center gap-2">
                      <span
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: COLORS[idx % COLORS.length] }}
                      />
                      {strategy.metadata.strategy.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td className={`px-4 py-3 whitespace-nowrap text-sm text-right ${
                    strategy.summary.total_return >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatPercent(strategy.summary.total_return * 100)}
                  </td>
                  <td className={`px-4 py-3 whitespace-nowrap text-sm text-right ${
                    strategy.summary.cagr >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatPercent(strategy.summary.cagr * 100)}
                  </td>
                  <td className={`px-4 py-3 whitespace-nowrap text-sm text-right ${
                    strategy.summary.sharpe_ratio >= 1 ? 'text-green-600' : 'text-gray-700'
                  }`}>
                    {strategy.summary.sharpe_ratio.toFixed(2)}
                  </td>
                  <td className={`px-4 py-3 whitespace-nowrap text-sm text-right ${
                    strategy.summary.sortino_ratio >= 1 ? 'text-green-600' : 'text-gray-700'
                  }`}>
                    {strategy.summary.sortino_ratio.toFixed(2)}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-right text-red-600">
                    {formatPercent(strategy.summary.max_drawdown * 100)}
                  </td>
                  <td className={`px-4 py-3 whitespace-nowrap text-sm text-right ${
                    strategy.summary.win_rate >= 0.5 ? 'text-green-600' : 'text-gray-700'
                  }`}>
                    {formatPercent(strategy.summary.win_rate * 100)}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-right text-gray-700">
                    {strategy.summary.total_trades}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default StrategyComparison
