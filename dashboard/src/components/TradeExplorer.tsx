import { useState, useMemo } from 'react'
import { Trade } from '../types'

interface Props {
  trades: Trade[]
}

type SortField = 'entry_date' | 'ticker' | 'pnl' | 'return_pct'
type SortDirection = 'asc' | 'desc'

function TradeExplorer({ trades }: Props) {
  const [sortField, setSortField] = useState<SortField>('entry_date')
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc')
  const [filterTicker, setFilterTicker] = useState('')
  const [filterDirection, setFilterDirection] = useState<'all' | 'long' | 'short'>('all')
  const [filterResult, setFilterResult] = useState<'all' | 'win' | 'loss'>('all')

  const uniqueTickers = useMemo(() => {
    return [...new Set(trades.map((t) => t.ticker))].sort()
  }, [trades])

  const filteredAndSorted = useMemo(() => {
    let result = [...trades]

    // Filter by ticker
    if (filterTicker) {
      result = result.filter((t) => t.ticker === filterTicker)
    }

    // Filter by direction
    if (filterDirection !== 'all') {
      result = result.filter((t) => t.direction === filterDirection)
    }

    // Filter by result
    if (filterResult === 'win') {
      result = result.filter((t) => t.pnl > 0)
    } else if (filterResult === 'loss') {
      result = result.filter((t) => t.pnl < 0)
    }

    // Sort
    result.sort((a, b) => {
      let comparison = 0
      switch (sortField) {
        case 'entry_date':
          comparison = new Date(a.entry_date).getTime() - new Date(b.entry_date).getTime()
          break
        case 'ticker':
          comparison = a.ticker.localeCompare(b.ticker)
          break
        case 'pnl':
          comparison = a.pnl - b.pnl
          break
        case 'return_pct':
          comparison = a.return_pct - b.return_pct
          break
      }
      return sortDirection === 'asc' ? comparison : -comparison
    })

    return result
  }, [trades, sortField, sortDirection, filterTicker, filterDirection, filterResult])

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('desc')
    }
  }

  const formatCurrency = (value: number) => {
    const formatted = Math.abs(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    return value >= 0 ? `$${formatted}` : `-$${formatted}`
  }

  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`

  const SortHeader = ({ field, label }: { field: SortField; label: string }) => (
    <th
      className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
      onClick={() => handleSort(field)}
    >
      <span className="flex items-center gap-1">
        {label}
        {sortField === field && (
          <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>
        )}
      </span>
    </th>
  )

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="bg-white rounded-lg shadow p-4 flex flex-wrap gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Ticker</label>
          <select
            value={filterTicker}
            onChange={(e) => setFilterTicker(e.target.value)}
            className="border rounded px-3 py-2 text-sm"
          >
            <option value="">All</option>
            {uniqueTickers.map((ticker) => (
              <option key={ticker} value={ticker}>{ticker}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Direction</label>
          <select
            value={filterDirection}
            onChange={(e) => setFilterDirection(e.target.value as any)}
            className="border rounded px-3 py-2 text-sm"
          >
            <option value="all">All</option>
            <option value="long">Long</option>
            <option value="short">Short</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Result</label>
          <select
            value={filterResult}
            onChange={(e) => setFilterResult(e.target.value as any)}
            className="border rounded px-3 py-2 text-sm"
          >
            <option value="all">All</option>
            <option value="win">Winners</option>
            <option value="loss">Losers</option>
          </select>
        </div>
        <div className="flex items-end">
          <span className="text-sm text-gray-500">
            Showing {filteredAndSorted.length} of {trades.length} trades
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <SortHeader field="ticker" label="Ticker" />
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Direction</th>
                <SortHeader field="entry_date" label="Entry Date" />
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Exit Date</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Entry Price</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Exit Price</th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Shares</th>
                <SortHeader field="pnl" label="P&L" />
                <SortHeader field="return_pct" label="Return" />
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Exit Reason</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredAndSorted.map((trade, idx) => (
                <tr key={idx} className="hover:bg-gray-50">
                  <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                    {trade.ticker}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      trade.direction === 'long' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {trade.direction.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                    {new Date(trade.entry_date).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                    {new Date(trade.exit_date).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 text-right">
                    ${trade.entry_price.toFixed(2)}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 text-right">
                    ${trade.exit_price.toFixed(2)}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 text-right">
                    {trade.shares.toFixed(2)}
                  </td>
                  <td className={`px-4 py-3 whitespace-nowrap text-sm font-medium text-right ${
                    trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatCurrency(trade.pnl)}
                  </td>
                  <td className={`px-4 py-3 whitespace-nowrap text-sm font-medium text-right ${
                    trade.return_pct >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatPercent(trade.return_pct)}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                    {trade.exit_reason.replace(/_/g, ' ')}
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

export default TradeExplorer
