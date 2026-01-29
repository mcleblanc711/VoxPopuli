import { useState, useEffect } from 'react'
import SummaryDashboard from './components/SummaryDashboard'
import TradeExplorer from './components/TradeExplorer'
import StrategyComparison from './components/StrategyComparison'
import ConfigDisplay from './components/ConfigDisplay'
import { BacktestResult } from './types'

function App() {
  const [results, setResults] = useState<BacktestResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'summary' | 'trades' | 'compare' | 'config'>('summary')

  useEffect(() => {
    loadResults()
  }, [])

  const loadResults = async () => {
    try {
      // Try to load from data directory
      const response = await fetch('./data/attention_momentum_results.json')
      if (!response.ok) {
        // Try sample data
        const sampleResponse = await fetch('./data/sample_results.json')
        if (!sampleResponse.ok) {
          setError('No backtest results found. Run a backtest first.')
          setLoading(false)
          return
        }
        const data = await sampleResponse.json()
        setResults(data)
      } else {
        const data = await response.json()
        setResults(data)
      }
      setLoading(false)
    } catch (err) {
      setError('Failed to load results')
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl text-gray-600">Loading...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl text-red-600">{error}</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-indigo-600 text-white py-4 px-6 shadow-lg">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold">VoxPopuli</h1>
          <span className="text-indigo-200">Reddit Sentiment Trading Backtester</span>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex space-x-8">
            {(['summary', 'trades', 'compare', 'config'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-4 px-2 border-b-2 font-medium text-sm capitalize ${
                  activeTab === tab
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab === 'compare' ? 'Strategy Comparison' : tab}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 px-6">
        {results && (
          <>
            {activeTab === 'summary' && <SummaryDashboard results={results} />}
            {activeTab === 'trades' && <TradeExplorer trades={results.trades} />}
            {activeTab === 'compare' && <StrategyComparison />}
            {activeTab === 'config' && <ConfigDisplay config={results.metadata.config} />}
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-400 py-4 px-6 mt-8">
        <div className="max-w-7xl mx-auto text-center text-sm">
          VoxPopuli - Built with React, Tailwind CSS, and Recharts
        </div>
      </footer>
    </div>
  )
}

export default App
