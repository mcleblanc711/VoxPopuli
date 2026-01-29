interface Props {
  config: Record<string, unknown>
}

function ConfigDisplay({ config }: Props) {
  const renderValue = (value: unknown): string => {
    if (value === null || value === undefined) return 'null'
    if (typeof value === 'boolean') return value ? 'true' : 'false'
    if (typeof value === 'number') return value.toString()
    if (typeof value === 'string') return value
    if (Array.isArray(value)) return value.join(', ')
    if (typeof value === 'object') return JSON.stringify(value, null, 2)
    return String(value)
  }

  const renderSection = (obj: Record<string, unknown>, depth = 0) => {
    return Object.entries(obj).map(([key, value]) => {
      const isObject = typeof value === 'object' && value !== null && !Array.isArray(value)

      return (
        <div key={key} className={depth > 0 ? 'ml-4' : ''}>
          {isObject ? (
            <div className="mb-2">
              <h4 className="font-medium text-gray-700 capitalize">{key.replace(/_/g, ' ')}</h4>
              {renderSection(value as Record<string, unknown>, depth + 1)}
            </div>
          ) : (
            <div className="flex justify-between py-1 border-b border-gray-100">
              <span className="text-gray-600 capitalize">{key.replace(/_/g, ' ')}</span>
              <span className="font-mono text-sm text-gray-900">{renderValue(value)}</span>
            </div>
          )}
        </div>
      )
    })
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Backtest Configuration</h3>
      <div className="space-y-4">
        {renderSection(config)}
      </div>
    </div>
  )
}

export default ConfigDisplay
