interface Props {
  data: Record<string, Record<string, number>>
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

function MonthlyHeatmap({ data }: Props) {
  const years = Object.keys(data).sort()

  const getColor = (value: number | undefined): string => {
    if (value === undefined) return 'bg-gray-100'
    if (value > 0.1) return 'bg-green-600 text-white'
    if (value > 0.05) return 'bg-green-500 text-white'
    if (value > 0.02) return 'bg-green-400'
    if (value > 0) return 'bg-green-200'
    if (value > -0.02) return 'bg-red-200'
    if (value > -0.05) return 'bg-red-400'
    if (value > -0.1) return 'bg-red-500 text-white'
    return 'bg-red-600 text-white'
  }

  const formatPercent = (value: number | undefined): string => {
    if (value === undefined) return '-'
    return `${(value * 100).toFixed(1)}%`
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full">
        <thead>
          <tr>
            <th className="px-2 py-1 text-left text-sm font-medium text-gray-500">Year</th>
            {MONTHS.map((month) => (
              <th key={month} className="px-2 py-1 text-center text-sm font-medium text-gray-500">
                {month}
              </th>
            ))}
            <th className="px-2 py-1 text-center text-sm font-medium text-gray-500">YTD</th>
          </tr>
        </thead>
        <tbody>
          {years.map((year) => {
            const yearData = data[year] || {}
            const ytd = Object.values(yearData).reduce((sum, val) => sum + val, 0)

            return (
              <tr key={year}>
                <td className="px-2 py-1 text-sm font-medium text-gray-700">{year}</td>
                {MONTHS.map((_, idx) => {
                  const monthNum = (idx + 1).toString()
                  const value = yearData[monthNum]

                  return (
                    <td
                      key={monthNum}
                      className={`px-2 py-1 text-center text-sm ${getColor(value)}`}
                    >
                      {formatPercent(value)}
                    </td>
                  )
                })}
                <td className={`px-2 py-1 text-center text-sm font-medium ${getColor(ytd)}`}>
                  {formatPercent(ytd)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default MonthlyHeatmap
