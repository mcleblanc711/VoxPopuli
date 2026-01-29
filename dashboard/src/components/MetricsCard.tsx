interface Props {
  label: string
  value: string
  positive?: boolean
  small?: boolean
}

function MetricsCard({ label, value, positive, small = false }: Props) {
  const colorClass = positive === undefined
    ? 'text-gray-900'
    : positive
    ? 'text-green-600'
    : 'text-red-600'

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <p className={`text-gray-500 ${small ? 'text-xs' : 'text-sm'}`}>{label}</p>
      <p className={`font-semibold ${colorClass} ${small ? 'text-lg' : 'text-2xl'}`}>
        {value}
      </p>
    </div>
  )
}

export default MetricsCard
