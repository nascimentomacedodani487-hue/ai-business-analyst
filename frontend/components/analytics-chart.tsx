"use client";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

type ChartType = "bar" | "line" | "pie";

interface AnalyticsChartProps {
  data: Record<string, unknown>[];
  type: ChartType;
  title: string;
  labelKey: string;
  valueKey: string;
}

const COLORS = [
  "#3B82F6",
  "#8B5CF6",
  "#06B6D4",
  "#10B981",
  "#F59E0B",
  "#EC4899",
];

const TOOLTIP_STYLE = {
  backgroundColor: "#18181b",
  border: "1px solid #3f3f46",
  borderRadius: "8px",
  color: "#fff",
};

export default function AnalyticsChart({
  data,
  type,
  title,
  labelKey,
  valueKey,
}: AnalyticsChartProps) {
  if (!data || data.length === 0) {
    return null;
  }

  // Verifica se as colunas configuradas existem nos resultados.
  const hasRequiredColumns = data.every(
    (row) =>
      Object.prototype.hasOwnProperty.call(row, labelKey) &&
      Object.prototype.hasOwnProperty.call(row, valueKey)
  );

  if (!hasRequiredColumns) {
    console.error(
      "AnalyticsChart: colunas configuradas não encontradas nos dados.",
      {
        labelKey,
        valueKey,
        availableColumns: Object.keys(data[0]),
      }
    );

    return null;
  }

  // Garante que todos os valores da métrica sejam numéricos.
  const hasValidNumericValues = data.every(
    (row) =>
      typeof row[valueKey] === "number" &&
      Number.isFinite(row[valueKey] as number)
  );

  if (!hasValidNumericValues) {
    console.error(
      "AnalyticsChart: a métrica selecionada contém valores inválidos.",
      {
        valueKey,
      }
    );

    return null;
  }

  return (
    <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-5">
      <h3 className="mb-6 text-sm font-semibold text-zinc-200">
        {title}
      </h3>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {type === "bar" ? (
            <BarChart data={data}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#27272a"
              />

              <XAxis
                dataKey={labelKey}
                stroke="#71717a"
                fontSize={11}
              />

              <YAxis
                stroke="#71717a"
                fontSize={11}
              />

              <Tooltip contentStyle={TOOLTIP_STYLE} />

              <Bar
                dataKey={valueKey}
                name={title}
                fill="#3B82F6"
                radius={[5, 5, 0, 0]}
              />
            </BarChart>
          ) : type === "line" ? (
            <LineChart data={data}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#27272a"
              />

              <XAxis
                dataKey={labelKey}
                stroke="#71717a"
                fontSize={11}
              />

              <YAxis
                stroke="#71717a"
                fontSize={11}
              />

              <Tooltip contentStyle={TOOLTIP_STYLE} />

              <Line
                type="monotone"
                dataKey={valueKey}
                name={title}
                stroke="#8B5CF6"
                strokeWidth={3}
                dot={{
                  fill: "#8B5CF6",
                  r: 4,
                }}
                activeDot={{
                  r: 6,
                }}
              />
            </LineChart>
          ) : (
            <PieChart>
              <Pie
                data={data}
                dataKey={valueKey}
                nameKey={labelKey}
                cx="50%"
                cy="50%"
                outerRadius={95}
                label
              >
                {data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>

              <Tooltip contentStyle={TOOLTIP_STYLE} />

              <Legend />
            </PieChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}