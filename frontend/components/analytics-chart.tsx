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

type MetricFormat = "currency" | "percentage" | "integer" | "number";

interface ChartSeries {
  data_key: string;
  label: string;
}

interface AnalyticsChartProps {
  data: Record<string, unknown>[];
  type: ChartType;
  title: string;
  labelKey: string;
  valueKey: string;
  series?: ChartSeries[];
}

const COLORS = [
  "#3B82F6",
  "#8B5CF6",
  "#06B6D4",
  "#10B981",
  "#F59E0B",
  "#EC4899",
];

const MONTHS = [
  "jan",
  "fev",
  "mar",
  "abr",
  "mai",
  "jun",
  "jul",
  "ago",
  "set",
  "out",
  "nov",
  "dez",
];

const TOOLTIP_STYLE = {
  backgroundColor: "#18181b",
  border: "1px solid #3f3f46",
  borderRadius: "10px",
  color: "#fff",
  padding: "10px 14px",
  boxShadow: "0 8px 24px rgba(0, 0, 0, 0.25)",
};

const TOOLTIP_LABEL_STYLE = {
  color: "#a1a1aa",
  fontSize: "12px",
  fontWeight: 600,
  marginBottom: "6px",
};

const TOOLTIP_ITEM_STYLE = {
  color: "#e4e4e7",
  fontSize: "12px",
  paddingTop: "3px",
  paddingBottom: "3px",
};

const AXIS_TICK = {
  fill: "#a1a1aa",
  fontSize: 11,
};

const GRID_STYLE = {
  strokeDasharray: "3 3",
  stroke: "#27272a",
};

const PERCENT_FORMATTER = new Intl.NumberFormat("pt-BR", {
  style: "percent",
  maximumFractionDigits: 1,
});

function normalizeMetricName(value: string): string {
  return value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[_-]+/g, " ");
}

/**
 * Identifica o formato da métrica usando seu nome e descrição.
 *
 * A identificação utiliza os nomes das colunas e das séries,
 * além do título do gráfico quando necessário.
 */
function resolveMetricFormat(identifier: string): MetricFormat {
  const normalized = normalizeMetricName(identifier);

  const currencyPattern =
    /\b(revenue|faturamento|receita|amount|valor|preco|price|ticket|cost|custo|despesa|expense|gmv|vendas valor|sales value)\b/;

  const percentagePattern =
    /\b(percent|percentage|percentual|porcentagem|taxa|rate|margin|margem|share|participacao|proportion|conversion|conversao)\b/;

  const integerPattern =
    /\b(count|quantidade|qtd|pedidos|orders|order|clientes|customers|customer|restaurantes|restaurants|motoristas|drivers|unidades|units|itens|items|volume)\b/;

  if (currencyPattern.test(normalized)) {
    return "currency";
  }

  if (percentagePattern.test(normalized)) {
    return "percentage";
  }

  if (integerPattern.test(normalized)) {
    return "integer";
  }

  return "number";
}

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);

    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return null;
}

/**
 * Alguns indicadores chegam como fração decimal (0.0362),
 * enquanto outros já chegam em escala percentual (3.62).
 *
 * Valores entre -1 e 1 são tratados como frações decimais.
 */
function normalizePercentage(value: number): number {
  if (value !== 0 && Math.abs(value) <= 1) {
    return value * 100;
  }

  return value;
}

function formatMetricValue(
  value: unknown,
  format: MetricFormat
): string {
  const numericValue = toFiniteNumber(value);

  if (numericValue === null) {
    return value === null || value === undefined
      ? "—"
      : String(value);
  }

  switch (format) {
    case "currency":
      return new Intl.NumberFormat("pt-BR", {
        style: "currency",
        currency: "BRL",
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(numericValue);

    case "percentage":
      return new Intl.NumberFormat("pt-BR", {
        minimumFractionDigits: 1,
        maximumFractionDigits: 2,
      }).format(normalizePercentage(numericValue)) + "%";

    case "integer":
      return new Intl.NumberFormat("pt-BR", {
        maximumFractionDigits: 0,
      }).format(numericValue);

    default:
      return new Intl.NumberFormat("pt-BR", {
        minimumFractionDigits: 0,
        maximumFractionDigits: 2,
      }).format(numericValue);
  }
}

/**
 * Formatação compacta para os eixos.
 *
 * Exemplos:
 *  4901 -> 4,9 mil
 *  1250000 -> 1,3 mi
 *  45000 -> R$ 45 mil
 */
function formatAxisValue(
  value: unknown,
  format: MetricFormat
): string {
  const numericValue = toFiniteNumber(value);

  if (numericValue === null) {
    return String(value ?? "");
  }

  if (format === "currency") {
    return new Intl.NumberFormat("pt-BR", {
      style: "currency",
      currency: "BRL",
      notation: "compact",
      maximumFractionDigits: 1,
    }).format(numericValue);
  }

  if (format === "percentage") {
    return (
      new Intl.NumberFormat("pt-BR", {
        minimumFractionDigits: 0,
        maximumFractionDigits: 1,
      }).format(normalizePercentage(numericValue)) + "%"
    );
  }

  return new Intl.NumberFormat("pt-BR", {
    notation: "compact",
    maximumFractionDigits: format === "integer" ? 0 : 1,
  }).format(numericValue);
}

/**
 * Melhora a apresentação dos períodos no eixo X.
 *
 * 2025-01    -> jan/2025
 * 2025-01-15 -> 15/01
 */
function formatCategoryLabel(value: unknown): string {
  const label = String(value ?? "");

  const monthlyMatch = /^(\d{4})-(\d{2})$/.exec(label);

  if (monthlyMatch) {
    const year = monthlyMatch[1];
    const month = Number(monthlyMatch[2]);

    if (month >= 1 && month <= 12) {
      return `${MONTHS[month - 1]}/${year}`;
    }
  }

  const dailyMatch = /^(\d{4})-(\d{2})-(\d{2})(?:T.*)?$/.exec(
    label
  );

  if (dailyMatch) {
    return `${dailyMatch[3]}/${dailyMatch[2]}`;
  }

  return label;
}

export default function AnalyticsChart({
  data,
  type,
  title,
  labelKey,
  valueKey,
  series,
}: AnalyticsChartProps) {
  if (!Array.isArray(data) || data.length === 0) {
    return null;
  }

  const hasGroupedSeries =
    type === "bar" && Array.isArray(series) && series.length > 0;

  const groupedSeries: ChartSeries[] = hasGroupedSeries
    ? series
    : [];

  // --------------------------------------------------
  // 1. VALIDAÇÃO DA DIMENSÃO DO EIXO X
  // --------------------------------------------------

  const hasLabelColumn = data.every((row) =>
    Object.prototype.hasOwnProperty.call(row, labelKey)
  );

  if (!hasLabelColumn) {
    console.error(
      "AnalyticsChart: coluna do eixo X não encontrada nos dados.",
      {
        labelKey,
        availableColumns: Object.keys(data[0]),
      }
    );

    return null;
  }

  // --------------------------------------------------
  // 2. VALIDAÇÃO DAS SÉRIES AGRUPADAS
  // --------------------------------------------------

  if (hasGroupedSeries) {
    const hasRequiredSeriesColumns = groupedSeries.every((item) =>
      data.every((row) =>
        Object.prototype.hasOwnProperty.call(row, item.data_key)
      )
    );

    if (!hasRequiredSeriesColumns) {
      console.error(
        "AnalyticsChart: uma ou mais colunas das séries não foram encontradas.",
        {
          series: groupedSeries,
          availableColumns: Object.keys(data[0]),
        }
      );

      return null;
    }

    const hasValidSeriesValues = groupedSeries.every((item) =>
      data.every(
        (row) =>
          typeof row[item.data_key] === "number" &&
          Number.isFinite(row[item.data_key] as number)
      )
    );

    if (!hasValidSeriesValues) {
      console.error(
        "AnalyticsChart: uma ou mais séries possuem valores inválidos.",
        {
          series: groupedSeries,
        }
      );

      return null;
    }
  } else {
    // --------------------------------------------------
    // 3. VALIDAÇÃO DAS SÉRIES SIMPLES
    // --------------------------------------------------

    const hasValueColumn = data.every((row) =>
      Object.prototype.hasOwnProperty.call(row, valueKey)
    );

    if (!hasValueColumn) {
      console.error(
        "AnalyticsChart: coluna da métrica não encontrada nos dados.",
        {
          valueKey,
          availableColumns: Object.keys(data[0]),
        }
      );

      return null;
    }

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
  }

  // --------------------------------------------------
  // 4. IDENTIFICAÇÃO DO FORMATO DAS MÉTRICAS
  // --------------------------------------------------

  const seriesFormats = groupedSeries.map((item) =>
    resolveMetricFormat(`${item.data_key} ${item.label}`)
  );

  const uniqueSeriesFormats = Array.from(new Set(seriesFormats));

  const axisFormat: MetricFormat = hasGroupedSeries
    ? uniqueSeriesFormats.length === 1
      ? uniqueSeriesFormats[0]
      : "number"
    : resolveMetricFormat(`${valueKey} ${title}`);

  // --------------------------------------------------
  // 5. FORMATAÇÃO DINÂMICA DOS TOOLTIPS
  // --------------------------------------------------

  const tooltipFormatter = (
    value: unknown,
    name: unknown
  ): [string, string] => {
    let metricFormat: MetricFormat;

    let displayName = String(name ?? title);

    if (hasGroupedSeries) {
      const selectedSeries = groupedSeries.find(
        (item) => item.label === String(name)
      );

      if (selectedSeries) {
        metricFormat = resolveMetricFormat(
          `${selectedSeries.data_key} ${selectedSeries.label}`
        );
      } else {
        metricFormat = axisFormat;
      }
    } else {
      metricFormat = resolveMetricFormat(`${valueKey} ${title}`);
    }

    if (!displayName) {
      displayName = title;
    }

    return [
      formatMetricValue(value, metricFormat),
      displayName,
    ];
  };

  // --------------------------------------------------
  // 6. CONFIGURAÇÃO DOS EIXOS
  // --------------------------------------------------

  const renderXAxis = () => (
    <XAxis
      dataKey={labelKey}
      stroke="#71717a"
      tick={AXIS_TICK}
      tickFormatter={formatCategoryLabel}
      tickLine={{ stroke: "#3f3f46" }}
      axisLine={{ stroke: "#3f3f46" }}
      minTickGap={20}
    />
  );

  const renderYAxis = () => (
    <YAxis
      stroke="#71717a"
      tick={AXIS_TICK}
      tickFormatter={(value) => formatAxisValue(value, axisFormat)}
      tickLine={{ stroke: "#3f3f46" }}
      axisLine={false}
      width={76}
    />
  );

  const renderGrid = () => (
    <CartesianGrid
      strokeDasharray={GRID_STYLE.strokeDasharray}
      stroke={GRID_STYLE.stroke}
      vertical={false}
    />
  );

  const renderTooltip = () => (
    <Tooltip
      contentStyle={TOOLTIP_STYLE}
      labelStyle={TOOLTIP_LABEL_STYLE}
      itemStyle={TOOLTIP_ITEM_STYLE}
      formatter={tooltipFormatter}
      labelFormatter={(label) => formatCategoryLabel(label)}
      cursor={{
        fill: "rgba(255, 255, 255, 0.04)",
      }}
    />
  );

  // --------------------------------------------------
  // 7. RENDERIZAÇÃO
  // --------------------------------------------------

  return (
    <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-5">
      <div className="mb-5 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-zinc-200">
          {title}
        </h3>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {type === "bar" ? (
            <BarChart
              data={data}
              barGap={4}
              barCategoryGap="20%"
              margin={{
                top: 8,
                right: 12,
                bottom: 8,
                left: 4,
              }}
            >
              {renderGrid()}

              {renderXAxis()}

              {renderYAxis()}

              {renderTooltip()}

              {hasGroupedSeries ? (
                <>
                  {groupedSeries.map((item, index) => (
                    <Bar
                      key={item.data_key}
                      dataKey={item.data_key}
                      name={item.label}
                      fill={COLORS[index % COLORS.length]}
                      radius={[5, 5, 0, 0]}
                      maxBarSize={48}
                    />
                  ))}

                  <Legend
                    verticalAlign="bottom"
                    align="center"
                    iconType="circle"
                    iconSize={8}
                    wrapperStyle={{
                      fontSize: "12px",
                      paddingTop: "12px",
                      color: "#a1a1aa",
                    }}
                  />
                </>
              ) : (
                <Bar
                  dataKey={valueKey}
                  name={title}
                  fill="#3B82F6"
                  radius={[5, 5, 0, 0]}
                  maxBarSize={56}
                />
              )}
            </BarChart>
          ) : type === "line" ? (
            <LineChart
              data={data}
              margin={{
                top: 8,
                right: 12,
                bottom: 8,
                left: 4,
              }}
            >
              {renderGrid()}

              {renderXAxis()}

              {renderYAxis()}

              {renderTooltip()}

              <Line
                type="monotone"
                dataKey={valueKey}
                name={title}
                stroke="#8B5CF6"
                strokeWidth={3}
                dot={{
                  fill: "#8B5CF6",
                  stroke: "#18181b",
                  strokeWidth: 2,
                  r: 4,
                }}
                activeDot={{
                  fill: "#8B5CF6",
                  stroke: "#fff",
                  strokeWidth: 2,
                  r: 6,
                }}
                connectNulls={false}
              />
            </LineChart>
          ) : (
            <PieChart>
              <Pie
                data={data}
                dataKey={valueKey}
                nameKey={labelKey}
                cx="50%"
                cy="45%"
                outerRadius={95}
                paddingAngle={2}
                stroke="#18181b"
                strokeWidth={2}
                label={
                  data.length <= 8
                    ? ({ name, percent }) =>
                        `${String(name ?? "")} (${PERCENT_FORMATTER.format(
                          Number(percent ?? 0)
                        )})`
                    : false
                }
                labelLine={false}
              >
                {data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>

              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                labelStyle={TOOLTIP_LABEL_STYLE}
                itemStyle={TOOLTIP_ITEM_STYLE}
                formatter={tooltipFormatter}
              />

              <Legend
                verticalAlign="bottom"
                align="center"
                iconType="circle"
                iconSize={8}
                wrapperStyle={{
                  fontSize: "12px",
                  paddingTop: "8px",
                  color: "#a1a1aa",
                }}
              />
            </PieChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}