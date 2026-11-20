
"use client";

import {
  useEffect,
  useRef,
  useState
} from "react";

import {
  compareBenchmarks,
  fetchBenchmarkDetail,
  fetchBenchmarkHistory,
  type BenchmarkComparisonResult,
  type BenchmarkHistoryFilters,
  type BenchmarkHistoryItem,
  type BenchmarkReport,
  type MetricComparison
} from "@/lib/evaluation";

const PAGE_SIZE = 20;

type DraftFilters = {
  provider: string;
  model: string;
  dataset_sha256: string;
};

const INITIAL_DRAFT_FILTERS: DraftFilters = {
  provider: "",
  model: "",
  dataset_sha256: ""
};

const METRIC_LABELS: Record<string, string> = {
  pass_rate: "Taxa de aprovação",
  average_score: "Pontuação média",
  row_accuracy: "Precisão de linhas",
  cell_accuracy: "Precisão de células",
  average_duration_ms: "Duração média",
  error_cases: "Cenários com erro",
  failed_cases: "Cenários reprovados",
  skipped_cases: "Cenários ignorados",
  "llm_usage.tokens.input": "Tokens de entrada",
  "llm_usage.tokens.output": "Tokens de saída",
  "llm_usage.tokens.thinking": "Tokens de raciocínio",
  "llm_usage.tokens.total": "Tokens totais",
  "llm_usage.tokens.unreported": "Tokens não reportados",
  "llm_usage.requests.failed": "Requisições com falha",
  "llm_usage.requests.retries": "Novas tentativas",
  "llm_usage.latency.average_request_ms": "Latência média"
};

const STATUS_STYLES: Record<string, string> = {
  stable:
    "border-white/10 bg-white/[0.04] text-zinc-300",
  compared:
    "border-blue-500/20 bg-blue-500/10 text-blue-300",
  improvement:
    "border-emerald-500/20 bg-emerald-500/10 text-emerald-400",
  regression:
    "border-red-500/20 bg-red-500/10 text-red-400",
  mixed:
    "border-amber-500/20 bg-amber-500/10 text-amber-300",
  not_comparable:
    "border-white/10 bg-white/[0.04] text-zinc-500",
  model_changed:
    "border-amber-500/20 bg-amber-500/10 text-amber-300",
  model_unknown:
    "border-amber-500/20 bg-amber-500/10 text-amber-300",
  candidate_only:
    "border-blue-500/20 bg-blue-500/10 text-blue-300",
  baseline_only:
    "border-zinc-500/20 bg-zinc-500/10 text-zinc-400"
};

const STATUS_LABELS: Record<string, string> = {
  stable: "Estável",
  compared: "Comparado",
  improvement: "Melhoria",
  regression: "Regressão",
  mixed: "Resultado misto",
  not_comparable: "Não comparável",
  incompatible: "Datasets incompatíveis",
  model_changed: "Modelo alterado",
  model_unknown: "Modelo desconhecido",
  candidate_only: "Somente candidato",
  baseline_only: "Somente referência",
  completed: "Concluído",
  error: "Falha"
};

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";

  return new Intl.NumberFormat("pt-BR", {
    maximumFractionDigits: 2
  }).format(value);
}

function formatDate(value: string): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value || "Data indisponível";
  }

  return date.toLocaleString("pt-BR", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function getProviderLabel(provider: string): string {
  switch (provider) {
    case "gemini":
      return "Gemini";
    case "openai":
      return "OpenAI";
    case "anthropic":
      return "Anthropic";
    default:
      return provider;
  }
}

function getStatusLabel(status: string): string {
  return STATUS_LABELS[status] ?? status.replaceAll("_", " ");
}

function StatusBadge({
  status
}: {
  status: string;
}) {
  const style =
    STATUS_STYLES[status] ??
    "border-white/10 bg-white/[0.04] text-zinc-400";

  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-medium ${style}`}
    >
      {getStatusLabel(status)}
    </span>
  );
}

function BenchmarkStatus({
  item
}: {
  item: BenchmarkHistoryItem;
}) {
  const status = item.summary.all_providers_completed
    ? "completed"
    : item.summary.completed_providers > 0
      ? "mixed"
      : "error";

  return <StatusBadge status={status} />;
}

function ProviderBadge({
  provider
}: {
  provider: string;
}) {
  return (
    <span className="rounded-full border border-violet-500/15 bg-violet-500/[0.06] px-2.5 py-1 text-[10px] text-violet-300">
      {getProviderLabel(provider)}
    </span>
  );
}

function formatMetricValue(
  metricName: string,
  value: unknown
): string {
  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value !== "number" || !Number.isFinite(value)) {
    return String(value);
  }

  if (
    metricName === "pass_rate" ||
    metricName === "row_accuracy" ||
    metricName === "cell_accuracy"
  ) {
    return `${formatNumber(value)}%`;
  }

  if (metricName === "average_duration_ms") {
    return `${formatNumber(value)} ms`;
  }

  if (metricName.endsWith("_ms")) {
    return `${formatNumber(value)} ms`;
  }

  return formatNumber(value);
}

function formatMetricDelta(
  metricName: string,
  metric: MetricComparison
): string {
  if (metric.delta === null || !Number.isFinite(metric.delta)) {
    return "—";
  }

  const prefix = metric.delta > 0 ? "+" : "";

  return `${prefix}${formatMetricValue(metricName, metric.delta)}`;
}

function MetricComparisonTable({
  metrics
}: {
  metrics: Record<string, MetricComparison>;
}) {
  const entries = Object.entries(metrics);

  if (entries.length === 0) {
    return (
      <p className="py-3 text-xs text-zinc-500">
        Não existem métricas comparáveis para este provedor.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[680px] text-left text-xs">
        <thead>
          <tr className="border-b border-white/[0.07] text-[10px] text-zinc-500">
            <th className="px-3 py-3 font-medium">Métrica</th>
            <th className="px-3 py-3 font-medium">Referência</th>
            <th className="px-3 py-3 font-medium">Candidato</th>
            <th className="px-3 py-3 font-medium">Variação</th>
            <th className="px-3 py-3 font-medium">Resultado</th>
          </tr>
        </thead>

        <tbody>
          {entries.map(([metricName, metric]) => (
            <tr
              key={metricName}
              className="border-b border-white/[0.04] last:border-0"
            >
              <td className="px-3 py-3 text-zinc-300">
                {METRIC_LABELS[metricName] ?? metricName}
              </td>

              <td className="px-3 py-3 tabular-nums text-zinc-400">
                {formatMetricValue(metricName, metric.baseline)}
              </td>

              <td className="px-3 py-3 tabular-nums text-zinc-200">
                {formatMetricValue(metricName, metric.candidate)}
              </td>

              <td className="px-3 py-3 tabular-nums text-zinc-300">
                <div>{formatMetricDelta(metricName, metric)}</div>

                {metric.delta_pct !== null && (
                  <p className="mt-1 text-[10px] text-zinc-600">
                    {metric.delta_pct > 0 ? "+" : ""}
                    {formatNumber(metric.delta_pct)}%
                  </p>
                )}
              </td>

              <td className="px-3 py-3">
                <StatusBadge status={metric.status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BenchmarkDetails({
  report
}: {
  report: BenchmarkReport;
}) {
  return (
    <div className="mt-5 rounded-xl border border-blue-500/15 bg-[#0d111a] p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-blue-400">
            Detalhes do benchmark
          </p>

          <h3 className="mt-1 break-all text-sm font-semibold text-white">
            {report.benchmark_id}
          </h3>
        </div>

        <StatusBadge
          status={
            report.summary.all_providers_completed
              ? "completed"
              : report.summary.completed_providers > 0
                ? "mixed"
                : "error"
          }
        />
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-lg bg-white/[0.03] p-3">
          <p className="text-[10px] text-zinc-500">Cenários</p>
          <p className="mt-2 text-xl font-semibold text-white">
            {formatNumber(report.dataset.case_count)}
          </p>
        </div>

        <div className="rounded-lg bg-white/[0.03] p-3">
          <p className="text-[10px] text-zinc-500">Provedores</p>
          <p className="mt-2 text-xl font-semibold text-white">
            {formatNumber(report.summary.provider_count)}
          </p>
        </div>

        <div className="rounded-lg bg-white/[0.03] p-3">
          <p className="text-[10px] text-zinc-500">Concluídos</p>
          <p className="mt-2 text-xl font-semibold text-emerald-400">
            {formatNumber(report.summary.completed_providers)}
          </p>
        </div>

        <div className="rounded-lg bg-white/[0.03] p-3">
          <p className="text-[10px] text-zinc-500">Com falha</p>
          <p className="mt-2 text-xl font-semibold text-red-400">
            {formatNumber(report.summary.failed_providers)}
          </p>
        </div>
      </div>

      <div className="mt-5">
        <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
          Identificação do dataset
        </p>

        <div className="rounded-lg bg-black/20 p-3">
          <p className="break-all font-mono text-[10px] text-zinc-400">
            SHA-256: {report.dataset.sha256}
          </p>
          <p className="mt-2 text-[10px] text-zinc-500">
            Versão {report.dataset.version} · {report.dataset.case_count} cenários
          </p>
        </div>
      </div>

      <div className="mt-5">
        <p className="mb-3 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
          Execução por provedor
        </p>

        <div className="space-y-3">
          {Object.entries(report.provider_runs).map(
            ([provider, run]) => {
              const summary = run.report?.summary ?? {};

              return (
                <div
                  key={provider}
                  className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <ProviderBadge provider={provider} />
                      <span className="text-xs text-zinc-400">
                        {run.report?.model ?? "Modelo não informado"}
                      </span>
                    </div>

                    <StatusBadge status={run.execution_status} />
                  </div>

                  {run.execution_status === "completed" && (
                    <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                      {[
                        {
                          label: "Aprovados",
                          value: summary.passed_cases
                        },
                        {
                          label: "Reprovados",
                          value: summary.failed_cases
                        },
                        {
                          label: "Erros",
                          value: summary.error_cases
                        },
                        {
                          label: "Taxa de aprovação",
                          value:
                            typeof summary.pass_rate === "number"
                              ? `${formatNumber(summary.pass_rate)}%`
                              : "—"
                        }
                      ].map((metric) => (
                        <div key={metric.label}>
                          <p className="text-[10px] text-zinc-500">
                            {metric.label}
                          </p>
                          <p className="mt-1 text-sm font-semibold text-zinc-200">
                            {typeof metric.value === "number"
                              ? formatNumber(metric.value)
                              : typeof metric.value === "string"
                                ? metric.value
                                : metric.value == null
                                  ? "—"
                                  : JSON.stringify(metric.value)}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}

                  {run.error && (
                    <p className="mt-3 break-words text-xs leading-6 text-red-300">
                      {run.error}
                    </p>
                  )}

                  {run.error_type && (
                    <p className="mt-2 text-[10px] text-zinc-600">
                      Tipo: {run.error_type}
                    </p>
                  )}
                </div>
              );
            }
          )}
        </div>
      </div>

      {report.repeatability && (
        <details className="mt-5 rounded-lg border border-white/[0.06] bg-white/[0.02]">
          <summary className="cursor-pointer p-3 text-xs text-zinc-400 hover:text-zinc-200">
            Ver informações de repetibilidade
          </summary>

          <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words border-t border-white/[0.06] p-3 text-[10px] leading-5 text-zinc-400">
            {JSON.stringify(report.repeatability, null, 2)}
          </pre>
        </details>
      )}

      {report.comparison && (
        <details className="mt-3 rounded-lg border border-white/[0.06] bg-white/[0.02]">
          <summary className="cursor-pointer p-3 text-xs text-zinc-400 hover:text-zinc-200">
            Ver comparações originais do benchmark
          </summary>

          <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words border-t border-white/[0.06] p-3 text-[10px] leading-5 text-zinc-400">
            {JSON.stringify(report.comparison, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}

function BenchmarkComparisonPanel({
  result
}: {
  result: BenchmarkComparisonResult;
}) {
  const overallStyle =
    STATUS_STYLES[result.summary.status] ??
    STATUS_STYLES.not_comparable;

  return (
    <section className="mt-6 rounded-xl border border-white/[0.08] bg-[#11141c] p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-wider text-violet-400">
            Comparação histórica
          </p>
          <h3 className="mt-1 text-lg font-semibold text-white">
            Resultado da comparação
          </h3>
        </div>

        <span
          className={`rounded-full border px-3 py-1.5 text-[10px] font-medium ${overallStyle}`}
        >
          {getStatusLabel(result.summary.status)}
        </span>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-lg bg-white/[0.03] p-4">
          <p className="text-[10px] text-zinc-500">Referência</p>
          <p className="mt-2 break-all font-mono text-xs text-zinc-200">
            {result.baseline.benchmark_id}
          </p>
          <p className="mt-2 text-[10px] text-zinc-500">
            {result.baseline.created_at
              ? formatDate(result.baseline.created_at)
              : "Data não disponível"}
          </p>
        </div>

        <div className="rounded-lg bg-white/[0.03] p-4">
          <p className="text-[10px] text-zinc-500">Candidato</p>
          <p className="mt-2 break-all font-mono text-xs text-zinc-200">
            {result.candidate.benchmark_id}
          </p>
          <p className="mt-2 text-[10px] text-zinc-500">
            {result.candidate.created_at
              ? formatDate(result.candidate.created_at)
              : "Data não disponível"}
          </p>
        </div>
      </div>

      {!result.compatibility.comparable && (
        <div className="mt-4 rounded-lg border border-amber-500/20 bg-amber-500/[0.05] p-4">
          <p className="text-xs font-semibold text-amber-300">
            Os benchmarks não são compatíveis
          </p>
          <p className="mt-2 text-xs leading-6 text-zinc-400">
            O comparador não encontrou compatibilidade suficiente entre
            os datasets. Nenhuma regressão ou melhoria foi presumida.
          </p>
        </div>
      )}

      <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          {
            label: "Provedores comparados",
            value: result.summary.providers_compared,
            color: "text-blue-400"
          },
          {
            label: "Melhorias",
            value: result.summary.improvements,
            color: "text-emerald-400"
          },
          {
            label: "Regressões",
            value: result.summary.regressions,
            color: "text-red-400"
          },
          {
            label: "Métricas estáveis",
            value: result.summary.stable_metrics,
            color: "text-zinc-200"
          }
        ].map((metric) => (
          <div
            key={metric.label}
            className="rounded-lg bg-white/[0.03] p-3"
          >
            <p className="text-[10px] text-zinc-500">
              {metric.label}
            </p>
            <p className={`mt-2 text-2xl font-semibold ${metric.color}`}>
              {formatNumber(metric.value)}
            </p>
          </div>
        ))}
      </div>

      <div className="mt-6 space-y-4">
        {Object.entries(result.providers).map(
          ([provider, providerResult]) => (
            <div
              key={provider}
              className="overflow-hidden rounded-lg border border-white/[0.07]"
            >
              <div className="flex flex-wrap items-center justify-between gap-3 bg-white/[0.025] p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <ProviderBadge provider={provider} />
                  <span className="text-[10px] text-zinc-500">
                    {providerResult.baseline_model ?? "—"}
                    {" → "}
                    {providerResult.candidate_model ?? "—"}
                  </span>
                </div>

                <StatusBadge status={providerResult.status} />
              </div>

              {providerResult.execution && (
                <div className="border-t border-white/[0.05] px-4 py-3">
                  <p className="text-[10px] text-zinc-500">
                    Execução: {providerResult.execution.baseline ?? "—"}
                    {" → "}
                    {providerResult.execution.candidate ?? "—"}
                  </p>
                </div>
              )}

              <div className="p-3">
                <MetricComparisonTable
                  metrics={providerResult.metrics}
                />
              </div>
            </div>
          )
        )}
      </div>

      <p className="mt-5 text-[10px] leading-5 text-zinc-600">
        A comparação utiliza as tolerâncias e direções definidas no
        BenchmarkComparator. Custos financeiros não são tratados como
        métricas de regressão ou melhoria.
      </p>
    </section>
  );
}

export default function BenchmarkHistory() {
  const [items, setItems] = useState<BenchmarkHistoryItem[]>([]);
  const [pagination, setPagination] = useState({
    total: 0,
    limit: PAGE_SIZE,
    offset: 0,
    has_more: false
  });

  const [page, setPage] = useState(0);
  const [filters, setFilters] = useState<BenchmarkHistoryFilters>({});
  const [draftFilters, setDraftFilters] = useState<DraftFilters>(
    INITIAL_DRAFT_FILTERS
  );

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  const [detail, setDetail] = useState<BenchmarkReport | null>(null);
  const [detailLoadingId, setDetailLoadingId] = useState("");
  const [detailError, setDetailError] = useState("");

  const [comparison, setComparison] =
    useState<BenchmarkComparisonResult | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState("");

  const requestSequence = useRef(0);

  useEffect(() => {
    let active = true;
    const currentRequest = ++requestSequence.current;

    fetchBenchmarkHistory({
      ...filters,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE
    })
      .then((response) => {
        if (!active || currentRequest !== requestSequence.current) {
          return;
        }

        setItems(response.items);
        setPagination(response.pagination);
        setError("");
      })
      .catch((err: unknown) => {
        if (!active || currentRequest !== requestSequence.current) {
          return;
        }

        setError(
          err instanceof Error
            ? err.message
            : "Não foi possível carregar o histórico."
        );
      })
      .finally(() => {
        if (active && currentRequest === requestSequence.current) {
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [filters, page, refreshKey]);

  function handleRefresh() {
    setLoading(true);
    setError("");
    setRefreshKey((current) => current + 1);
  }

  function handleApplyFilters() {
    const nextFilters: BenchmarkHistoryFilters = {};

    if (draftFilters.provider) {
      nextFilters.provider = draftFilters.provider;
    }

    if (draftFilters.model.trim()) {
      nextFilters.model = draftFilters.model.trim();
    }

    if (draftFilters.dataset_sha256.trim()) {
      nextFilters.dataset_sha256 =
        draftFilters.dataset_sha256.trim();
    }

    setLoading(true);
    setError("");
    setFilters(nextFilters);
    setPage(0);
    setSelectedIds([]);
    setDetail(null);
    setComparison(null);
    setComparisonError("");
  }

  function handleClearFilters() {
    setLoading(true);
    setError("");
    setDraftFilters(INITIAL_DRAFT_FILTERS);
    setFilters({});
    setPage(0);
    setSelectedIds([]);
    setDetail(null);
    setComparison(null);
    setComparisonError("");
  }

  function handleSelectBenchmark(
    benchmarkId: string,
    checked: boolean
  ) {
    setSelectedIds((current) => {
      if (!checked) {
        return current.filter((id) => id !== benchmarkId);
      }

      if (current.includes(benchmarkId) || current.length >= 2) {
        return current;
      }

      return [...current, benchmarkId];
    });

    setComparison(null);
    setComparisonError("");
  }

  async function handleOpenDetails(benchmarkId: string) {
    if (detail?.benchmark_id === benchmarkId) {
      setDetail(null);
      setDetailError("");
      return;
    }

    setDetail(null);
    setDetailError("");
    setDetailLoadingId(benchmarkId);

    try {
      const report = await fetchBenchmarkDetail(benchmarkId);
      setDetail(report);
    } catch (err) {
      setDetailError(
        err instanceof Error
          ? err.message
          : "Não foi possível recuperar os detalhes."
      );
    } finally {
      setDetailLoadingId("");
    }
  }

  async function handleCompare() {
    if (selectedIds.length !== 2 || comparisonLoading) {
      return;
    }

    setComparisonLoading(true);
    setComparisonError("");
    setComparison(null);

    try {
      const result = await compareBenchmarks({
        baseline_id: selectedIds[0],
        candidate_id: selectedIds[1]
      });

      setComparison(result);
    } catch (err) {
      setComparisonError(
        err instanceof Error
          ? err.message
          : "Não foi possível comparar os benchmarks."
      );
    } finally {
      setComparisonLoading(false);
    }
  }

  const completedProviders = items.reduce(
    (total, item) => total + item.summary.completed_providers,
    0
  );

  const failedBenchmarks = items.filter(
    (item) => item.summary.failed_providers > 0
  ).length;

  const distinctDatasets = new Set(
    items.map((item) => item.dataset.sha256)
  ).size;

  const totalPages = Math.max(
    1,
    Math.ceil(pagination.total / PAGE_SIZE)
  );

  const selectionLabels = selectedIds.map((id, index) => ({
    id,
    role: index === 0 ? "Referência" : "Candidato"
  }));

  return (
    <div className="w-full space-y-6">
      <section className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
          <p className="text-xs text-zinc-500">
            Benchmarks encontrados
          </p>
          <p className="mt-3 text-3xl font-semibold text-white">
            {formatNumber(pagination.total)}
          </p>
          <p className="mt-2 text-[10px] text-zinc-600">
            Conforme os filtros aplicados
          </p>
        </div>

        <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
          <p className="text-xs text-zinc-500">
            Provedores concluídos
          </p>
          <p className="mt-3 text-3xl font-semibold text-emerald-400">
            {formatNumber(completedProviders)}
          </p>
          <p className="mt-2 text-[10px] text-zinc-600">
            Somente na página carregada
          </p>
        </div>

        <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
          <p className="text-xs text-zinc-500">
            Benchmarks parciais
          </p>
          <p className="mt-3 text-3xl font-semibold text-amber-300">
            {formatNumber(failedBenchmarks)}
          </p>
          <p className="mt-2 text-[10px] text-zinc-600">
            Somente na página carregada
          </p>
        </div>

        <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
          <p className="text-xs text-zinc-500">
            Datasets distintos
          </p>
          <p className="mt-3 text-3xl font-semibold text-blue-400">
            {formatNumber(distinctDatasets)}
          </p>
          <p className="mt-2 text-[10px] text-zinc-600">
            Somente na página carregada
          </p>
        </div>
      </section>

      <section className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs font-semibold text-zinc-200">
              Filtrar benchmarks
            </p>
            <p className="mt-1 text-[10px] text-zinc-500">
              Combine provedor, modelo e fingerprint do dataset.
            </p>
          </div>

          <button
            type="button"
            onClick={handleRefresh}
            disabled={loading}
            className="rounded-lg border border-white/10 px-3 py-2 text-xs text-zinc-400 transition hover:bg-white/[0.05] disabled:opacity-40"
          >
            {loading ? "Atualizando..." : "↻ Atualizar"}
          </button>
        </div>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
          <label className="block">
            <span className="mb-2 block text-[10px] text-zinc-500">
              Provedor
            </span>

            <select
              value={draftFilters.provider}
              onChange={(event) =>
                setDraftFilters((current) => ({
                  ...current,
                  provider: event.target.value
                }))
              }
              className="w-full rounded-lg border border-white/10 bg-[#0d0f15] px-3 py-3 text-xs text-zinc-200 outline-none focus:border-blue-500/40"
            >
              <option value="">Todos os provedores</option>
              <option value="gemini">Gemini</option>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
            </select>
          </label>

          <label className="block">
            <span className="mb-2 block text-[10px] text-zinc-500">
              Modelo
            </span>

            <input
              value={draftFilters.model}
              onChange={(event) =>
                setDraftFilters((current) => ({
                  ...current,
                  model: event.target.value
                }))
              }
              placeholder="Ex.: gemini-2.5-flash"
              className="w-full rounded-lg border border-white/10 bg-[#0d0f15] px-3 py-3 text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-blue-500/40"
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-[10px] text-zinc-500">
              SHA-256 do dataset
            </span>

            <input
              value={draftFilters.dataset_sha256}
              onChange={(event) =>
                setDraftFilters((current) => ({
                  ...current,
                  dataset_sha256: event.target.value
                }))
              }
              placeholder="Fingerprint completo"
              maxLength={64}
              className="w-full rounded-lg border border-white/10 bg-[#0d0f15] px-3 py-3 font-mono text-xs text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-blue-500/40"
            />
          </label>
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={handleApplyFilters}
            className="rounded-lg bg-blue-600 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-blue-500"
          >
            Aplicar filtros
          </button>

          <button
            type="button"
            onClick={handleClearFilters}
            className="rounded-lg border border-white/10 px-4 py-2.5 text-xs text-zinc-400 transition hover:bg-white/[0.04]"
          >
            Limpar filtros
          </button>
        </div>
      </section>

      <section className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-white">
              Histórico de benchmarks
            </p>
            <p className="mt-1 text-[10px] text-zinc-500">
              {pagination.total} registro(s) · {PAGE_SIZE} por página
            </p>
          </div>

          <span className="rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1.5 text-[10px] text-zinc-500">
            Histórico persistente · SQLite
          </span>
        </div>

        {error && (
          <div className="mb-5 rounded-lg border border-red-500/20 bg-red-500/[0.05] p-4">
            <p className="text-xs font-semibold text-red-400">
              Não foi possível carregar os benchmarks
            </p>
            <p className="mt-2 text-xs leading-6 text-zinc-400">
              {error}
            </p>
            <button
              type="button"
              onClick={handleRefresh}
              className="mt-3 rounded-lg border border-red-500/20 px-3 py-2 text-xs text-red-300 hover:bg-red-500/10"
            >
              Tentar novamente
            </button>
          </div>
        )}

        {loading && items.length === 0 && (
          <div className="flex items-center gap-3 py-10">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-blue-400/30 border-t-blue-400" />
            <p className="text-xs text-zinc-400">
              Consultando o histórico de benchmarks...
            </p>
          </div>
        )}

        {!loading && !error && items.length === 0 && (
          <div className="rounded-lg border border-dashed border-white/10 p-8 text-center">
            <div className="text-2xl text-zinc-600">⌁</div>
            <p className="mt-3 text-sm font-medium text-zinc-300">
              Nenhum benchmark encontrado
            </p>
            <p className="mx-auto mt-2 max-w-md text-xs leading-6 text-zinc-500">
              Os relatórios persistidos pela CLI aparecerão aqui.
              A página não executa benchmarks automaticamente.
            </p>
          </div>
        )}

        {items.length > 0 && (
          <div className="space-y-3">
            {items.map((item) => {
              const isSelected = selectedIds.includes(item.benchmark_id);
              const isDisabled =
                selectedIds.length >= 2 && !isSelected;

              return (
                <article
                  key={item.benchmark_id}
                  className={`rounded-xl border p-4 transition ${
                    isSelected
                      ? "border-blue-500/30 bg-blue-500/[0.035]"
                      : "border-white/[0.06] bg-white/[0.015]"
                  }`}
                >
                  <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
                    <div className="flex min-w-0 items-start gap-3">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        disabled={isDisabled}
                        onChange={(event) =>
                          handleSelectBenchmark(
                            item.benchmark_id,
                            event.target.checked
                          )
                        }
                        aria-label={`Selecionar benchmark ${item.benchmark_id} para comparação`}
                        className="mt-1 h-4 w-4 shrink-0 accent-blue-500 disabled:cursor-not-allowed disabled:opacity-30"
                      />

                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="break-all font-mono text-xs font-semibold text-zinc-200">
                            {item.benchmark_id}
                          </h3>
                          <BenchmarkStatus item={item} />
                        </div>

                        <p className="mt-2 text-[10px] text-zinc-500">
                          {formatDate(item.created_at)}
                        </p>

                        <div className="mt-3 flex flex-wrap gap-2">
                          {item.providers.map((provider) => (
                            <ProviderBadge
                              key={provider}
                              provider={provider}
                            />
                          ))}
                        </div>

                        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-[10px] text-zinc-500">
                          <span>
                            {formatNumber(item.dataset.case_count)} cenários
                          </span>
                          <span>
                            {formatNumber(item.summary.completed_providers)} concluído(s)
                          </span>
                          <span>
                            {formatNumber(item.summary.failed_providers)} com falha
                          </span>
                          <span>
                            Dataset v{item.dataset.version}
                          </span>
                        </div>

                        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2">
                          {Object.entries(item.models).map(
                            ([provider, model]) => (
                              <p
                                key={provider}
                                className="text-[10px] text-zinc-600"
                              >
                                {getProviderLabel(provider)}:{" "}
                                <span className="text-zinc-400">
                                  {model ?? "Modelo não informado"}
                                </span>
                              </p>
                            )
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="flex shrink-0 flex-wrap items-center gap-2 xl:justify-end">
                      <button
                        type="button"
                        onClick={() =>
                          void handleOpenDetails(item.benchmark_id)
                        }
                        disabled={detailLoadingId === item.benchmark_id}
                        className="rounded-lg border border-white/10 px-3 py-2 text-xs text-zinc-300 transition hover:bg-white/[0.05] disabled:opacity-40"
                      >
                        {detailLoadingId === item.benchmark_id
                          ? "Carregando..."
                          : detail?.benchmark_id === item.benchmark_id
                            ? "Fechar detalhes"
                            : "Ver detalhes"}
                      </button>
                    </div>
                  </div>

                  {detailError &&
                    detailLoadingId === "" &&
                    detail === null && (
                      <p className="mt-3 text-xs text-red-300">
                        {detailError}
                      </p>
                    )}

                  {detail?.benchmark_id === item.benchmark_id && (
                    <BenchmarkDetails report={detail} />
                  )}
                </article>
              );
            })}
          </div>
        )}

        {detailError && (
          <div className="mt-4 rounded-lg border border-red-500/20 bg-red-500/[0.05] p-3">
            <p className="text-xs text-red-300">{detailError}</p>
          </div>
        )}

        {pagination.total > PAGE_SIZE && (
          <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-white/[0.06] pt-5">
            <p className="text-[10px] text-zinc-500">
              Página {page + 1} de {totalPages}
            </p>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  setLoading(true);
                  setError("");
                  setPage((current) => Math.max(0, current - 1));
                }}
                disabled={page === 0 || loading}
                className="rounded-lg border border-white/10 px-3 py-2 text-xs text-zinc-400 transition hover:bg-white/[0.05] disabled:cursor-not-allowed disabled:opacity-30"
              >
                ← Anterior
              </button>

              <button
                type="button"
                onClick={() => {
                  setLoading(true);
                  setError("");
                  setPage((current) =>
                    Math.min(totalPages - 1, current + 1)
                  );
                }}
                disabled={!pagination.has_more || loading}
                className="rounded-lg border border-white/10 px-3 py-2 text-xs text-zinc-400 transition hover:bg-white/[0.05] disabled:cursor-not-allowed disabled:opacity-30"
              >
                Próxima →
              </button>
            </div>
          </div>
        )}
      </section>

      <section className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-zinc-200">
              Comparar benchmarks
            </p>
            <p className="mt-1 text-[10px] leading-5 text-zinc-500">
              Selecione dois relatórios. O primeiro será a referência e o
              segundo será o candidato.
            </p>
          </div>

          <span className="rounded-full border border-violet-500/20 bg-violet-500/[0.06] px-3 py-1.5 text-[10px] text-violet-300">
            {selectedIds.length}/2 selecionados
          </span>
        </div>

        {selectionLabels.length > 0 && (
          <div className="mt-4 space-y-2">
            {selectionLabels.map((selection) => (
              <div
                key={selection.id}
                className="flex flex-wrap items-center gap-2 rounded-lg bg-white/[0.03] p-3"
              >
                <span className="text-[10px] text-zinc-500">
                  {selection.role}
                </span>
                <span className="break-all font-mono text-[10px] text-zinc-300">
                  {selection.id}
                </span>
              </div>
            ))}
          </div>
        )}

        <button
          type="button"
          onClick={() => void handleCompare()}
          disabled={selectedIds.length !== 2 || comparisonLoading}
          className="mt-4 flex items-center gap-2 rounded-lg bg-violet-600 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-35"
        >
          {comparisonLoading
            ? "Comparando relatórios..."
            : "Comparar selecionados"}
        </button>

        {comparisonError && (
          <div className="mt-4 rounded-lg border border-red-500/20 bg-red-500/[0.05] p-3">
            <p className="text-xs text-red-300">
              {comparisonError}
            </p>
          </div>
        )}

        {comparison && (
          <BenchmarkComparisonPanel result={comparison} />
        )}

        <p className="mt-4 text-[10px] leading-5 text-zinc-600">
          Esta operação consulta somente benchmarks já persistidos.
          Nenhuma chamada aos provedores de IA é realizada.
        </p>
      </section>
    </div>
  );
}
