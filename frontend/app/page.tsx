"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AnalyticsChart from "@/components/analytics-chart";
import AgentTimeline from "@/components/agent-timeline";
import HistoryManager from "@/components/history-manager";
import ResultsTable from "@/components/results-table";
import {
  fetchAnalysisHistory,
  fetchAnalysisDetail,
  restoreAnalysis,
  type Analysis,
  type HistoryItem,
  type LLMProvider
} from "@/lib/analysis-history";

const API_URL = process.env.NEXT_PUBLIC_API_URL;
const HISTORY_STORAGE_KEY = "ai-business-analyst-history-v1";
const MAX_HISTORY_ITEMS = 50;

const PROVIDERS: {
  value: LLMProvider;
  label: string;
  model: string;
  description: string;
}[] = [
  {
    value: "gemini",
    label: "Google Gemini",
    model: "gemini-2.5-flash",
    description: "Modelo padrão"
  },
  {
    value: "openai",
    label: "OpenAI",
    model: "gpt-4.1-mini",
    description: "GPT"
  },
  {
    value: "anthropic",
    label: "Anthropic Claude",
    model: "claude-haiku-4-5-20251001",
    description: "Claude"
  }
];

const suggestions = [
  {
    icon: "↗",
    title: "Análise de vendas",
    description: "Como foram as vendas no período?",
    question: "Como foram as vendas no período?"
  },
  {
    icon: "◉",
    title: "Desempenho regional",
    description: "Qual cidade teve mais pedidos entregues?",
    question: "Qual cidade teve mais pedidos entregues?"
  },
  {
    icon: "⌁",
    title: "Análise de restaurantes",
    description: "Quais restaurantes tiveram melhor desempenho?",
    question: "Quais restaurantes tiveram melhor desempenho?"
  },
  {
    icon: "◇",
    title: "Indicadores operacionais",
    description: "Qual é nossa taxa de cancelamento?",
    question: "Qual é nossa taxa de cancelamento?"
  }
];

type Page = "overview" | "analyst" | "data" | "history";

type CatalogColumn = {
  name: string;
  type: string;
  nullable: boolean;
  description: string;
};

type CatalogTable = {
  name: string;
  description: string;
  total_rows: number;
  columns: CatalogColumn[];
  sample: Record<string, unknown>[];
};

type DataCatalog = {
  database: string;
  dataset: string;
  period: string;
  tables_count: number;
  tables: CatalogTable[];
};

type LLMMetrics = {
  provider: string;
  model: string;
  pricing_tier: string;
  cost_estimate_usd: number | null;
  requests: {
    total: number;
    successful: number;
    failed: number;
    retries: number;
  };
  tokens: {
    input: number;
    output: number;
    thinking: number;
    total: number;
    unreported: number;
  };
  latency: {
    average_ms: number;
  };
  quota?: {
    status: string;
    note: string;
  };
};

type ApiRequestError = Error & {
  analysisId?: string;
};

type ApiErrorDetail = {
  message?: unknown;
  analysis_id?: unknown;
};

function isLLMProvider(value: unknown): value is LLMProvider {
  return (
    value === "gemini" ||
    value === "openai" ||
    value === "anthropic"
  );
}

function getProviderLabel(provider?: string | null) {
  switch (provider) {
    case "gemini":
      return "Google Gemini";
    case "openai":
      return "OpenAI";
    case "anthropic":
      return "Anthropic Claude";
    default:
      return provider || "Provedor não identificado";
  }
}

function getProviderModel(provider: LLMProvider) {
  return PROVIDERS.find((item) => item.value === provider)?.model ?? "";
}

function formatTime(value?: number | null) {
  if (
    value === undefined ||
    value === null ||
    !Number.isFinite(value) ||
    value < 0
  ) {
    return "—";
  }

  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }

  return `${(value / 1000).toFixed(2)} s`;
}

function formatNumber(value: number) {
  if (!Number.isFinite(value)) return "—";

  return new Intl.NumberFormat("pt-BR").format(value);
}

function formatValue(value: unknown) {
  if (value === null || value === undefined) return "—";

  if (typeof value === "boolean") {
    return value ? "Sim" : "Não";
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}

function extractApiError(
  data: unknown,
  fallback: string
): ApiRequestError {
  if (!data || typeof data !== "object") {
    return new Error(fallback);
  }

  const responseData = data as {
    detail?: unknown;
  };

  const detail = responseData.detail;

  if (typeof detail === "string" && detail.trim()) {
    return new Error(detail);
  }

  if (detail && typeof detail === "object") {
    const errorDetail = detail as ApiErrorDetail;

    const message =
      typeof errorDetail.message === "string" &&
      errorDetail.message.trim()
        ? errorDetail.message
        : fallback;

    const error = new Error(message) as ApiRequestError;

    if (
      typeof errorDetail.analysis_id === "string" &&
      errorDetail.analysis_id.trim()
    ) {
      error.analysisId = errorDetail.analysis_id;
    }

    return error;
  }

  return new Error(fallback);
}

async function readJsonResponse(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function isValidAnalysis(value: unknown): value is Analysis {
  if (!value || typeof value !== "object") return false;

  const item = value as Partial<Analysis>;

  return (
    typeof item.question === "string" &&
    typeof item.sql === "string" &&
    Array.isArray(item.results) &&
    typeof item.total_results === "number" &&
    typeof item.answer === "string" &&
    typeof item.cache_hit === "boolean" &&
    Array.isArray(item.steps) &&
    Boolean(item.metrics) &&
    typeof item.metrics?.total_time_ms === "number"
  );
}

function isValidHistoryItem(value: unknown): value is HistoryItem {
  if (!value || typeof value !== "object") return false;

  const item = value as Partial<HistoryItem>;

  return (
    typeof item.question === "string" &&
    typeof item.total_time_ms === "number" &&
    Number.isFinite(item.total_time_ms) &&
    item.total_time_ms >= 0 &&
    typeof item.cache_hit === "boolean" &&
    typeof item.timestamp === "string"
  );
}

function LLMMetricsPanel({
  metrics,
  loading,
  error,
  onRefresh
}: {
  metrics: LLMMetrics | null;
  loading: boolean;
  error: string;
  onRefresh: () => void;
}) {
  const isFreeTier = metrics?.pricing_tier === "free";

  const formattedCost =
    metrics?.cost_estimate_usd !== null &&
    metrics?.cost_estimate_usd !== undefined &&
    Number.isFinite(metrics.cost_estimate_usd)
      ? `US$ ${metrics.cost_estimate_usd.toFixed(2)}`
      : "Não calculado";

  return (
    <section className="mb-8 rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-violet-400">
            Monitoramento de IA
          </p>
          <h2 className="mt-1 text-lg font-semibold text-white">
            Utilização dos modelos
          </h2>
          <p className="mt-1 text-xs text-zinc-500">
            Métricas agregadas das chamadas ao provedor de IA selecionado.
          </p>
        </div>

        <button
          onClick={onRefresh}
          disabled={loading}
          className="rounded-lg border border-white/10 px-3 py-2 text-xs text-zinc-400 transition hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? "Atualizando..." : "↻ Atualizar"}
        </button>
      </div>

      {loading && !metrics && (
        <div className="flex items-center gap-3 py-6">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-violet-400/30 border-t-violet-400" />
          <p className="text-xs text-zinc-400">
            Carregando métricas do provedor...
          </p>
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-500/20 bg-red-500/[0.06] p-4">
          <p className="text-xs font-medium text-red-400">
            Não foi possível carregar as métricas de IA
          </p>
          <p className="mt-2 text-xs text-zinc-400">{error}</p>
        </div>
      )}

      {metrics && (
        <>
          <div className="mb-5 flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-violet-500/20 bg-violet-500/10 px-3 py-1 text-[10px] text-violet-300">
              {getProviderLabel(metrics.provider)}
            </span>

            <span className="rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1 text-[10px] text-zinc-400">
              {metrics.model}
            </span>

            <span
              className={`rounded-full border px-3 py-1 text-[10px] ${
                isFreeTier
                  ? "border-emerald-500/20 bg-emerald-500/[0.06] text-emerald-400"
                  : "border-amber-500/20 bg-amber-500/[0.06] text-amber-300"
              }`}
            >
              {isFreeTier ? "Camada gratuita configurada" : "Cobrança por uso"}
            </span>
          </div>

          <div className="mb-3 grid grid-cols-2 gap-3 lg:grid-cols-4">
            <div className="rounded-lg bg-white/[0.03] p-4">
              <p className="text-[10px] text-zinc-500">
                Requisições totais
              </p>
              <p className="mt-2 text-2xl font-semibold text-white">
                {formatNumber(metrics.requests.total)}
              </p>
              <p className="mt-1 text-[10px] text-zinc-600">
                Chamadas registradas
              </p>
            </div>

            <div className="rounded-lg bg-white/[0.03] p-4">
              <p className="text-[10px] text-zinc-500">Sucessos</p>
              <p className="mt-2 text-2xl font-semibold text-emerald-400">
                {formatNumber(metrics.requests.successful)}
              </p>
              <p className="mt-1 text-[10px] text-zinc-600">
                Chamadas concluídas
              </p>
            </div>

            <div className="rounded-lg bg-white/[0.03] p-4">
              <p className="text-[10px] text-zinc-500">Falhas</p>
              <p className="mt-2 text-2xl font-semibold text-red-400">
                {formatNumber(metrics.requests.failed)}
              </p>
              <p className="mt-1 text-[10px] text-zinc-600">
                Chamadas com erro
              </p>
            </div>

            <div className="rounded-lg bg-white/[0.03] p-4">
              <p className="text-[10px] text-zinc-500">Retries</p>
              <p className="mt-2 text-2xl font-semibold text-amber-400">
                {formatNumber(metrics.requests.retries)}
              </p>
              <p className="mt-1 text-[10px] text-zinc-600">
                Novas tentativas programadas
              </p>
            </div>
          </div>

          <div className="mb-3 rounded-lg bg-white/[0.03] p-4">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs font-semibold text-zinc-300">
                Consumo de tokens
              </p>
              <span className="text-[10px] text-zinc-500">
                {formatNumber(metrics.tokens.total)} tokens contabilizados
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <div>
                <p className="text-[10px] text-zinc-500">Entrada</p>
                <p className="mt-2 text-xl font-semibold text-blue-400">
                  {formatNumber(metrics.tokens.input)}
                </p>
              </div>

              <div>
                <p className="text-[10px] text-zinc-500">Saída</p>
                <p className="mt-2 text-xl font-semibold text-violet-400">
                  {formatNumber(metrics.tokens.output)}
                </p>
              </div>

              <div>
                <p className="text-[10px] text-zinc-500">Raciocínio</p>
                <p className="mt-2 text-xl font-semibold text-amber-400">
                  {formatNumber(metrics.tokens.thinking)}
                </p>
              </div>

              <div>
                <p className="text-[10px] text-zinc-500">
                  Não reportados
                </p>
                <p className="mt-2 text-xl font-semibold text-zinc-300">
                  {formatNumber(metrics.tokens.unreported)}
                </p>
              </div>
            </div>

            {metrics.tokens.unreported > 0 && (
              <p className="mt-3 text-[10px] text-zinc-600">
                Algumas chamadas não disponibilizaram informações de
                utilização de tokens.
              </p>
            )}
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="rounded-lg bg-white/[0.03] p-4">
              <p className="text-[10px] text-zinc-500">Latência média</p>
              <p className="mt-2 text-2xl font-semibold text-white">
                {formatTime(metrics.latency.average_ms)}
              </p>
              <p className="mt-1 text-[10px] text-zinc-600">
                Por requisição registrada
              </p>
            </div>

            <div className="rounded-lg bg-white/[0.03] p-4">
              <p className="text-[10px] text-zinc-500">
                Custo monetário estimado
              </p>
              <p
                className={`mt-2 text-2xl font-semibold ${
                  metrics.cost_estimate_usd === null
                    ? "text-zinc-300"
                    : "text-emerald-400"
                }`}
              >
                {formattedCost}
              </p>
              <p className="mt-1 text-[10px] text-zinc-600">
                {isFreeTier
                  ? "Estimativa da configuração atual"
                  : "Estimativa financeira indisponível"}
              </p>
            </div>
          </div>

          {metrics.quota && (
            <div className="mt-4 rounded-lg border border-amber-500/15 bg-amber-500/[0.04] p-4">
              <div className="flex items-center gap-2">
                <span className="text-amber-400">ⓘ</span>
                <p className="text-xs font-semibold text-amber-300">
                  Informações sobre a cota gratuita
                </p>
              </div>

              <p className="mt-2 text-xs leading-6 text-zinc-400">
                {metrics.quota.note}
              </p>

              <p className="mt-2 text-[10px] text-zinc-600">
                O custo monetário estimado não representa o saldo ou a
                disponibilidade da cota da API.
              </p>
            </div>
          )}

          {!isFreeTier && (
            <div className="mt-4 rounded-lg border border-white/[0.07] bg-white/[0.02] p-4">
              <p className="text-xs leading-6 text-zinc-400">
                O provedor utiliza cobrança baseada no consumo. O custo
                financeiro real depende dos preços vigentes e da quantidade
                de tokens faturáveis. Este painel ainda não calcula esse valor.
              </p>
            </div>
          )}
        </>
      )}

      {!metrics && !loading && !error && (
        <p className="py-4 text-xs text-zinc-500">
          Nenhuma métrica disponível no momento.
        </p>
      )}
    </section>
  );
}

export default function Home() {
  const [activePage, setActivePage] = useState<Page>("analyst");
  const [question, setQuestion] = useState("");
  const [sentQuestion, setSentQuestion] = useState("");
  const [selectedProvider, setSelectedProvider] =
    useState<LLMProvider>("gemini");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [errorAnalysisId, setErrorAnalysisId] = useState("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);

  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [historyDetailLoadingId, setHistoryDetailLoadingId] =
    useState("");

  const [catalog, setCatalog] = useState<DataCatalog | null>(null);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState("");

  const [llmMetrics, setLlmMetrics] = useState<LLMMetrics | null>(null);
  const [llmMetricsLoading, setLlmMetricsLoading] = useState(false);
  const [llmMetricsError, setLlmMetricsError] = useState("");

  // Identifica a consulta de métricas mais recente e evita respostas antigas.
  const llmMetricsRequestId = useRef(0);

  // Recupera o histórico antigo do navegador como alternativa.
  useEffect(() => {
    try {
      const storedHistory = window.localStorage.getItem(
        HISTORY_STORAGE_KEY
      );

      if (storedHistory) {
        const parsed: unknown = JSON.parse(storedHistory);

        if (Array.isArray(parsed)) {
          const validItems = parsed
            .filter(isValidHistoryItem)
            .map((item) => ({
              ...item,
              analysis: isValidAnalysis(item.analysis)
                ? item.analysis
                : undefined
            }));

          // Sincronização intencional com o armazenamento local após a hidratação.
          // eslint-disable-next-line react-hooks/set-state-in-effect
          setHistory(validItems.slice(0, MAX_HISTORY_ITEMS));
          setHistoryTotal(validItems.length);
        }
      }
    } catch (storageError) {
      console.warn(
        "Não foi possível recuperar o histórico salvo.",
        storageError
      );
    } finally {
      setHistoryLoaded(true);
    }
  }, []);

  // Persiste no navegador para manter compatibilidade e fallback.
  useEffect(() => {
    if (!historyLoaded) return;

    try {
      window.localStorage.setItem(
        HISTORY_STORAGE_KEY,
        JSON.stringify(history.slice(0, MAX_HISTORY_ITEMS))
      );
    } catch (storageError) {
      console.warn(
        "Não foi possível salvar o histórico no navegador.",
        storageError
      );
    }
  }, [history, historyLoaded]);

  // O backend é a fonte principal do histórico.
  const loadHistory = useCallback(async () => {
    if (!API_URL) {
      setHistoryError(
        "A variável NEXT_PUBLIC_API_URL não está configurada."
      );
      return;
    }

    setHistoryLoading(true);
    setHistoryError("");

    try {
      const response = await fetchAnalysisHistory(
        MAX_HISTORY_ITEMS,
        0
      );

      setHistory(response.items);
      setHistoryTotal(response.total);
    } catch (err) {
      setHistoryError(
        err instanceof Error
          ? err.message
          : "Não foi possível carregar o histórico do servidor."
      );

      // Não apagamos o histórico local quando o backend falha.
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  // Carrega o histórico persistente ao abrir a aplicação.
  useEffect(() => {
    // A chamada inicia uma sincronização assíncrona com a API.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void loadHistory();
  }, [loadHistory]);

  // Consulta as métricas do provedor solicitado.
  // O parâmetro permite consultar o provedor confirmado pela API sem
  // depender de uma atualização de estado ainda não renderizada.
  const loadLLMMetrics = useCallback(
    async (provider: LLMProvider = selectedProvider) => {
      const requestId = ++llmMetricsRequestId.current;

      if (!API_URL) {
        setLlmMetrics(null);
        setLlmMetricsError(
          "A variável NEXT_PUBLIC_API_URL não está configurada."
        );
        setLlmMetricsLoading(false);
        return;
      }

      setLlmMetricsLoading(true);
      setLlmMetricsError("");
      setLlmMetrics(null);

      try {
        const response = await fetch(
          `${API_URL}/metrics/llm?provider=${encodeURIComponent(provider)}`,
          {
            cache: "no-store"
          }
        );

        const data = await readJsonResponse(response);

        if (!response.ok) {
          throw extractApiError(
            data,
            "Não foi possível carregar as métricas do provedor de IA."
          );
        }

        // Descarta respostas de consultas anteriores.
        if (requestId !== llmMetricsRequestId.current) return;

        if (!data || typeof data !== "object") {
          throw new Error(
            "A API retornou métricas em formato inesperado."
          );
        }

        const receivedMetrics = data as LLMMetrics;

        if (
          typeof receivedMetrics.provider !== "string" ||
          typeof receivedMetrics.model !== "string" ||
          !receivedMetrics.requests ||
          !receivedMetrics.tokens ||
          !receivedMetrics.latency
        ) {
          throw new Error(
            "A API retornou métricas incompletas ou inválidas."
          );
        }

        // Evita exibir métricas de um provedor diferente do solicitado.
        if (receivedMetrics.provider !== provider) {
          throw new Error(
            "O provedor retornado pela API não corresponde ao provedor selecionado."
          );
        }

        setLlmMetrics(receivedMetrics);
      } catch (err) {
        if (requestId !== llmMetricsRequestId.current) return;

        setLlmMetrics(null);
        setLlmMetricsError(
          err instanceof Error
            ? err.message
            : "Erro ao consultar as métricas de IA."
        );
      } finally {
        if (requestId === llmMetricsRequestId.current) {
          setLlmMetricsLoading(false);
        }
      }
    },
    [selectedProvider]
  );

  useEffect(() => {
    if (activePage === "overview" || activePage === "analyst") {
      // A chamada inicia uma sincronização assíncrona com a API.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      void loadLLMMetrics();
    }
  }, [activePage, loadLLMMetrics]);

  async function loadCatalog() {
    if (catalog || catalogLoading) return;

    if (!API_URL) {
      setCatalogError(
        "A variável NEXT_PUBLIC_API_URL não está configurada."
      );
      return;
    }

    setCatalogLoading(true);
    setCatalogError("");

    try {
      const response = await fetch(`${API_URL}/data/catalog`, {
        cache: "no-store"
      });

      const data = await readJsonResponse(response);

      if (!response.ok) {
        throw extractApiError(
          data,
          "Não foi possível carregar o catálogo de dados."
        );
      }

      setCatalog(data as DataCatalog);
    } catch (err) {
      setCatalogError(
        err instanceof Error
          ? err.message
          : "Erro ao conectar com a API."
      );
    } finally {
      setCatalogLoading(false);
    }
  }

  function navigateTo(page: Page) {
    setActivePage(page);

    if (page === "data") {
      void loadCatalog();
    }
  }

  function handleNewAnalysis() {
    setQuestion("");
    setSentQuestion("");
    setAnalysis(null);
    setError("");
    setErrorAnalysisId("");
    setActivePage("analyst");
  }

  // Abre uma análise persistida ou recupera uma entrada antiga local.
  async function handleSelectHistory(item: HistoryItem) {
    if (loading || historyDetailLoadingId) return;

    setActivePage("analyst");
    setQuestion("");
    setSentQuestion("");
    setAnalysis(null);
    setError("");
    setErrorAnalysisId("");

    if (item.analysis_id) {
      setHistoryDetailLoadingId(item.analysis_id);

      try {
        const detail = await fetchAnalysisDetail(item.analysis_id);

        setSentQuestion(detail.question);

        if (isLLMProvider(detail.provider)) {
          setSelectedProvider(detail.provider);
        }

        if (detail.status === "failure") {
          setAnalysis(null);
          setError(
            detail.error_message ||
              "Esta análise foi registrada como uma execução com falha."
          );
          setErrorAnalysisId(detail.analysis_id);
          return;
        }

        const restored = restoreAnalysis(detail);

        if (!isValidAnalysis(restored)) {
          throw new Error(
            "O histórico retornou um resultado em formato inesperado."
          );
        }

        if (isLLMProvider(restored.provider)) {
          setSelectedProvider(restored.provider);
        }

        setAnalysis(restored);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Não foi possível recuperar esta análise."
        );
        setErrorAnalysisId(item.analysis_id);
        setSentQuestion(item.question);
      } finally {
        setHistoryDetailLoadingId("");
      }

      return;
    }

    // Compatibilidade com registros antigos do localStorage.
    if (isValidAnalysis(item.analysis)) {
      setSentQuestion(item.question);

      if (isLLMProvider(item.analysis.provider)) {
        setSelectedProvider(item.analysis.provider);
      }

      setAnalysis(item.analysis);
      return;
    }

    setQuestion(item.question);
    setError(
      "Esta entrada antiga não possui o resultado completo salvo. A pergunta foi recuperada; envie-a novamente para executar a análise."
    );
  }

  // Atalho Ctrl + K ou Cmd + K para iniciar uma nova análise.
  useEffect(() => {
    function handleKeyboardShortcut(event: KeyboardEvent) {
      const isNewAnalysisShortcut =
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "k";

      if (!isNewAnalysisShortcut) return;

      event.preventDefault();
      handleNewAnalysis();
    }

    window.addEventListener("keydown", handleKeyboardShortcut);

    return () => {
      window.removeEventListener("keydown", handleKeyboardShortcut);
    };
  }, []);

  async function handleSubmit(
    event: React.FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    const currentQuestion = question.trim();

    if (!currentQuestion || loading) return;

    if (!API_URL) {
      setError(
        "A variável NEXT_PUBLIC_API_URL não está configurada."
      );
      setErrorAnalysisId("");
      return;
    }

    const requestedProvider = selectedProvider;

    setSentQuestion(currentQuestion);
    setQuestion("");
    setAnalysis(null);
    setError("");
    setErrorAnalysisId("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question: currentQuestion,
          provider: requestedProvider
        })
      });

      const data = await readJsonResponse(response);

      if (!response.ok) {
        throw extractApiError(
          data,
          "Não foi possível realizar a análise."
        );
      }

      const result = data as Analysis;

      if (!isValidAnalysis(result)) {
        throw new Error(
          "A API retornou uma resposta em formato inesperado."
        );
      }

      setAnalysis(result);

      const confirmedProvider = isLLMProvider(result.provider)
        ? result.provider
        : requestedProvider;

      // Mantém o seletor sincronizado com o provedor confirmado pela API.
      setSelectedProvider(confirmedProvider);

      // Atualização otimista: a resposta aparece imediatamente.
      setHistory((previous) =>
        [
          {
            analysis_id: result.analysis_id,
            question: currentQuestion,
            total_time_ms: result.metrics.total_time_ms,
            cache_hit: result.cache_hit,
            timestamp: new Date().toLocaleString("pt-BR", {
              day: "2-digit",
              month: "2-digit",
              year: "numeric",
              hour: "2-digit",
              minute: "2-digit"
            }),
            status: "success" as const,
            provider: confirmedProvider,
            model: result.model,
            analysis: result
          },
          ...previous.filter(
            (item) =>
              !result.analysis_id ||
              item.analysis_id !== result.analysis_id
          )
        ].slice(0, MAX_HISTORY_ITEMS)
      );

      // Sincroniza com o registro realmente persistido no backend.
      void loadHistory();
      void loadLLMMetrics(confirmedProvider);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);

        const apiError = err as ApiRequestError;
        setErrorAnalysisId(apiError.analysisId ?? "");
      } else {
        setError("Erro ao conectar com a API.");
        setErrorAnalysisId("");
      }

      // O backend também persiste as execuções com falha.
      void loadHistory();
      void loadLLMMetrics(requestedProvider);
    } finally {
      setLoading(false);
    }
  }

  // Total de execuções registradas, incluindo as que falharam.
  const totalAnalyses =
    historyTotal > 0 ? historyTotal : history.length;

  // Compatibilidade com registros antigos: status ausente não é
  // classificado como falha, pois esses registros podem não ter esse campo.
  const successfulHistory = history.filter(
    (item) => item.status !== "failure"
  );

  const cacheHits = successfulHistory.filter(
    (item) => item.cache_hit
  ).length;

  // Taxa calculada somente sobre registros que não estão marcados como falha.
  const cacheRate =
    successfulHistory.length > 0
      ? Math.round(
          (cacheHits / successfulHistory.length) * 100
        )
      : 0;

  // Falhas não entram no cálculo de desempenho médio.
  const averageTime =
    successfulHistory.length > 0
      ? successfulHistory.reduce(
          (sum, item) => sum + item.total_time_ms,
          0
        ) / successfulHistory.length
      : 0;

  // O histórico retornado pelo backend é apresentado do mais recente
  // para o mais antigo; usamos o primeiro registro concluído.
  const latestSuccessfulAnalysis = successfulHistory[0] ?? null;

  const totalRows =
    catalog?.tables.reduce(
      (sum, table) => sum + table.total_rows,
      0
    ) ?? 0;

  function renderHistoryStatus(item: HistoryItem) {
    if (item.status === "failure") {
      return (
        <span className="rounded-full bg-red-500/10 px-2 py-1 text-[9px] text-red-400">
          Falha
        </span>
      );
    }

    if (item.partial_success) {
      return (
        <span className="rounded-full bg-amber-500/10 px-2 py-1 text-[9px] text-amber-300">
          Concluída parcialmente
        </span>
      );
    }

    return (
      <span
        className={`rounded-full px-2 py-1 text-[9px] ${
          item.cache_hit
            ? "bg-blue-500/10 text-blue-400"
            : "bg-emerald-500/10 text-emerald-400"
        }`}
      >
        {item.cache_hit ? "Cache" : "Concluída"}
      </span>
    );
  }

  return (
    <main className="flex min-h-screen bg-[#090b10] text-zinc-100">
      {/* Sidebar */}
      <aside className="hidden w-[260px] shrink-0 flex-col border-r border-white/[0.07] bg-[#0d0f15] md:flex">
        <div className="flex h-[76px] items-center gap-3 border-b border-white/[0.07] px-6">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 text-lg font-bold shadow-lg shadow-blue-500/20">
            A
          </div>

          <div>
            <h1 className="text-sm font-semibold tracking-wide">
              AI Business Analyst
            </h1>
            <p className="text-[10px] tracking-wider text-zinc-500">
              INTELLIGENCE PLATFORM
            </p>
          </div>
        </div>

        <div className="p-4">
          <button
            onClick={handleNewAnalysis}
            className="flex w-full items-center gap-3 rounded-xl border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-zinc-200 transition hover:bg-white/[0.08]"
          >
            <span className="text-lg">＋</span>
            Nova análise
            <span className="ml-auto text-xs text-zinc-600">
              Ctrl K
            </span>
          </button>
        </div>

        <div className="px-5 pt-5">
          <p className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-600">
            Workspace
          </p>

          <button
            onClick={() => navigateTo("overview")}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition ${
              activePage === "overview"
                ? "bg-blue-500/10 text-blue-400"
                : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200"
            }`}
          >
            <span>▦</span>
            Visão geral
          </button>

          <button
            onClick={() => navigateTo("analyst")}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition ${
              activePage === "analyst"
                ? "bg-blue-500/10 text-blue-400"
                : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200"
            }`}
          >
            <span>✳</span>
            Analista de IA
          </button>

          <button
            onClick={() => navigateTo("data")}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition ${
              activePage === "data"
                ? "bg-blue-500/10 text-blue-400"
                : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200"
            }`}
          >
            <span>◫</span>
            Dados e consultas
          </button>

          <button
            onClick={() => navigateTo("history")}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition ${
              activePage === "history"
                ? "bg-blue-500/10 text-blue-400"
                : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200"
            }`}
          >
            <span>◷</span>
            Histórico
          </button>
        </div>

        <div className="mt-8 px-5">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-600">
              Recentes
            </p>
            <span className="text-xs text-zinc-600">
              {totalAnalyses}
            </span>
          </div>

          {historyLoading && history.length === 0 ? (
            <p className="rounded-lg px-3 py-2 text-xs text-zinc-500">
              Carregando histórico...
            </p>
          ) : history.length > 0 ? (
            <div className="space-y-1">
              {history.slice(0, 5).map((item, index) => (
                <button
                  key={item.analysis_id ?? `${item.timestamp}-${index}`}
                  onClick={() => void handleSelectHistory(item)}
                  disabled={Boolean(historyDetailLoadingId)}
                  className="w-full truncate rounded-lg px-3 py-2 text-left text-xs text-zinc-500 transition hover:bg-white/[0.04] hover:text-zinc-300 disabled:opacity-50"
                  title={item.question}
                >
                  {historyDetailLoadingId === item.analysis_id
                    ? "Carregando análise..."
                    : item.question}
                </button>
              ))}
            </div>
          ) : (
            <p className="rounded-lg px-3 py-2 text-xs text-zinc-500">
              Suas análises aparecerão aqui.
            </p>
          )}
        </div>

        <div className="mt-auto border-t border-white/[0.07] p-4">
          <div className="flex items-center gap-3 rounded-xl p-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-blue-500 text-xs font-bold">
              AI
            </div>

            <div>
              <p className="text-xs font-medium text-zinc-200">
                AI Workspace
              </p>
              <p className="text-[10px] text-zinc-500">
                Ambiente de análise
              </p>
            </div>

            <span className="ml-auto text-zinc-600">⚙</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <section className="flex min-h-screen min-w-0 flex-1 flex-col">
        <header className="flex h-[76px] shrink-0 items-center justify-between border-b border-white/[0.07] px-5 md:px-10">
          <div>
            <p className="text-xs text-zinc-500">
              Workspace /{" "}
              {activePage === "overview"
                ? "Visão geral"
                : activePage === "data"
                  ? "Dados e consultas"
                  : activePage === "history"
                    ? "Histórico"
                    : "Analista de IA"}
            </p>

            <h2 className="mt-1 text-sm font-medium text-zinc-200">
              {activePage === "overview"
                ? "Visão geral"
                : activePage === "data"
                  ? "Catálogo de dados"
                  : activePage === "history"
                    ? "Histórico de análises"
                    : "Nova análise"}
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/[0.06] px-3 py-1.5 sm:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              <span className="text-[11px] text-emerald-400">
                Ambiente local
              </span>
            </div>

            <button
              onClick={() =>
                window.open(
                  "http://localhost:8000/docs",
                  "_blank",
                  "noopener,noreferrer"
                )
              }
              className="rounded-lg border border-white/10 px-3 py-2 text-xs text-zinc-400 hover:bg-white/5"
            >
              Documentação ↗
            </button>
          </div>
        </header>

        <div className="flex flex-1 flex-col items-center px-5 pb-8 pt-12 md:pt-16">
          <div className="w-full max-w-[1000px]">
            {/* OVERVIEW */}
            {activePage === "overview" && (
              <div>
                <div className="mb-8">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-blue-400">
                    Workspace
                  </p>

                  <h1 className="mt-2 text-3xl font-semibold text-white">
                    Visão geral
                  </h1>

                  <p className="mt-3 max-w-xl text-sm leading-7 text-zinc-500">
                    Acompanhe as atividades do agente, o desempenho
                    das análises e a estrutura dos dados disponíveis.
                  </p>
                </div>

                <div className="mb-8 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
                    <p className="text-xs text-zinc-500">
                      Execuções registradas
                    </p>
                    <p className="mt-3 text-3xl font-semibold text-white">
                      {totalAnalyses}
                    </p>
                    <p className="mt-2 text-[10px] text-zinc-600">
                      Inclui execuções com falha
                    </p>
                  </div>

                  <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
                    <p className="text-xs text-zinc-500">
                      Tempo médio
                    </p>
                    <p className="mt-3 text-3xl font-semibold text-white">
                      {successfulHistory.length > 0
                        ? formatTime(averageTime)
                        : "—"}
                    </p>
                    <p className="mt-2 text-[10px] text-zinc-600">
                      Média das execuções concluídas · registros carregados
                    </p>
                  </div>

                  <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
                    <p className="text-xs text-zinc-500">Taxa de cache</p>
                    <p className="mt-3 text-3xl font-semibold text-blue-400">
                      {successfulHistory.length > 0
                        ? `${cacheRate}%`
                        : "—"}
                    </p>
                    <p className="mt-2 text-[10px] text-zinc-600">
                      {cacheHits} reutilização(ões) · registros carregados
                    </p>
                  </div>

                  <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
                    <p className="text-xs text-zinc-500">
                      Última análise concluída
                    </p>
                    <p className="mt-3 text-3xl font-semibold text-white">
                      {latestSuccessfulAnalysis
                        ? formatTime(latestSuccessfulAnalysis.total_time_ms)
                        : "—"}
                    </p>
                    <p className="mt-2 text-[10px] text-zinc-600">
                      Tempo total da última execução bem-sucedida
                    </p>
                  </div>
                </div>

                {historyError && (
                  <div className="mb-6 rounded-lg border border-amber-500/20 bg-amber-500/[0.05] p-3">
                    <p className="text-xs text-amber-300">
                      Não foi possível sincronizar o histórico com o
                      backend. Exibindo os registros disponíveis no
                      navegador.
                    </p>
                    <p className="mt-1 text-[10px] text-zinc-500">
                      {historyError}
                    </p>
                  </div>
                )}

                <LLMMetricsPanel
                  metrics={llmMetrics}
                  loading={llmMetricsLoading}
                  error={llmMetricsError}
                  onRefresh={() => void loadLLMMetrics()}
                />

                <div className="mb-6 rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-semibold text-zinc-200">
                        Base de dados
                      </p>
                      <p className="mt-1 text-xs text-zinc-500">
                        Synthetic Delivery Dataset · 2025
                      </p>
                    </div>

                    <button
                      onClick={() => navigateTo("data")}
                      className="rounded-lg border border-blue-500/20 bg-blue-500/10 px-3 py-2 text-xs text-blue-400 transition hover:bg-blue-500/20"
                    >
                      Explorar dados ↗
                    </button>
                  </div>

                  <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
                    {[
                      { label: "Restaurantes", value: "200" },
                      { label: "Clientes", value: "5.000" },
                      { label: "Entregadores", value: "300" },
                      { label: "Pedidos", value: "60.000" }
                    ].map((item) => (
                      <div
                        key={item.label}
                        className="rounded-lg bg-white/[0.03] p-3"
                      >
                        <p className="text-[10px] text-zinc-500">
                          {item.label}
                        </p>
                        <p className="mt-2 text-xl font-semibold text-zinc-200">
                          {item.value}
                        </p>
                      </div>
                    ))}
                  </div>

                  <p className="mt-4 text-[10px] text-zinc-600">
                    Dados sintéticos utilizados para desenvolvimento
                    e demonstração do produto.
                  </p>
                </div>

                <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
                  <div className="mb-4 flex items-center justify-between">
                    <p className="text-xs font-semibold text-zinc-200">
                      Atividade recente
                    </p>

                    <button
                      onClick={() => navigateTo("analyst")}
                      className="text-[10px] text-blue-400 hover:text-blue-300"
                    >
                      Nova análise ↗
                    </button>
                  </div>

                  {history.length > 0 ? (
                    <div className="space-y-3">
                      {history.slice(0, 5).map((item, index) => (
                        <button
                          key={item.analysis_id ?? `${item.timestamp}-${index}`}
                          onClick={() => void handleSelectHistory(item)}
                          className="flex w-full items-center justify-between gap-3 border-b border-white/[0.05] pb-3 text-left last:border-0 last:pb-0"
                        >
                          <span className="min-w-0 truncate text-xs text-zinc-300">
                            {item.question}
                          </span>
                          <span className="shrink-0 text-[10px] text-zinc-600">
                            {item.timestamp}
                          </span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-zinc-500">
                      Nenhuma atividade registrada ainda.
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* DATA CATALOG */}
            {activePage === "data" && (
              <div>
                <div className="mb-8">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-blue-400">
                    Fonte de dados
                  </p>

                  <h1 className="mt-2 text-3xl font-semibold text-white">
                    Dados e consultas
                  </h1>

                  <p className="mt-3 max-w-xl text-sm leading-7 text-zinc-500">
                    Explore as tabelas, os campos e os registros
                    disponíveis para o agente de análise.
                  </p>
                </div>

                {catalogLoading && (
                  <div className="flex items-center gap-3 rounded-xl border border-white/[0.07] bg-[#11141c] p-6">
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-blue-400/30 border-t-blue-400" />
                    <p className="text-sm text-zinc-400">
                      Consultando o catálogo do DuckDB...
                    </p>
                  </div>
                )}

                {catalogError && (
                  <div className="rounded-xl border border-red-500/20 bg-red-500/[0.06] p-5">
                    <p className="text-sm font-semibold text-red-400">
                      Não foi possível carregar os dados
                    </p>

                    <p className="mt-2 text-xs text-zinc-400">
                      {catalogError}
                    </p>

                    <button
                      onClick={() => {
                        setCatalog(null);
                        void loadCatalog();
                      }}
                      className="mt-4 rounded-lg border border-red-500/20 px-3 py-2 text-xs text-red-300 hover:bg-red-500/10"
                    >
                      Tentar novamente
                    </button>
                  </div>
                )}

                {catalog && (
                  <>
                    <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3">
                      <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-4">
                        <p className="text-[10px] text-zinc-500">Banco</p>
                        <p className="mt-2 text-lg font-semibold text-white">
                          {catalog.database}
                        </p>
                      </div>

                      <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-4">
                        <p className="text-[10px] text-zinc-500">
                          Tabelas
                        </p>
                        <p className="mt-2 text-lg font-semibold text-white">
                          {catalog.tables_count}
                        </p>
                      </div>

                      <div className="col-span-2 rounded-xl border border-white/[0.07] bg-[#11141c] p-4 sm:col-span-1">
                        <p className="text-[10px] text-zinc-500">
                          Registros totais
                        </p>
                        <p className="mt-2 text-lg font-semibold text-white">
                          {formatNumber(totalRows)}
                        </p>
                      </div>
                    </div>

                    <div className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-blue-500/15 bg-blue-500/[0.05] p-4">
                      <div>
                        <p className="text-xs font-medium text-blue-300">
                          {catalog.dataset}
                        </p>
                        <p className="mt-1 text-[10px] text-zinc-500">
                          Período de referência: {catalog.period}
                        </p>
                      </div>

                      <span className="rounded-full border border-blue-500/20 px-3 py-1 text-[10px] text-blue-300">
                        Dados sintéticos
                      </span>
                    </div>

                    <div className="space-y-5">
                      {catalog.tables.map((table) => (
                        <details
                          key={table.name}
                          open={table.name === "orders"}
                          className="group overflow-hidden rounded-xl border border-white/[0.07] bg-[#11141c]"
                        >
                          <summary className="flex cursor-pointer list-none items-center justify-between gap-4 p-5 hover:bg-white/[0.02]">
                            <div className="min-w-0">
                              <div className="flex flex-wrap items-center gap-3">
                                <h2 className="text-sm font-semibold text-white">
                                  {table.name}
                                </h2>

                                <span className="rounded-full bg-blue-500/10 px-2 py-1 text-[10px] text-blue-400">
                                  {formatNumber(table.total_rows)} registros
                                </span>
                              </div>

                              <p className="mt-2 text-xs text-zinc-500">
                                {table.description}
                              </p>
                            </div>

                            <span className="shrink-0 text-zinc-500 transition group-open:rotate-180">
                              ⌄
                            </span>
                          </summary>

                          <div className="border-t border-white/[0.06] p-5">
                            <div className="mb-5">
                              <p className="mb-3 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                                Estrutura da tabela · {table.columns.length} campos
                              </p>

                              <div className="overflow-x-auto">
                                <table className="w-full text-left text-xs">
                                  <thead>
                                    <tr className="border-b border-white/[0.08] text-zinc-500">
                                      <th className="px-3 py-3 font-medium">
                                        Coluna
                                      </th>
                                      <th className="px-3 py-3 font-medium">
                                        Tipo
                                      </th>
                                      <th className="px-3 py-3 font-medium">
                                        Nulos
                                      </th>
                                      <th className="px-3 py-3 font-medium">
                                        Descrição
                                      </th>
                                    </tr>
                                  </thead>

                                  <tbody>
                                    {table.columns.map((column) => (
                                      <tr
                                        key={column.name}
                                        className="border-b border-white/[0.04] text-zinc-300 last:border-0"
                                      >
                                        <td className="px-3 py-3 font-mono text-blue-300">
                                          {column.name}
                                        </td>
                                        <td className="px-3 py-3 text-violet-300">
                                          {column.type}
                                        </td>
                                        <td className="px-3 py-3 text-zinc-500">
                                          {column.nullable ? "Sim" : "Não"}
                                        </td>
                                        <td className="min-w-[220px] px-3 py-3 text-zinc-500">
                                          {column.description}
                                        </td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </div>

                            <div>
                              <p className="mb-3 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                                Amostra de registros
                              </p>

                              {table.sample.length > 0 ? (
                                <div className="overflow-x-auto rounded-lg border border-white/[0.05]">
                                  <table className="w-full text-left text-xs">
                                    <thead>
                                      <tr className="bg-white/[0.03] text-zinc-500">
                                        {table.columns.map((column) => (
                                          <th
                                            key={column.name}
                                            className="whitespace-nowrap px-3 py-3 font-medium"
                                          >
                                            {column.name}
                                          </th>
                                        ))}
                                      </tr>
                                    </thead>

                                    <tbody>
                                      {table.sample.map((row, index) => (
                                        <tr
                                          key={index}
                                          className="border-t border-white/[0.04] text-zinc-300"
                                        >
                                          {table.columns.map((column) => (
                                            <td
                                              key={column.name}
                                              className="whitespace-nowrap px-3 py-3"
                                            >
                                              {formatValue(row[column.name])}
                                            </td>
                                          ))}
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              ) : (
                                <p className="text-xs text-zinc-500">
                                  Nenhum registro de amostra disponível.
                                </p>
                              )}
                            </div>
                          </div>
                        </details>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            {/* AI ANALYST */}
            {activePage === "analyst" && (
              <>
                <div className="mb-10">
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-blue-400">
                        Observabilidade
                      </p>
                      <h2 className="mt-1 text-lg font-semibold text-white">
                        Desempenho do agente
                      </h2>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1 text-[10px] text-zinc-500">
                        Histórico persistente
                      </span>

                      <button
                        onClick={() => void loadHistory()}
                        disabled={historyLoading}
                        className="rounded-lg border border-white/10 px-2.5 py-1.5 text-[10px] text-zinc-400 transition hover:bg-white/5 disabled:opacity-40"
                      >
                        {historyLoading ? "..." : "↻"}
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                    <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-4">
                      <p className="text-[10px] text-zinc-500">
                        Execuções registradas
                      </p>
                      <p className="mt-3 text-2xl font-semibold text-white">
                        {totalAnalyses}
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-600">
                        Inclui execuções com falha
                      </p>
                    </div>

                    <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-4">
                      <p className="text-[10px] text-zinc-500">
                        Tempo médio
                      </p>
                      <p className="mt-3 text-2xl font-semibold text-white">
                        {successfulHistory.length > 0
                          ? formatTime(averageTime)
                          : "—"}
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-600">
                        Média das execuções concluídas · registros carregados
                      </p>
                    </div>

                    <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-4">
                      <p className="text-[10px] text-zinc-500">
                        Taxa de cache
                      </p>
                      <p className="mt-3 text-2xl font-semibold text-blue-400">
                        {successfulHistory.length > 0
                          ? `${cacheRate}%`
                          : "—"}
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-600">
                        {cacheHits} reutilização(ões) · registros carregados
                      </p>
                    </div>

                    <div className="rounded-xl border border-white/[0.07] bg-[#11141c] p-4">
                      <p className="text-[10px] text-zinc-500">
                        Última análise concluída
                      </p>
                      <p className="mt-3 text-2xl font-semibold text-white">
                        {latestSuccessfulAnalysis
                          ? formatTime(latestSuccessfulAnalysis.total_time_ms)
                          : "—"}
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-600">
                        Tempo total da última execução bem-sucedida
                      </p>
                    </div>
                  </div>

                  {historyError && (
                    <div className="mt-4 rounded-lg border border-amber-500/20 bg-amber-500/[0.05] p-3">
                      <p className="text-xs text-amber-300">
                        Falha ao sincronizar o histórico com o backend.
                      </p>
                      <p className="mt-1 text-[10px] text-zinc-500">
                        {historyError}
                      </p>
                    </div>
                  )}

                  <div className="mt-5">
                    <LLMMetricsPanel
                      metrics={llmMetrics}
                      loading={llmMetricsLoading}
                      error={llmMetricsError}
                      onRefresh={() => void loadLLMMetrics()}
                    />
                  </div>

                  {analysis && (
                    <div className="mt-3 rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
                      <div className="mb-4 flex items-center justify-between">
                        <p className="text-xs font-semibold text-zinc-300">
                          Detalhamento da última análise
                        </p>
                        <span className="text-[10px] text-zinc-600">
                          Tempo em milissegundos
                        </span>
                      </div>

                      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                        <div>
                          <p className="text-[10px] text-zinc-500">
                            Geração SQL ·{" "}
                            {getProviderLabel(analysis.provider ?? selectedProvider)}
                          </p>
                          <p className="mt-1 text-lg font-semibold text-blue-400">
                            {formatTime(
                              analysis.metrics.sql_generation_ms
                            )}
                          </p>
                        </div>

                        <div>
                          <p className="text-[10px] text-zinc-500">
                            Execução · DuckDB
                          </p>
                          <p className="mt-1 text-lg font-semibold text-violet-400">
                            {formatTime(
                              analysis.metrics.sql_execution_ms
                            )}
                          </p>
                        </div>

                        <div>
                          <p className="text-[10px] text-zinc-500">
                            Interpretação ·{" "}
                            {getProviderLabel(analysis.provider ?? selectedProvider)}
                          </p>
                          <p className="mt-1 text-lg font-semibold text-emerald-400">
                            {formatTime(
                              analysis.metrics.response_generation_ms
                            )}
                          </p>
                        </div>
                      </div>

                      {analysis.cache_hit && (
                        <p className="mt-4 border-t border-white/[0.06] pt-3 text-[10px] text-blue-400">
                          ⚡ Resultado recuperado do cache. As etapas de
                          geração SQL, execução e interpretação não foram
                          executadas novamente.
                        </p>
                      )}
                    </div>
                  )}

                  {analysis?.steps && analysis.steps.length > 0 && (
                    <div className="mt-3">
                      <AgentTimeline steps={analysis.steps} />
                    </div>
                  )}

                  {history.length > 0 && (
                    <div className="mt-3 rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
                      <div className="mb-4 flex items-center justify-between">
                        <p className="text-xs font-semibold text-zinc-300">
                          Histórico de desempenho
                        </p>
                        <span className="text-[10px] text-zinc-600">
                          {history.length} de {totalAnalyses} registro(s)
                        </span>
                      </div>

                      <div className="space-y-3">
                        {history.slice(0, 5).map((item, index) => (
                          <button
                            key={item.analysis_id ?? `${item.timestamp}-${index}`}
                            onClick={() => void handleSelectHistory(item)}
                            disabled={Boolean(historyDetailLoadingId)}
                            className="flex w-full items-center justify-between gap-3 border-b border-white/[0.04] pb-3 text-left last:border-0 last:pb-0 disabled:opacity-50"
                          >
                            <div className="min-w-0">
                              <p className="truncate text-xs text-zinc-300">
                                {item.question}
                              </p>
                              <p className="mt-1 text-[10px] text-zinc-600">
                                {item.timestamp}
                                {item.provider
                                  ? ` · ${getProviderLabel(item.provider)}`
                                  : ""}
                              </p>
                            </div>

                            <div className="flex shrink-0 items-center gap-3">
                              <span className="text-xs font-medium text-zinc-300">
                                {formatTime(item.total_time_ms)}
                              </span>

                              {renderHistoryStatus(item)}
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="mb-8 flex justify-center">
                  <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-blue-400/20 bg-gradient-to-br from-blue-500/20 to-violet-500/20 text-2xl text-blue-300 shadow-xl shadow-blue-500/10">
                    ✳
                  </div>
                </div>

                <div className="mb-3 text-center">
                  <p className="mb-3 text-[11px] font-semibold uppercase tracking-[0.25em] text-blue-400">
                    Business intelligence powered by AI
                  </p>

                  <h1 className="text-3xl font-semibold tracking-tight text-white md:text-[40px]">
                    Transforme dados em
                    <span className="block bg-gradient-to-r from-blue-400 to-violet-400 bg-clip-text text-transparent">
                      decisões inteligentes.
                    </span>
                  </h1>

                  <p className="mx-auto mt-5 max-w-lg text-sm leading-7 text-zinc-500">
                    Faça perguntas sobre seu negócio em linguagem natural.
                    Nossa inteligência artificial transforma perguntas
                    em análises de dados.
                  </p>
                </div>

                <form
                  onSubmit={handleSubmit}
                  className="mt-10 rounded-2xl border border-white/10 bg-[#11141c] p-3 shadow-2xl shadow-black/30 transition focus-within:border-blue-500/40"
                >
                  <div className="mb-3 flex flex-col gap-2 border-b border-white/[0.06] px-3 pb-3 sm:flex-row sm:items-center sm:justify-between">
                    <label
                      htmlFor="provider"
                      className="text-xs font-medium text-zinc-400"
                    >
                      Modelo de inteligência artificial
                    </label>

                    <select
                      id="provider"
                      value={selectedProvider}
                      onChange={(event) =>
                        setSelectedProvider(event.target.value as LLMProvider)
                      }
                      disabled={loading}
                      className="w-full rounded-lg border border-white/10 bg-[#0d0f15] px-3 py-2 text-xs text-zinc-200 outline-none transition focus:border-blue-500/50 disabled:cursor-not-allowed disabled:opacity-50 sm:w-auto sm:min-w-[220px]"
                    >
                      {PROVIDERS.map((provider) => (
                        <option
                          key={provider.value}
                          value={provider.value}
                          className="bg-[#11141c]"
                        >
                          {provider.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="px-3 pb-2">
                    <p className="text-[10px] text-zinc-600">
                      Modelo: {getProviderModel(selectedProvider)}
                    </p>
                  </div>

                  <textarea
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    onKeyDown={(event) => {
                      if (
                        event.key === "Enter" &&
                        !event.shiftKey
                      ) {
                        event.preventDefault();
                        event.currentTarget.form?.requestSubmit();
                      }
                    }}
                    placeholder="Pergunte qualquer coisa sobre seu negócio..."
                    rows={3}
                    className="w-full resize-none bg-transparent px-3 py-2 text-sm leading-6 text-zinc-100 outline-none placeholder:text-zinc-600"
                  />

                  <div className="flex items-center justify-between border-t border-white/[0.06] pt-3">
                    <span className="px-2 text-[11px] text-zinc-600">
                      ↵ Enviar · Shift + ↵ Nova linha
                    </span>

                    <button
                      type="submit"
                      disabled={!question.trim() || loading}
                      className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                      Analisar
                      <span>↑</span>
                    </button>
                  </div>
                </form>

                {historyDetailLoadingId && (
                  <div className="mt-5 flex items-center gap-3 rounded-xl border border-white/[0.07] bg-white/[0.02] p-4">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-400/30 border-t-blue-400" />
                    <p className="text-xs text-zinc-400">
                      Recuperando análise salva...
                    </p>
                  </div>
                )}

                {sentQuestion && (
                  <div className="mt-5 space-y-4">
                    <div className="rounded-xl border border-blue-500/20 bg-blue-500/[0.06] p-4">
                      <p className="text-[10px] font-semibold uppercase tracking-wider text-blue-400">
                        Sua pergunta
                      </p>
                      <p className="mt-2 text-sm text-zinc-200">
                        {sentQuestion}
                      </p>
                    </div>

                    {loading && (
                      <div className="flex items-center gap-3 rounded-xl border border-white/[0.07] bg-white/[0.02] p-5">
                        <div className="h-5 w-5 animate-spin rounded-full border-2 border-blue-400/30 border-t-blue-400" />
                        <div>
                          <p className="text-sm font-medium text-zinc-200">
                            Analisando seus dados...
                          </p>
                          <p className="mt-1 text-xs text-zinc-500">
                            O agente está gerando e executando uma consulta SQL.
                          </p>
                        </div>
                      </div>
                    )}

                    {error && (
                      <div className="rounded-xl border border-red-500/20 bg-red-500/[0.06] p-4">
                        <p className="text-xs font-semibold text-red-400">
                          Não foi possível concluir a análise
                        </p>

                        <p className="mt-2 text-sm text-zinc-300">
                          {error}
                        </p>

                        {errorAnalysisId && (
                          <div className="mt-4 rounded-lg border border-red-500/10 bg-black/20 p-3">
                            <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                              Identificador da análise
                            </p>

                            <p className="mt-2 break-all font-mono text-[11px] text-red-300">
                              {errorAnalysisId}
                            </p>

                            <button
                              type="button"
                              onClick={() => {
                                void navigator.clipboard.writeText(
                                  errorAnalysisId
                                );
                              }}
                              className="mt-3 rounded-md border border-white/10 px-3 py-1.5 text-[10px] text-zinc-400 transition hover:bg-white/5 hover:text-zinc-200"
                            >
                              Copiar identificador
                            </button>
                          </div>
                        )}
                      </div>
                    )}

                    {analysis && (
                      <>
                        <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.04] p-5">
                          <div className="flex flex-wrap items-center justify-between gap-3">
                            <p className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400">
                              Análise concluída
                            </p>

                            <div className="flex items-center gap-2">
                              <span
                                className={`rounded-full px-2.5 py-1 text-[10px] font-medium ${
                                  analysis.cache_hit
                                    ? "border border-blue-500/20 bg-blue-500/10 text-blue-400"
                                    : "border border-emerald-500/20 bg-emerald-500/10 text-emerald-400"
                                }`}
                              >
                                {analysis.cache_hit
                                  ? "⚡ Cache"
                                  : "✳ Nova análise"}
                              </span>

                              <span className="rounded-full bg-emerald-500/10 px-2 py-1 text-[10px] text-emerald-400">
                                {analysis.total_results} resultado(s)
                              </span>
                            </div>
                          </div>

                          <div className="mt-3 flex flex-wrap items-center gap-2">
                            <span className="rounded-full border border-violet-500/20 bg-violet-500/10 px-2.5 py-1 text-[10px] text-violet-300">
                              {getProviderLabel(analysis.provider ?? selectedProvider)}
                            </span>

                            {analysis.model && (
                              <span className="rounded-full border border-white/[0.08] bg-white/[0.03] px-2.5 py-1 text-[10px] text-zinc-500">
                                {analysis.model}
                              </span>
                            )}
                          </div>

                          <p className="mt-4 text-sm leading-7 text-zinc-200">
                            {analysis.answer}
                          </p>

                          <p className="mt-3 text-[10px] text-zinc-500">
                            {analysis.cache_hit
                              ? "Resultado recuperado do cache. Nenhuma nova chamada ao provedor de IA foi necessária."
                              : "Resultado processado pelo agente de IA."}
                          </p>
                        </div>

                        {/* MULTI-QUERY RESULTS */}
                        {analysis.queries && analysis.queries.length > 0 ? (
                          <div className="space-y-5">
                            <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-4">
                              <p className="text-xs font-semibold text-zinc-300">
                                Resumo das consultas
                              </p>

                              <div className="mt-3 flex flex-wrap gap-2">
                                <span className="rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1.5 text-[10px] text-zinc-300">
                                  {analysis.query_count ?? analysis.queries.length} consulta(s)
                                </span>

                                <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-[10px] text-emerald-400">
                                  {analysis.successful_queries ?? 0} concluída(s)
                                </span>

                                {(analysis.unsupported_queries ?? 0) > 0 && (
                                  <span className="rounded-full border border-amber-500/20 bg-amber-500/10 px-3 py-1.5 text-[10px] text-amber-300">
                                    {analysis.unsupported_queries} não suportada(s)
                                  </span>
                                )}

                                {(analysis.failed_queries ?? 0) > 0 && (
                                  <span className="rounded-full border border-red-500/20 bg-red-500/10 px-3 py-1.5 text-[10px] text-red-400">
                                    {analysis.failed_queries} falha(s)
                                  </span>
                                )}
                              </div>

                              {analysis.partial_success && (
                                <p className="mt-3 text-xs leading-6 text-amber-400">
                                  {(analysis.unsupported_queries ?? 0) > 0
                                    ? (analysis.failed_queries ?? 0) > 0
                                      ? `A análise foi concluída parcialmente: ${analysis.unsupported_queries} consulta(s) não suportada(s) pelos dados e ${analysis.failed_queries} falha(s) operacional(is). Os resultados válidos foram preservados.`
                                      : analysis.unsupported_queries === 1
                                        ? "A análise foi concluída parcialmente. Uma consulta não pôde ser realizada devido a uma limitação dos dados disponíveis. Os demais resultados foram preservados."
                                        : `A análise foi concluída parcialmente. ${analysis.unsupported_queries} consultas não puderam ser realizadas devido a limita??es dos dados disponíveis. Os demais resultados foram preservados.`
                                    : "A análise foi concluída parcialmente. Uma ou mais consultas apresentaram falhas operacionais, mas os resultados das demais foram preservados."}
                                </p>
                              )}
                            </div>

                            {analysis.queries.map((query, index) => (
                              <div
                                key={query.id}
                                className="space-y-4 rounded-xl border border-white/[0.07] bg-white/[0.02] p-5"
                              >
                                <div className="flex flex-wrap items-center justify-between gap-3">
                                  <div>
                                    <p className="text-[10px] font-semibold uppercase tracking-wider text-violet-400">
                                      Consulta {index + 1}
                                    </p>

                                    <h3 className="mt-2 text-sm font-semibold text-zinc-200">
                                      {query.objective}
                                    </h3>
                                  </div>

                                  <span
                                    className={`rounded-full px-2.5 py-1 text-[10px] font-medium ${
                                      query.failure_type === "unsupported_capability"
                                        ? "border border-amber-500/20 bg-amber-500/10 text-amber-300"
                                        : query.status === "success"
                                          ? "border border-emerald-500/20 bg-emerald-500/10 text-emerald-400"
                                          : "border border-red-500/20 bg-red-500/10 text-red-400"
                                    }`}
                                  >
                                    {query.failure_type === "unsupported_capability"
                                      ? "Não suportada"
                                      : query.status === "success"
                                        ? "Concluída"
                                        : "Falhou"}
                                  </span>
                                </div>

                                <p className="text-xs leading-6 text-zinc-400">
                                  {query.question}
                                </p>

                                {query.status === "failure" &&
                                  query.failure_type === "unsupported_capability" && (
                                    <div className="rounded-lg border border-amber-500/20 bg-amber-500/[0.05] p-4">
                                      <p className="text-xs font-semibold text-amber-300">
                                        Limitação dos dados
                                      </p>

                                      <p className="mt-2 break-words text-xs leading-6 text-zinc-400">
                                        {query.error || "Os dados disponíveis não permitem realizar esta análise."}
                                      </p>
                                    </div>
                                )}

                                {query.status === "failure" &&
                                  query.failure_type !== "unsupported_capability" && (
                                    <div className="rounded-lg border border-red-500/20 bg-red-500/[0.05] p-4">
                                      <p className="text-xs font-semibold text-red-400">
                                        Erro ao executar esta consulta
                                      </p>

                                      <p className="mt-2 break-words text-xs leading-6 text-zinc-400">
                                        {query.error || "Não foi possível executar esta consulta."}
                                      </p>
                                    </div>
                                  )}

                                {query.status === "success" && (
                                  <>
                                    <div className="flex flex-wrap gap-2">
                                      <span className="rounded-full bg-emerald-500/10 px-2.5 py-1 text-[10px] text-emerald-400">
                                        {query.total_results} resultado(s)
                                      </span>

                                      {typeof query.metrics?.total_time_ms === "number" && (
                                        <span className="rounded-full bg-white/[0.04] px-2.5 py-1 text-[10px] text-zinc-400">
                                          {query.metrics.total_time_ms.toFixed(2)} ms
                                        </span>
                                      )}
                                    </div>

                                    {query.chart && (
                                      <AnalyticsChart
                                        data={query.chart.data ?? query.results}
                                        type={query.chart.type}
                                        title={query.chart.title}
                                        labelKey={query.chart.label_key}
                                        valueKey={query.chart.value_key}
                                        series={query.chart.series}
                                      />
                                    )}

                                    <div className="rounded-lg border border-white/[0.06] bg-black/10 p-4">
                                      <p className="text-xs font-semibold text-zinc-300">
                                        Dados consultados
                                      </p>

                                      {query.results.length > 0 ? (
                                        <ResultsTable
                                          rows={query.results}
                                          totalResults={query.total_results}
                                          formatValue={formatValue}
                                        />
                                      ) : (
                                        <p className="mt-3 text-xs text-zinc-500">
                                          Nenhum registro encontrado.
                                        </p>
                                      )}
                                    </div>
                                  </>
                                )}

                                {query.sql && (
                                  <details className="group rounded-lg border border-white/[0.07] bg-white/[0.02]">
                                    <summary className="flex cursor-pointer items-center justify-between p-4 text-xs font-medium text-zinc-400 hover:text-zinc-200">
                                      <span>Consultar SQL desta consulta</span>
                                      <span className="transition group-open:rotate-180">
                                        ⌄
                                      </span>
                                    </summary>

                                    <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words border-t border-white/[0.06] p-4 text-xs leading-6 text-blue-300">
                                      <code>{query.sql}</code>
                                    </pre>
                                  </details>
                                )}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <>
                            {analysis.chart && (
                              <AnalyticsChart
                                data={analysis.chart.data ?? analysis.results}
                                type={analysis.chart.type}
                                title={analysis.chart.title}
                                labelKey={analysis.chart.label_key}
                                valueKey={analysis.chart.value_key}
                                series={analysis.chart.series}
                              />
                            )}

                            <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-5">
                              <p className="text-xs font-semibold text-zinc-300">
                                Dados consultados
                              </p>

                              {analysis.results.length > 0 ? (
                                <ResultsTable
                                  rows={analysis.results}
                                  totalResults={analysis.total_results}
                                  formatValue={formatValue}
                                />
                              ) : (
                                <p className="mt-3 text-xs text-zinc-500">
                                  Nenhum registro encontrado.
                                </p>
                              )}
                            </div>

                            <details className="group rounded-xl border border-white/[0.07] bg-white/[0.02]">
                              <summary className="flex cursor-pointer items-center justify-between p-4 text-xs font-medium text-zinc-400 hover:text-zinc-200">
                                <span>Consultar SQL gerado pela IA</span>
                                <span className="transition group-open:rotate-180">
                                  ⌄
                                </span>
                              </summary>

                              <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words border-t border-white/[0.06] p-4 text-xs leading-6 text-blue-300">
                                <code>{analysis.sql}</code>
                              </pre>
                            </details>
                          </>
                        )}
                      </>
                    )}
                  </div>
                )}

                <div className="mt-12">
                  <div className="mb-4 flex items-center justify-between">
                    <p className="text-xs font-medium text-zinc-400">
                      Experimente perguntar
                    </p>
                    <p className="text-[10px] text-zinc-600">
                      SUGESTÕES
                    </p>
                  </div>

                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    {suggestions.map((item) => (
                      <button
                        key={item.title}
                        onClick={() => setQuestion(item.question)}
                        className="group flex items-start gap-3 rounded-xl border border-white/[0.07] bg-white/[0.02] p-4 text-left transition hover:border-blue-500/30 hover:bg-blue-500/[0.04]"
                      >
                        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white/[0.05] text-lg text-blue-400 transition group-hover:bg-blue-500/10">
                          {item.icon}
                        </span>

                        <span>
                          <span className="block text-xs font-medium text-zinc-300">
                            {item.title}
                          </span>
                          <span className="mt-1 block text-[11px] leading-5 text-zinc-500">
                            {item.description}
                          </span>
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="mt-14 flex flex-wrap items-center justify-center gap-x-6 gap-y-3 border-t border-white/[0.06] pt-6">
                  <span className="text-[10px] text-zinc-600">
                    POWERED BY
                  </span>
                  <span className="text-xs font-medium text-zinc-500">
                    Gemini
                  </span>
                  <span className="text-xs font-medium text-zinc-500">
                    OpenAI
                  </span>
                  <span className="text-xs font-medium text-zinc-500">
                    Anthropic
                  </span>
                  <span className="text-xs font-medium text-zinc-500">
                    Python
                  </span>
                  <span className="text-xs font-medium text-zinc-500">
                    FastAPI
                  </span>
                  <span className="text-xs font-medium text-zinc-500">
                    DuckDB
                  </span>
                </div>
              </>
            )}

            {/* HISTORY MANAGER */}
            {activePage === "history" && (
              <HistoryManager
                onSelect={handleSelectHistory}
                onChanged={() => void loadHistory()}
              />
            )}
          </div>
        </div>
      </section>
    </main>
  );
}