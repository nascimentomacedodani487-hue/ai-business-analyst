
const API_URL = process.env.NEXT_PUBLIC_API_URL;

export type BenchmarkProvider = "gemini" | "openai" | "anthropic";

export type BenchmarkHistoryItem = {
  benchmark_id: string;
  created_at: string;
  schema_version: number;
  dataset: {
    version: number;
    source: string | null;
    case_count: number;
    sha256: string;
  };
  providers: string[];
  models: Record<string, string | null>;
  summary: {
    provider_count: number;
    completed_providers: number;
    failed_providers: number;
    all_providers_completed: boolean;
  };
};

export type BenchmarkHistoryResponse = {
  items: BenchmarkHistoryItem[];
  pagination: {
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
  };
};

export type BenchmarkProviderRun = {
  execution_status: "completed" | "error";
  report?: {
    run_id?: string;
    provider?: string | null;
    model?: string | null;
    summary?: Record<string, unknown>;
    results?: Record<string, unknown>[];
    token_cost_summary?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
  } | null;
  error?: string | null;
  error_type?: string | null;
};

export type BenchmarkReport = {
  schema_version: number;
  benchmark_id: string;
  created_at: string;
  dataset: {
    version: number;
    source: string | null;
    case_count: number;
    case_ids?: string[];
    sha256: string;
  };
  configuration: {
    providers: string[];
    provider_count: number;
    scenario_count: number;
    execution_strategy: string;
    same_dataset_for_all_providers: boolean;
    repetitions_per_provider?: number;
    total_requested_runs?: number;
  };
  summary: {
    provider_count: number;
    completed_providers: number;
    failed_providers: number;
    all_providers_completed: boolean;
    total_repetitions?: number;
    completed_repetitions?: number;
    failed_repetitions?: number;
  };
  provider_runs: Record<string, BenchmarkProviderRun>;
  comparison?: {
    by_provider?: Record<string, Record<string, unknown>>;
    by_category?: Record<string, Record<string, unknown>>;
    by_case?: Record<string, Record<string, unknown>>;
  };
  repeatability?: Record<string, unknown>;
};

export type BenchmarkHistoryFilters = {
  limit?: number;
  offset?: number;
  provider?: string;
  model?: string;
  dataset_sha256?: string;
};

export type MetricComparison = {
  baseline: number | null;
  candidate: number | null;
  delta: number | null;
  delta_pct: number | null;
  threshold: number;
  status:
    | "stable"
    | "improvement"
    | "regression"
    | "not_comparable";
};

export type ProviderBenchmarkComparison = {
  status: string;
  baseline_model: string | null;
  candidate_model: string | null;
  execution?: {
    baseline: string | null;
    candidate: string | null;
    status: string;
  };
  metrics: Record<string, MetricComparison>;
};

export type BenchmarkComparisonResult = {
  schema_version: number;
  baseline: {
    benchmark_id: string;
    created_at: string | null;
    dataset_sha256: string | null;
    dataset_case_count: number | null;
  };
  candidate: {
    benchmark_id: string;
    created_at: string | null;
    dataset_sha256: string | null;
    dataset_case_count: number | null;
  };
  compatibility: {
    comparable: boolean;
    [key: string]: unknown;
  };
  summary: {
    status: string;
    providers_total: number;
    providers_compared: number;
    providers_with_regressions: number;
    providers_with_improvements: number;
    regressions: number;
    improvements: number;
    stable_metrics: number;
    not_comparable_metrics: number;
  };
  providers: Record<string, ProviderBenchmarkComparison>;
};

type BenchmarkComparisonRequest = {
  baseline_id: string;
  candidate_id: string;
};

async function readJsonResponse(
  response: Response
): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function getApiUrl(): string {
  if (!API_URL) {
    throw new Error(
      "A variável NEXT_PUBLIC_API_URL não está configurada."
    );
  }

  return API_URL.replace(/\/+$/, "");
}

function extractError(
  data: unknown,
  fallback: string
): string {
  if (!data || typeof data !== "object") {
    return fallback;
  }

  const detail = (data as { detail?: unknown }).detail;

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (detail && typeof detail === "object") {
    const message = (detail as { message?: unknown }).message;

    if (typeof message === "string" && message.trim()) {
      return message;
    }
  }

  return fallback;
}

async function apiRequest<T>(
  path: string,
  options: RequestInit = {},
  fallback = "Não foi possível concluir a operação."
): Promise<T> {
  const response = await fetch(`${getApiUrl()}${path}`, {
    ...options,
    headers: {
      ...(options.body
        ? { "Content-Type": "application/json" }
        : {}),
      ...options.headers
    },
    cache: "no-store"
  });

  const data = await readJsonResponse(response);

  if (!response.ok) {
    throw new Error(extractError(data, fallback));
  }

  return data as T;
}

/**
 * Lista benchmarks persistidos no backend.
 *
 * Não executa benchmarks nem realiza chamadas a provedores LLM.
 */
export async function fetchBenchmarkHistory(
  filters: BenchmarkHistoryFilters = {}
): Promise<BenchmarkHistoryResponse> {
  const params = new URLSearchParams({
    limit: String(filters.limit ?? 20),
    offset: String(filters.offset ?? 0)
  });

  if (filters.provider?.trim()) {
    params.set("provider", filters.provider.trim());
  }

  if (filters.model?.trim()) {
    params.set("model", filters.model.trim());
  }

  if (filters.dataset_sha256?.trim()) {
    params.set(
      "dataset_sha256",
      filters.dataset_sha256.trim()
    );
  }

  return apiRequest<BenchmarkHistoryResponse>(
    `/evaluation/benchmarks?${params.toString()}`,
    {},
    "Não foi possível carregar o histórico de benchmarks."
  );
}

/**
 * Recupera um benchmark completo pelo identificador.
 */
export async function fetchBenchmarkDetail(
  benchmarkId: string
): Promise<BenchmarkReport> {
  if (!benchmarkId.trim()) {
    throw new Error(
      "O identificador do benchmark não pode ser vazio."
    );
  }

  return apiRequest<BenchmarkReport>(
    `/evaluation/benchmarks/${encodeURIComponent(benchmarkId)}`,
    {},
    "Não foi possível recuperar os detalhes do benchmark."
  );
}

/**
 * Compara dois relatórios históricos.
 *
 * A operação utiliza somente dados persistidos.
 * Não executa chamadas aos provedores de IA.
 */
export async function compareBenchmarks(
  request: BenchmarkComparisonRequest
): Promise<BenchmarkComparisonResult> {
  if (
    !request.baseline_id.trim() ||
    !request.candidate_id.trim()
  ) {
    throw new Error(
      "Selecione os dois benchmarks para comparação."
    );
  }

  return apiRequest<BenchmarkComparisonResult>(
    "/evaluation/benchmarks/compare",
    {
      method: "POST",
      body: JSON.stringify({
        baseline_id: request.baseline_id,
        candidate_id: request.candidate_id
      })
    },
    "Não foi possível comparar os benchmarks."
  );
}
