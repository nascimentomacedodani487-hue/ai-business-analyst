
const API_URL = process.env.NEXT_PUBLIC_API_URL;

export type LLMProvider = "gemini" | "openai" | "anthropic";

export type AnalysisStatus = "success" | "failure";

export type AgentStep = {
  name: string;
  status: "completed" | "failed" | "unsupported";
  duration_ms: number;
  details?: string | null;
};

export type AnalysisChartSeries = {
  data_key: string;
  label: string;
};

export type AnalysisChart = {
  type: "bar" | "line" | "pie";
  title: string;
  label_key: string;
  value_key: string;
  data?: Record<string, unknown>[];

  // Séries opcionais para gráficos de barras agrupadas.
  // Mantém compatibilidade com gráficos históricos.
  series?: AnalysisChartSeries[];
};

export type AnalysisQuery = {
  id: string;
  objective: string;
  question: string;
  status: "success" | "failure";
  failure_type?: string | null;
  sql: string | null;
  results: Record<string, unknown>[];
  total_results: number;
  chart: AnalysisChart | null;
  error: string | null;
  metrics: {
    sql_generation_ms?: number;
    sql_correction_ms?: number;
    sql_execution_ms?: number;
    total_time_ms?: number;
  };
};

export type Analysis = {
  analysis_id?: string;
  question: string;
  provider?: LLMProvider;
  model?: string;
  sql: string;
  results: Record<string, unknown>[];
  total_results: number;
  answer: string;
  cache_hit: boolean;
  steps: AgentStep[];
  metrics: {
    total_time_ms: number;
    sql_generation_ms?: number;
    sql_execution_ms?: number;
    response_generation_ms?: number;
    cache_hit: boolean;
  };
  chart: AnalysisChart | null;

  // Consultas independentes de uma análise composta.
  queries?: AnalysisQuery[];
  query_count?: number;
  successful_queries?: number;
  failed_queries?: number;
  unsupported_queries?: number;
  partial_success?: boolean;
  analysis_mode?: "single_query" | "multi_query";
};

export type HistoryItem = {
  analysis_id?: string;
  question: string;
  title?: string | null;
  provider?: LLMProvider | null;
  model?: string | null;
  total_time_ms: number;
  cache_hit: boolean;
  timestamp: string;
  status?: AnalysisStatus;
  partial_success?: boolean;
  error_type?: string | null;
  error_message?: string | null;
  is_pinned?: boolean;
  is_locked?: boolean;
  is_archived?: boolean;
  project_id?: string | null;
  analysis?: Analysis;
};

export type HistoryFilters = {
  status?: AnalysisStatus;
  search?: string;
  include_archived?: boolean;
  project_id?: string | null;
};

export type HistoryPage = {
  items: HistoryItem[];
  total: number;
  has_more: boolean;
};

export type AnalysisProject = {
  project_id: string;
  name: string;
  description?: string | null;
  created_at: string;
  updated_at?: string | null;
  analysis_count?: number;
};

export type BackendHistoryRecord = {
  analysis_id: string;
  created_at: string;
  question: string;
  title?: string | null;
  status: AnalysisStatus;
  provider?: LLMProvider | null;
  model?: string | null;
  queries?: AnalysisQuery[] | null;
  total_time_ms: number;
  cache_hit: boolean;
  error_type?: string | null;
  error_message?: string | null;
  is_pinned?: boolean;
  is_locked?: boolean;
  is_archived?: boolean;
  project_id?: string | null;
};

type BackendHistoryResponse = {
  items?: BackendHistoryRecord[];
  pagination?: {
    total?: number;
    limit?: number;
    offset?: number;
    has_more?: boolean;
  };
  total?: number;
  limit?: number;
  offset?: number;
  has_more?: boolean;
};

type BackendAnalysisDetail = {
  analysis_id: string;
  created_at: string;
  question: string;
  title?: string | null;
  status: AnalysisStatus;
  provider?: LLMProvider | null;
  model?: string | null;
  total_time_ms: number;
  cache_hit: boolean;
  sql?: string | null;
  results?: Record<string, unknown>[] | null;
  total_results?: number;
  answer?: string | null;
  chart?: Analysis["chart"];
  queries?: AnalysisQuery[] | null;
  query_count?: number;
  successful_queries?: number;
  failed_queries?: number;
  unsupported_queries?: number;
  partial_success?: boolean;
  analysis_mode?: "single_query" | "multi_query";
  steps?: AgentStep[];
  error_type?: string | null;
  error_message?: string | null;
  is_pinned?: boolean;
  is_locked?: boolean;
  is_archived?: boolean;
  project_id?: string | null;
};

type BackendProjectResponse = {
  items?: AnalysisProject[];
};

type BackendMutationResponse = {
  message: string;
  analysis_id?: string;
  project_id?: string;
  title?: string;
  name?: string;
  is_pinned?: boolean;
  is_locked?: boolean;
  is_archived?: boolean;
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

function formatTimestamp(value: string): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString("pt-BR", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });
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
      ...(options.body ? { "Content-Type": "application/json" } : {}),
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

function mapHistoryRecord(
  record: BackendHistoryRecord
): HistoryItem {
  const queries = Array.isArray(record.queries)
    ? record.queries
    : [];

  const successfulCount = queries.filter(
    (query) => query.status === "success"
  ).length;

  const unsupportedCount = queries.filter(
    (query) => query.failure_type === "unsupported_capability"
  ).length;

  const failedCount = queries.filter(
    (query) =>
      query.status === "failure" &&
      query.failure_type !== "unsupported_capability"
  ).length;

  return {
    analysis_id: record.analysis_id,
    question: record.question,
    title: record.title ?? null,
    provider: record.provider ?? null,
    model: record.model ?? null,
    total_time_ms: record.total_time_ms,
    cache_hit: record.cache_hit,
    timestamp: formatTimestamp(record.created_at),
    status: record.status,
    partial_success:
      successfulCount > 0 &&
      (unsupportedCount > 0 || failedCount > 0),
    error_type: record.error_type ?? null,
    error_message: record.error_message ?? null,
    is_pinned: Boolean(record.is_pinned),
    is_locked: Boolean(record.is_locked),
    is_archived: Boolean(record.is_archived),
    project_id: record.project_id ?? null
  };
}

/**
 * Consulta o histórico persistido no backend.
 *
 * Mantém compatibilidade com as chamadas antigas:
 * fetchAnalysisHistory(limit, offset).
 *
 * Os filtros são opcionais.
 */
export async function fetchAnalysisHistory(
  limit = 50,
  offset = 0,
  filters: HistoryFilters = {}
): Promise<HistoryPage> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset)
  });

  if (filters.status) {
    params.set("status", filters.status);
  }

  if (filters.search?.trim()) {
    params.set("search", filters.search.trim());
  }

  if (filters.include_archived) {
    params.set("include_archived", "true");
  }

  if (filters.project_id) {
    params.set("project_id", filters.project_id);
  }

  const payload = await apiRequest<BackendHistoryResponse>(
    `/analyses/history?${params.toString()}`,
    {},
    "Não foi possível carregar o histórico de análises."
  );

  const records = Array.isArray(payload.items)
    ? payload.items
    : [];

  const pagination = payload.pagination;

  const items = records.map(mapHistoryRecord);

  return {
    items,
    total:
      typeof pagination?.total === "number"
        ? pagination.total
        : typeof payload.total === "number"
          ? payload.total
          : items.length,
    has_more:
      typeof pagination?.has_more === "boolean"
        ? pagination.has_more
        : Boolean(payload.has_more)
  };
}

/**
 * Recupera os detalhes completos de uma análise.
 */
export async function fetchAnalysisDetail(
  analysisId: string
): Promise<BackendAnalysisDetail> {
  return apiRequest<BackendAnalysisDetail>(
    `/analyses/history/${encodeURIComponent(analysisId)}`,
    {},
    "Não foi possível recuperar os detalhes da análise."
  );
}

/**
 * Converte o registro persistido para o formato
 * utilizado pelos componentes atuais do frontend.
 *
 * Mantém compatibilidade com análises antigas
 * que ainda não possuem consultas múltiplas.
 */
export function restoreAnalysis(
  detail: BackendAnalysisDetail
): Analysis {
  if (detail.status !== "success") {
    throw new Error(
      detail.error_message ||
        "Esta análise foi registrada como uma execução com falha."
    );
  }

  const queries = Array.isArray(detail.queries)
    ? detail.queries
    : [];

  const successfulQueryCount = queries.filter(
    (query) => query.status === "success"
  ).length;

  const unsupportedQueryCount = queries.filter(
    (query) => query.failure_type === "unsupported_capability"
  ).length;

  const failedQueryCount = queries.filter(
    (query) =>
      query.status === "failure" &&
      query.failure_type !== "unsupported_capability"
  ).length;

  // Reconstrói as métricas a partir das etapas persistidas.
  // Isso mantém a observabilidade ao restaurar análises do histórico.
  const restoredSteps = Array.isArray(detail.steps)
    ? detail.steps
    : [];

  const sumStepDurations = (
    matches: (name: string) => boolean
  ): number | undefined => {
    const matchingSteps = restoredSteps.filter(
      (step) =>
        matches(step.name) &&
        Number.isFinite(step.duration_ms)
    );

    if (matchingSteps.length === 0) {
      return undefined;
    }

    return matchingSteps.reduce(
      (total, step) => total + step.duration_ms,
      0
    );
  };

  const sqlGenerationMs = sumStepDurations(
    (name) =>
      name === "sql_generation" ||
      /^query_\d+_sql_generation$/.test(name)
  );

  const sqlExecutionMs = sumStepDurations(
    (name) =>
      name === "sql_execution" ||
      /^query_\d+_sql_execution$/.test(name) ||
      /^sql_execution_attempt_\d+$/.test(name)
  );

  const responseGenerationMs = sumStepDurations(
    (name) =>
      name === "response_generation" ||
      name === "multi_response_generation"
  );

  return {
    analysis_id: detail.analysis_id,
    question: detail.question,
    provider: detail.provider ?? undefined,
    model: detail.model ?? undefined,
    sql: detail.sql ?? "",
    results: Array.isArray(detail.results)
      ? detail.results
      : [],
    total_results:
      typeof detail.total_results === "number"
        ? detail.total_results
        : 0,
    answer: detail.answer ?? "",
    cache_hit: Boolean(detail.cache_hit),
    steps: restoredSteps,
    metrics: {
      total_time_ms: detail.total_time_ms,
      sql_generation_ms: sqlGenerationMs,
      sql_execution_ms: sqlExecutionMs,
      response_generation_ms: responseGenerationMs,
      cache_hit: Boolean(detail.cache_hit)
    },
    chart: detail.chart ?? null,

    // Dados das consultas independentes.
    // Os gráficos e suas séries são preservados integralmente.
    queries,
    query_count: detail.query_count ?? queries.length,
    successful_queries:
      detail.successful_queries ?? successfulQueryCount,
    failed_queries:
      detail.failed_queries ?? failedQueryCount,
    unsupported_queries:
      detail.unsupported_queries ?? unsupportedQueryCount,
    partial_success:
      detail.partial_success ??
      (
        successfulQueryCount > 0 &&
        (failedQueryCount > 0 || unsupportedQueryCount > 0)
      ),
    analysis_mode:
      detail.analysis_mode ??
      (queries.length > 0 ? "multi_query" : "single_query")
  };
}

/* -------------------------------------------------------------------------- */
/* Gerenciamento de análises                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Renomeia uma análise.
 */
export async function renameAnalysis(
  analysisId: string,
  title: string
): Promise<BackendMutationResponse> {
  return apiRequest(
    `/analyses/history/${encodeURIComponent(analysisId)}/title`,
    {
      method: "PATCH",
      body: JSON.stringify({ title })
    },
    "Não foi possível renomear a análise."
  );
}

/**
 * Fixa ou desafixa uma análise.
 */
export async function setAnalysisPinned(
  analysisId: string,
  enabled: boolean
): Promise<BackendMutationResponse> {
  return apiRequest(
    `/analyses/history/${encodeURIComponent(analysisId)}/pin`,
    {
      method: "PATCH",
      body: JSON.stringify({ enabled })
    },
    "Não foi possível atualizar o destaque da análise."
  );
}

/**
 * Bloqueia ou desbloqueia uma análise.
 */
export async function setAnalysisLocked(
  analysisId: string,
  enabled: boolean
): Promise<BackendMutationResponse> {
  return apiRequest(
    `/analyses/history/${encodeURIComponent(analysisId)}/lock`,
    {
      method: "PATCH",
      body: JSON.stringify({ enabled })
    },
    "Não foi possível atualizar o bloqueio da análise."
  );
}

/**
 * Arquiva ou restaura uma análise.
 */
export async function setAnalysisArchived(
  analysisId: string,
  enabled: boolean
): Promise<BackendMutationResponse> {
  return apiRequest(
    `/analyses/history/${encodeURIComponent(analysisId)}/archive`,
    {
      method: "PATCH",
      body: JSON.stringify({ enabled })
    },
    "Não foi possível atualizar o arquivamento da análise."
  );
}

/**
 * Move uma análise para um projeto.
 * Use null para removê-la de um projeto.
 */
export async function moveAnalysisToProject(
  analysisId: string,
  projectId: string | null
): Promise<BackendMutationResponse> {
  return apiRequest(
    `/analyses/history/${encodeURIComponent(analysisId)}/project`,
    {
      method: "PATCH",
      body: JSON.stringify({ project_id: projectId })
    },
    "Não foi possível mover a análise."
  );
}

/**
 * Exclui permanentemente uma análise.
 */
export async function deleteAnalysis(
  analysisId: string
): Promise<BackendMutationResponse> {
  return apiRequest(
    `/analyses/history/${encodeURIComponent(analysisId)}`,
    {
      method: "DELETE"
    },
    "Não foi possível excluir a análise."
  );
}

/* -------------------------------------------------------------------------- */
/* Gerenciamento de projetos                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Lista os projetos cadastrados.
 */
export async function fetchAnalysisProjects(): Promise<
  AnalysisProject[]
> {
  const payload = await apiRequest<BackendProjectResponse>(
    "/analysis-projects",
    {},
    "Não foi possível carregar os projetos."
  );

  return Array.isArray(payload.items) ? payload.items : [];
}

/**
 * Cria um projeto.
 */
export async function createAnalysisProject(
  name: string,
  description?: string
): Promise<AnalysisProject> {
  return apiRequest(
    "/analysis-projects",
    {
      method: "POST",
      body: JSON.stringify({
        name,
        description: description ?? ""
      })
    },
    "Não foi possível criar o projeto."
  );
}

/**
 * Renomeia um projeto.
 */
export async function renameAnalysisProject(
  projectId: string,
  name: string
): Promise<BackendMutationResponse> {
  return apiRequest(
    `/analysis-projects/${encodeURIComponent(projectId)}`,
    {
      method: "PATCH",
      body: JSON.stringify({ name })
    },
    "Não foi possível renomear o projeto."
  );
}

/**
 * Exclui um projeto.
 * As análises associadas são preservadas pelo backend.
 */
export async function deleteAnalysisProject(
  projectId: string
): Promise<BackendMutationResponse> {
  return apiRequest(
    `/analysis-projects/${encodeURIComponent(projectId)}`,
    {
      method: "DELETE"
    },
    "Não foi possível excluir o projeto."
  );
}
