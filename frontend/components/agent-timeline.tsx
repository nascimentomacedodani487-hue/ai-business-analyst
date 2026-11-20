type AgentStep = {
  name: string;
  status: "completed" | "failed" | "unsupported";
  duration_ms: number;
  details?: string | null;
};

type AgentTimelineProps = {
  steps: AgentStep[];
};

const UNKNOWN_STAGE_LABEL = "Etapa de processamento";

const stageLabels: Record<string, string> = {
  database_version: "Identificação da base de dados",
  cache_lookup: "Consulta ao cache",
  cache_hit: "Recuperação do resultado em cache",
  ai_initialization: "Inicialização da inteligência artificial",
  sql_generation: "Geração da consulta SQL",
  sql_validation: "Validação de segurança SQL",
  sql_execution: "Execução da consulta no DuckDB",
  response_generation: "Interpretação dos resultados",
  chart_generation: "Preparação do gráfico",
};

const stageDescriptions: Record<string, string> = {
  database_version: "Verifica a versão atual da base de dados.",
  cache_lookup: "Procura uma análise anterior que possa ser reutilizada.",
  cache_hit: "A análise foi recuperada sem novas chamadas à IA.",
  ai_initialization: "Inicializa os recursos necessários para a análise.",
  sql_generation: "Transforma a pergunta em uma consulta SQL.",
  sql_validation: "Verifica se a consulta respeita as regras de segurança.",
  sql_execution: "Executa a consulta validada no banco de dados.",
  response_generation: "Converte os resultados em uma resposta compreensível.",
  chart_generation: "Prepara a configuração de visualização dos resultados.",
};

function formatDuration(value: number) {
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }

  return `${(value / 1000).toFixed(2)} s`;
}

function getStageLabel(name: string): string {
  if (stageLabels[name]) {
    return stageLabels[name];
  }

  const capabilityMatch = name.match(
    /^query_(\d+)_capability_validation$/
  );

  if (capabilityMatch) {
    return `Validação de capacidade — consulta ${capabilityMatch[1]}`;
  }

  const correctionMatch = name.match(/^sql_correction_(\d+)$/);

  if (correctionMatch) {
    return `Correção SQL — tentativa ${correctionMatch[1]}`;
  }

  const executionMatch = name.match(/^sql_execution_attempt_(\d+)$/);

  if (executionMatch) {
    return `Execução SQL — tentativa ${executionMatch[1]}`;
  }

  // Não expõe identificadores internos de etapas desconhecidas.
  return UNKNOWN_STAGE_LABEL;
}

function getStageDescription(name: string): string | undefined {
  if (stageDescriptions[name]) {
    return stageDescriptions[name];
  }

  if (/^query_\d+_capability_validation$/.test(name)) {
    return "Verifica se a base possui os dados e a granularidade necessários antes de gerar SQL.";
  }

  const correctionMatch = name.match(/^sql_correction_(\d+)$/);

  if (correctionMatch) {
    return "Solicita à inteligência artificial uma nova versão da consulta após identificar um erro na tentativa anterior.";
  }

  const executionMatch = name.match(/^sql_execution_attempt_(\d+)$/);

  if (executionMatch) {
    return "Executa a versão atual da consulta SQL no DuckDB.";
  }

  return "Etapa registrada durante o processamento da análise.";
}

function getStageIcon(name: string) {
  const icons: Record<string, string> = {
    database_version: "◫",
    cache_lookup: "⌕",
    cache_hit: "⚡",
    ai_initialization: "✳",
    sql_generation: "⌘",
    sql_validation: "✓",
    sql_execution: "▤",
    response_generation: "✧",
    chart_generation: "▥",
  };

  if (/^sql_correction_\d+$/.test(name)) {
    return "↻";
  }

  if (/^sql_execution_attempt_\d+$/.test(name)) {
    return "▤";
  }

  return icons[name] ?? "•";
}

export default function AgentTimeline({ steps }: AgentTimelineProps) {
  if (!steps || steps.length === 0) {
    return null;
  }

  return (
    <section className="rounded-xl border border-white/[0.07] bg-[#11141c] p-5">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold text-zinc-200">
            Agent Execution Timeline
          </p>
          <p className="mt-1 text-[10px] text-zinc-500">
            Etapas registradas durante o processamento da análise
          </p>
        </div>

        <span className="rounded-full border border-white/[0.08] bg-white/[0.03] px-3 py-1 text-[10px] text-zinc-500">
          {steps.length} etapas
        </span>
      </div>

      <div>
        {steps.map((step, index) => {
          const completed = step.status === "completed";
          const unsupported = step.status === "unsupported";
          const isCache = step.name === "cache_hit";
          const title = getStageLabel(step.name);
          const description = getStageDescription(step.name);

          return (
            <div
              key={`${step.name}-${index}`}
              className="flex gap-3"
            >
              <div className="flex w-8 shrink-0 flex-col items-center">
                <div
                  className={`flex h-8 w-8 items-center justify-center rounded-full border text-sm ${
                    unsupported
                      ? "border-amber-500/30 bg-amber-500/10 text-amber-300"
                      : !completed
                        ? "border-red-500/30 bg-red-500/10 text-red-400"
                        : isCache
                          ? "border-blue-500/30 bg-blue-500/10 text-blue-400"
                          : "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                  }`}
                >
                  {unsupported ? "?" : completed ? getStageIcon(step.name) : "!"}
                </div>

                {index < steps.length - 1 && (
                  <div className="my-1 min-h-7 w-px flex-1 bg-white/[0.10]" />
                )}
              </div>

              <div className="min-w-0 flex-1 pb-5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="text-xs font-medium text-zinc-200">
                      {title}
                    </p>

                    <span
                      className={`rounded-full px-2 py-0.5 text-[9px] ${
                        unsupported
                          ? "bg-amber-500/10 text-amber-300"
                          : !completed
                            ? "bg-red-500/10 text-red-400"
                            : isCache
                              ? "bg-blue-500/10 text-blue-400"
                              : "bg-emerald-500/10 text-emerald-400"
                      }`}
                    >
                      {unsupported
                        ? "Não suportada"
                        : !completed
                          ? "Falhou"
                          : isCache
                            ? "Cache hit"
                            : "Concluído"}
                    </span>
                  </div>

                  <span className="shrink-0 text-[10px] tabular-nums text-zinc-500">
                    {formatDuration(step.duration_ms)}
                  </span>
                </div>

                {description && (
                  <p className="mt-1 text-[10px] leading-5 text-zinc-500">
                    {description}
                  </p>
                )}

                {step.details && (
                  <details className="group mt-2">
                    <summary className="cursor-pointer text-[10px] text-blue-400 hover:text-blue-300">
                      Ver detalhes
                    </summary>
                    <p className="mt-2 rounded-lg border border-white/[0.05] bg-white/[0.02] p-3 text-[10px] leading-5 text-zinc-400">
                      {step.details}
                    </p>
                  </details>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}