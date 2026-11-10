
"use client";

import { useCallback, useEffect, useState } from "react";

import {
  fetchAnalysisHistory,
  fetchAnalysisProjects,
  renameAnalysis,
  setAnalysisPinned,
  setAnalysisLocked,
  setAnalysisArchived,
  moveAnalysisToProject,
  deleteAnalysis,
  createAnalysisProject,
  renameAnalysisProject,
  deleteAnalysisProject,
  type HistoryItem,
  type AnalysisProject,
  type AnalysisStatus
} from "@/lib/analysis-history";

type HistoryManagerProps = {
  onSelect: (item: HistoryItem) => void;
  onChanged?: () => void;
};

const PAGE_SIZE = 10;

function formatDuration(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "—";
  }

  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }

  return `${(value / 1000).toFixed(1)} s`;
}

function getStatusLabel(
  status?: AnalysisStatus,
  partialSuccess = false
): string {
  if (status !== "success") {
    return "Falhou";
  }

  return partialSuccess
    ? "Concluída parcialmente"
    : "Concluída";
}

function getStatusStyle(
  status?: AnalysisStatus,
  partialSuccess = false
): string {
  if (status !== "success") {
    return "border-red-500/20 bg-red-500/10 text-red-400";
  }

  return partialSuccess
    ? "border-amber-500/20 bg-amber-500/10 text-amber-300"
    : "border-emerald-500/20 bg-emerald-500/10 text-emerald-400";
}

export default function HistoryManager({
  onSelect,
  onChanged
}: HistoryManagerProps) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [projects, setProjects] = useState<AnalysisProject[]>([]);

  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<"" | AnalysisStatus>("");
  const [projectId, setProjectId] = useState("");
  const [includeArchived, setIncludeArchived] = useState(false);

  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [busyId, setBusyId] = useState("");
  const [expandedErrorId, setExpandedErrorId] = useState("");

  const [refreshKey, setRefreshKey] = useState(0);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setSearch(searchInput.trim());
      setPage(1);
    }, 350);

    return () => window.clearTimeout(timer);
  }, [searchInput]);

  const refresh = useCallback(() => {
    setRefreshKey((current) => current + 1);
  }, []);

  useEffect(() => {
    let active = true;

    async function loadData() {
      setLoading(true);
      setError("");

      try {
        const [historyResult, projectResult] = await Promise.all([
          fetchAnalysisHistory(PAGE_SIZE, (page - 1) * PAGE_SIZE, {
            ...(status ? { status } : {}),
            ...(search ? { search } : {}),
            ...(projectId ? { project_id: projectId } : {}),
            ...(includeArchived ? { include_archived: true } : {})
          }),
          fetchAnalysisProjects()
        ]);

        if (!active) return;

        setItems(historyResult.items);
        setTotal(historyResult.total);
        setProjects(projectResult);
      } catch (err) {
        if (!active) return;

        setError(
          err instanceof Error
            ? err.message
            : "Não foi possível carregar o histórico."
        );
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadData();

    return () => {
      active = false;
    };
  }, [
    page,
    status,
    search,
    projectId,
    includeArchived,
    refreshKey
  ]);

  const runAction = useCallback(
    async (
      id: string,
      action: () => Promise<unknown>,
      message: string
    ) => {
      setBusyId(id);
      setError("");
      setSuccessMessage("");

      try {
        await action();

        setSuccessMessage(message);
        refresh();
        onChanged?.();
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Não foi possível concluir a operação."
        );
      } finally {
        setBusyId("");
      }
    },
    [refresh, onChanged]
  );

  async function handleRename(item: HistoryItem) {
    if (!item.analysis_id || item.is_locked) return;

    const currentTitle =
      item.title?.trim() || item.question.slice(0, 80);

    const newTitle = window.prompt(
      "Digite o novo título da análise:",
      currentTitle
    );

    if (newTitle === null) return;

    if (!newTitle.trim()) {
      setError("O título não pode ficar vazio.");
      return;
    }

    await runAction(
      item.analysis_id,
      () => renameAnalysis(item.analysis_id!, newTitle.trim()),
      "Análise renomeada."
    );
  }

  async function handleDelete(item: HistoryItem) {
    if (!item.analysis_id || item.is_locked) return;

    const title = item.title || item.question;

    const confirmed = window.confirm(
      `Excluir permanentemente esta análise?\n\n"${title}"\n\nEssa ação não poderá ser desfeita.`
    );

    if (!confirmed) return;

    await runAction(
      item.analysis_id,
      () => deleteAnalysis(item.analysis_id!),
      "Análise excluída."
    );
  }

  async function handleCreateProject() {
    const name = window.prompt("Nome do novo projeto:");

    if (name === null) return;

    if (!name.trim()) {
      setError("O nome do projeto não pode ficar vazio.");
      return;
    }

    setError("");
    setSuccessMessage("");

    try {
      await createAnalysisProject(name.trim());

      setSuccessMessage("Projeto criado.");
      refresh();
      onChanged?.();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Não foi possível criar o projeto."
      );
    }
  }

  async function handleRenameProject(project: AnalysisProject) {
    const name = window.prompt(
      "Digite o novo nome do projeto:",
      project.name
    );

    if (name === null) return;

    if (!name.trim()) {
      setError("O nome do projeto não pode ficar vazio.");
      return;
    }

    setBusyId(project.project_id);
    setError("");
    setSuccessMessage("");

    try {
      await renameAnalysisProject(
        project.project_id,
        name.trim()
      );

      setSuccessMessage("Projeto renomeado.");
      refresh();
      onChanged?.();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Não foi possível renomear o projeto."
      );
    } finally {
      setBusyId("");
    }
  }

  async function handleDeleteProject(project: AnalysisProject) {
    const confirmed = window.confirm(
      `Excluir o projeto "${project.name}"?\n\nAs análises serão preservadas, mas ficarão sem projeto associado.`
    );

    if (!confirmed) return;

    setBusyId(project.project_id);
    setError("");
    setSuccessMessage("");

    try {
      await deleteAnalysisProject(project.project_id);

      if (projectId === project.project_id) {
        setProjectId("");
        setPage(1);
      }

      setSuccessMessage("Projeto excluído. As análises foram preservadas.");
      refresh();
      onChanged?.();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Não foi possível excluir o projeto."
      );
    } finally {
      setBusyId("");
    }
  }

  async function handleMove(
    item: HistoryItem,
    newProjectId: string
  ) {
    if (!item.analysis_id || item.is_locked) return;

    await runAction(
      item.analysis_id,
      () =>
        moveAnalysisToProject(
          item.analysis_id!,
          newProjectId || null
        ),
      "Projeto da análise atualizado."
    );
  }

  function changeFilter(callback: () => void) {
    callback();
    setPage(1);
  }

  return (
    <section className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-semibold text-white">
              Histórico de análises
            </h2>

            <span className="rounded-full border border-white/10 bg-white/[0.04] px-2.5 py-1 text-xs text-zinc-400">
              {total} registros
            </span>
          </div>

          <p className="mt-1 text-sm text-zinc-500">
            Pesquise, organize e gerencie suas análises salvas.
          </p>
        </div>

        <button
          type="button"
          onClick={handleCreateProject}
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-white/10 bg-white/[0.06] px-4 py-2.5 text-sm font-medium text-zinc-200 transition hover:bg-white/10"
        >
          <span className="text-base">＋</span>
          Novo projeto
        </button>
      </div>

      {error && (
        <div
          role="alert"
          className="rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-300"
        >
          {error}
        </div>
      )}

      {successMessage && (
        <div
          role="status"
          className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-300"
        >
          {successMessage}
        </div>
      )}

      <div className="rounded-xl border border-white/[0.08] bg-[#111318] p-4">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
          <div className="xl:col-span-2">
            <label
              htmlFor="history-search"
              className="mb-2 block text-xs font-medium text-zinc-400"
            >
              Buscar análises
            </label>

            <div className="relative">
              <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500">
                ⌕
              </span>

              <input
                id="history-search"
                type="search"
                value={searchInput}
                onChange={(event) =>
                  setSearchInput(event.target.value)
                }
                placeholder="Pesquisar por título ou pergunta..."
                className="w-full rounded-lg border border-white/10 bg-black/20 py-2.5 pl-9 pr-3 text-sm text-white outline-none transition placeholder:text-zinc-600 focus:border-violet-500/50"
              />
            </div>
          </div>

          <div>
            <label
              htmlFor="history-status"
              className="mb-2 block text-xs font-medium text-zinc-400"
            >
              Status
            </label>

            <select
              id="history-status"
              value={status}
              onChange={(event) =>
                changeFilter(() =>
                  setStatus(event.target.value as "" | AnalysisStatus)
                )
              }
              className="w-full rounded-lg border border-white/10 bg-[#191b22] px-3 py-2.5 text-sm text-zinc-200 outline-none focus:border-violet-500/50"
            >
              <option value="">Todos os status</option>
              <option value="success">Concluídas</option>
              <option value="failure">Falhas</option>
            </select>
          </div>

          <div>
            <label
              htmlFor="history-project"
              className="mb-2 block text-xs font-medium text-zinc-400"
            >
              Projeto
            </label>

            <select
              id="history-project"
              value={projectId}
              onChange={(event) =>
                changeFilter(() => setProjectId(event.target.value))
              }
              className="w-full rounded-lg border border-white/10 bg-[#191b22] px-3 py-2.5 text-sm text-zinc-200 outline-none focus:border-violet-500/50"
            >
              <option value="">Todos os projetos</option>
              <option value="__unassigned__">
                Sem projeto
              </option>

              {projects.map((project) => (
                <option
                  key={project.project_id}
                  value={project.project_id}
                >
                  {project.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="mt-4 flex flex-col gap-3 border-t border-white/[0.06] pt-4 sm:flex-row sm:items-center sm:justify-between">
          <label className="flex cursor-pointer items-center gap-2 text-sm text-zinc-400">
            <input
              type="checkbox"
              checked={includeArchived}
              onChange={(event) =>
                changeFilter(() =>
                  setIncludeArchived(event.target.checked)
                )
              }
              className="h-4 w-4 accent-violet-500"
            />
            Incluir análises arquivadas
          </label>

          <button
            type="button"
            onClick={() => {
              setSearchInput("");
              setSearch("");
              setStatus("");
              setProjectId("");
              setIncludeArchived(false);
              setPage(1);
              setExpandedErrorId("");
            }}
            className="text-left text-xs text-zinc-500 transition hover:text-white sm:text-right"
          >
            Limpar filtros
          </button>
        </div>
      </div>

      {projects.length > 0 && (
        <div className="rounded-xl border border-white/[0.08] bg-[#111318] p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-zinc-200">
                Seus projetos
              </h3>
              <p className="mt-0.5 text-xs text-zinc-500">
                Organize análises por assunto ou finalidade.
              </p>
            </div>

            <span className="text-xs text-zinc-500">
              {projects.length} projeto(s)
            </span>
          </div>

          <div className="flex flex-wrap gap-2">
            {projects.map((project) => (
              <div
                key={project.project_id}
                className="flex items-center gap-2 rounded-lg border border-white/[0.08] bg-white/[0.025] px-3 py-2"
              >
                <button
                  type="button"
                  onClick={() => {
                    setProjectId(project.project_id);
                    setPage(1);
                  }}
                  className="text-left text-sm text-zinc-300 transition hover:text-white"
                  title="Filtrar por este projeto"
                >
                  <span className="mr-1.5 text-violet-400">▦</span>
                  {project.name}
                  <span className="ml-2 text-xs text-zinc-600">
                    {project.analysis_count ?? 0}
                  </span>
                </button>

                <button
                  type="button"
                  disabled={busyId === project.project_id}
                  onClick={() => handleRenameProject(project)}
                  title="Renomear projeto"
                  className="text-xs text-zinc-500 hover:text-white disabled:opacity-40"
                >
                  ✎
                </button>

                <button
                  type="button"
                  disabled={busyId === project.project_id}
                  onClick={() => handleDeleteProject(project)}
                  title="Excluir projeto"
                  className="text-xs text-zinc-500 hover:text-red-400 disabled:opacity-40"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-3">
        {loading ? (
          <div className="rounded-xl border border-white/[0.08] bg-[#111318] px-5 py-12 text-center">
            <div className="mx-auto mb-3 h-6 w-6 animate-spin rounded-full border-2 border-violet-500/20 border-t-violet-400" />
            <p className="text-sm text-zinc-500">
              Carregando histórico...
            </p>
          </div>
        ) : items.length === 0 ? (
          <div className="rounded-xl border border-dashed border-white/10 bg-[#111318] px-5 py-12 text-center">
            <div className="mb-3 text-3xl text-zinc-600">◫</div>
            <h3 className="text-sm font-medium text-zinc-300">
              Nenhuma análise encontrada
            </h3>
            <p className="mt-1 text-xs text-zinc-500">
              Tente alterar os filtros ou realizar uma nova análise.
            </p>
          </div>
        ) : (
          items.map((item) => {
            const id = item.analysis_id || "";
            const locked = Boolean(item.is_locked);
            const archived = Boolean(item.is_archived);
            const pinned = Boolean(item.is_pinned);
            const busy = busyId === id;
            const errorExpanded = expandedErrorId === id;

            const title =
              item.title?.trim() ||
              item.question.slice(0, 90) ||
              "Análise sem título";

            const assignedProject = projects.find(
              (project) => project.project_id === item.project_id
            );

            return (
              <article
                key={id || `${item.timestamp}-${item.question}`}
                className={`rounded-xl border bg-[#111318] p-4 transition ${
                  pinned
                    ? "border-violet-500/30"
                    : "border-white/[0.08]"
                }`}
              >
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0 flex-1">
                      <div className="mb-2 flex flex-wrap items-center gap-2">
                        <span
                          className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${getStatusStyle(item.status, item.partial_success)}`}
                        >
                          {getStatusLabel(item.status, item.partial_success)}
                        </span>

                        {pinned && (
                          <span className="rounded-full border border-violet-500/20 bg-violet-500/10 px-2 py-0.5 text-[10px] text-violet-300">
                            Fixada
                          </span>
                        )}

                        {locked && (
                          <span className="rounded-full border border-amber-500/20 bg-amber-500/10 px-2 py-0.5 text-[10px] text-amber-300">
                            Bloqueada
                          </span>
                        )}

                        {archived && (
                          <span className="rounded-full border border-zinc-500/20 bg-zinc-500/10 px-2 py-0.5 text-[10px] text-zinc-400">
                            Arquivada
                          </span>
                        )}
                      </div>

                      <h3 className="truncate text-sm font-semibold text-zinc-100">
                        {title}
                      </h3>

                      <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-zinc-500">
                        {item.question}
                      </p>

                      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] text-zinc-500">
                        <span>{item.timestamp}</span>

                        {item.provider && (
                          <span>
                            {item.provider}
                            {item.model ? ` · ${item.model}` : ""}
                          </span>
                        )}

                        <span>
                          {formatDuration(item.total_time_ms)}
                        </span>

                        {item.cache_hit && (
                          <span className="text-emerald-500">
                            Cache
                          </span>
                        )}

                        {assignedProject && (
                          <span className="text-violet-300">
                            ▦ {assignedProject.name}
                          </span>
                        )}
                      </div>
                    </div>

                    {item.status === "success" ? (
                      <button
                        type="button"
                        onClick={() => onSelect(item)}
                        className="shrink-0 rounded-lg border border-violet-500/20 bg-violet-500/10 px-3 py-2 text-xs font-medium text-violet-300 transition hover:bg-violet-500/20"
                      >
                        Abrir análise
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() =>
                          setExpandedErrorId((current) =>
                            current === id ? "" : id
                          )
                        }
                        className="shrink-0 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2 text-xs font-medium text-red-300 transition hover:bg-red-500/20"
                      >
                        {errorExpanded ? "Ocultar erro" : "Ver erro"}
                      </button>
                    )}
                  </div>

                  {item.status === "failure" && errorExpanded && (
                    <div className="rounded-lg border border-red-500/20 bg-red-500/[0.05] p-4">
                      <h4 className="text-xs font-semibold text-red-300">
                        Detalhes da falha
                      </h4>

                      <div className="mt-3 space-y-3">
                        <div>
                          <p className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                            Tipo do erro
                          </p>
                          <p className="mt-1 break-words text-xs text-zinc-300">
                            {item.error_type || "Não informado"}
                          </p>
                        </div>

                        <div>
                          <p className="text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                            Mensagem
                          </p>
                          <p className="mt-1 whitespace-pre-wrap break-words text-xs leading-relaxed text-zinc-400">
                            {item.error_message ||
                              "Nenhuma mensagem de erro foi registrada."}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="flex flex-col gap-3 border-t border-white/[0.06] pt-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        type="button"
                        disabled={locked || busy || !id}
                        onClick={() => handleRename(item)}
                        className="rounded-md border border-white/10 px-2.5 py-1.5 text-xs text-zinc-400 transition hover:bg-white/[0.06] hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
                      >
                        Renomear
                      </button>

                      <button
                        type="button"
                        disabled={locked || busy || !id}
                        onClick={() =>
                          runAction(
                            id,
                            () => setAnalysisPinned(id, !pinned),
                            pinned ? "Análise desafixada." : "Análise fixada."
                          )
                        }
                        className="rounded-md border border-white/10 px-2.5 py-1.5 text-xs text-zinc-400 transition hover:bg-white/[0.06] hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
                      >
                        {pinned ? "Desafixar" : "Fixar"}
                      </button>

                      <button
                        type="button"
                        disabled={busy || !id}
                        onClick={() =>
                          runAction(
                            id,
                            () => setAnalysisLocked(id, !locked),
                            locked ? "Análise desbloqueada." : "Análise bloqueada."
                          )
                        }
                        className="rounded-md border border-white/10 px-2.5 py-1.5 text-xs text-zinc-400 transition hover:bg-white/[0.06] hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
                      >
                        {locked ? "Desbloquear" : "Bloquear"}
                      </button>

                      <button
                        type="button"
                        disabled={locked || busy || !id}
                        onClick={() =>
                          runAction(
                            id,
                            () => setAnalysisArchived(id, !archived),
                            archived ? "Análise restaurada." : "Análise arquivada."
                          )
                        }
                        className="rounded-md border border-white/10 px-2.5 py-1.5 text-xs text-zinc-400 transition hover:bg-white/[0.06] hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
                      >
                        {archived ? "Restaurar" : "Arquivar"}
                      </button>

                      <button
                        type="button"
                        disabled={locked || busy || !id}
                        onClick={() => handleDelete(item)}
                        className="rounded-md border border-red-500/15 px-2.5 py-1.5 text-xs text-red-400/80 transition hover:bg-red-500/10 hover:text-red-300 disabled:cursor-not-allowed disabled:opacity-30"
                      >
                        Excluir
                      </button>

                      {busy && (
                        <span className="text-xs text-zinc-600">
                          Atualizando...
                        </span>
                      )}
                    </div>

                    <div className="flex flex-col gap-1 sm:max-w-xs">
                      <label
                        htmlFor={`project-${id}`}
                        className="text-[10px] text-zinc-600"
                      >
                        Organizar em projeto
                      </label>

                      <select
                        id={`project-${id}`}
                        value={item.project_id || ""}
                        disabled={locked || busy || !id}
                        onChange={(event) =>
                          handleMove(item, event.target.value)
                        }
                        className="rounded-lg border border-white/10 bg-[#191b22] px-3 py-2 text-xs text-zinc-300 outline-none focus:border-violet-500/50 disabled:cursor-not-allowed disabled:opacity-30"
                      >
                        <option value="">Sem projeto</option>

                        {projects.map((project) => (
                          <option
                            key={project.project_id}
                            value={project.project_id}
                          >
                            {project.name}
                          </option>
                        ))}
                      </select>
                    </div>

                    {locked && (
                      <p className="text-[11px] text-amber-500/70">
                        Esta análise está bloqueada. Desbloqueie-a para editar, organizar, arquivar ou excluir.
                      </p>
                    )}
                  </div>
                </div>
              </article>
            );
          })
        )}
      </div>

      {!loading && total > 0 && (
        <div className="flex flex-col gap-3 border-t border-white/[0.06] pt-4 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-zinc-500">
            Exibindo{" "}
            {Math.min((page - 1) * PAGE_SIZE + 1, total)}–
            {Math.min(page * PAGE_SIZE, total)} de {total} análises
          </p>

          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={page <= 1}
              onClick={() => setPage((current) => current - 1)}
              className="rounded-lg border border-white/10 px-3 py-2 text-xs text-zinc-300 transition hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-30"
            >
              Anterior
            </button>

            <span className="min-w-20 text-center text-xs text-zinc-500">
              Página {page} de {totalPages}
            </span>

            <button
              type="button"
              disabled={page >= totalPages}
              onClick={() => setPage((current) => current + 1)}
              className="rounded-lg border border-white/10 px-3 py-2 text-xs text-zinc-300 transition hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-30"
            >
              Próxima
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
