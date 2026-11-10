"use client";

import { useState } from "react";
import type { ReactNode } from "react";

type ResultsTableProps = {
  rows: Record<string, unknown>[];
  totalResults?: number;
  formatValue: (value: unknown) => ReactNode;
  pageSize?: number;
};

export default function ResultsTable({
  rows,
  totalResults,
  formatValue,
  pageSize = 10,
}: ResultsTableProps) {
  const [pagination, setPagination] = useState({
    rows,
    page: 1,
  });

  // Reinicia a página quando uma nova lista de resultados é recebida.
  // A página é derivada durante a renderização, sem precisar de efeito.
  const currentPage =
    pagination.rows === rows ? pagination.page : 1;

  const safePageSize =
    Number.isFinite(pageSize) && pageSize > 0
      ? Math.floor(pageSize)
      : 10;

  const totalPages = Math.max(
    1,
    Math.ceil(rows.length / safePageSize)
  );

  const visiblePage = Math.min(currentPage, totalPages);
  const startIndex = (visiblePage - 1) * safePageSize;
  const endIndex = Math.min(
    startIndex + safePageSize,
    rows.length
  );

  const visibleRows = rows.slice(startIndex, endIndex);

  const resultSummary =
    typeof totalResults === "number" &&
    totalResults !== rows.length
      ? `${rows.length} carregados de ${totalResults} resultados`
      : `${rows.length} resultados`;

  return (
    <div className="mt-4 space-y-3">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-white/[0.08] text-zinc-500">
              {Object.keys(rows[0] ?? {}).map((column) => (
                <th
                  key={column}
                  className="whitespace-nowrap px-3 py-3 font-medium"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {visibleRows.map((row, rowIndex) => (
              <tr
                key={startIndex + rowIndex}
                className="border-b border-white/[0.04] text-zinc-300"
              >
                {Object.values(row).map((value, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="whitespace-nowrap px-3 py-3"
                  >
                    {formatValue(value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-[10px] text-zinc-500">
          Exibindo {rows.length > 0 ? startIndex + 1 : 0}–{endIndex} de{" "}
          {resultSummary}.
        </p>

        {totalPages > 1 && (
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() =>
                setPagination({
                  rows,
                  page: Math.max(1, visiblePage - 1),
                })
              }
              disabled={visiblePage === 1}
              className="rounded-lg border border-white/[0.08] px-3 py-1.5 text-[10px] text-zinc-300 transition hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-30"
            >
              Anterior
            </button>

            <span className="text-[10px] tabular-nums text-zinc-400">
              Página {visiblePage} de {totalPages}
            </span>

            <button
              type="button"
              onClick={() =>
                setPagination({
                  rows,
                  page: Math.min(totalPages, visiblePage + 1),
                })
              }
              disabled={visiblePage === totalPages}
              className="rounded-lg border border-white/[0.08] px-3 py-1.5 text-[10px] text-zinc-300 transition hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-30"
            >
              Próxima
            </button>
          </div>
        )}
      </div>
    </div>
  );
}