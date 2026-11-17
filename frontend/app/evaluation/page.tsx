
import type { Metadata } from "next";
import Link from "next/link";

import BenchmarkHistory from "@/components/evaluation/benchmark-history";

export const metadata: Metadata = {
  title: "Evaluation Engine | AI Business Analyst",
  description:
    "Acompanhe benchmarks, inspecione resultados e compare o desempenho histórico dos provedores de IA."
};

export default function EvaluationPage() {
  return (
    <main className="min-h-screen bg-[#090b10] text-zinc-100">
      <div className="mx-auto w-full max-w-[1600px] px-4 py-8 sm:px-6 lg:px-8">
        <header className="mb-8">
          <div className="mb-6">
            <Link
              href="/"
              className="inline-flex items-center gap-2 rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-2 text-xs text-zinc-400 transition hover:border-white/[0.14] hover:bg-white/[0.05] hover:text-zinc-200"
            >
              <span aria-hidden="true">←</span>
              Voltar ao workspace
            </Link>
          </div>

          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="mb-3 flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-violet-400" />

                <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-violet-300">
                  AI Quality & Benchmarking
                </span>
              </div>

              <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                Evaluation Engine
              </h1>

              <p className="mt-3 max-w-2xl text-sm leading-7 text-zinc-400">
                Monitore a qualidade das análises, consulte resultados
                persistidos e compare benchmarks históricos para identificar
                melhorias, regressões e mudanças de desempenho nos provedores
                de IA.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-2 rounded-full border border-emerald-500/15 bg-emerald-500/[0.05] px-3 py-1.5 text-[10px] text-emerald-300">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                Histórico persistente
              </span>

              <span className="rounded-full border border-white/[0.08] bg-white/[0.025] px-3 py-1.5 text-[10px] text-zinc-500">
                Evaluation Engine
              </span>
            </div>
          </div>
        </header>

        <section
          aria-label="Histórico e comparação de benchmarks"
          className="min-w-0"
        >
          <BenchmarkHistory />
        </section>

        <footer className="mt-10 border-t border-white/[0.06] pt-5">
          <div className="flex flex-col gap-2 text-[10px] leading-5 text-zinc-600 sm:flex-row sm:items-center sm:justify-between">
            <p>
              AI Business Analyst · Evaluation Engine
            </p>

            <p>
              Resultados persistidos e comparação histórica
            </p>
          </div>
        </footer>
      </div>
    </main>
  );
}
