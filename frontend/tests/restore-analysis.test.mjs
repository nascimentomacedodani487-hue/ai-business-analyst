import assert from "node:assert/strict";
import { test } from "node:test";

import { restoreAnalysis } from "../lib/analysis-history.ts";

function makeDetail(overrides = {}) {
  return {
    analysis_id: "analysis-test-001",
    created_at: "2026-11-08T12:15:00",
    question: "Analise o desempenho do marketplace.",
    status: "success",
    provider: "gemini",
    model: "gemini-2.5-flash",
    total_time_ms: 10940,
    cache_hit: false,
    steps: [],
    queries: [],
    ...overrides,
  };
}

function makeQuery(id, status, failureType = null) {
  return {
    id,
    objective: `Consulta ${id}`,
    question: `Pergunta ${id}`,
    status,
    failure_type: failureType,
    sql: null,
    results: [],
    total_results: 0,
    chart: null,
    error: null,
    metrics: {},
  };
}

test("restaura metricas de uma analise simples", () => {
  const detail = makeDetail({
    steps: [
      {
        name: "sql_generation",
        status: "completed",
        duration_ms: 1200,
      },
      {
        name: "sql_execution",
        status: "completed",
        duration_ms: 300,
      },
      {
        name: "response_generation",
        status: "completed",
        duration_ms: 800,
      },
    ],
  });

  const restored = restoreAnalysis(detail);

  assert.equal(restored.metrics.sql_generation_ms, 1200);
  assert.equal(restored.metrics.sql_execution_ms, 300);
  assert.equal(restored.metrics.response_generation_ms, 800);
  assert.equal(restored.metrics.total_time_ms, 10940);
  assert.deepEqual(restored.steps, detail.steps);
});

test("soma metricas de consultas independentes e recupera a sintese", () => {
  const detail = makeDetail({
    steps: [
      {
        name: "database_version",
        status: "completed",
        duration_ms: 1,
      },
      {
        name: "query_planning",
        status: "completed",
        duration_ms: 3160,
      },
      {
        name: "query_1_sql_generation",
        status: "completed",
        duration_ms: 956,
      },
      {
        name: "query_1_sql_validation",
        status: "completed",
        duration_ms: 14,
      },
      {
        name: "query_1_sql_execution",
        status: "completed",
        duration_ms: 170,
      },
      {
        name: "query_2_sql_generation",
        status: "completed",
        duration_ms: 1070,
      },
      {
        name: "query_2_sql_execution",
        status: "completed",
        duration_ms: 115,
      },
      {
        name: "query_3_capability_validation",
        status: "unsupported",
        duration_ms: 1,
      },
      {
        name: "multi_response_generation",
        status: "completed",
        duration_ms: 3340,
      },
    ],
    queries: [
      makeQuery("query-1", "success"),
      makeQuery("query-2", "success"),
      makeQuery(
        "query-3",
        "failure",
        "unsupported_capability"
      ),
    ],
  });

  const restored = restoreAnalysis(detail);

  assert.equal(restored.metrics.sql_generation_ms, 2026);
  assert.equal(restored.metrics.sql_execution_ms, 285);
  assert.equal(restored.metrics.response_generation_ms, 3340);

  assert.equal(restored.query_count, 3);
  assert.equal(restored.successful_queries, 2);
  assert.equal(restored.unsupported_queries, 1);
  assert.equal(restored.failed_queries, 0);
  assert.equal(restored.partial_success, true);
  assert.equal(restored.analysis_mode, "multi_query");
});

test("nao inventa metricas quando as etapas correspondentes nao existem", () => {
  const restored = restoreAnalysis(
    makeDetail({
      steps: [
        {
          name: "database_version",
          status: "completed",
          duration_ms: 1,
        },
      ],
    })
  );

  assert.equal(restored.metrics.sql_generation_ms, undefined);
  assert.equal(restored.metrics.sql_execution_ms, undefined);
  assert.equal(restored.metrics.response_generation_ms, undefined);
  assert.equal(restored.metrics.total_time_ms, 10940);
});

test("mantem compatibilidade com historico antigo sem etapas", () => {
  const restored = restoreAnalysis(
    makeDetail({
      steps: undefined,
      queries: undefined,
    })
  );

  assert.deepEqual(restored.steps, []);
  assert.deepEqual(restored.queries, []);
  assert.equal(restored.metrics.total_time_ms, 10940);
  assert.equal(restored.metrics.cache_hit, false);
  assert.equal(restored.metrics.sql_generation_ms, undefined);
  assert.equal(restored.analysis_mode, "single_query");
});

test("rejeita a restauracao de uma analise que falhou", () => {
  const detail = makeDetail({
    status: "failure",
    error_message: "Falha simulada.",
  });

  assert.throws(
    () => restoreAnalysis(detail),
    /Falha simulada/
  );
});
