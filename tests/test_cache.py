
from ai_analyst.agent.cache import AnalysisCache


def test_cache_stores_and_retrieves_analysis():
    cache = AnalysisCache()

    result = {"answer": "Foram encontrados 100 pedidos."}

    cache.set("Quantos pedidos tivemos?", "v1", result)

    cached_result = cache.get("Quantos pedidos tivemos?", "v1")

    assert cached_result == result


def test_cache_normalizes_questions():
    cache = AnalysisCache()

    result = {"answer": "Análise concluída."}

    cache.set("  Quantos pedidos tivemos?  ", "v1", result)

    cached_result = cache.get("quantos pedidos tivemos?", "v1")

    assert cached_result == result


def test_cache_respects_data_version():
    cache = AnalysisCache()

    result = {"answer": "100 pedidos"}

    cache.set("Quantos pedidos?", "v1", result)

    assert cache.get("Quantos pedidos?", "v2") is None


def test_cache_returns_independent_copy():
    cache = AnalysisCache()

    result = {"answer": "Resultado original"}

    cache.set("Minha pergunta", "v1", result)

    cached_result = cache.get("Minha pergunta", "v1")
    cached_result["answer"] = "Resultado alterado"

    assert cache.get("Minha pergunta", "v1")["answer"] == "Resultado original"


def test_cache_respects_max_entries():
    cache = AnalysisCache(max_entries=2)

    cache.set("Pergunta 1", "v1", {"answer": "1"})
    cache.set("Pergunta 2", "v1", {"answer": "2"})
    cache.set("Pergunta 3", "v1", {"answer": "3"})

    assert cache.get("Pergunta 1", "v1") is None
    assert cache.get("Pergunta 2", "v1") is not None
    assert cache.get("Pergunta 3", "v1") is not None


def test_cache_expires_results(monkeypatch):
    import ai_analyst.agent.cache as cache_module

    current_time = 100.0

    monkeypatch.setattr(
        cache_module,
        "monotonic",
        lambda: current_time,
    )

    cache = AnalysisCache(ttl_seconds=30)

    cache.set("Minha pergunta", "v1", {"answer": "Resultado"})

    current_time += 31

    assert cache.get("Minha pergunta", "v1") is None
