"""
Cache em memória para armazenar análises de negócio.
"""

from collections import OrderedDict
from copy import deepcopy
from time import monotonic


class AnalysisCache:
    """Armazena resultados de análises temporariamente."""

    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_entries: int = 100,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._cache = OrderedDict()

    def _build_key(
        self,
        question: str,
        data_version: str,
        provider: str = "gemini",
        model: str = "default",
    ) -> tuple:
        """Cria uma chave isolada por pergunta, dados e modelo."""

        normalized_question = " ".join(question.strip().lower().split())

        normalized_provider = provider.strip().lower()
        normalized_model = model.strip().lower()

        # Compatibilidade com registros antigos do cache Gemini.
        if normalized_provider == "gemini" and normalized_model == "default":
            normalized_model = "gemini-2.5-flash"

        return (
            normalized_question,
            data_version,
            normalized_provider,
            normalized_model,
        )

    def get(
        self,
        question: str,
        data_version: str,
        provider: str = "gemini",
        model: str = "default",
    ) -> dict | None:
        """Recupera uma análise válida do cache."""

        key = self._build_key(
            question,
            data_version,
            provider,
            model,
        )

        if key not in self._cache:
            return None

        result, created_at = self._cache[key]

        # Verificar se o resultado expirou.
        if monotonic() - created_at > self.ttl_seconds:
            del self._cache[key]
            return None

        # Atualizar a posição do item para indicar uso recente.
        self._cache.move_to_end(key)

        return deepcopy(result)

    def set(
        self,
        question: str,
        data_version: str,
        result: dict,
        provider: str = "gemini",
        model: str = "default",
    ) -> None:
        """Armazena uma análise no cache."""

        key = self._build_key(
            question,
            data_version,
            provider,
            model,
        )

        self._cache[key] = (deepcopy(result), monotonic())
        self._cache.move_to_end(key)

        # Remover o item mais antigo quando atingir o limite.
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Limpa todas as análises armazenadas."""

        self._cache.clear()
