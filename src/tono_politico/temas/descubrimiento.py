"""Ejecución de BERTopic y extracción de metadata de tópicos.

Encapsula la interacción con BERTopic en una función pura que:
    1. Configura el modelo (UMAP, HDBSCAN, embeddings LFM2.5)
    2. Ejecuta fit_transform sobre los textos
    3. Extrae tópicos, palabras clave y probabilidades
"""

from __future__ import annotations

import logging
from typing import Any

from ..segmentacion.models import Segmento
from .models import ResultadoTemas, SegmentoTematizado, TopicoInfo

logger = logging.getLogger(__name__)


def descubrir_temas(
    segmentos: list[Segmento],
    embedding_model: Any,
    min_topic_size: int = 3,
    n_neighbors: int = 10,
    n_components: int = 5,
    random_state: int = 42,
) -> ResultadoTemas:
    """Ejecuta BERTopic sobre los segmentos y devuelve resultados estructurados.

    Args:
        segmentos: Lista de Segmento del Componente 2.
        embedding_model: Instancia de sentence-transformers (LFM2.5-Embedding-350M).
        min_topic_size: Mínimo de documentos por tópico (HDBSCAN min_cluster_size).
            Si ``len(segmentos) < min_topic_size``, todos van a un único tópico
            controlado con warning estructurado.
        n_neighbors: UMAP n_neighbors.
        n_components: UMAP n_components (dimensionalidad de reducción).
        random_state: Semilla para reproducibilidad de UMAP (default: 42).

    Returns:
        ResultadoTemas con segmentos tematizados, tópicos y metadata.
    """
    from bertopic import BERTopic  # type: ignore[import-not-found]
    from hdbscan import HDBSCAN  # type: ignore[import-not-found]
    from umap import UMAP  # type: ignore[import-not-found]

    textos = [s.texto for s in segmentos]

    if len(textos) < min_topic_size:
        logger.warning(
            f"Dataset pequeño: {len(textos)} segmentos < min_topic_size={min_topic_size}. "
            "Asignando todos a un único tópico controlado (id=0)."
        )
        return _resultado_topico_unico(segmentos)

    # Configurar componentes del pipeline de BERTopic
    umap_model = UMAP(
        n_neighbors=min(n_neighbors, len(textos) - 1),
        n_components=min(n_components, len(textos) - 1),
        metric="cosine",
        random_state=random_state,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min(min_topic_size, len(textos)),
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="spanish",
        calculate_probabilities=False,
        verbose=False,
    )

    logger.info(f"Ejecutando BERTopic sobre {len(textos)} segmentos")

    topics, probabilities = topic_model.fit_transform(textos)

    # Extraer metadata de tópicos
    topic_info_df = topic_model.get_topic_info()
    topicos: list[TopicoInfo] = []

    for _, row in topic_info_df.iterrows():
        tid = int(row["Topic"])
        num_docs = int(row["Count"])

        if tid == -1:
            palabras: list[str] = []
            nombre = "Outlier"
        else:
            palabras_raw = topic_model.get_topic(tid)
            if isinstance(palabras_raw, list):
                palabras = [
                    str(p[0])
                    for p in list(palabras_raw)[:10]  # type: ignore[arg-type]
                    if isinstance(p, (list, tuple)) and len(p) > 0
                ]
            else:
                palabras = []
            nombre = str(row.get("Name", f"Tópico_{tid}"))

        topicos.append(
            TopicoInfo(
                id=tid,
                nombre=nombre,
                palabras_clave=palabras,
                num_segmentos=num_docs,
                representatividad=num_docs / len(textos),
            )
        )

    # Construir segmentos tematizados
    segmentos_tematizados: list[SegmentoTematizado] = []
    for i, segmento in enumerate(segmentos):
        tid = int(topics[i])

        if probabilities is not None and len(probabilities) > 0:
            # BERTopic devuelve probabilidades como array [N, num_topics]
            # o como lista de probs por documento
            prob = _extraer_probabilidad(probabilities, i, tid, topic_model)
        else:
            prob = 1.0 if tid != -1 else 0.0

        segmentos_tematizados.append(
            SegmentoTematizado(
                segmento=segmento,
                topico_id=tid,
                probabilidad=prob,
            )
        )

    num_topicos = len([t for t in topicos if t.id != -1])

    logger.info(f"BERTopic descubrió {num_topicos} tópicos de {len(textos)} segmentos")

    return ResultadoTemas(
        segmentos=segmentos_tematizados,
        topicos=topicos,
        num_topicos=num_topicos,
    )


def _extraer_probabilidad(
    probabilities: Any,
    idx: int,
    topico_id: int,
    topic_model: Any,
) -> float:
    """Extrae la probabilidad de un documento para su tópico asignado."""
    try:
        import numpy as np

        prob_array = probabilities[idx]

        if hasattr(prob_array, "__len__") and len(prob_array) > 0:
            prob_array = np.asarray(prob_array)
            if topico_id >= 0 and topico_id < len(prob_array):
                return float(prob_array[topico_id])
            return float(np.max(prob_array))
    except (IndexError, TypeError, ValueError):
        pass

    return 1.0 if topico_id != -1 else 0.0


def _resultado_topico_unico(segmentos: list[Segmento]) -> ResultadoTemas:
    """Construye un ResultadoTemas con un único tópico controlado (id=0).

    Se usa cuando no hay suficientes segmentos para BERTopic.
    Todos los segmentos se asignan al tópico 0 en vez de outlier (-1),
    para que el pipeline pueda filtrarlos y analizarlos.
    """
    return ResultadoTemas(
        segmentos=[
            SegmentoTematizado(
                segmento=s,
                topico_id=0,
                probabilidad=1.0,
            )
            for s in segmentos
        ],
        topicos=[
            TopicoInfo(
                id=0,
                nombre="Tópico único (dataset pequeño)",
                palabras_clave=[],
                num_segmentos=len(segmentos),
                representatividad=1.0,
            )
        ],
        num_topicos=1,
    )
