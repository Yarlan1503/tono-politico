"""BERTopic sobre list[Argumento]."""

from __future__ import annotations

import logging
from typing import Any

from ..argument_shape.models import Argumento
from .models import ArgumentoTematizado, ResultadoTemas, TopicoInfo

logger = logging.getLogger(__name__)


def descubrir_temas(
    argumentos: list[Argumento],
    embedding_model: Any,
    min_topic_size: int = 3,
    n_neighbors: int = 10,
    n_components: int = 5,
    random_state: int = 42,
) -> ResultadoTemas:
    """Ejecuta BERTopic y devuelve ResultadoTemas con ArgumentoTematizado."""
    from bertopic import BERTopic  # type: ignore[import-not-found]
    from hdbscan import HDBSCAN  # type: ignore[import-not-found]
    from umap import UMAP  # type: ignore[import-not-found]

    textos = [a.texto for a in argumentos]

    if len(textos) < min_topic_size:
        logger.warning(
            "Dataset pequeño: %s argumentos < min_topic_size=%s. Asignando tópico controlado id=0.",
            len(textos),
            min_topic_size,
        )
        return _resultado_topico_unico(argumentos)

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

    logger.info("BERTopic sobre %s argumentos", len(textos))
    topics, probabilities = topic_model.fit_transform(textos)

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
                num_argumentos=num_docs,
                representatividad=num_docs / len(textos),
            )
        )

    tematizados: list[ArgumentoTematizado] = []
    for i, argumento in enumerate(argumentos):
        tid = int(topics[i])
        if probabilities is not None and len(probabilities) > 0:
            prob = _extraer_probabilidad(probabilities, i, tid)
        else:
            prob = 1.0 if tid != -1 else 0.0
        tematizados.append(
            ArgumentoTematizado(argumento=argumento, topico_id=tid, probabilidad=prob)
        )

    num_topicos = len([t for t in topicos if t.id != -1])
    return ResultadoTemas(
        argumentos=tematizados,
        topicos=topicos,
        num_topicos=num_topicos,
    )


def _extraer_probabilidad(probabilities: Any, idx: int, topico_id: int) -> float:
    try:
        import numpy as np

        prob_array = probabilities[idx]
        if hasattr(prob_array, "__len__") and len(prob_array) > 0:
            prob_array = np.asarray(prob_array)
            if 0 <= topico_id < len(prob_array):
                return float(prob_array[topico_id])
            return float(np.max(prob_array))
    except (IndexError, TypeError, ValueError):
        pass
    return 1.0 if topico_id != -1 else 0.0


def _resultado_topico_unico(argumentos: list[Argumento]) -> ResultadoTemas:
    return ResultadoTemas(
        argumentos=[
            ArgumentoTematizado(argumento=a, topico_id=0, probabilidad=1.0) for a in argumentos
        ],
        topicos=[
            TopicoInfo(
                id=0,
                nombre="Tópico único (dataset pequeño)",
                palabras_clave=[],
                num_argumentos=len(argumentos),
                representatividad=1.0,
            )
        ],
        num_topicos=1,
    )
