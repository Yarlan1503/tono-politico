"""DTOs del Componente 5: Tono.

Modela la salida del análisis de tono político con 6 dimensiones:

1. Stance (apoyo/rechazo)        — vía LLM
2. Intensidad antagónica (1-5)   — vía LLM
3. Lógica política (6 labels)    — vía embeddings
4. Sentimiento (5 emociones)     — vía embeddings
5. Estilo discursivo (6 estilos) — vía embeddings
6. Función discursiva (3 labels) — vía embeddings
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..segmentacion.models import Segmento


@dataclass(frozen=True)
class EtiquetaScore:
    """Una etiqueta con su score de similitud (0.0 a 1.0).

    Atributos:
        etiqueta: Nombre de la etiqueta (ej. "populista").
        score: Score de similitud coseno (0.0 a 1.0).
    """

    etiqueta: str
    score: float


@dataclass
class ResultadoLogicaPolitica:
    """Resultado de las 6 dimensiones de lógica política.

    Cada valor es un score de similitud coseno (0.0 a 1.0) que indica
    qué tanto el segmento se alinea con cada lógica política.

    Atributos:
        nacionalista: Defensa de la soberanía frente a intereses extranjeros.
        globalista: Apertura a intereses extranjeros e integración global.
        populista: Alineación con valores del pueblo por encima de expertos.
        tecnocrata: Confianza en expertise técnica sobre voluntad popular.
        corporativista: Apoyo a empresas privadas y mercado libre.
        estatista: Defensa del Estado como rector de la economía.
    """

    nacionalista: float
    globalista: float
    populista: float
    tecnocrata: float
    corporativista: float
    estatista: float

    def to_scores(self) -> list[EtiquetaScore]:
        """Devuelve las 6 etiquetas como lista ordenada de EtiquetaScore."""
        return [
            EtiquetaScore("nacionalista", self.nacionalista),
            EtiquetaScore("globalista", self.globalista),
            EtiquetaScore("populista", self.populista),
            EtiquetaScore("tecnocrata", self.tecnocrata),
            EtiquetaScore("corporativista", self.corporativista),
            EtiquetaScore("estatista", self.estatista),
        ]

    def dominante(self) -> EtiquetaScore:
        """Devuelve la etiqueta con el score más alto."""
        return max(self.to_scores(), key=lambda e: e.score)


@dataclass
class ResultadoSentimiento:
    """Resultado de las 5 emociones políticas detectadas.

    Atributos:
        esperanza: Optimismo, promesa de un mañana mejor.
        angustia: Preocupación, temor, descripción de peligro.
        indignacion: Rabia moral ante una injusticia.
        orgullo: Exaltación de la grandeza nacional o de identidad.
        empatia: Compasión hacia quienes sufren.
    """

    esperanza: float
    angustia: float
    indignacion: float
    orgullo: float
    empatia: float

    def to_scores(self) -> list[EtiquetaScore]:
        return [
            EtiquetaScore("esperanza", self.esperanza),
            EtiquetaScore("angustia", self.angustia),
            EtiquetaScore("indignacion", self.indignacion),
            EtiquetaScore("orgullo", self.orgullo),
            EtiquetaScore("empatia", self.empatia),
        ]

    def dominante(self) -> EtiquetaScore:
        return max(self.to_scores(), key=lambda e: e.score)


@dataclass
class ResultadoEstiloDiscursivo:
    """Resultado de los 6 estilos discursivos.

    Atributos:
        directo: Lenguaje cotidiano, frases cortas, sin adornos.
        academico: Datos, citas, estructura argumentativa formal.
        confrontativo: Provocador, busca el choque frontal.
        conciliador: Diplomático, inclusivo, busca consensos.
        catastrofista: Alarmista, todo es crisis inminente.
        testimonial: Anécdotas, historias de gente común.
    """

    directo: float
    academico: float
    confrontativo: float
    conciliador: float
    catastrofista: float
    testimonial: float

    def to_scores(self) -> list[EtiquetaScore]:
        return [
            EtiquetaScore("directo", self.directo),
            EtiquetaScore("academico", self.academico),
            EtiquetaScore("confrontativo", self.confrontativo),
            EtiquetaScore("conciliador", self.conciliador),
            EtiquetaScore("catastrofista", self.catastrofista),
            EtiquetaScore("testimonial", self.testimonial),
        ]

    def dominante(self) -> EtiquetaScore:
        return max(self.to_scores(), key=lambda e: e.score)


@dataclass
class ResultadoFuncionDiscursiva:
    """Resultado de las 3 funciones discursivas.

    Atributos:
        critica: El discurso ataca, denuncia o señala responsables.
        propuesta: El discurso ofrece soluciones o plantea alternativas.
        narrativa_personal: El discurso construye la imagen del político.
    """

    critica: float
    propuesta: float
    narrativa_personal: float

    def to_scores(self) -> list[EtiquetaScore]:
        return [
            EtiquetaScore("critica", self.critica),
            EtiquetaScore("propuesta", self.propuesta),
            EtiquetaScore("narrativa_personal", self.narrativa_personal),
        ]

    def dominante(self) -> EtiquetaScore:
        return max(self.to_scores(), key=lambda e: e.score)


@dataclass
class ResultadoStance:
    """Resultado del análisis de stance sobre un tema específico.

    Atributos:
        stance: "apoyo" o "rechazo" respecto al tema evaluado.
        confianza: Score de confianza (0.0 a 1.0).
    """

    stance: str
    confianza: float


@dataclass
class SegmentoConTono:
    """Segmento enriquecido con su análisis de tono completo.

    Atributos:
        segmento: Segmento original del Componente 2.
        stance: Posición (apoyo/rechazo) sobre el tema evaluado.
        intensidad_antagonica: Nivel 1-5 de confrontación.
        logica_politica: Scores de las 6 lógicas políticas.
        sentimiento: Scores de las 5 emociones políticas.
        estilo_discursivo: Scores de los 6 estilos.
        funcion_discursiva: Scores de las 3 funciones.
    """

    segmento: Segmento
    stance: ResultadoStance
    intensidad_antagonica: int
    logica_politica: ResultadoLogicaPolitica
    sentimiento: ResultadoSentimiento
    estilo_discursivo: ResultadoEstiloDiscursivo
    funcion_discursiva: ResultadoFuncionDiscursiva


@dataclass
class ResultadoTono:
    """Salida completa del Componente 5: Tono.

    Atributos:
        tema: Tema/objetivo evaluado (ej. "fracking").
        actor: Nombre del actor político analizado (ej. "AMLO").
        segmentos: Lista de segmentos con su análisis de tono.
    """

    tema: str
    actor: str
    segmentos: list[SegmentoConTono] = field(default_factory=list)
