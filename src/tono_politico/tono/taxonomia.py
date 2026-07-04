"""Taxonomía de prototipos para análisis de tono político.

Cada prototipo es una descripción rica (multi-oración) que ejemplifica un label.
El sistema de embeddings compara el texto del segmento contra cada prototipo
mediante similitud coseno: a mayor similitud, más presente está esa dimensión.

Diseño:
- Los prototipos son lo suficientemente largos (>30 chars) para dar contexto rico.
- Cada label de intensidad describe un nivel claramente distinto.
- Las etiquetas en español se mapean a los DTOs de models.py.
"""

from __future__ import annotations

# Dimensión Stance: binaria, evaluada por LLM (no por embeddings).
# Se declara aquí para centralizar todo el vocabulario de labels.
STANCE_LABELS: list[str] = ["apoyo", "rechazo"]


# Lista de las dimensiones evaluadas por embeddings (no por LLM).
DIMENSIONES_EMBEDDINGS: list[str] = [
    "logica_politica",
    "sentimiento",
    "estilo_discursivo",
    "funcion_discursiva",
    "intensidad",
]


# ============================================================
# PROTOTIPOS — LOGICA POLITICA
# ============================================================
_PROTO_LOGICA: dict[str, str] = {
    "nacionalista": (
        "El discurso defiende con firmeza la soberanía nacional frente a intereses extranjeros. "
        "Resalta que los recursos naturales, la energía y el patrimonio pertenecen exclusivamente "
        "a la nación y no deben ser controlados por potencias externas ni empresas trasnacionales. "
        "Apela al orgullo nacional, a la independencia económica y a la autodeterminación del pueblo."
    ),
    "globalista": (
        "El discurso favorece la apertura a la inversión extranjera, el libre comercio y la "
        "integración económica global. Promueve la cooperación internacional, los tratados comerciales "
        "y la interdependencia entre países como motor del desarrollo. Ve la entrada de capital "
        "internacional y la apertura de mercados como algo positivo y necesario."
    ),
    "populista": (
        "El discurso alinea su visión política con los valores, las necesidades y la voluntad del "
        "pueblo trabajador. El pueblo es el sujeto moral principal: las decisiones deben servir "
        "a las mayorías, no a las élites ni a los tecnócratas. Defiende que el bienestar del pueblo "
        "está por encima de cualquier consideración técnica, académica o corporativa."
    ),
    "tecnocrata": (
        "El discurso confía en la expertise técnica, científica y académica como base fundamental "
        "para las decisiones públicas. Recurre a estudios, datos, estadísticas y comités de expertos "
        "para sustentar las políticas. Sostiene que el conocimiento especializado debe guiar al Estado "
        "por encima de la opinión popular o la presión política."
    ),
    "corporativista": (
        "El discurso muestra un respaldo explícito a las empresas privadas, el mercado libre "
        "y los intereses corporativos. Promueve la inversión privada, la desregulación, "
        "la iniciativa empresarial y la competencia como motores del desarrollo económico. "
        "Defiende que las empresas generan riqueza, empleo y progreso para el país."
    ),
    "estatista": (
        "El discurso defiende al Estado como rector de la economía y la vida pública. Sostiene "
        "que el sector público debe controlar los recursos estratégicos, las industrias clave "
        "y los servicios esenciales. Las instituciones del Estado deben regular, dirigir "
        "y garantizar el bienestar colectivo por encima del mercado."
    ),
}


# ============================================================
# PROTOTIPOS — SENTIMIENTO (emociones políticas)
# ============================================================
_PROTO_SENTIMIENTO: dict[str, str] = {
    "esperanza": (
        "El discurso transmite optimismo y fe en el futuro. Promete un mañana mejor, habla de "
        "transformación positiva, de posibilidades y de un horizonte luminoso. Inspira confianza "
        "en que las cosas van a mejorar para el pueblo y para el país con las políticas propuestas."
    ),
    "angustia": (
        "El discurso transmite preocupación, temor y angustia genuina. Describe situaciones de "
        "peligro, deterioro o amenaza. Genera ansiedad sobre el presente o el futuro del país. "
        "El tono refleja sufrimiento, zozobra y una sensación de riesgo real e inminente."
    ),
    "indignacion": (
        "El discurso transmite indignación y rabia moral ante una injusticia. Hay furia contenida "
        "frente a un atropello, un abuso de poder o una traición al pueblo. Expresa cólera legítima "
        "y exige que los responsables rindan cuentas. No es tristeza: es active moral outrage."
    ),
    "orgullo": (
        "El discurso transmite orgullo nacional, cultural o de identidad. Exalta la grandeza del "
        "país y la dignidad de su pueblo. Celebra logros colectivos, hazañas históricas y la "
        "fortaleza de la nación. Habla de honor, de pertenencia y de la valía del pueblo."
    ),
    "empatia": (
        "El discurso transmite empatía y compasión hacia quienes sufren. Reconoce el dolor de "
        "los demás, se pone en el lugar del vulnerable y conecta emocionalmente con quienes "
        "han sido marginados o lastimados. Escucha y valida la experiencia del pueblo."
    ),
}


# ============================================================
# PROTOTIPOS — ESTILO DISCURSIVO
# ============================================================
_PROTO_ESTILO: dict[str, str] = {
    "directo": (
        "El político habla de manera llana y directa, como habla la gente común en la calle. "
        "Usa frases cortas, lenguaje cotidiano y llama a las cosas por su nombre sin adornos "
        "ni circunloquios. No usa tecnicismos ni jerga académica: es claro, crudo y sin filtro."
    ),
    "academico": (
        "El político recurre a un lenguaje formal y estructurado. Cita estudios, presenta datos, "
        "estadísticas y evidencia rigurosa para sustentar sus argumentos. Construye una "
        "argumentación metódica con referencias técnicas, informes institucionales y terminología "
        "especializada propia de expertos en la materia."
    ),
    "confrontativo": (
        "El político busca el choque frontal y la provocación. Ataca a sus oponentes de manera "
        "agresiva y desafiante, sin titubeos ni diplomacia. Confronta directamente, señala "
        "enemigos por nombre y usa el conflicto como herramienta política para movilizar a su base."
    ),
    "conciliador": (
        "El político usa un tono diplomático, inclusivo y constructivo. Busca puentes y consensos "
        "entre sectores enfrentados. Reconoce la validez de diferentes posiciones, convoca a la "
        "unidad nacional y su lenguaje une en lugar de dividir. Prioriza el diálogo sobre el conflicto."
    ),
    "catastrofista": (
        "El político presenta la situación como una crisis apocalíptica e irreversible. Todo "
        "está al borde del colapso: la economía, la seguridad, las instituciones. Usa un lenguaje "
        "alarmista que sugiere que el desastre es inminente y solo una intervención radical puede "
        "evitar la catástrofe definitiva."
    ),
    "testimonial": (
        "El político recurre a anécdotas personales e historias de gente común. Cita testimonios "
        "de ciudadanos anónimos que conoció en sus recorridos. Usa narrativas individuales y "
        "vivencias concretas para ilustrar problemas generales. Humaniza la política poniendo "
        "rostros y nombres a las estadísticas."
    ),
}


# ============================================================
# PROTOTIPOS — FUNCION DISCURSIVA
# ============================================================
_PROTO_FUNCION: dict[str, str] = {
    "critica": (
        "El discurso cumple la función de atacar, denunciar o señalar responsables. Critica "
        "duramente a adversarios políticos, instituciones, gobiernos anteriores o políticas "
        "específicas. Es un acto de acusación: expone fracasos, corrupción o traiciones "
        "para desacreditar al oponente."
    ),
    "propuesta": (
        "El discurso cumple la función de ofrecer soluciones, plantear alternativas o presentar "
        "un plan de acción concreto. Propone medidas, programas o reformas para resolver "
        "problemas. Es un acto constructivo que mira hacia adelante y dice qué se va a hacer."
    ),
    "narrativa_personal": (
        "El discurso cumple la función de construir la imagen personal del político. Presenta "
        "su trayectoria, sus valores, su identidad política y su carácter de liderazgo. "
        "Es un acto de construcción de marca personal: el político se muestra como héroe, "
        "transformador o salvador de la nación."
    ),
}


# ============================================================
# PROTOTIPOS — INTENSIDAD ANTIGONICA (5 niveles)
# ============================================================
_PROTO_INTENSIDAD: dict[str, str] = {
    "1": (
        "Tono conciliador, propositivo y colaborativo. No hay confrontación alguna. "
        "El político reconoce la validez de diferentes perspectivas y busca diálogo abierto. "
        "Su lenguaje es mesurado, respetuoso y constructivo. Invita al consenso y evita señalar "
        "culpables. La atmósfera es de cooperación."
    ),
    "2": (
        "Tono firme pero respetuoso. El político disiente claramente de sus oponentes pero "
        "reconoce su legitimidad democrática. Argumenta con solidez sin caer en la agresividad "
        "personal. Hay desacuerdo claro expresado de forma civil y ordenada."
    ),
    "3": (
        "Tono combativo. El político señala responsables y critica con dureza, pero sin "
        "agresión personal directa ni insultos. Hay una disputa ideológica clara y frontal. "
        "Se enfrentan ideas y políticas con energía, manteniendo un límite en la decencia verbal."
    ),
    "4": (
        "Tono confrontacional. El político ataca directamente a adversarios identificados por "
        "nombre o por grupo. Señala culpables con dureza, los acusa de forma explícita y no "
        "busca diálogo alguno. Quiere someter y desacreditar al oponente de forma contundente."
    ),
    "5": (
        "Tono beligerante y maximalista. El político divide el mundo entre buenos y malos "
        "con una enemistad existencial. El adversario es un enemigo absoluto que debe ser "
        "derrotado a toda costa. Lenguaje inflamatorio, acusaciones graves, llamado a la "
        "movilización. No hay punto medio ni reconciliación posible."
    ),
}


# ============================================================
# REGISTRO CENTRAL
# ============================================================
_PROTO_PROTOTIPOS: dict[str, dict[str, str]] = {
    "logica_politica": _PROTO_LOGICA,
    "sentimiento": _PROTO_SENTIMIENTO,
    "estilo_discursivo": _PROTO_ESTILO,
    "funcion_discursiva": _PROTO_FUNCION,
    "intensidad": _PROTO_INTENSIDAD,
}


def prototipos_de(dimension: str) -> dict[str, str]:
    """Devuelve el diccionario de prototipos para una dimensión.

    Args:
        dimension: Una de: "logica_politica", "sentimiento",
            "estilo_discursivo", "funcion_discursiva", "intensidad".

    Returns:
        Dict[label, texto_prototipo].

    Raises:
        KeyError: Si la dimensión no existe.
    """
    return _PROTO_PROTOTIPOS[dimension]


def todas_las_dimensiones() -> list[str]:
    """Devuelve los nombres de todas las dimensiones con prototipos."""
    return list(_PROTO_PROTOTIPOS.keys())
