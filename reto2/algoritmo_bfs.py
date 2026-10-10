r"""
algoritmo_bfs.py
================
Heurística voraz local para el Reto 2.

Notación formal según el documento del Reto 2 y variables binarias:
- Q_t: Vértices quemados acumulados hasta el turno t (en el documento, B_{t-1}).
- S_t: Subconjunto de vértices susceptibles o sanos (S_t = V \ Q_t).
- I_t: Frente activo de fuego (vértices recién quemados N_{t-1}).
- C_t: Calendario acumulado de cortes por cortafuegos (C_{t-1} subconjunto de C_t).
- delta(I_t, S_t): Cociclo frontera saliente entre el fuego activo y la zona susceptible.
- k: Presupuesto máximo de aristas a cortar por turno (k = 2).
- radio: Profundidad máxima r de exploración local en el subgrafo susceptible.

Fundamento metodológico:
Resuelve un problema de asignación de tipo mochila discreta para k = 2 sobre las aristas
del cociclo delta(I_t, S_t), guiado por un ratio beneficio/coste obtenido mediante una
exploración en anchura (BFS) acotada a radio r dentro del subgrafo inducido G[S_t].
"""

import collections


def estrategia_bfs_local(adj, Q_t, I_t, C_t, k=2, radio=2):
    """
    Selecciona hasta k cortafuegos en el turno t mediante una heurística voraz
    basada en la conectividad local acotada (radio r) en el subgrafo susceptible S_t.

    Parámetros:
      adj: dict {nodo: iterable/set(vecinos)} con la lista de adyacencia del grafo G = (V, E).
      Q_t: set con los vértices quemados acumulados hasta el momento (Q_t = B_{t-1}).
      I_t: iterable/set con los vértices que arden novedosamente en el turno actual (frente N_{t-1}).
      C_t: set de tuplas canónicas (u, v) con u < v representando aristas ya cortadas (C_{t-1}).
      k: int, presupuesto máximo de cortafuegos por etapa (por defecto k=2).
      radio: int, profundidad máxima r de exploración BFS desde cada candidato susceptible.

    Retorna:
      list de hasta k tuplas (u, v) con las aristas seleccionadas para ser cortadas.
    """
    # 1. IDENTIFICACIÓN DEL CONJUNTO SUSCEPTIBLE (S_t = V \ Q_t)
    # Solo consideramos vértices que aún no han sido alcanzados por el fuego.
    S_t = set(adj.keys()) - Q_t
    candidatos_aristas = {}

    # 2. DELIMITACIÓN DEL FRENTE ACTIVO (Cociclo delta(I_t, S_t))
    # Por optimalidad local, los cortafuegos deben ubicarse en aristas que comunican
    # directamente el frente de fuego I_t con la zona susceptible S_t.
    # Se agrupan las aristas incidentes por cada nodo de destino susceptible:
    # candidatos_aristas[destino] = [aristas del frente I_t que entran a 'destino'].
    for origen in I_t:
        for destino in adj[origen]:
            arista = (origen, destino) if origen < destino else (destino, origen)
            if destino in S_t and arista not in C_t:
                candidatos_aristas.setdefault(destino, []).append(arista)

    # Si el frente de fuego ya no tiene caminos abiertos hacia nodos sanos,
    # el incendio está contenido y no se requiere colocar defensas.
    if not candidatos_aristas:
        return []

    puntuaciones = {}

    # 3. FILTRADO DE FACTIBILIDAD Y EVALUACIÓN MEDIANTE BFS ACOTADO EN G[S_t]
    for nodo, aristas_incidentes in candidatos_aristas.items():
        # El coste de blindar completamente este nodo es el número de vías directas
        # que le llegan desde I_t: d_{I_t}(nodo) = |delta({nodo}, I_t)|.
        coste = len(aristas_incidentes)

        # Regla de poda: Si un nodo recibe más aristas desde el fuego de las que podemos
        # cortar en este turno (coste > k), es matemáticamente imposible evitar su quema inmediata.
        # Cortar solo una fracción de ellas desperdiciaría presupuesto.
        if coste > k:
            continue

        # A) Exploración BFS acotada a radio r dentro del subgrafo intacto G[S_t]
        # Mide la masa crítica de vértices susceptibles alcanzables a corto alcance (bola B_{S_t}(nodo, r)).
        visitados = {nodo}
        cola = collections.deque([(nodo, 0)])
        grado_susceptible = 0
        while cola:
            actual, distancia = cola.popleft()
            if actual != nodo and distancia >= radio:
                continue
            for vecino in adj[actual]:
                arista = (actual, vecino) if actual < vecino else (vecino, actual)
                if vecino in S_t and arista not in C_t:
                    if actual == nodo:
                        grado_susceptible += 1
                    if vecino not in visitados and distancia < radio:
                        visitados.add(vecino)
                        cola.append((vecino, distancia + 1))

        # C) Ratio Beneficio / Coste (Función de puntuación voraz)
        # Numerador: Impacto preventivo estimado (volumen del cono local |B_{S_t}(nodo, r)| + deg_{S_t}(nodo)).
        # Denominador: Inversión en cortafuegos requerida (1 o 2).
        puntuaciones[nodo] = (len(visitados) + grado_susceptible) / coste

    # 4. RESOLUCIÓN DEL PROBLEMA DE ASIGNACIÓN (MOCHILA DISCRETA 0-1)
    factibles = list(puntuaciones.keys())
    return resolver_mochila_cortafuegos(
        factibles, candidatos_aristas, puntuaciones, adj, S_t, k
    )

def resolver_mochila_cortafuegos(
    factibles, candidatos_aristas, puntuaciones, adj, S_t, k
):
    """Resuelve la selección óptima de cortafuegos como una Mochila Discreta 0-1.

    Cada nodo susceptible v actúa como un elemento indivisible:
      - Peso / Coste: w_v = d_{I_t}(v) = |delta({v}, I_t)| (aristas incidentes
      desde el fuego).
      - Beneficio: val_v = puntuaciones[v] * w_v (nodos protegidos en la bola
      local más salidas).
      - Capacidad: Presupuesto k de cortafuegos por etapa.

    Parámetros:
      factibles: list con los nodos candidatos que cumplen w_v <= k.
      candidatos_aristas: dict {nodo: [aristas incidentes desde I_t]}.
      puntuaciones: dict {nodo: ratio_beneficio_coste}.
      adj: dict {nodo: set(vecinos)} con la lista de adyacencia de G = (V, E).
      S_t: set con los vértices susceptibles o sanos (V \\ Q_t).
      k: int, presupuesto total de aristas a cortar por turno.

    Retorna:
      list de hasta k tuplas (u, v) con las mejores aristas seleccionadas.
    """
    if not factibles or k <= 0:
        return []

    # 1. CONSTRUCCIÓN DE LOS ELEMENTOS (ITEMS) DE LA MOCHILA
    # Se recupera el valor neto a proteger: val_v = puntuaciones[v] * coste
    items = []
    for nodo in factibles:
        coste = len(candidatos_aristas[nodo])
        beneficio = puntuaciones[nodo] * coste
        items.append((nodo, coste, beneficio))

    # 2. PROGRAMACIÓN DINÁMICA CLÁSICA 0-1 (CAPACIDAD k)
    # dp[cap] almacena la tupla: (mejor_beneficio_acumulado, lista_nodos_blindados)
    dp = [(0.0, []) for _ in range(k + 1)]

    for nodo, w, val in items:
        for cap in range(k, w - 1, -1):
            if dp[cap - w][0] + val > dp[cap][0]:
                dp[cap] = (dp[cap - w][0] + val, dp[cap - w][1] + [nodo])

    # 3. EXTRACCIÓN DE ARISTAS DE LOS NODOS SELECCIONADOS
    _, nodos_salvados = dp[k]
    mejores_aristas = []
    for nodo in nodos_salvados:
        mejores_aristas.extend(candidatos_aristas[nodo])

    # 4. RELLENO DE PRESUPUESTO RESIDUAL (Si la mochila no consumió exactamente k aristas)
    if len(mejores_aristas) < k:
        aristas_usadas = set(mejores_aristas)
        aristas_libres = [
            e
            for lista in candidatos_aristas.values()
            for e in lista
            if e not in aristas_usadas
        ]
        # Priorizar aristas libres que conectan con mayor número de nodos sanos en S_t
        aristas_libres.sort(
            key=lambda e: len((adj[e[0]] | adj[e[1]]) & S_t), reverse=True
        )
        mejores_aristas.extend(aristas_libres[: (k - len(mejores_aristas))])

    # 5. CASO DE RESPALDO (FALLBACK)
    # Si ningún nodo individual pudo ser blindado (todos tenían d_{I_t}(v) > k)
    if not mejores_aristas:
        todas_aristas = list(
            {e for lista in candidatos_aristas.values() for e in lista}
        )
        todas_aristas.sort(
            key=lambda e: len((adj[e[0]] | adj[e[1]]) & S_t), reverse=True
        )
        mejores_aristas = todas_aristas[:k]

    return mejores_aristas[:k]