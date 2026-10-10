r"""
algoritmo_rollout.py
====================
Implementación de la política de decisión basada en Rollout (anticipación de 1 paso
y simulación con horizonte completo) para el Reto 2.

Notación formal según el documento del Reto 2 y variables binarias:
- Q_t: Vértices quemados acumulados hasta el turno t (en el documento, B_{t-1}).
- S_t: Subconjunto de vértices susceptibles o sanos (S_t = V \ Q_t).
- I_t: Frente activo de fuego (vértices recién quemados N_{t-1}).
- C_t: Calendario acumulado de cortes por cortafuegos (C_{t-1} subconjunto de C_t).
- delta(I_t, S_t): Cociclo frontera saliente del frente activo hacia la zona susceptible.
- k: Presupuesto máximo de aristas a cortar por turno (k = 2).
- H: Horizonte temporal máximo hasta la extinción (H = n - 1 por el Lema 1).

Utiliza como política base de simulación (rollout policy) la heurística voraz de
BFS local acotado y acelera la búsqueda mediante:
  1. Ordenación de acciones (Move Ordering) usando la decisión base como pivote.
  2. Poda por cota superior (Branch & Bound / Alpha Cutoff) fundamentada en la
     monotonía del conjunto de quemados (|Q_0| <= |Q_1| <= ... <= |Q_H|).
  3. Descarte anticipado en el Paso 1 (Zero-BFS cutoff).
  4. Optimización canónica de aristas no dirigidas mediante comparación ternaria directa.
"""

import collections, itertools

try:
    from .algoritmo_bfs import estrategia_bfs_local
except ImportError:
    from algoritmo_bfs import estrategia_bfs_local
def _evaluar_arista_bfs(adj, S_t, C_t, arista, radio=2):
    """
    Calcula el potencial defensivo de una arista (u, v) en delta(I_t, S_t)
    mediante la masa de nodos susceptibles y conectividad residual protegidos
    con BFS local.

    Parámetros:
      adj: dict {nodo: set(vecinos)} con la lista de adyacencia de G = (V, E).
    S_t: set con los vértices susceptibles o sanos (S_t = V \\ Q_t).
      C_t: set de tuplas canónicas (u, v) con u < v de aristas cortadas previas (C_{t-1}).
      arista: tupla (u, v) perteneciente al cociclo frontera delta(I_t, S_t).
      radio: int, profundidad máxima r de exploración local en el subgrafo susceptible.

    Retorna:
      float/int con la puntuación de impacto: volumen de la bola local |B_{S_t}(v, r)|
      más el grado residual deg_{S_t}(v).
    """
    # 1. IDENTIFICACIÓN DEL VÉRTICE RECEPTOR EN EL SUBGRAFO SANO S_t
    # De los dos extremos de la arista en delta(I_t, S_t), se localiza el nodo que aún no ha ardido.
    u, v = arista
    nodo = v if v in S_t else u

    # 2. EXPLORACIÓN BFS ACOTADA A RADIO r EN EL SUBGRAFO INDUCIDO G[S_t]
    # Mide la masa crítica de vértices susceptibles alcanzables a corto alcance (bola B_{S_t}(nodo, r)).
    visitados = {nodo}
    cola = collections.deque([(nodo, 0)])
    while cola:
        actual, dist = cola.popleft()
        if dist >= radio:
            continue
        for vecino in adj[actual]:
            if vecino in S_t and vecino not in visitados:
                # Comprobación de corte canónico en la arista interna entre nodos sanos
                e = (actual, vecino) if actual < vecino else (vecino, actual)
                if e not in C_t:
                    visitados.add(vecino)
                    cola.append((vecino, dist + 1))

    # 3. GRADO RESIDUAL SUSCEPTIBLE INMEDIATO: deg_{S_t}(nodo)
    # Cuantifica las vías directas de dispersión hacia el resto del bosque no quemado.
    grado_susc = sum(
        1
        for w in adj[nodo]
        if w in S_t
        and (((nodo, w) if nodo < w else (w, nodo)) not in C_t)
    )

    # 4. FUNCIÓN DE VALORACIÓN DEFENSIVA (SCORE)
    # Suma la masa protegida en la bola local con la conectividad saliente residual.
    return len(visitados) + grado_susc

def _simular_trayectoria_rollout(
    adj,
    Q_t_inicial,
    I_t_inicial,
    C_t_inicial,
    accion_candidata,
    k,
    heuristica_base,
    radio_base,
    cota_superior=float("inf"),
    cota_infecciones=float("inf"),
):
    """
    Simula la evolución determinista completa del incendio desde el estado actual
    tras aplicar la acción candidata, proyectando las etapas futuras mediante la
    heurística base hasta alcanzar la extinción o activar la poda anticipada.

    Parámetros:
      adj: dict {nodo: set(vecinos)} con la lista de adyacencia del grafo G = (V, E).
      Q_t_inicial: set con los vértices quemados acumulados hasta el turno actual (Q_t = B_{t-1}).
      I_t_inicial: set con el frente de fuego activo en el turno actual (I_t = N_{t-1}).
      C_t_inicial: set de tuplas (u, v) con u < v representando cortafuegos previos (C_{t-1}).
      accion_candidata: iterable de tuplas (u, v) con los cortes a probar en este turno.
      k: int, presupuesto máximo de cortafuegos por etapa.
      heuristica_base: función voraz utilizada para decidir los cortes en turnos futuros.
      radio_base: int, profundidad de exploración r para la heurística base.
      cota_superior: int/float, menor número total de quemados hallado hasta el momento (cota alfa).
      cota_infecciones: int/float, menor número de contagios en el turno inmediato para desempatar.

    Retorna:
      tuple (total_quemados, infecciones_inmediatas) al final de la trayectoria o al podarse.
    """
    # 1. APLICACIÓN DE LA ACCIÓN CANDIDATA (Paso 1 del turno en el modelo formal)
    # Se incorporan los nuevos cortafuegos al conjunto histórico asegurando orientación canónica u < v.
    C_t = set(C_t_inicial)
    for u, v in accion_candidata:
        C_t.add((u, v) if u < v else (v, u))

    Q_t = set(Q_t_inicial)

    # 2. PASO 1 DE LA SIMULACIÓN: PROPAGACIÓN INMEDIATA DEL FUEGO (Paso 2 del turno)
    # El frente actual I_t_inicial intenta infectar a todos sus vecinos no quemados (S_t)
    # a través de las aristas abiertas del cociclo delta(I_t, S_t).
    nuevos_quemados = set()
    for origen in I_t_inicial:
        for destino in adj[origen]:
            if destino not in Q_t:
                e = (origen, destino) if origen < destino else (destino, origen)
                if e not in C_t:
                    nuevos_quemados.add(destino)

    # Si la acción candidata bloqueó todas las vías de escape del cociclo,
    # el fuego se extingue de inmediato (Definición 1 y Lema 1 del documento).
    if not nuevos_quemados:
        return len(Q_t), 0

    Q_t.update(nuevos_quemados)
    I_t = nuevos_quemados
    infecciones_inmediatas = len(nuevos_quemados)

    # PODA ALFA PASO 1 (Zero-BFS Cutoff):
    # Por monotonía del fuego (|Q_0| <= |Q_1| <= ... <= |Q_H|), si tras el primer paso
    # ya igualamos o empeoramos la cota conocida, es matemáticamente imposible mejorar
    # la mejor solución hallada. Abortamos de inmediato sin invocar al bucle de BFS.
    if (len(Q_t) > cota_superior) or (
        len(Q_t) == cota_superior and infecciones_inmediatas >= cota_infecciones
    ):
        return len(Q_t), infecciones_inmediatas

    # 3. PROYECCIÓN MULTI-ETAPA HACIA EL FUTURO (BUCLE DETERMINISTA HASTA EXTINCIÓN)
    # A partir del turno siguiente, los bomberos actúan guiados por la política base (BFS local).
    while True:
        # A) Decisión de los bomberos en la simulación: selección de cortes en delta(I_t, S_t)
        cortes_siguientes = heuristica_base(
            adj, Q_t, I_t, C_t, k=k, radio=radio_base
        )
        for u, v in cortes_siguientes:
            C_t.add((u, v) if u < v else (v, u))

        # B) Expansión del fuego en la simulación: avance por aristas abiertas
        siguientes_quemados = set()
        for origen in I_t:
            for destino in adj[origen]:
                if destino not in Q_t:
                    e = (origen, destino) if origen < destino else (destino, origen)
                    if e not in C_t:
                        siguientes_quemados.add(destino)

        # Criterio de parada: I_t = vacío (ningún vértice nuevo alcanzado, fuego extinguido)
        if not siguientes_quemados:
            break

        Q_t.update(siguientes_quemados)
        I_t = siguientes_quemados

        # PODA ALFA MULTI-ETAPA:
        # Si el acumulado de quemados en etapas intermedias supera la cota alfa, se interrumpe
        # la simulación inmediatamente sin continuar calculando hasta el horizonte H.
        if (len(Q_t) > cota_superior) or (
            len(Q_t) == cota_superior and infecciones_inmediatas >= cota_infecciones
        ):
            return len(Q_t), infecciones_inmediatas

    return len(Q_t), infecciones_inmediatas


def estrategia_rollout(
    adj,
    Q_t,
    I_t,
    C_t,
    k=2,
    heuristica_base=estrategia_bfs_local,
    radio_base=2,
    max_candidatos_aristas=16,
):
    """
    Selecciona hasta k cortafuegos en el turno t evaluando el desenlace global de cada
    acción candidata mediante simulación completa (Rollout) con poda Branch & Bound.

    Parámetros:
      adj: dict {nodo: set(vecinos)} con la lista de adyacencia del grafo G = (V, E).
      Q_t: set con los vértices quemados acumulados hasta el momento (Q_t = B_{t-1}).
      I_t: iterable/set con los vértices recién quemados en el turno actual (frente N_{t-1}).
      C_t: set de tuplas canónicas (u, v) con u < v de aristas cortadas previas (C_{t-1}).
      k: int, presupuesto máximo de cortafuegos por etapa (por defecto k=2).
      heuristica_base: política base para simular los turnos futuros (por defecto BFS local).
      radio_base: int, profundidad de exploración de la heurística base.
      max_candidatos_aristas: int, cota máxima de aristas frontera para acotar la combinatoria.

    Retorna:
      list de hasta k tuplas (u, v) con las aristas óptimas seleccionadas para cortar.
    """
    # 1. IDENTIFICACIÓN DEL CONJUNTO SUSCEPTIBLE (S_t = V \ Q_t)
    S_t = set(adj.keys()) - Q_t

    # 2. DELIMITACIÓN DEL FRENTE ACTIVO (Cociclo delta(I_t, S_t))
    # Solo consideramos aristas que comunican directamente el frente con nodos sanos.
    aristas_frontera = set()
    for origen in I_t:
        for destino in adj[origen]:
            if destino in S_t:
                e = (origen, destino) if origen < destino else (destino, origen)
                if e not in C_t:
                    aristas_frontera.add(e)

    # Si no hay caminos abiertos hacia nodos susceptibles, el incendio está contenido.
    if not aristas_frontera:
        return []

    # 3. OBTENCIÓN DE LA DECISIÓN BASE Y MOVE ORDERING
    # Consultamos primero la propuesta de la heurística voraz para utilizarla como pivote.
    decision_base = heuristica_base(
        adj, Q_t, I_t, C_t, k=k, radio=radio_base
    )
    decision_base_canon = tuple(
        sorted((u, v) if u < v else (v, u) for u, v in decision_base)
    )

    # Filtrado y ordenación de aristas frontera mediante el criterio de BFS local acotado
    # Prioriza aristas que conducen a la mayor masa de bosque sano protegido
    lista_aristas = list(aristas_frontera)
    lista_aristas.sort(
        key=lambda e: _evaluar_arista_bfs(
            adj, S_t, C_t, e, radio=radio_base
        ),
        reverse=True,
    )
    if len(lista_aristas) > max_candidatos_aristas:
        lista_aristas = lista_aristas[:max_candidatos_aristas]

    # 4. FIJACIÓN DE LA COTA ALFA INICIAL (Move Ordering)
    # Simulamos en primer lugar la decisión base; esto establece un umbral de coste estricto
    # desde la primera iteración, garantizando formalmente la propiedad de Policy Improvement:
    # J(pi_{rollout}) <= J(pi_{bfs}).
    mejor_accion = list(decision_base)
    if decision_base_canon:
        mejor_coste, menor_dano_inmediato = _simular_trayectoria_rollout(
            adj,
            Q_t,
            I_t,
            C_t,
            decision_base_canon,
            k,
            heuristica_base,
            radio_base,
            cota_superior=float("inf"),
            cota_infecciones=float("inf"),
        )
    else:
        mejor_coste = float("inf")
        menor_dano_inmediato = float("inf")

    # 5. GENERACIÓN DEL ESPACIO DE ACCIONES COMBINATORIAS ADMISIBLES
    # Se construyen todas las alternativas admisibles de tamaño k y k-1 sin duplicar la base.
    acciones_candidatas = []
    visitadas = {decision_base_canon} if decision_base_canon else set()

    # Combinaciones de k = 2 cortafuegos (prioritarias por capacidad de bloqueo conjunto)
    if k >= 2:
        for a1, a2 in itertools.combinations(lista_aristas, 2):
            cand = (a1, a2) if a1 < a2 else (a2, a1)
            if cand not in visitadas:
                visitadas.add(cand)
                acciones_candidatas.append(cand)

    # Evaluar k=1 únicamente si no hay suficientes aristas para formar pares.
    if len(lista_aristas) == 1:
        cand = (lista_aristas[0],)
        if cand not in visitadas:
            visitadas.add(cand)
            acciones_candidatas.append(cand)

    # 6. EVALUACIÓN Y SELECCIÓN MEDIANTE ROLLOUT CON PODA ACTIVA
    for accion in acciones_candidatas:
        total_quemados, infecciones_paso1 = _simular_trayectoria_rollout(
            adj,
            Q_t,
            I_t,
            C_t,
            accion,
            k,
            heuristica_base,
            radio_base,
            cota_superior=mejor_coste,
            cota_infecciones=menor_dano_inmediato,
        )

        # Criterio estricto de mejora de política:
        # Primario: Minimizar el daño total acumulado |Q_H| al extinguirse el incendio.
        # Secundario (desempate): Minimizar infecciones inmediatas |I_{t+1}| en el turno t+1.
        if (total_quemados < mejor_coste) or (
            total_quemados == mejor_coste
            and infecciones_paso1 < menor_dano_inmediato
        ):
            mejor_coste = total_quemados
            menor_dano_inmediato = infecciones_paso1
            mejor_accion = list(accion)

    return mejor_accion[:k]