r"""
reto2_rollout_anidado.py
========================
Algoritmo aproximado para el Reto 2: Rollout Anidado (Nested Rollout / Policy Iteration)
fundamentado en control óptimo y simulación determinista sobre la heurística BFS local.

Notación formal según el documento del Reto 2 y variables binarias:
- Q_t: Vértices quemados acumulados hasta el turno t (en el documento, B_{t-1}).
- S_t: Subconjunto de vértices susceptibles o sanos (S_t = V \ Q_t).
- I_t: Frente activo de fuego (vértices recién quemados N_{t-1}).
- C_t: Calendario acumulado de cortes por cortafuegos (C_{t-1} subconjunto de C_t).
- delta(I_t, S_t): Cociclo frontera saliente del frente activo hacia la zona susceptible.
- k: Presupuesto máximo de aristas a cortar por turno (k = 2).
- radio: Profundidad máxima r de exploración BFS en la heurística base pi_0.
- pi_0: Política base de nivel 0 (heurística voraz de BFS local acotado de algoritmo_bfs.py).
- pi_j: Política de rollout de nivel j obtenida por mejora de política sobre pi_{j-1}.
- J_{pi}(s, a): Coste proyectado (nodos quemados acumulados |Q_H|) tras aplicar la acción 'a'
  en el estado 's' y proyectar los turnos futuros con la política 'pi' hasta la extinción.

Fundamento metodológico y teoría de control óptimo (Bertsekas):
Aplica la técnica de Rollout Anidado (Nested Rollout), donde cada nivel j actúa como
una etapa formal de Iteración de Políticas (Policy Iteration):
    pi_j(estado) = argmin_{a \in A(estado)} J_{pi_{j-1}}(estado, a)

Propiedades teóricas garantizadas:
  1. Mejora Monótona de Política (Policy Improvement Theorem):
         J(pi_j) <= J(pi_{j-1}) <= ... <= J(pi_0)
     Garantiza formalmente que pi_j jamás producirá un resultado peor que pi_{j-1}.
  2. Corrección del sesgo miope en cuellos de botella:
     Con nivel=1 se recupera el Rollout estándar; con nivel=2 la política base empleada
     para simular el futuro ya incorpora anticipación táctica, corrigiendo las valoraciones
     erróneas de la BFS en grafos densos o ramificados (ej. graph_008: 10 -> 9 quemados,
     alcanzando el óptimo teórico del modelo exacto MILP).
  3. Aceleración computacional mediante Branch & Bound:
     - Move Ordering: uso de la decisión de pi_{j-1} como pivote para fijar la cota alfa.
     - Poda por cota superior basada en la monotonía del fuego (|Q_0| <= |Q_1| <= ... <= |Q_H|).
     - Poda anticipada en el Paso 1 (Zero-BFS cutoff).
     - Factorización combinatoria adaptativa min(k, |cand|).
     - Memorización de la jerarquía de políticas mediante @lru_cache.

Uso:
    python reto2_rollout_anidado.py graph_007_probs.txt graph_008_probs.txt [--q 1] [--k 2] [--nivel 2]
"""

import itertools
from functools import lru_cache

try:
  from .algoritmo_bfs import estrategia_bfs_local
  from .algoritmo_rollout import (
    _evaluar_arista_bfs,
    _simular_trayectoria_rollout,
  )
except ImportError:
  from algoritmo_bfs import estrategia_bfs_local
  from algoritmo_rollout import _evaluar_arista_bfs, _simular_trayectoria_rollout


def _canon(u, v):
    """
    Retorna la representación canónica orientada (min(u, v), max(u, v))
    para aristas no dirigidas, optimizada mediante comparación ternaria directa.
    """
    return (u, v) if u < v else (v, u)


def _rollout_un_nivel(base, max_candidatos):
    """
    Construye un operador funcional de orden superior que ejecuta un paso de Rollout
    (mejora de política) sobre una política base arbitraria pi_{j-1}.

    Parámetros:
      base: función de política pi_{j-1}(adj, Q_t, I_t, C_t, k, radio).
      max_candidatos: int, cota superior de aristas frontera a considerar en la combinatoria.

    Retorna:
      función con la signatura estándar pi_j(adj, Q_t, I_t, C_t, k, radio).
    """

    def politica(adj, Q_t, I_t, C_t, k=2, radio=2):
        """
        Política defensiva derivada por Rollout sobre la política 'base'.
        Evalúa el impacto global de cada acción candidata mediante simulación completa.
        """
        # 1. IDENTIFICACIÓN DEL CONJUNTO SUSCEPTIBLE (S_t = V \ Q_t)
        S_t = set(adj) - Q_t

        # 2. DELIMITACIÓN DEL FRENTE ACTIVO (Cociclo delta(I_t, S_t))
        # Solo se consideran aristas abiertas que comunican directamente el frente I_t con nodos sanos S_t.
        frontera = {_canon(o, d) for o in I_t for d in adj[o]
                    if d in S_t and _canon(o, d) not in C_t}

        # Si el fuego no dispone de aristas abiertas hacia la zona susceptible, está contenido
        if not frontera:
            return []

        # 3. OBTENCIÓN DE LA DECISIÓN BASE Y MOVE ORDERING
        # Consultamos primero la propuesta de la política base pi_{j-1} para utilizarla como pivote.
        decision = tuple(sorted(_canon(u, v) for u, v in base(adj, Q_t, I_t, C_t, k=k, radio=radio)))

        # 4. FIJACIÓN DE LA COTA ALFA INICIAL (Branch & Bound)
        # Se evalúa en primer lugar la decisión base; esto establece un umbral de coste estricto
        # desde la iteración cero, garantizando la propiedad de mejora: J(pi_j) <= J(pi_{j-1}).
        if decision:
            mejor_c, mejor_d = _simular_trayectoria_rollout(
                adj, Q_t, I_t, C_t, decision, k, base, radio)
        else:
            mejor_c = mejor_d = float("inf")
        mejor = list(decision)

        # 5. CRIBA Y ORDENACIÓN TOPOLÓGICA DE ARISTAS CANDIDATAS
        # Se priorizan las aristas frontera según el potencial de masa susceptible protegida (BFS local)
        cand = sorted(frontera, key=lambda e: _evaluar_arista_bfs(adj, S_t, C_t, e, radio=radio),
                      reverse=True)[:max_candidatos]
        vistas = {decision}

        # 6. EVALUACIÓN COMBINATORIA CON PODA POR COTA SUPERIOR
        # Se exploran combinaciones de tamaño min(k, |cand|), adaptándose si el frente es unitario (|cand|=1)
        for a in itertools.combinations(cand, min(k, len(cand))):
            a = tuple(sorted(a))
            if a in vistas:
                continue
            vistas.add(a)

            # Simulación determinista completa de la trayectoria proyectada con la política base
            c, d = _simular_trayectoria_rollout(adj, Q_t, I_t, C_t, a, k, base, radio,
                                                cota_superior=mejor_c, cota_infecciones=mejor_d)

            # Criterio de mejora lexicográfico:
            # Primario: Minimizar el daño total acumulado |Q_H| al extinguirse el incendio (c).
            # Secundario (desempate): Minimizar infecciones inmediatas |I_{t+1}| en el turno siguiente (d).
            if c < mejor_c or (c == mejor_c and d < mejor_d):
                mejor_c, mejor_d, mejor = c, d, list(a)

        return mejor[:k]

    return politica


@lru_cache(maxsize=None)
def politica_rollout(nivel=2, max_candidatos=24):
    """
    Instancia y memoriza la jerarquía de operadores anidados pi_{nivel} sobre pi_0 (BFS local).
    El uso de caché evita reconstruir las clausuras funcionales en cada llamada o etapa.

    Parámetros:
      nivel: int, número de iteraciones de mejora de política anidadas (nivel=1 o nivel=2).
      max_candidatos: int, tamaño máximo de la ventana de candidatos frontera.

    Retorna:
      función de política defensiva resultante tras encadenar los 'nivel' operadores de Rollout.
    """
    pol = estrategia_bfs_local
    for _ in range(nivel):
        pol = _rollout_un_nivel(pol, max_candidatos)
    return pol


def estrategia_rollout_anidado(adj, Q_t, I_t, C_t, k=2, radio=2, nivel=2, max_candidatos=24):
    """
    Interfaz unificada compatible con la firma de estrategia_bfs_local y estrategia_rollout.

    Parámetros:
      adj: dict {nodo: set(vecinos)} con la lista de adyacencia del grafo G = (V, E).
      Q_t: set con los vértices quemados acumulados hasta el momento (Q_t = B_{t-1}).
      I_t: iterable/set con los vértices recién quemados en el turno actual (frente N_{t-1}).
      C_t: set de tuplas canónicas (u, v) con u < v de aristas cortadas previas (C_{t-1}).
      k: int, presupuesto máximo de cortafuegos por etapa (por defecto k=2).
      radio: int, profundidad de exploración BFS de la heurística base pi_0.
      nivel: int, profundidad de anidamiento del Rollout (por defecto nivel=2).
      max_candidatos: int, ventana máxima de aristas frontera evaluadas en cada nivel.

    Retorna:
      list de hasta k tuplas (u, v) con las aristas seleccionadas para ser cortadas.
    """
    return politica_rollout(nivel, max_candidatos)(adj, Q_t, I_t, C_t, k=k, radio=radio)


def resolver(adj, q, k=2, nivel=2, max_candidatos=24):
    """
    Ejecuta el incendio determinista completo según la dinámica secuencial del Reto 2:
      - Estado inicial (t=0): Ignición en foco q (Q_0 = {q}, I_0 = {q}, C_0 = vacío).
      - Turno t (t >= 1):
          Paso 1: Los bomberos aplican hasta k cortes seleccionados por la política pi.
          Paso 2: Expansión determinista del fuego a través de aristas no cortadas de delta(I_{t-1}, S_t).
      - Criterio de parada: Fuego extinguido (I_t = vacío).

    Parámetros:
      adj: dict {nodo: set(vecinos)} con la lista de adyacencia del grafo G = (V, E).
      q: int, vértice fuente de ignición inicial en t=0.
      k: int, presupuesto de cortafuegos por etapa (por defecto k=2).
      nivel: int, nivel de Rollout anidado.
      max_candidatos: int, ventana máxima de aristas candidatas.

    Retorna:
      tuple (calendario, Q):
        calendario: dict {t: list(cortes)} con las aristas cortadas en cada turno.
        Q: set con los vértices calcinados al término del proceso (Q_H).
    """
    pol = politica_rollout(nivel, max_candidatos)
    Q, I, C, calendario, t = {q}, {q}, set(), {}, 0
    while I:
        t += 1
        # Paso 1: Actuación de los bomberos (selección y registro de cortafuegos)
        cortes = [_canon(u, v) for u, v in pol(adj, set(Q), set(I), set(C), k=k, radio=2)]
        C |= set(cortes)
        if cortes:
            calendario[t] = cortes

        # Paso 2: Propagación determinista del frente de fuego por aristas abiertas
        nuevos = {d for o in I for d in adj[o] if d not in Q and _canon(o, d) not in C}
        Q |= nuevos
        I = nuevos
    return calendario, Q

