r"""
reto2_exacto.py
===============
Método exacto para el Reto 2: Formulación de Programación Lineal Entera Mixta (MILP)
mediante variables binarias acumuladas del documento "Reto 2: formulación mediante
variables binarias", resuelto como un problema ILP con el solver HiGHS (scipy.optimize.milp).

Formulación matemática del Modelo (P):
--------------------------------------
Sea G = (V, E) un grafo no dirigido con n = |V| vértices y m = |E| aristas simples.
Sea \vec{E} el conjunto de aristas orientadas bidireccionales:
    \vec{E} = {(u, v), (v, u) : {u, v} \in E}
Sea q \in V el foco de ignición inicial y k el presupuesto de cortafuegos por turno (k = 2).
Por el Lema 1, el horizonte temporal máximo admisible es H = n - 1 etapas.

Variables de decisión binarias acumuladas (§2 del documento):
  - X_{e, t} \in {0, 1}: Indicadora de corte acumulado. Vale 1 si la arista e \in E
    está cortada en el turno t o antes (e \in C_t), y 0 si permanece abierta.
    (t = 1, ..., H; con X_{e, 0} = 0).
  - b_{v, t} \in {0, 1}: Indicadora de infección acumulada. Vale 1 si el vértice v \in V
    está quemado en el instante t o antes (v \in B_t = Q_t), y 0 si sigue susceptible.
    (t = 0, ..., H).

Modelo algebraico (P) con Límite Estricto por Turno (§5.1):
  Minimizar:
      \sum_{v \in V} b_{v, H}                                      (1) Función objetivo

  Sujeto a:
      b_{q, 0} = 1,  b_{v, 0} = 0   \forall v \in V \setminus {q}  (2) Estado inicial foco
      X_{e, 0} = 0                  \forall e \in E                (3) Estado inicial cortafuegos
      b_{u, t-1} \le b_{v, t} + X_{uv, t}  \forall (u, v) \in \vec{E}, t=1..H  (4) Propagación
      \sum_{e \in E} (X_{e, t} - X_{e, t-1}) \le k   \forall t=1..H  (5') Límite estricto por turno
      X_{e, t-1} \le X_{e, t}       \forall e \in E, t=1..H        (6) Permanencia de cortes
      b_{v, t-1} \le b_{v, t}       \forall v \in V, t=1..H        (7) Monotonía de quemados
      X_{e, t} \in {0, 1}           \forall e \in E, t=1..H        (8) Dominio binario cortes
      b_{v, t} \in {0, 1}           \forall v \in V, t=0..H        (9) Dominio binario vértices

Fundamentos teóricos y teoremas demostrados en el documento:
  - Lema 1 (Horizonte finito): El incendio se extingue en a lo sumo n - 1 etapas; fijar
    H = n - 1 garantiza capturar la extinción completa sin truncar soluciones.
  - Lema 2 (Validez de la propagación acumulada): Usar todos los quemados b_{u, t-1} en la
    restricción (4) es exacto porque los cortafuegos son permanentes (C_{t-1} \subseteq C_t),
    impidiendo que vértices contagiados en s < t-1 infecten indebidamente a través de cortes.
  - Lema 3 y Teorema 1 (Equivalencia del óptimo): Adelantar cortes nunca perjudica la defensa;
    la variante estricta (5') produce directamente calendarios admisibles turno a turno
    respetando el óptimo teórico global del modelo acumulado (5).
  - §5.2 y Lema 4 (Verificación con Definición 1): Las restricciones lineales describen
    "incendios posibles o peores" (pueden inflar b_{v, t} en etapas intermedias sin coste),
    pero la minimización en t = H selecciona el incendio real |B_H|. El calendario extraído
    se valida independientemente mediante la Definición 1 (frente N_t).

Uso:
    python reto2_exacto.py graph_007_probs.txt [--q 1] [--k 2] [--tiempo 300]
    python reto2_exacto.py graph_003_probs.txt graph_004_probs.txt ...
"""

import argparse
from typing import Any, cast

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


def cargar_grafo(ruta):
    """
    Lee un fichero de arcos y construye la estructura de adyacencia del grafo no dirigido G = (V, E).

    Formato de entrada:
      Líneas con el esquema "u v p" o "u v". Para el Reto 2 (p_ij = 1, determinista),
      únicamente se extrae la conectividad topológica: la arista no dirigida {u, v}
      existe si aparece registrada en cualquier orientación.

    Parámetros:
      ruta: str con la ruta al fichero de datos.

    Retorna:
      adj: dict {nodo: set(vecinos)} con las listas de adyacencia de G = (V, E).
    """
    # 1. LECTURA DE ARCOS Y DETERMINACIÓN DEL NÚMERO DE VÉRTICES n
    arcos = []
    with open(ruta) as f:
        for linea in f:
            partes = linea.split()
            if len(partes) >= 2:
                arcos.append((int(partes[0]), int(partes[1])))
    n = max(max(a) for a in arcos) + 1

    # 2. CONSTRUCCIÓN DE LA ESTRUCTURA DE ADYACENCIA SIMÉTRICA (Grafo no dirigido simple)
    adj = {v: set() for v in range(n)}
    for u, v in arcos:
        if u != v:  # Exclusión de lazos reflexivos
            adj[u].add(v)
            adj[v].add(u)
    return adj


def propagar(adj, q, calendario):
    r"""
    Simulación canónica de la dinámica determinista del incendio según la Definición 1 del documento.
    Verifica de manera independiente el calendario de cortafuegos obtenido por el optimizador exacto.

    Evolución temporal del sistema:
      - Estado inicial (t = 0):
          B_0 = {q}   (vértices quemados acumulados Q_0)
          N_0 = {q}   (frente activo inicial I_0)
          C_0 = \emptyset (cortafuegos previos)
      - Para cada turno t >= 1 mientras N_{t-1} \neq \emptyset:
          Paso 1: Se aplican los cortes del calendario: C_t = C_{t-1} \cup calendario[t].
          Paso 2: Transmisión determinista por aristas abiertas del cociclo delta(N_{t-1}, V \ B_{t-1}):
                  N_t = { d \in V \ B_{t-1} : \exists o \in N_{t-1}, {o, d} \in E \setminus C_t }
                  B_t = B_{t-1} \cup N_t

    Parámetros:
      adj: dict {nodo: set(vecinos)} con la lista de adyacencia del grafo.
      q: int, foco de ignición inicial en t = 0.
      calendario: dict {t: list((u, v))} con los cortes aplicados exactamente en cada etapa t.

    Retorna:
      tuple (B, frentes):
        B: set con el conjunto total de vértices calcinados al extinguirse el fuego (|B_H|).
        frentes: list de sets [N_0, N_1, ..., N_T] con los vértices que ardieron en cada turno.
    """
    # 1. CONDICIONES INICIALES EN t = 0 (Ignición del foco q)
    C, B, N = set(), {q}, {q}
    frentes = [set(N)]
    t = 0

    # 2. BUCLE DETERMINISTA POR ETAPAS HASTA LA EXTINCIÓN (N = vacío)
    while N:
        t += 1
        # Paso 1: Incorporación de los cortes del calendario programados para el turno t
        for u, v in calendario.get(t, []):
            C.add((u, v) if u < v else (v, u))

        # Paso 2: Propagación del frente activo N hacia nodos sanos susceptibles no cortados
        nuevos = {d for o in N for d in adj[o]
                  if d not in B and ((o, d) if o < d else (d, o)) not in C}

        # Criterio de parada: No se producen nuevos contagios (incendio extinguido y contenido)
        if not nuevos:
            break

        B |= nuevos
        N = nuevos
        frentes.append(set(nuevos))

    return B, frentes


def resolver_exacto(adj, q, k=2, H=None, tiempo=300):
    r"""
    Construye y resuelve el modelo de optimización lineal entera binaria (P) del Reto 2
    utilizando el solver de Branch-and-Cut HiGHS a través de la interfaz scipy.optimize.milp.

    Parámetros:
      adj: dict {nodo: set(vecinos)} con la lista de adyacencia de G = (V, E).
      q: int, foco inicial de ignición (por defecto q=1).
      k: int, presupuesto de aristas a cortar por turno (por defecto k=2).
      H: int, horizonte temporal máximo (por Lema 1, por defecto H = n - 1).
      tiempo: float, límite de tiempo de CPU en segundos asignado al solver HiGHS.

    Retorna:
      dict con el resultado de la optimización:
        - "estado": int, código de terminación de HiGHS (0 = óptimo global probado).
        - "objetivo": int, valor de la función objetivo \sum b_{v, H}.
        - "quemados_verificados": int, nodos calcinados confirmados tras simular con Def. 1.
        - "calendario": dict {t: list(cortes)} con el plan de intervención turno a turno.
        - "frentes": list con los conjuntos de infección N_t verificados.
    """
    n = len(adj)
    # Lema 1: En un grafo de n vértices, el fuego se propaga a lo sumo durante n - 1 etapas
    H = H or n - 1

    # Extracción e indexación canónica de aristas no dirigidas E = {(u, v) : u < v}
    E = sorted({(u, v) for u in adj for v in adj[u] if u < v})
    m = len(E)

    # 1. MAPEO DE VARIABLES BINARIAS A ÍNDICES DEL VECTOR UNIDIMENSIONAL
    # Espacio total de variables: nvar = H * m (cortes X) + (H + 1) * n (quemados b)
    def X(e, t):            # Variable X_{e, t}: arista e cortada en o antes del turno t (t = 1..H)
        return (t - 1) * m + e

    def b(v, t):            # Variable b_{v, t}: nodo v quemado en o antes del turno t (t = 0..H)
        return H * m + t * n + v

    nvar = H * m + (H + 1) * n

    # 2. CONSTRUCCIÓN DE LA MATRIZ DE RESTRICCIONES EN FORMATO DISPERSO (COO Matrix)
    filas, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def fila(coefs, cota_inf, cota_sup):
        """Añade una restricción lineal: cota_inf <= sum(coef * x_c) <= cota_sup."""
        nonlocal r
        for c, v in coefs:
            filas.append(r); cols.append(c); vals.append(v)
        lo.append(cota_inf); hi.append(cota_sup); r += 1

    # 3. GENERACIÓN DE LAS RESTRICCIONES DINÁMICAS TEMPORALES (t = 1, ..., H)
    for t in range(1, H + 1):
        # Ecuación (4): Propagación bidireccional a través de aristas abiertas
        # b_{u, t-1} <= b_{v, t} + X_{uv, t}  ==>  b_{u, t-1} - b_{v, t} - X_{uv, t} <= 0
        for e, (u, v) in enumerate(E):
            for a, c in ((u, v), (v, u)):
                fila([(b(a, t - 1), 1), (b(c, t), -1), (X(e, t), -1)], -np.inf, 0)

        # Ecuación (5') / §5.1: Presupuesto estricto por turno (cortafuegos colocados en t <= k)
        # \sum_{e \in E} (X_{e, t} - X_{e, t-1}) <= k
        coefs = [(X(e, t), 1) for e in range(m)]
        if t > 1:
            coefs += [(X(e, t - 1), -1) for e in range(m)]
        fila(coefs, -np.inf, k)

        # Ecuación (6): Permanencia de cortafuegos (los cortafuegos no se retiran)
        # X_{e, t-1} <= X_{e, t}  ==>  X_{e, t-1} - X_{e, t} <= 0
        if t > 1:
            for e in range(m):
                fila([(X(e, t - 1), 1), (X(e, t), -1)], -np.inf, 0)

        # Ecuación (7): Monotonía temporal de quemados (un vértice quemado permanece quemado)
        # b_{v, t-1} <= b_{v, t}  ==>  b_{v, t-1} - b_{v, t} <= 0
        for v in range(n):
            fila([(b(v, t - 1), 1), (b(v, t), -1)], -np.inf, 0)

    # Conversión eficiente a formato CSR (Compressed Sparse Row) para el solver HiGHS
    A = coo_matrix((vals, (filas, cols)), shape=(r, nvar)).tocsr()

    # 4. FUNCIÓN OBJETIVO: Ecuación (1)
    # Minimizar \sum_{v \in V} b_{v, H} (número total de vértices quemados al final del horizonte)
    c = np.zeros(nvar)
    for v in range(n):
        c[b(v, H)] = 1

    # 5. LÍMITES DE VARIABLES Y CONDICIONES INICIALES: Ecuaciones (2), (8) y (9)
    lb, ub = np.zeros(nvar), np.ones(nvar)
    for v in range(n):
        # En t = 0 solo el foco q arde (b_{q, 0} = 1); el resto permanece sano (b_{v, 0} = 0)
        lb[b(v, 0)] = ub[b(v, 0)] = 1.0 if v == q else 0.0

    # 6. RESOLUCIÓN EXACTA MEDIANTE BRANCH-AND-BOUND / BRANCH-AND-CUT (HiGHS)
    restricciones = LinearConstraint(cast(Any, A), cast(Any, lo), cast(Any, hi))
    limites = Bounds(cast(Any, lb), cast(Any, ub))
    res = milp(c, constraints=restricciones, integrality=np.ones(nvar),
               bounds=limites, options={"time_limit": tiempo})
    if res.x is None:
        return {"estado": res.status, "mensaje": res.message}

    # 7. EXTRACCIÓN DEL CALENDARIO EJECUTABLE DE CORTAFUEGOS
    # Identifica cuándo una arista cambia de estado: X_{e, t} - X_{e, t-1} = 1
    x = np.round(res.x).astype(int)
    calendario = {}
    for t in range(1, H + 1):
        nuevos = [E[e] for e in range(m)
                  if x[X(e, t)] == 1 and (t == 1 or x[X(e, t - 1)] == 0)]
        if nuevos:
            calendario[t] = nuevos

    # 8. VERIFICACIÓN INDEPENDIENTE MEDIANTE LA DEFINICIÓN 1
    # Purga cualquier variable b_{v, t} sobredimensionada en etapas intermedias (§5.2)
    quemados, frentes = propagar(adj, q, calendario)
    return {"estado": res.status,                      # 0 = Optimalidad global matemática garantizada
            "objetivo": int(round(res.fun)),
            "quemados_verificados": len(quemados),
            "calendario": calendario,
            "frentes": frentes}

if __name__ == "__main__":
    # Configuración de interfaz de línea de comandos
    ap = argparse.ArgumentParser()
    ap.add_argument("ficheros", nargs="+")
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--tiempo", type=float, default=300)
    a = ap.parse_args()

    # Evaluación y resolución exacta para cada grafo suministrado
    for ruta in a.ficheros:
        adj = cargar_grafo(ruta)
        s = resolver_exacto(adj, a.q, a.k, tiempo=a.tiempo)
        print(f"== {ruta}  (n={len(adj)}, q={a.q}, k={a.k})")
        if "objetivo" not in s:
            print("   sin solución:", s); continue
        print(f"   óptimo: {s['objetivo']} quemados, {len(adj) - s['objetivo']} salvados "
              f"(verificado con Def. 1: {s['quemados_verificados']}; estado HiGHS={s['estado']})")
        for t, cortes in sorted(s["calendario"].items()):
            print(f"    turno {t}: cortar {cortes}")
        for t, N in enumerate(s["frentes"]):
            print(f"    arden en t={t}: {sorted(N)}")