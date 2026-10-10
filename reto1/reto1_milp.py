#!/usr/bin/env python3
"""Reto 1: optimizacion EXACTA de E[|Q_2|], k cortes no dirigidos.

Uso: python reto1_milp.py Datos/graph_003_probs.txt --k 4
Dependencias: python -m pip install numpy scipy

El fuego empieza en t=0 y se propaga en t=1,2. y_e=1 conserva la
arista; y_e=0 corta ambos sentidos. Se exige exactamente k cortes.

Para l != v, S_lv = (1-p_lv*y_lv) * prod_u(1-p_lu*p_uv*y_lu*y_uv).
Construimos ese producto sucesivamente. Si q es el producto parcial,
q_nuevo = q-c*w, con w=q*y_e (salto directo) o w=q*y_e*y_f
(dos saltos). Cada producto x*y, con x en [0,1] e y binaria,
se representa EXACTAMENTE mediante w<=x, w<=y, w>=x+y-1, w>=0.
Solo las |E| variables y son binarias; las auxiliares son continuas.
Objetivo lineal: 1 + sum_{l!=v} pi_l*(1-S_lv).
No se muestrean escenarios ni se usa una relajacion aproximada.

Referencia de la interfaz del solver:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from time import perf_counter

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit('Falta una dependencia. Ejecuta: python -m pip install numpy scipy') from exc


@dataclass
class Graph:
    name: str
    nodes: list[int]
    arcs: dict[tuple[int, int], float]
    pi: dict[int, float]

    @property
    def edges(self):
        return sorted({tuple(sorted(arc)) for arc in self.arcs})


def rows(text, columns, name):
    for lineno, line in enumerate(text.splitlines(), 1):
        fields = line.split('#', 1)[0].strip().split()
        if not fields:
            continue
        if len(fields) != columns:
            raise ValueError(f'{name}:{lineno}: se esperaban {columns} columnas.')
        yield fields


def parse_graph(graph_text, inicio_text, name='grafo'):
    pi = {}
    for node, value in rows(inicio_text, 2, 'inicio'):
        node, value = int(node), float(value)
        if node in pi or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Probabilidad de inicio invalida o nodo duplicado.')
        pi[node] = value
    if not pi or not math.isclose(math.fsum(pi.values()), 1.0, abs_tol=1e-8, rel_tol=0):
        raise ValueError('Las probabilidades de inicio deben sumar 1; no se normalizan automaticamente.')
    arcs = {}
    for a, b, p in rows(graph_text, 3, name):
        a, b, p = int(a), int(b), float(p)
        if a == b or a not in pi or b not in pi:
            raise ValueError(f'Arco ({a},{b}) con bucle o nodo ausente de inicio.txt.')
        if (a, b) in arcs or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f'Arco ({a},{b}) duplicado o probabilidad invalida.')
        arcs[a, b] = p
    if any((b, a) not in arcs for a, b in arcs):
        raise ValueError('Cada arco debe tener su arco inverso, aunque sus probabilidades difieran.')
    return Graph(name, sorted(pi), arcs, pi)


def load_graph(graph_path, inicio_path=None):
    path = Path(graph_path)
    inicio = Path(inicio_path) if inicio_path else path.parent / 'inicio.txt'
    return parse_graph(path.read_text(encoding='utf-8-sig'),
                       inicio.read_text(encoding='utf-8-sig'), path.name)


def expected_burned(graph, cuts=()):
    """Evaluacion independiente y directa de la formula (9) del PDF."""
    cut_set = {tuple(sorted(e)) for e in cuts}
    p = {arc: value for arc, value in graph.arcs.items()
         if tuple(sorted(arc)) not in cut_set}
    terms = []
    for l in graph.nodes:
        for v in graph.nodes:
            if v == l or graph.pi[l] == 0:
                continue
            survival = 1 - p.get((l, v), 0.0)
            for u in graph.nodes:
                if u != l and u != v:
                    survival *= 1 - p.get((l, u), 0.0) * p.get((u, v), 0.0)
            terms.append(graph.pi[l] * (1 - survival))
    return 1.0 + math.fsum(terms)


class LinearModel:
    def __init__(self):
        self.c, self.integrality, self.lb, self.ub = [], [], [], []
        self.row, self.col, self.data, self.lower, self.upper = [], [], [], [], []

    def var(self, cost=0, binary=False, lower=0, upper=1):
        idx = len(self.c)
        self.c.append(cost)
        self.integrality.append(int(binary))
        self.lb.append(lower)
        self.ub.append(upper)
        return idx

    def constraint(self, terms, lower=-np.inf, upper=np.inf):
        row = len(self.lower)
        for col, value in terms:
            self.row.append(row)
            self.col.append(col)
            self.data.append(value)
        self.lower.append(lower)
        self.upper.append(upper)

    def times_binary(self, x, y):
        if x is None:  # El producto parcial inicial es la constante 1.
            return y
        w = self.var()
        self.constraint([(w, 1), (x, -1)], upper=0)
        self.constraint([(w, 1), (y, -1)], upper=0)
        self.constraint([(w, 1), (x, -1), (y, -1)], lower=-1)
        return w

    def matrices(self):
        from scipy.optimize import Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
        a = coo_matrix((self.data, (self.row, self.col)),
                       shape=(len(self.lower), len(self.c))).tocsc()
        return dict(c=np.asarray(self.c), integrality=np.asarray(self.integrality),
                    bounds=Bounds(self.lb, self.ub),
                    constraints=LinearConstraint(a, self.lower, self.upper))


def build_model(graph, k):
    edges = graph.edges
    if not 0 <= k <= len(edges):
        raise ValueError(f'k debe estar entre 0 y {len(edges)}.')
    model = LinearModel()
    y = {e: model.var(binary=True) for e in edges}
    model.constraint([(i, 1) for i in y.values()], len(edges)-k, len(edges)-k)
    model.var(cost=1, lower=1, upper=1)  # El origen siempre arde; incluido en el gap.
    p = graph.arcs
    for l in graph.nodes:
        if graph.pi[l] == 0:
            continue
        for v in graph.nodes:
            if v == l:
                continue
            factors = []
            if p.get((l, v), 0) > 0:
                factors.append((p[l, v], [y[tuple(sorted((l, v)))]]))
            for u in graph.nodes:
                if u == l or u == v:
                    continue
                coefficient = p.get((l, u), 0) * p.get((u, v), 0)
                if coefficient > 0:
                    factors.append((coefficient, [y[tuple(sorted((l, u)))],
                                                   y[tuple(sorted((u, v)))]]))
            if not factors:
                continue  # v no es alcanzable desde l en dos pasos.
            # Factores grandes primero: orden determinista; el producto es el mismo.
            factors.sort(key=lambda item: -item[0])
            q = None
            for coefficient, binary_factors in factors:
                w = q
                for binary in binary_factors:
                    w = model.times_binary(w, binary)
                next_q = model.var()
                terms = [(next_q, 1), (w, coefficient)]
                rhs = 1 if q is None else 0
                if q is not None:
                    terms.append((q, -1))
                model.constraint(terms, rhs, rhs)
                q = next_q
            burned = model.var(cost=graph.pi[l])
            model.constraint([(burned, 1), (q, 1)], 1, 1)
    return model, y


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def solve_graph(graph, k=4, time_limit=None, relative_gap=0.0, log=False):
    try:
        import scipy
        from scipy.optimize import milp
    except ImportError as exc:
        raise RuntimeError('Se necesita SciPy >= 1.9. Ejecuta: python -m pip install "scipy>=1.9"') from exc
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError('El limite de tiempo debe ser positivo y finito.')
    if not math.isfinite(relative_gap) or relative_gap < 0:
        raise ValueError('El gap relativo debe ser finito y no negativo.')
    start = perf_counter()
    model, y = build_model(graph, k)
    matrices = model.matrices()
    build_seconds = perf_counter() - start
    options = {'mip_rel_gap': relative_gap, 'disp': log}
    if time_limit is not None:
        options['time_limit'] = time_limit
    solver_start = perf_counter()
    result = milp(**matrices, options=options)
    solver_seconds = perf_counter() - solver_start
    baseline = expected_burned(graph)
    gap = finite(getattr(result, 'mip_gap', None))
    proven = result.status == 0 and gap is not None and gap <= 1e-8
    output = dict(grafo=graph.name, n=len(graph.nodes), aristas=len(y), k=k, s=2,
                  estado=('optimo' if proven else {0:'terminado_con_tolerancia',
                          1:'limite_alcanzado', 2:'inviable', 3:'no_acotado',
                          4:'error_solver'}.get(result.status, 'desconocido')),
                  optimo_certificado=proven, mensaje=str(result.message),
                  quemadas_sin_cortes=baseline, quemadas=None, salvadas=None,
                  no_quemadas=None, cortes=None, objetivo_milp=None,
                  cota_inferior=finite(getattr(result, 'mip_dual_bound', None)),
                  gap_relativo=gap, tiempo_modelo_s=build_seconds,
                  tiempo_solver_s=solver_seconds, tiempo_total_s=None,
                  variables_binarias=len(y), variables_continuas=len(model.c)-len(y),
                  restricciones=len(model.lower), version_scipy=scipy.__version__)
    if result.x is not None:
        decisions = np.array([result.x[i] for i in y.values()])
        if np.max(np.abs(decisions - np.rint(decisions)), initial=0) > 1e-5:
            raise RuntimeError('El solver devolvio decisiones no enteras.')
        cuts = [e for e, idx in y.items() if result.x[idx] < 0.5]
        if len(cuts) != k:
            raise RuntimeError('La solucion no respeta el numero de cortes.')
        burned = expected_burned(graph, cuts)
        if not math.isclose(burned, result.fun, abs_tol=1e-5, rel_tol=1e-7):
            raise RuntimeError(f'La formula original ({burned}) y el MILP ({result.fun}) no coinciden.')
        output.update(cortes=[list(e) for e in cuts], quemadas=burned,
                      salvadas=baseline-burned, no_quemadas=len(graph.nodes)-burned,
                      objetivo_milp=float(result.fun))
    output['tiempo_total_s'] = perf_counter() - start
    return output


def solver_arguments(parser):
    parser.add_argument('--k', type=int, default=4, help='Numero exacto de cortes (4).')
    parser.add_argument('--tiempo', type=float, default=None,
                        help='Limite en segundos para el solver por grafo; por defecto sin limite.')
    parser.add_argument('--gap', type=float, default=0.0,
                        help='Gap relativo permitido (0 exige cerrar la brecha).')
    parser.add_argument('--log', action='store_true', help='Mostrar el registro de HiGHS.')


def show_result(r):
    print(f"\nGrafo: {r['grafo']} | n={r['n']} | aristas={r['aristas']} | k={r['k']} | s=2")
    print(f"Estado: {r['estado']} (certificacion sujeta a tolerancias numericas)")
    if r['cortes'] is None:
        print('No se ha obtenido una solucion factible. ' + r['mensaje'])
    else:
        print('Cortes (etiquetas originales; ambos sentidos): ' +
              ', '.join(f'{{{a},{b}}}' for a, b in r['cortes']))
        print(f"Quemadas sin cortes:        {r['quemadas_sin_cortes']:.9f}")
        print(f"Quemadas con cortes:        {r['quemadas']:.9f}")
        print(f"Salvadas gracias a cortes:  {r['salvadas']:.9f}")
        print(f"Total que no se quema:      {r['no_quemadas']:.9f}")
    print(f"Cota inferior: {r['cota_inferior']} | Gap relativo: {r['gap_relativo']}")
    print(f"Tiempo modelo: {r['tiempo_modelo_s']:.3f} s | solver: {r['tiempo_solver_s']:.3f} s | total: {r['tiempo_total_s']:.3f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('grafo', type=Path, help='Fichero graph_XXX_probs.txt (o graph_XXX.txt).')
    parser.add_argument('--inicio', type=Path, help='Por defecto: inicio.txt junto al grafo.')
    parser.add_argument('--json', type=Path, help='Guardar tambien el resultado en JSON.')
    solver_arguments(parser)
    args = parser.parse_args()
    try:
        result = solve_graph(load_graph(args.grafo, args.inicio), args.k, args.tiempo, args.gap, args.log)
        show_result(result)
        if args.json:
            args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return 0 if result['cortes'] is not None else 2
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
