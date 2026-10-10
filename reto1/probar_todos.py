#!/usr/bin/env python3
"""Resuelve todos los graph_*.txt de una carpeta o ZIP con reto1_milp.py.

Uso: python probar_todos.py Datos.zip
     python probar_todos.py Datos --tiempo 120 --csv resultados.csv
Colocar este script junto a reto1_milp.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import statistics
import sys
from time import perf_counter
import zipfile

from reto1_milp import load_graph, parse_graph, solve_graph, solver_arguments


def load_all(source):
    source = Path(source)
    if source.is_dir():
        paths = sorted(source.rglob('graph_*.txt'))
        for path in paths:
            yield load_graph(path)
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            names = {n for n in archive.namelist() if not n.startswith('__MACOSX/')}
            graphs = sorted(n for n in names if PurePosixPath(n).match('graph_*.txt'))
            for name in graphs:
                inicio = str(PurePosixPath(name).parent / 'inicio.txt')
                if inicio not in names:
                    raise ValueError(f'Falta {inicio} en el ZIP.')
                yield parse_graph(archive.read(name).decode('utf-8-sig'),
                                  archive.read(inicio).decode('utf-8-sig'),
                                  PurePosixPath(name).name)
    else:
        raise ValueError('Indica una carpeta de datos o un archivo ZIP valido.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datos', type=Path, help='Carpeta Datos o archivo Datos.zip.')
    parser.add_argument('--csv', type=Path, default=Path('resultados_reto1.csv'))
    parser.add_argument('--json', type=Path, help='Guardar tambien los detalles en JSON.')
    solver_arguments(parser)
    args = parser.parse_args()
    started = perf_counter()
    results = []
    try:
        graphs = list(load_all(args.datos))
        if not graphs:
            raise ValueError('No se han encontrado ficheros graph_*.txt.')
        print(f'{len(graphs)} grafos. k={args.k}, s=2. Esperanzas exactas; solver HiGHS.', flush=True)
        print(f"{'Grafo':24} {'Quemadas':>10} {'Salvadas':>10} {'Solver(s)':>10} {'Total(s)':>10} Estado", flush=True)
        for graph in graphs:
            r = solve_graph(graph, args.k, args.tiempo, args.gap, args.log)
            results.append(r)
            burned = '-' if r['quemadas'] is None else f"{r['quemadas']:.6f}"
            saved = '-' if r['salvadas'] is None else f"{r['salvadas']:.6f}"
            print(f"{r['grafo']:24} {burned:>10} {saved:>10} {r['tiempo_solver_s']:10.3f} {r['tiempo_total_s']:10.3f} {r['estado']}", flush=True)
            print('  Cortes: ' + str(r['cortes']), flush=True)
            # Guardar tras cada grafo para conservar los ya terminados.
            with args.csv.open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(results[0]))
                writer.writeheader()
                for item in results:
                    writer.writerow({**item, 'cortes': json.dumps(item['cortes'])})
            if args.json:
                args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        feasible = [r for r in results if r['quemadas'] is not None]
        if len(feasible) == len(results):
            print(f"\nMedia de zonas quemadas: {statistics.mean(r['quemadas'] for r in feasible):.9f}")
            print(f"Media de zonas salvadas: {statistics.mean(r['salvadas'] for r in feasible):.9f}")
        else:
            print('\nNo se calcula la media global: faltan soluciones factibles.')
        print(f"Optimos certificados: {sum(r['optimo_certificado'] for r in results)}/{len(results)}")
        print(f'Tiempo total del lote (lectura y escritura incluidas): {perf_counter()-started:.3f} s')
        print(f'Resultados: {args.csv.resolve()}')
        return 0 if len(feasible) == len(results) else 2
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    sys.exit(main())
