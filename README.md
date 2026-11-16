# Modelización, Optimización y Simulación de Incendios en Grafos (CMM-IMI 2026)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Optimization](https://img.shields.io/badge/Solver-HiGHS%20(SciPy%20MILP)-green.svg)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
[![Memoria PDF](https://img.shields.io/badge/Documento-memoria.pdf-red.svg)](memoria.pdf)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Este repositorio contiene la implementación computacional, modelos matemáticos y la memoria técnica desarrollados en el marco del proyecto **CMM-IMI 2026**.

El proyecto aborda variantes estocásticas y deterministas del **Problema del Bombero (*Firefighter Problem*)**, modelando la propagación de incendios sobre redes complejas y diseñando estrategias óptimas de intervención mediante cortafuegos utilizando:
1. **Programación Lineal Entera Mixta (MILP)** exacta resuelta con el solver **HiGHS** (`scipy.optimize.milp`).
2. **Control Óptimo e Iteración de Políticas:** heurística voraz BFS sobre mochila discreta y **Rollout Anidado (*Nested Rollout*)** con poda Branch & Bound.
3. **Simulación Estocástica:** métodos de **Monte Carlo** y algoritmos voraces alternantes multi-etapa.

> [!IMPORTANT]
> **Memoria Completa del Proyecto:**  
> Todas las deducciones teóricas, demostraciones matemáticas, modelos algebraicos formales y la discusión detallada de resultados se encuentran consolidados en el documento:  
> 📄 **[memoria.pdf](memoria.pdf)**

---

## Índice

- [Memoria del Proyecto](#memoria-del-proyecto)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Descripción de los Retos](#descripción-de-los-retos)
  - [Reto 1: Optimización Exacta MILP con Probabilidades](#reto-1-optimización-exacta-milp-con-probabilidades)
  - [Reto 1 (Extra): Enfoque Voraz y Simulación](#reto-1-extra-enfoque-voraz-y-simulación)
  - [Reto 2: Contención Dinámica Determinista (MILP y Rollout Anidado)](#reto-2-contención-dinámica-determinista-milp-y-rollout-anidado)
  - [Retos 3, 4 y 5: Dinámica Estocástica y Voraz Alternante](#retos-3-4-y-5-dinámica-estocástica-y-voraz-alternante)
- [Instalación y Requisitos](#instalación-y-requisitos)
- [Guía Rápida de Ejecución](#guía-rápida-de-ejecución)
- [Autores y Colaboradores](#autores-y-colaboradores)
- [Licencia](#licencia)

---

## Memoria del Proyecto

El documento principal del trabajo es [**`memoria.pdf`**](memoria.pdf), que reúne a lo largo de sus 42 páginas:
- **Capítulo 1 (Reto 1):** Formalización del espacio probabilístico $\Omega$, equivalencia matemática del proceso de contagio, deducción cerrada de la esperanza $\mathbb{E}[|Q_2|]$ y linealización exacta del producto continuo-binario para el solver MILP.
- **Capítulo 2 (Reto 2):** Demostración del horizonte finito $H = n - 1$ (Lema 1), formulación exacta acumulada $(P)$, teoremas de equivalencia y diseño del algoritmo de **Rollout Anidado** ($\pi_2$) con mejora monótona garantizada.
- **Capítulo 3 (Reto 3):** Algoritmo *Alternating Greedy* estocástico con Monte Carlo y propuesta de optimización mediante números aleatorios comunes y podas en la frontera.
- **Capítulo 4 (Reto 4):** Modelado bi-nivel de la perspectiva del incendiario (búsqueda del origen $q^\star$ que maximiza el daño frente a la respuesta dinámica).
- **Capítulo 5 (Reto 5):** Extensiones realistas del modelo: probabilidades variables con viento y meteorología, zonas con ponderación de valor ($w_i$), focos activos durante múltiples etapas y quemas de ensanche.

---

## Estructura del Repositorio

```text
CMM-IMI-2026/
├── memoria.pdf                             # Memoria técnica y matemática completa del proyecto (42 págs.)
├── README.md                               # Presentación del proyecto y guía de reproducción
├── requirements.txt                        # Dependencias de Python unificadas
├── LICENSE                                 # Licencia de código abierto MIT
├── reto1/                                  # Reto 1: Formulación MILP exacta (s=2, probabilidades)
│   ├── reto1_milp.py                       # Modelo lineal exacto con solver HiGHS
│   ├── probar_todos.py                     # Ejecución automatizada en lote sobre datasets
│   ├── referencia_enumeracion.json         # Validación cruzada por enumeración exhaustiva
│   ├── resultados_reto1.csv                # Resultados certificados obtenidos
│   └── Datos/                              # Grafos de prueba (graph_*_probs.txt, inicio.txt)
├── reto1_extra/                            # Reto 1 Extra: Heurística voraz y Monte Carlo (s=5, k=4)
│   ├── reto_1_extra.py                     # Algoritmo voraz multi-etapa con Monte Carlo
│   ├── montecarlo.py                       # Motor de simulación estocástica
│   ├── plot_functions.py                   # Visualización y graficación planar
│   ├── io_functions.py                     # Parseo de ficheros de red
│   └── Datos/                              # Datos de grafos
├── reto2/                                  # Reto 2: Propagación determinista y cortafuegos por turno
│   ├── reto2_exacto.py                     # MILP acumulado por etapas (Lema 1 y horizonte H=n-1)
│   ├── reto2_rollout_anidado.py            # Nested Rollout / Policy Iteration (Bertsekas)
│   ├── algoritmo_bfs.py                    # Heurística voraz de mochila discreta sobre BFS
│   ├── algoritmo_rollout.py                # Rollout base nivel 1
│   └── Datos/                              # Grafos deterministas y pruebas
└── reto3_4_5/                              # Retos 3 a 5: Simulación estocástica y greedy alternante
    ├── reto_3.py                           # Reto 3: Alternating Greedy base con N=100, s=20, k=1
    ├── reto_3_alternativo.py               # Reto 3 optimizado: podas en frontera, CRN y DP cache
    ├── reto_4.py                           # Reto 4: Perspectiva del incendiario (búsqueda de q*)
    ├── reto_5_1.py                         # Reto 5.1: Análisis en topologías complejas (graph_010)
    ├── reto_5_2.py                         # Reto 5.2: Simulación masiva N=1000 con inicio variable
    ├── montecarlo.py                       # Módulo común de simulación
    ├── plot_functions.py                   # Representación gráfica de evoluciones
    ├── io_functions.py                     # Utilidades de lectura
    └── Datos/                              # Grafos de experimentación
```

---

## Descripción de los Retos

### Reto 1: Optimización Exacta MILP con Probabilidades
- **Objetivo:** Minimizar el número esperado de nodos quemados $\mathbb{E}[|Q_2|]$ en un horizonte de dos etapas ($t=0,1,2$), seleccionando exactamente $k=4$ cortes de aristas no dirigidas de forma preventiva.
- **Fundamento Matemático:** Para cada nodo susceptible, la probabilidad de no contagio $S_{lv}$ se factoriza en caminos directos e indirectos de dos saltos. El modelo linealiza exactamente los productos continuo-binario ($w = q \cdot y_e$) mediante envolventes exactas sin recurrir a discretizaciones ni relajaciones aproximadas.
- **Solver:** HiGHS via `scipy.optimize.milp`. Los resultados óptimos han sido verificados contra enumeración exhaustiva ([referencia_enumeracion.json](reto1/referencia_enumeracion.json)).

### Reto 1 (Extra): Enfoque Voraz y Simulación
- Extensión para horizontes más largos ($s=5$) y presupuesto $k=4$, utilizando selección voraz guiada por estimaciones de Monte Carlo ($N=1000$) y visualización comparativa de grafos salvados frente a indefensos.

### Reto 2: Contención Dinámica Determinista (MILP y Rollout Anidado)
- **Objetivo:** Contener un incendio determinista ($p_{ij}=1$) que inicia en el foco $q=1$, disponiendo de un presupuesto estricto de $k=2$ cortafuegos por turno colocados de manera reactiva.
- **Modelo Exacto ([reto2_exacto.py](reto2/reto2_exacto.py)):** Formulación ILP en variables binarias acumuladas ($X_{e,t}$ para cortes y $b_{v,t}$ para nodos quemados) sobre un horizonte finito $H = n - 1$.
- **Heurística Voraz BFS ([algoritmo_bfs.py](reto2/algoritmo_bfs.py)):** Resuelve una mochila discreta 0-1 en el cociclo frontera del fuego, ponderada por la conectividad local de radio $r$.
- **Rollout Anidado ([reto2_rollout_anidado.py](reto2/reto2_rollout_anidado.py)):** Implementa iteración de políticas (*Nested Rollout*, Bertsekas) de nivel 2 con podas *Branch & Bound*. Corrige la miopía de la BFS en grafos complejos (por ejemplo, en `graph_008` reduce de 10 a 9 quemados, igualando el óptimo teórico exacto del MILP).

### Retos 3, 4 y 5: Dinámica Estocástica y Voraz Alternante
- **Reto 3 ([reto_3.py](reto3_4_5/reto_3.py)):** Algoritmo *Alternating Greedy* sobre simulaciones de Monte Carlo ($N=100$, $s=20$, $k=1$). En cada turno se elige el mejor cortafuegos local y se avanza la propagación estocástica, contrastando contra la red sin defensas.
- **Reto 3 Alternativo ([reto_3_alternativo.py](reto3_4_5/reto_3_alternativo.py)):** Versión altamente optimizada mediante podas del cociclo frontera, reducción de varianza con Números Aleatorios Comunes (CRN) y memoización de estados (*DP Cache*), acelerando drásticamente el cálculo y mejorando la tasa de nodos salvados.
- **Reto 4 ([reto_4.py](reto3_4_5/reto_4.py)):** Perspectiva del incendiario: análisis bi-nivel para identificar la zona de inicio $q^\star$ que maximiza el daño frente a la defensa dinámica.
- **Reto 5.1 & 5.2 ([reto_5_1.py](reto3_4_5/reto_5_1.py), [reto_5_2.py](reto3_4_5/reto_5_2.py)):** Sensibilidad frente a perturbaciones estocásticas por etapas (`graph_010`) y simulación a gran escala ($N=1000$) con distribuciones generales (`inicio.txt`).

---

## Instalación y Requisitos

Se recomienda **Python 3.12 de 64 bits** (también compatible con Python 3.11).

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/pedro-villa/CMM-IMI-2026.git
   cd CMM-IMI-2026
   ```

2. **Crear y activar un entorno virtual:**
   - En Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - En Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

> [!NOTE]
> El solver **HiGHS** viene integrado dentro de SciPy (`scipy.optimize.milp`), por lo que **no** requiere instalar solvers externos ni licencias comerciales.

---

## Guía Rápida de Ejecución

Cada script se ejecuta situándose en su correspondiente directorio de trabajo:

### 1. Reto 1 (MILP Exacto)
```bash
cd reto1

# Resolver todos los grafos del directorio Datos:
python probar_todos.py Datos

# Resolver un grafo individual (ejemplo con k=4 cortes):
python reto1_milp.py Datos/graph_003_probs.txt --k 4

cd ..
```

### 2. Reto 2 (Exacto vs. Rollout Anidado)
```bash
cd reto2

# Resolución exacta con MILP (límite de tiempo: 300 s):
python reto2_exacto.py Datos/graph_007_probs.txt --q 1 --k 2 --tiempo 300

# Resolución aproximada con Nested Rollout (nivel 2):
python reto2_rollout_anidado.py Datos/graph_007_probs.txt --q 1 --k 2 --nivel 2

cd ..
```

### 3. Reto 1 Extra (Voraz con Monte Carlo)
```bash
cd reto1_extra
python reto_1_extra.py
cd ..
```

### 4. Retos 3, 4 y 5 (Monte Carlo y Greedy Alternante)
```bash
cd reto3_4_5

# Reto 3 base (N=100, s=20):
python reto_3.py

# Reto 3 optimizado (podas y CRN):
python reto_3_alternativo.py

# Reto 4 (N=50, s=5):
python reto_4.py

# Reto 5.1 (Grafo 10) y Reto 5.2 (N=1000, inicio variable):
python reto_5_1.py
python reto_5_2.py

cd ..
```

---

## Autores y Colaboradores

Proyecto desarrollado en colaboración por:

1. **Jesús González** - [@abriljg](https://github.com/abriljg)
2. **Mario Gallego** - [@Mario5572](https://github.com/Mario5572)
3. **Pedro Villa** - [@pedro-villa](https://github.com/pedro-villa)

---

## Licencia

Este proyecto se distribuye bajo la Licencia **MIT**. Consulta el archivo `LICENSE` para más detalles.
