# alns_loggi.py
import argparse
import math
import random
import time
from collections import defaultdict
from typing import List, Tuple
import numpy as np
import os

# ------------- TSPLIB PARSER (Loggi) ------------- #

def parse_vrp_file(path: str):
    """
    Parse a TSPLIB-like VRP file (Loggi format variants).
    Supports:
      - DIMENSION
      - CAPACITY
      - EDGE_WEIGHT_TYPE: EXPLICIT (EDGE_WEIGHT_FORMAT: FULL_MATRIX or LOWER_ROW)
      - EDGE_WEIGHT_SECTION
      - DEMAND_SECTION
      - DEPOT_SECTION
    Returns:
      {
        'n': int,
        'capacity': int,
        'depot': int (0-based),
        'demands': np.ndarray shape (n,),
        'dist': np.ndarray shape (n, n)
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Meta
    n = None
    capacity = None
    edge_weight_type = None  # EXPLICIT
    edge_weight_format = None  # FULL_MATRIX, LOWER_ROW, maybe UPPER_ROW etc.
    section = None

    # Buffers
    dist = None
    demands = None
    depot_idx = None

    # Helpers to collect EDGE_WEIGHT_SECTION numbers
    ew_values: List[int] = []

    # First pass: read headers
    i = 0
    while i < len(lines):
        line = lines[i]

        # Header keys
        if line.startswith("DIMENSION"):
            # e.g. DIMENSION : 401
            n = int(line.split(":")[1].strip())
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            # Usually 'EXPLICIT'
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("EDGE_WEIGHT_FORMAT"):
            edge_weight_format = line.split(":")[1].strip()
        elif line == "EDGE_WEIGHT_SECTION":
            section = "EDGE_WEIGHT_SECTION"
            # read following numeric rows until another section token
            i += 1
            while i < len(lines):
                ln = lines[i]
                if ln in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF"):
                    # section ends
                    i -= 1  # step back to re-handle this line in outer loop
                    break
                # defensive: blank lines already stripped
                parts = ln.split()
                # ensure all are ints; if not, break safely
                try:
                    ew_values.extend(map(int, parts))
                except ValueError:
                    # if any non-numeric appears, end section
                    i -= 1
                    break
                i += 1
        elif line == "DEMAND_SECTION":
            section = "DEMAND_SECTION"
            if demands is None:
                demands = np.zeros(n, dtype=int)
            i += 1
            while i < len(lines):
                ln = lines[i]
                if ln in ("DEPOT_SECTION", "EOF", "NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION"):
                    i -= 1
                    break
                # lines like: "1 0" or "2 14"
                parts = ln.split()
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1  # 0-based
                    dem = int(parts[1])
                    demands[idx] = dem
                i += 1
        elif line == "DEPOT_SECTION":
            section = "DEPOT_SECTION"
            i += 1
            depots = []
            while i < len(lines):
                ln = lines[i]
                # TSPLIB ends depot list with -1
                if ln.startswith("-1") or ln == "EOF":
                    break
                # could be single depot id (1-based)
                try:
                    depots.append(int(ln))
                except ValueError:
                    # ignore weird lines
                    pass
                i += 1
            depot_idx = (depots[0] - 1) if depots else 0
        # ignore NODE_COORD_SECTION (we don't build distances from coords here)
        i += 1

    if n is None or capacity is None:
        raise ValueError("Missing DIMENSION or CAPACITY in file.")

    # Build distance matrix from ew_values
    if edge_weight_type is None or edge_weight_type.upper() != "EXPLICIT":
        # For Loggi we expect EXPLICIT. If not, try to guess as FULL_MATRIX
        edge_weight_format = edge_weight_format or "FULL_MATRIX"

    if edge_weight_format is None:
        # Try to infer by count
        L = len(ew_values)
        if L == n * n:
            edge_weight_format = "FULL_MATRIX"
        elif L == (n * (n - 1)) // 2:
            edge_weight_format = "LOWER_ROW"
        else:
            raise ValueError("Cannot infer EDGE_WEIGHT_FORMAT from EDGE_WEIGHT_SECTION length.")

    dist = np.zeros((n, n), dtype=float)

    if edge_weight_format.upper() in ("FULL_MATRIX", "FULL_MATRIX_DIAG", "FUNCTION"):
        # Use FULL_MATRIX if ew_values length == n*n
        if len(ew_values) != n * n:
            raise ValueError("EDGE_WEIGHT_SECTION does not match FULL_MATRIX size.")
        dist = np.array(ew_values, dtype=float).reshape(n, n)
    elif edge_weight_format.upper() in ("LOWER_ROW", "LOWER_ROW_DIAG", "LOWER_TRIANGLE"):
        # We have strictly the lower triangle (row 2 has 1 number, row 3 has 2, ...)
        # We'll fill symmetrically.
        expected = (n * (n - 1)) // 2
        if len(ew_values) != expected:
            raise ValueError("EDGE_WEIGHT_SECTION length does not match LOWER_ROW expected size.")
        k = 0
        for r in range(1, n):
            for c in range(r):
                w = float(ew_values[k])
                dist[r, c] = w
                dist[c, r] = w
                k += 1
    else:
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}")

    # Ensure zero diagonal
    np.fill_diagonal(dist, 0.0)

    # Demands default (if absent) = 0 for depot, 1 for customers (but Loggi should have DEMAND_SECTION!)
    if demands is None:
        demands = np.ones(n, dtype=int)
        demands[0] = 0

    # Depot default = 0 if absent
    if depot_idx is None:
        depot_idx = 0

    # If depot isn't 0, we can either:
    # (A) keep as is and remember depot index; or
    # (B) reorder nodes so that depot becomes index 0.
    # We'll do (A) to avoid reindexing confusion, but note it downstream.
    return {
        "n": n,
        "capacity": capacity,
        "depot": depot_idx,
        "demands": demands,
        "dist": dist,
    }


# ------------- VRP Helpers ------------- #

def route_cost(dist: np.ndarray, depot: int, route: List[int]) -> float:
    """Cost of a single route visiting 'route' (list of nodes, each != depot)."""
    if not route:
        return 0.0
    c = dist[depot, route[0]]
    for i in range(len(route) - 1):
        c += dist[route[i], route[i + 1]]
    c += dist[route[-1], depot]
    return c

def total_cost(dist: np.ndarray, depot: int, routes: List[List[int]]) -> float:
    return sum(route_cost(dist, depot, r) for r in routes)

def route_load(demands: np.ndarray, route: List[int]) -> int:
    return int(demands[route].sum()) if route else 0

def is_feasible_route(demands: np.ndarray, cap: int, route: List[int]) -> bool:
    return route_load(demands, route) <= cap

def flatten(routes: List[List[int]]) -> List[int]:
    out = []
    for r in routes:
        out.extend(r)
    return out

# ------------- Initialization (Greedy Cheapest Insertion) ------------- #

def initial_solution(n, depot, demands, cap, dist) -> List[List[int]]:
    """
    Build a feasible solution:
      - start with empty list of routes
      - insert each customer by cheapest feasible insertion
      - if not possible, open new route
    """
    customers = [i for i in range(n) if i != depot]
    random.shuffle(customers)
    routes: List[List[int]] = []

    for cust in customers:
        best_delta = math.inf
        best_pos = None
        best_rid = None

        for rid, r in enumerate(routes):
            if route_load(demands, r) + demands[cust] > cap:
                continue
            # try all insertion positions
            for pos in range(len(r) + 1):
                prev_node = depot if pos == 0 else r[pos - 1]
                next_node = depot if pos == len(r) else r[pos]
                delta = dist[prev_node, cust] + dist[cust, next_node] - dist[prev_node, next_node]
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos
                    best_rid = rid

        if best_rid is None:
            # open new route
            routes.append([cust])
        else:
            routes[best_rid].insert(best_pos, cust)

    # Final trim: split routes that exceed capacity (defensive, shouldn't happen)
    fixed = []
    for r in routes:
        cur = []
        w = 0
        for v in r:
            if w + demands[v] <= cap:
                cur.append(v)
                w += demands[v]
            else:
                # start a new route
                fixed.append(cur)
                cur = [v]
                w = demands[v]
        if cur:
            fixed.append(cur)
    return fixed

# ------------- Destroy operators ------------- #

def destroy_random(routes: List[List[int]], d_frac: float) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a fraction of customers uniformly at random.
    Returns (new_routes, removed_customers)
    """
    all_customers = flatten(routes)
    k = max(1, int(len(all_customers) * d_frac))
    removed = set(random.sample(all_customers, k))

    new_routes = []
    for r in routes:
        nr = [v for v in r if v not in removed]
        if nr:
            new_routes.append(nr)
    return new_routes, list(removed)

def destroy_worst(dist: np.ndarray, depot: int, routes: List[List[int]], d_frac: float) -> Tuple[List[List[int]], List[int]]:
    """
    Remove customers whose removal gives largest cost decrease (greedy estimate).
    """
    scored = []
    for rid, r in enumerate(routes):
        if not r:
            continue
        for pos, v in enumerate(r):
            prev_node = depot if pos == 0 else r[pos - 1]
            next_node = depot if pos == len(r) - 1 else r[pos + 1]
            # delta cost if we remove v
            delta = dist[prev_node, next_node] - dist[prev_node, v] - dist[v, next_node]
            # higher delta => better to remove
            scored.append((delta, rid, pos, v))

    if not scored:
        return routes, []

    scored.sort(reverse=True, key=lambda x: x[0])
    all_customers = flatten(routes)
    k = max(1, int(len(all_customers) * d_frac))
    remove_set = set()
    picked = []
    for delta, rid, pos, v in scored:
        if v in remove_set:
            continue
        remove_set.add(v)
        picked.append(v)
        if len(picked) >= k:
            break

    new_routes = []
    for r in routes:
        nr = [v for v in r if v not in remove_set]
        if nr:
            new_routes.append(nr)
    return new_routes, picked

# ------------- Repair operator ------------- #

def repair_greedy(dist: np.ndarray, depot: int, demands: np.ndarray, cap: int,
                  routes: List[List[int]], removed: List[int]) -> List[List[int]]:
    """
    Cheapest-feasible insertion for each removed node.
    Opens new routes if necessary.
    """
    routes = [list(r) for r in routes]
    random.shuffle(removed)

    for cust in removed:
        best_delta = math.inf
        best_pos = None
        best_rid = None

        for rid, r in enumerate(routes):
            if route_load(demands, r) + demands[cust] > cap:
                continue
            for pos in range(len(r) + 1):
                prev_node = depot if pos == 0 else r[pos - 1]
                next_node = depot if pos == len(r) else r[pos]
                delta = dist[prev_node, cust] + dist[cust, next_node] - dist[prev_node, next_node]
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos
                    best_rid = rid

        if best_rid is None:
            routes.append([cust])
        else:
            routes[best_rid].insert(best_pos, cust)

    # Optional: remove empty routes (shouldn't exist)
    routes = [r for r in routes if r]
    return routes

# ------------- Acceptance (SA) ------------- #

def accept_sa(current_cost, candidate_cost, T) -> bool:
    if candidate_cost < current_cost:
        return True
    if T <= 1e-9:
        return False
    prob = math.exp(-(candidate_cost - current_cost) / T)
    return random.random() < prob

# ------------- ALNS main loop ------------- #

def alns(dist, depot, demands, cap, iters=1000, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    routes = initial_solution(len(demands), depot, demands, cap, dist)
    best_routes = [list(r) for r in routes]
    best_cost = total_cost(dist, depot, routes)

    # temperature (SA)
    T0 = 0.02 * best_cost if best_cost > 0 else 1.0
    T = T0
    cooling = 0.995

    # weights for destroy ops
    destroy_ops = ["random", "worst"]
    w = {op: 1.0 for op in destroy_ops}
    scores = {op: 0.0 for op in destroy_ops}
    counts = {op: 1e-9 for op in destroy_ops}

    # removal fraction schedule
    d_fracs = [0.05, 0.08, 0.12, 0.15]


    for it in range(1, iters + 1):
        # pick operator by roulette
        totw = sum(w.values())
        r = random.random() * totw
        cum = 0.0
        op = destroy_ops[0]
        for k in destroy_ops:
            cum += w[k]
            if r <= cum:
                op = k
                break

        d_frac = random.choice(d_fracs)

        if op == "random":
            partial, removed = destroy_random(routes, d_frac)
        else:
            partial, removed = destroy_worst(dist, depot, routes, d_frac)

        candidate = repair_greedy(dist, depot, demands, cap, partial, removed)
        cand_cost = total_cost(dist, depot, candidate)
        cur_cost = total_cost(dist, depot, routes)

        if accept_sa(cur_cost, cand_cost, T):
            routes = candidate
            cur_cost = cand_cost

        # Local update to best
        improved = False
        if cand_cost < best_cost - 1e-9:
            best_cost = cand_cost
            best_routes = [list(r) for r in candidate]
            improved = True

        # Update operator score
        if improved:
            scores[op] += 5.0
        elif cand_cost < cur_cost + 1e-9:
            scores[op] += 1.0
        counts[op] += 1.0
        # Every few iters adapt weights
        if it % 50 == 0:
            for k in destroy_ops:
                w[k] = 0.8 * w[k] + 0.2 * (scores[k] / counts[k])
                scores[k] = 0.0
                counts[k] = 1e-9

        T *= cooling

    return best_routes, best_cost

# ------------- Runner ------------- #

def run_instance(path: str, iters: int, seed: int):
    print(f"Loading: {path}")
    t0 = time.time()
    data = parse_vrp_file(path)
    n = data["n"]
    cap = data["capacity"]
    depot = data["depot"]
    demands = data["demands"]
    dist = data["dist"]

    print(f"  n={n} cap={cap} depot={depot+1}")
    print(f"  distance matrix loaded ({dist.shape})")

    routes, best = alns(dist, depot, demands, cap, iters=iters, seed=seed)
    dt = time.time() - t0
    print(f"✔ Done {os.path.basename(path)} | best={best:.2f} | time={dt:.2f}s")
    return routes, best, dt

# ------------- CLI ------------- #

def main():
    parser = argparse.ArgumentParser(description="ALNS for Loggi (TSPLIB-like).")
    parser.add_argument("--folder", type=str, default=".", help="Folder containing instances")
    parser.add_argument("--instances", nargs="+", required=True, help="List of .vrp files to run")
    parser.add_argument("--iters", type=int, default=1000, help="Number of ALNS iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    for ins in args.instances:
        path = os.path.join(args.folder, ins)
        run_instance(path, args.iters, args.seed)

if __name__ == "__main__":
    main()
