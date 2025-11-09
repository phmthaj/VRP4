import json
import random
import re
import heapq
from math import ceil, floor, exp
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Any
import argparse
import sys
import os
import copy

SEED = 42
random.seed(SEED)

SCALE = 1000
CHUNK_SAFETY = 0.95
HUGE = 1_000_000_000

SEGMENT_LENGTH = 50
RHO = 0.6
REWARD_GLOBAL_BEST = 6.0
REWARD_IMPROVED = 3.0
REWARD_ACCEPTED = 1.0
EARLY_STOP_PATIENCE = 2000
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_id_to_idx(nodes_json) -> Dict[int, int]:
    return {int(n["id"]): idx for idx, n in enumerate(nodes_json["nodes"])}
def get_node_demand(
    nodes_json, GOODS: List[str], node_type: str = "vendor"
) -> Dict[int, Dict[str, float]]:
    out = {}
    key_primary = "collected_goods_dict" if node_type == "vendor" else "order_dict"
    key_fallbacks = [
        "collected_goods", "order_dict", "demand_dict", "goods_dict"
    ]
    
    for n in nodes_json["nodes"]:
        if n.get("type") != node_type:
            continue
        nid = int(n["id"])
        gdict = n.get(key_primary)
        if not gdict:
            for key in key_fallbacks:
                gdict = n.get(key)
                if gdict: break
        if not gdict:
            gdict = {}
            
        out[nid] = {g: float(gdict.get(g, 0.0)) for g in GOODS}
    return out
def vehicles_of_depot(vehicles_json, depot_internal: int) -> List[dict]:
    depot_internal_str = str(depot_internal)
    depot_internal_int = int(depot_internal)
    return [
        v for v in vehicles_json["vehicles"] 
        if str(v["depot_id"]) == depot_internal_str or int(v["depot_id"]) == depot_internal_int
    ]
def vehicle_capacity_vec(v: dict, GOODS: List[str]) -> List[int]:
    return [int(round(float(v["capacity_dict"][g]) * SCALE)) for g in GOODS]
def vehicle_cost_per_km(v: dict, vehicle_costs_table: Dict[str, float]) -> float:
    t = v.get("vehicle_type", "type_1")
    return float(vehicle_costs_table.get(t, 1.0))
def _parse_vehicle_idx_as_int(v: dict) -> int:
    raw = str(v.get("vehicle_idx", v.get("vehicle_id", "0")))
    m = re.search(r'\d+', raw)
    return int(m.group()) if m else 0
def pick_cheapest_vehicle(vehicles_json, depot_internal: int, GOODS: List[str]):
    vehs = vehicles_of_depot(vehicles_json, depot_internal)
    if not vehs:
        return None, None, None, None
    costs = vehicles_json["meta"]["vehicle_costs"]
    def _cost(v): return vehicle_cost_per_km(v, costs)
    best = min(vehs, key=_cost)
    cap_vec = vehicle_capacity_vec(best, GOODS)
    veh_cost = _cost(best)
    veh_idx = _parse_vehicle_idx_as_int(best)
    return best, cap_vec, veh_cost, veh_idx
def sub_vector_inplace(a: List[int], b: List[int]) -> None:
    for i in range(len(a)): a[i] -= b[i]

def add_vector_inplace(a: List[int], b: List[int]) -> None:
    for i in range(len(a)): a[i] += b[i]

def leq_vec(a: List[int], b: List[int]) -> bool:
    return all(a[i] <= b[i] for i in range(len(a)))
def fit_alloc_into_cap(alloc_vec: List[int], cap_vec: List[int]) -> Tuple[List[int], List[int]]:
    fit = [min(alloc_vec[i], cap_vec[i]) for i in range(len(alloc_vec))]
    rem = [alloc_vec[i] - fit[i] for i in range(len(alloc_vec))]
    return fit, rem
def _norm_val(x) -> float:
    if x is None: return float("inf")
    if isinstance(x, str) and x.lower() == "infinity": return float("inf")
    return float(x)

def build_graph_from_matrix(distance_matrix: List[List[Any]]) -> List[List[float]]:
    n = len(distance_matrix)
    G = [[float("inf")] * n for _ in range(n)]
    for i in range(n):
        G[i][i] = 0.0
        row = distance_matrix[i]
        for j in range(n):
            val = _norm_val(row[j])
            if val < float("inf"):
                G[i][j] = val
    return G
def dijkstra_path(G: List[List[float]], src: int, dst: int) -> Tuple[float, List[int]]:
    n = len(G)
    dist = [float("inf")] * n
    parent = [-1] * n
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]: 
            continue
        if u == dst:
            break
        for v in range(n):
            w = G[u][v]
            if w == float("inf"): 
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    if dist[dst] == float("inf"):
        return float("inf"), []
    path = []
    cur = dst
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return dist[dst], path
def _preprocess_blocked_pairs(
    nodes_json, clusters_json, matrix_json, output_path: Path, 
    node_type: str, cluster_key: str
) -> Tuple[List[List[float]], Dict[Tuple[int,int], List[int]]]:
    print(f"--- Bắt đầu rút gọn cạnh cấm cho {node_type} ---")
    distance_matrix = matrix_json["distance_matrix"]
    id_to_idx = build_id_to_idx(nodes_json)
    idx_to_id = {idx: nid for nid, idx in id_to_idx.items()}
    G = build_graph_from_matrix(distance_matrix)

    depots = [int(n["id"]) for n in nodes_json["nodes"] if n.get("type") == "depot"]
    nodes_of_type = [int(n["id"]) for n in nodes_json["nodes"] if n.get("type") == node_type]

    n = len(distance_matrix)
    dist_out = [[_norm_val(distance_matrix[i][j]) for j in range(n)] for i in range(n)]
    path_map: Dict[Tuple[int,int], List[int]] = {}

    assigned = set()
    for c in clusters_json.get("clusters", []):
        items = c.get(cluster_key, [])
        if isinstance(items, dict):
            assigned.update(int(k) for k in items.keys())
        elif isinstance(items, list):
            assigned.update(int(v) for v in items)
            
    nodes_to_check = [nid for nid in nodes_of_type if nid in assigned]
    print(f"Tìm thấy {len(nodes_to_check)} {node_type}s được gán để kiểm tra cạnh cấm.")

    for depot_id in depots:
        for node_id in nodes_to_check:
            iu = id_to_idx.get(depot_id)
            iv = id_to_idx.get(node_id)
            if iu is None or iv is None: continue

            if dist_out[iu][iv] == float("inf"):
                d, p_idx = dijkstra_path(G, iu, iv)
                if d < float("inf"):
                    dist_out[iu][iv] = d
                    path_map[(depot_id, node_id)] = [idx_to_id[k] for k in p_idx]
            if dist_out[iv][iu] == float("inf"):
                d, p_idx = dijkstra_path(G, iv, iu)
                if d < float("inf"):
                    dist_out[iv][iu] = d
                    path_map[(node_id, depot_id)] = [idx_to_id[k] for k in p_idx]

    dist_out_json = {
        "distance_matrix": [
            [("Infinity" if x == float("inf") else x) for x in row]
            for row in dist_out
        ],
        "path_reconstruction": {
            f"{u}_{v}": path_map[(u,v)] for (u,v) in path_map
        }
    }
    save_json(output_path, dist_out_json)
    print(f"Đã lưu ma trận rút gọn vào: {output_path}")
    return dist_out, path_map
def dist_accessor(distance_matrix_num, id_to_idx, path_map: Dict[Tuple[int,int], List[int]]) -> Callable[[int,int], float]:
    n = len(distance_matrix_num)
    cache_cost: Dict[Tuple[int,int], float] = {}

    def pair_cost(u: int, v: int) -> float:
        key = (u, v)
        if key in cache_cost:
            return cache_cost[key]
        if key in path_map:
            path_ids = path_map[key]
            total = 0.0
            for i in range(len(path_ids)-1):
                a = id_to_idx.get(int(path_ids[i]), None)
                b = id_to_idx.get(int(path_ids[i+1]), None)
                if a is None or b is None: 
                    cache_cost[key] = float("inf"); return cache_cost[key]
                w = distance_matrix_num[a][b]
                if w == float("inf"):
                    cache_cost[key] = float("inf"); return cache_cost[key]
                total += w
            cache_cost[key] = total
            return total
        
        ia = id_to_idx.get(int(u), None)
        ib = id_to_idx.get(int(v), None)
        if ia is None or ib is None: 
            return float("inf")
        val = distance_matrix_num[ia][ib]
        return val

    def d(a: int, b: int) -> float:
        val = pair_cost(a, b)
        if val == float("inf"):
            return HUGE
        return float(val)

    return d
def integer_split_scaled_safe(
    vendors: List[int],
    demand_map: Dict[int, Dict[str, float]],
    min_cap_scaled: int,
    GOODS: List[str]
):
    if min_cap_scaled <= 0:
        min_cap_scaled = 1
    limit = int(max(1, CHUNK_SAFETY * min_cap_scaled))
    chunks_nodes: List[int] = []
    chunks_qty_scaled: List[int] = []
    for vid in vendors:
        total = float(sum(demand_map.get(vid, {}).get(g, 0.0) for g in GOODS))
        total_scaled = int(round(total * SCALE))
        if total_scaled <= 0:
            continue
        need = ceil(total_scaled / limit)
        if need == 0: continue
        base = total_scaled // need
        rem  = total_scaled % need
        for i in range(need):
            take = base + (1 if i < rem else 0)
            take = min(take, limit)
            take = max(1, take)
            chunks_nodes.append(vid)
            chunks_qty_scaled.append(int(take))
    return chunks_nodes, chunks_qty_scaled, None
def allocate_chunk_vector(vendor_remaining_vec: List[int], qty_scaled: int, GOODS: List[str]) -> List[int]:
    safe_rem = [max(0, x) for x in vendor_remaining_vec]
    total_rem = sum(safe_rem)
    if total_rem <= 0:
        return [0]*len(GOODS)

    alloc = []
    for rem_i in safe_rem:
        portion = (rem_i / total_rem) if total_rem > 0 else 0.0
        part = int(round(qty_scaled * portion))
        alloc.append(part)
    diff = qty_scaled - sum(alloc)
    
    iters = 0
    while diff != 0 and iters < len(alloc) * 2:
        idx = iters % len(alloc)
        if diff > 0:
            alloc[idx] += 1; diff -= 1
        elif alloc[idx] > 0:
            alloc[idx] -= 1; diff += 1
        iters += 1
        
    for i in range(len(alloc)):
        alloc[i] = min(alloc[i], safe_rem[i])
        alloc[i] = max(0, alloc[i])
    return alloc
@dataclass
class Route:
    depot: int
    vehicle_idx: int
    veh_cost: float
    cap_vec: List[int]
    n_goods: int
    vehicle_type: str = "unknown"
    path: List[int] = field(default_factory=list)
    load_vec: List[int] = field(default_factory=list)
    assigned_tasks: List[Dict[str, Any]] = field(default_factory=list)
    dist_cache: float = 0.0

    @staticmethod
    def new(depot_node_id: int, vehicle_idx: int, veh_cost_per_km: float, cap_vec: List[int], GOODS: List[str], vehicle_type: str = "unknown") -> "Route":
        n_goods = len(GOODS)
        return Route(
            depot=int(depot_node_id),
            vehicle_idx=vehicle_idx,
            veh_cost=float(veh_cost_per_km),
            cap_vec=cap_vec[:],
            n_goods=n_goods,
            vehicle_type=vehicle_type,
            path=[int(depot_node_id), int(depot_node_id)],
            load_vec=[0]*n_goods,
            assigned_tasks=[],
            dist_cache=0.0
        )

    def clone_shallow(self) -> "Route":
        return Route(
            depot=self.depot,
            vehicle_idx=self.vehicle_idx,
            veh_cost=self.veh_cost,
            cap_vec=self.cap_vec[:],
            n_goods=self.n_goods,
            vehicle_type=self.vehicle_type,
            path=self.path[:],
            load_vec=self.load_vec[:],
            assigned_tasks=[{"node_id": c["node_id"], "alloc": c["alloc"][:]} for c in self.assigned_tasks],
            dist_cache=self.dist_cache
        )

@dataclass
class Solution:
    routes: List[Route]
    unassigned: set
    task_remaining: Dict[int, List[int]]
    n_goods: int
    cost: float = 0.0

    def clone(self) -> "Solution":
        return Solution(
            routes=[r.clone_shallow() for r in self.routes],
            unassigned=set(self.unassigned),
            task_remaining={vid: vec[:] for vid, vec in self.task_remaining.items()},
            n_goods=self.n_goods,
            cost=self.cost
        )
def compute_route_distance(route: Route, dist_fn: Callable[[int,int], float]) -> float:
    total = 0.0
    p = route.path
    for i in range(len(p)-1):
        d = dist_fn(p[i], p[i+1])
        if d >= HUGE:
            return float(HUGE)
        total += d
    return total

def recompute_all_cost(sol: Solution, dist_fn: Callable[[int,int], float]) -> float:
    total = 0.0
    for r in sol.routes:
        d = compute_route_distance(r, dist_fn)
        if d >= HUGE:
            sol.cost = float(HUGE)
            return float(HUGE)
        r.dist_cache = d
        total += d * r.veh_cost
    sol.cost = total
    return total
def delta_insert_edge(route: Route, node_id: int, dist_fn: Callable[[int,int], float], insert_pos: int) -> float:
    p = route.path
    u = p[insert_pos-1]
    v = p[insert_pos] 
    duv = dist_fn(u, v)
    dux = dist_fn(u, node_id)
    dxv = dist_fn(node_id, v)
    if dux >= HUGE or dxv >= HUGE or duv >= HUGE:
        return float(HUGE)
    return (dux + dxv - duv)

def apply_insert_at(route: Route, node_id: int, alloc_vec: List[int], delta_d: float, insert_pos: int) -> None:
    route.path.insert(insert_pos, node_id)
    add_vector_inplace(route.load_vec, alloc_vec)
    route.assigned_tasks.append({"node_id": node_id, "alloc": alloc_vec[:]})
    route.dist_cache += float(delta_d)

def feasible_add_task_to_route(route: Route, alloc_vec: List[int]) -> bool:
    new_load = [route.load_vec[i] + alloc_vec[i] for i in range(route.n_goods)]
    return leq_vec(new_load, route.cap_vec)
def list_all_visits(sol: Solution) -> List[Tuple[int,int,int]]:
    locs: List[Tuple[int,int,int]] = []
    for ri, r in enumerate(sol.routes):
        for pos in range(1, len(r.path)-1):
            vid = r.path[pos]
            if vid != r.depot:
                locs.append((ri, pos, vid))
    return locs

def remove_visit(sol: Solution, ri: int, pos: int) -> Tuple[int, List[int]]:
    r = sol.routes[ri]
    vid = r.path[pos]
    alloc = None
    for i in reversed(range(len(r.assigned_tasks))):
        if r.assigned_tasks[i]["node_id"] == vid:
            alloc = r.assigned_tasks.pop(i)["alloc"]
            break
    if alloc is None:
        alloc = [0]*sol.n_goods

    for j in range(sol.n_goods):
        r.load_vec[j] -= alloc[j]
        if vid in sol.task_remaining:
            sol.task_remaining[vid][j] += alloc[j]

    r.path.pop(pos)
    sol.unassigned.discard(vid)
    return (vid, alloc)
def random_removal(sol: Solution, dist_fn, remove_ratio=0.15):
    locs = list_all_visits(sol)
    if not locs:
        return []
    k = max(1, int(len(locs)*remove_ratio))
    picked = random.sample(locs, k)
    picked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    removed = []
    for (ri, pos, vid) in picked:
        removed.append(remove_visit(sol, ri, pos))
    return removed

def worst_removal(sol: Solution, dist_fn, remove_ratio=0.15):
    locs = list_all_visits(sol)
    if not locs:
        return []
    contribs = []
    for (ri, pos, vid) in locs:
        r = sol.routes[ri]
        u = r.path[pos-1]; x = r.path[pos]; v = r.path[pos+1]
        duv = dist_fn(u, v)
        c = (dist_fn(u, x) + dist_fn(x, v) - duv) * r.veh_cost
        contribs.append(((ri, pos, vid), c))
    contribs.sort(key=lambda t: t[1], reverse=True)
    k = max(1, int(len(locs)*remove_ratio))
    picked_idx = [it[0] for it in contribs[:k]]
    picked_idx.sort(key=lambda t: (t[0], t[1]), reverse=True)
    removed = []
    for (ri, pos, vid) in picked_idx:
        removed.append(remove_visit(sol, ri, pos))
    return removed

def shaw_removal(sol: Solution, dist_fn, remove_ratio=0.15):
    locs = list_all_visits(sol)
    if not locs:
        return []
    k = max(1, int(len(locs)*remove_ratio))
    seed_ri, seed_pos, seed_vid = random.choice(locs)
    cand = []
    for (ri, pos, vid) in locs:
        if (ri, pos, vid) == (seed_ri, seed_pos, seed_vid):
            continue
        dist = dist_fn(seed_vid, vid)
        cand.append(((ri, pos, vid), dist))
    cand.sort(key=lambda t: t[1])
    picked_idx = [(seed_ri, seed_pos, seed_vid)] + [c[0] for c in cand[:k-1]]
    picked_idx.sort(key=lambda t: (t[0], t[1]), reverse=True)
    removed = []
    for (ri, pos, vid) in picked_idx:
        removed.append(remove_visit(sol, ri, pos))
    return removed
def _open_new_trip_and_fit(sol: Solution, vid: int, alloc_vec: List[int],
                           vehicles_json, DEPOT_NODE_TO_INTERNAL, dist_fn, GOODS: List[str]) -> Tuple[bool, List[int]]:
    if not sol.routes and not DEPOT_NODE_TO_INTERNAL:
         return False, alloc_vec
    
    depot_node = sol.routes[0].depot if sol.routes else next(iter(DEPOT_NODE_TO_INTERNAL))
    depot_internal = DEPOT_NODE_TO_INTERNAL.get(depot_node)
    if depot_internal is None:
        return False, alloc_vec

    best_v, base_cap_vec, veh_cost, veh_idx = pick_cheapest_vehicle(vehicles_json, depot_internal, GOODS)
    if best_v is None:
        return False, alloc_vec
    
    veh_type = best_v.get("vehicle_type", "type_1")

    fit_vec, rem_vec = fit_alloc_into_cap(alloc_vec, base_cap_vec)
    if sum(fit_vec) <= 0:
        return False, alloc_vec

    new_r = Route.new(depot_node, veh_idx, veh_cost, base_cap_vec, GOODS, vehicle_type=veh_type)

    if dist_fn(depot_node, vid) >= HUGE or dist_fn(vid, depot_node) >= HUGE:
        return False, alloc_vec

    delta_d = delta_insert_edge(new_r, vid, dist_fn, insert_pos=1)
    if delta_d >= HUGE:
        return False, alloc_vec

    apply_insert_at(new_r, vid, fit_vec, delta_d, insert_pos=1)
    sol.routes.append(new_r)

    if vid in sol.task_remaining:
        for j in range(sol.n_goods):
            sol.task_remaining[vid][j] -= fit_vec[j]

    return (sum(rem_vec) == 0), rem_vec
def greedy_best_position_insertion(sol: Solution, removed_list, dist_fn, GOODS: List[str],
                                   vehicles_json=None, DEPOT_NODE_TO_INTERNAL=None) -> bool:
    processing_list = removed_list[:]
    while processing_list:
        (vid, alloc_vec) = processing_list.pop(0)
        if sum(alloc_vec) <= 0:
            continue
        best = None
        for r in sol.routes:
            if not feasible_add_task_to_route(r, alloc_vec):
                continue
            # *** SỬA LỖI: Bỏ '+ 1' ***
            for pos in range(1, len(r.path)):
                delta_d = delta_insert_edge(r, vid, dist_fn, pos)
                if delta_d >= HUGE:
                    continue
                delta_cost = delta_d * r.veh_cost
                if (best is None) or (delta_cost < best[0]):
                    best = (delta_cost, r, delta_d, pos)

        if best is not None:
            _, r, delta_d, pos = best
            apply_insert_at(r, vid, alloc_vec, delta_d, pos)
            if vid in sol.task_remaining:
                for j in range(sol.n_goods):
                    sol.task_remaining[vid][j] -= alloc_vec[j]
        else:
            done, rem_vec = _open_new_trip_and_fit(sol, vid, alloc_vec, vehicles_json, DEPOT_NODE_TO_INTERNAL, dist_fn, GOODS)
            if not done and sum(rem_vec) > 0:
                processing_list.insert(0, (vid, rem_vec))
    return True

def regret2_best_position_insertion(sol: Solution, removed_list, dist_fn, GOODS: List[str],
                                    vehicles_json=None, DEPOT_NODE_TO_INTERNAL=None) -> bool:
    processing_list = removed_list[:]
    while processing_list:
        candidates = []
        remainders_for_next_round = []

        for (vid, alloc_vec) in processing_list:
            if sum(alloc_vec) <= 0:
                continue
            two_best = []
            for r in sol.routes:
                if not feasible_add_task_to_route(r, alloc_vec):
                    continue
                # *** SỬA LỖI: Bỏ '+ 1' ***
                for pos in range(1, len(r.path)):
                    delta_d = delta_insert_edge(r, vid, dist_fn, pos)
                    if delta_d >= HUGE:
                        continue
                    delta_cost = delta_d * r.veh_cost
                    two_best.append((delta_cost, r, delta_d, pos))
            two_best.sort(key=lambda x: x[0])

            if len(two_best) == 0:
                done, rem_vec = _open_new_trip_and_fit(sol, vid, alloc_vec, vehicles_json, DEPOT_NODE_TO_INTERNAL, dist_fn, GOODS)
                if not done and sum(rem_vec) > 0:
                    remainders_for_next_round.append((vid, rem_vec))
            else:
                candidates.append((vid, alloc_vec, two_best[:2]))

        def regret_score(lst):
            if not lst: return -HUGE
            if len(lst) == 1: return 1e9
            return lst[1][0] - lst[0][0]

        candidates.sort(key=lambda item: regret_score(item[2]), reverse=True)
        for (vid, alloc_vec, best_list) in candidates:
            if not best_list:
                remainders_for_next_round.append((vid, alloc_vec))
                continue
            delta_cost, r, delta_d, pos = best_list[0]
            if not feasible_add_task_to_route(r, alloc_vec):
                remainders_for_next_round.append((vid, alloc_vec))
                continue
            apply_insert_at(r, vid, alloc_vec, delta_d, pos)
            if vid in sol.task_remaining:
                for j in range(sol.n_goods):
                    sol.task_remaining[vid][j] -= alloc_vec[j]

        processing_list = remainders_for_next_round
    return True
def force_feasible_all(sol: Solution, vehicles_json, dist_fn, DEPOT_NODE_TO_INTERNAL: Dict[int,int], GOODS: List[str]) -> None:
    if not sol.routes and not DEPOT_NODE_TO_INTERNAL:
        return
    
    depot_node = sol.routes[0].depot if sol.routes else next(iter(DEPOT_NODE_TO_INTERNAL))
    depot_internal = DEPOT_NODE_TO_INTERNAL.get(depot_node)
    if depot_internal is None: return

    best_v, base_cap_vec, veh_cost, veh_idx = pick_cheapest_vehicle(vehicles_json, depot_internal, GOODS)
    if best_v is None:
        for vid, vec in sol.task_remaining.items():
            if sum(vec) > 0:
                sol.unassigned.add(vid)
                sol.task_remaining[vid] = [0]*sol.n_goods
        return
    
    veh_type = best_v.get("vehicle_type", "type_1")

    while True:
        todo = [(vid, vec) for vid, vec in sol.task_remaining.items() if sum(vec) > 0]
        if not todo:
            break
        vid, rem_vec = min(todo, key=lambda t: dist_fn(depot_node, t[0]))

        if dist_fn(depot_node, vid) >= HUGE or dist_fn(vid, depot_node) >= HUGE:
            sol.unassigned.add(vid)
            sol.task_remaining[vid] = [0]*sol.n_goods
            continue

        fit_vec, _ = fit_alloc_into_cap(rem_vec, base_cap_vec)
        if sum(fit_vec) <= 0:
            sol.unassigned.add(vid)
            sol.task_remaining[vid] = [0]*sol.n_goods
            continue

        new_r = Route.new(depot_node, veh_idx, veh_cost, base_cap_vec, GOODS, vehicle_type=veh_type)
        delta_d = delta_insert_edge(new_r, vid, dist_fn, insert_pos=1)
        if delta_d >= HUGE:
            sol.unassigned.add(vid)
            sol.task_remaining[vid] = [0]*sol.n_goods
            continue

        apply_insert_at(new_r, vid, fit_vec, delta_d, insert_pos=1)
        sol.routes.append(new_r)
        sub_vector_inplace(sol.task_remaining[vid], fit_vec)
def build_initial_solution_for_cluster(
    cluster,
    vehicles_json,
    vehicle_costs,
    dist_fn,
    demand_map,
    min_cap_scaled,
    DEPOT_NODE_TO_INTERNAL: Dict[int, int],
    GOODS: List[str]
) -> Solution:
    depot_node = int(cluster["depot_node_id"])
    depot_internal = DEPOT_NODE_TO_INTERNAL[depot_node]
    vendors = [int(v) for v in cluster.get("assigned_vendors", [])]
    n_goods = len(GOODS)

    vehicles = vehicles_of_depot(vehicles_json, depot_internal)
    if not vehicles:
        print(f"Warning: Không có xe cho depot {depot_node} (internal {depot_internal}).")
        task_remaining = {}
        for vid in vendors:
            vec = [int(round(float(demand_map.get(vid, {}).get(g, 0.0))*SCALE)) for g in GOODS]
            task_remaining[vid] = vec
        sol = Solution(routes=[], unassigned=set(vendors), task_remaining=task_remaining, n_goods=n_goods, cost=0.0)
        return sol

    routes: List[Route] = []
    for vi, v in enumerate(vehicles):
        cap_vec = vehicle_capacity_vec(v, GOODS)
        veh_cost = vehicle_cost_per_km(v, vehicles_json["meta"]["vehicle_costs"])
        veh_type = v.get("vehicle_type", "type_1")
        routes.append(Route.new(depot_node, vi, veh_cost, cap_vec, GOODS, vehicle_type=veh_type))

    per_caps_total = [sum(vehicle_capacity_vec(v, GOODS)) for v in vehicles]
    min_cap = int(max(1, floor(min(per_caps_total))))
    
    chunk_nodes, chunk_qty_scaled, _ = integer_split_scaled_safe(vendors, demand_map, min_cap_scaled=min_cap, GOODS=GOODS)

    task_remaining: Dict[int, List[int]] = {}
    for vid in vendors:
        vec = [int(round(float(demand_map.get(vid, {}).get(g, 0.0))*SCALE)) for g in GOODS]
        task_remaining[vid] = vec

    processing_list = list(zip(chunk_nodes, chunk_qty_scaled))
    random.shuffle(processing_list)

    unassigned_chunks = []

    while processing_list:
        (vid, qty_scaled) = processing_list.pop(0)
        alloc = allocate_chunk_vector(task_remaining[vid], qty_scaled, GOODS)
        if sum(alloc) <= 0:
            continue

        best = None
        for r in routes:
            if not feasible_add_task_to_route(r, alloc):
                continue
            # *** SỬA LỖI: Bỏ '+ 1' ***
            for pos in range(1, len(r.path)):
                delta_d = delta_insert_edge(r, vid, dist_fn, pos)
                if delta_d >= HUGE:
                    continue
                delta_cost = delta_d * r.veh_cost
                if (best is None) or (delta_cost < best[0]):
                    best = (delta_cost, r, delta_d, pos)

        if best is not None:
            _, r, delta_d, pos = best
            apply_insert_at(r, vid, alloc, delta_d, pos)
            sub_vector_inplace(task_remaining[vid], alloc)
        else:
            best_v, base_cap_vec, veh_cost, veh_idx = pick_cheapest_vehicle(vehicles_json, depot_internal, GOODS)
            if best_v is None: 
                unassigned_chunks.append((vid, qty_scaled))
                continue
            
            veh_type = best_v.get("vehicle_type", "type_1")
            new_r = Route.new(depot_node, veh_idx, veh_cost, base_cap_vec, GOODS, vehicle_type=veh_type)
            
            fit_vec, rem_vec = fit_alloc_into_cap(alloc, new_r.cap_vec)
            if sum(fit_vec) <= 0: 
                unassigned_chunks.append((vid, qty_scaled))
                continue
            if dist_fn(depot_node, vid) >= HUGE or dist_fn(vid, depot_node) >= HUGE:
                continue
            
            delta_d = delta_insert_edge(new_r, vid, dist_fn, insert_pos=1)
            if delta_d < HUGE:
                apply_insert_at(new_r, vid, fit_vec, delta_d, insert_pos=1)
                sub_vector_inplace(task_remaining[vid], fit_vec)
                routes.append(new_r)
                if sum(rem_vec) > 0:
                    rem_qty_scaled = sum(rem_vec)
                    processing_list.insert(0, (vid, rem_qty_scaled))
            else:
                 unassigned_chunks.append((vid, qty_scaled))

    sol = Solution(routes, set(), task_remaining, n_goods)
    force_feasible_all(sol, vehicles_json, dist_fn, DEPOT_NODE_TO_INTERNAL, GOODS)
    recompute_all_cost(sol, dist_fn)
    return sol
def build_initial_solution_for_market_cluster(
    cluster,
    vehicles_json,
    vehicle_costs,
    dist_fn,
    DEPOT_NODE_TO_INTERNAL: Dict[int, int],
    GOODS: List[str]
) -> Solution:
    depot_node = int(cluster["depot_node_id"])
    depot_internal = DEPOT_NODE_TO_INTERNAL[depot_node]
    n_goods = len(GOODS)

    market_tasks = cluster.get("assigned_markets_demand", {})
    market_ids = [int(mid) for mid in market_tasks.keys()]

    vehicles = vehicles_of_depot(vehicles_json, depot_internal)
    if not vehicles:
        print(f"Warning: Không có xe cho depot {depot_node} (internal {depot_internal}).")
        task_remaining = {}
        for mid, vec_f in market_tasks.items():
            # SỬA LỖI TƯƠNG TỰ Ở ĐÂY:
            alloc_vec = [int(round(v * SCALE)) if v > 1e-5 else 0 for v in vec_f]
            # Đảm bảo padding nếu cần (dù trường hợp này n_goods = 4)
            current_len = len(alloc_vec)
            if current_len < n_goods:
                alloc_vec.extend([0] * (n_goods - current_len))
            elif current_len > n_goods:
                alloc_vec = alloc_vec[:n_goods]
            task_remaining[int(mid)] = alloc_vec
        return Solution(routes=[], unassigned=set(market_ids), task_remaining=task_remaining, n_goods=n_goods, cost=0.0)

    routes: List[Route] = []
    for vi, v in enumerate(vehicles):
        cap_vec = vehicle_capacity_vec(v, GOODS)
        veh_cost = vehicle_cost_per_km(v, vehicles_json["meta"]["vehicle_costs"])
        veh_type = v.get("vehicle_type", "type_1")
        routes.append(Route.new(depot_node, vi, veh_cost, cap_vec, GOODS, vehicle_type=veh_type))

    task_remaining: Dict[int, List[int]] = {}
    processing_list: List[Tuple[int, List[int]]] = []
    
    for mid_str, vec_f in market_tasks.items():
        mid = int(mid_str)
        
        # ======================= SỬA LỖI QUAN TRỌNG TẠI ĐÂY =======================
        # Giữ lại các giá trị 0 để đảm bảo độ dài vector là 4
        alloc_vec = [int(round(v * SCALE)) if v > 1e-5 else 0 for v in vec_f]
        # ======================= KẾT THÚC PHẦN SỬA LỖI =======================
        
        # (Phòng ngừa) Dù n_goods = 4, nhưng giữ lại logic padding này là tốt
        current_len = len(alloc_vec)
        if current_len < n_goods:
            alloc_vec.extend([0] * (n_goods - current_len))
        elif current_len > n_goods:
            alloc_vec = alloc_vec[:n_goods]
        
        if sum(alloc_vec) <= 0:
            continue
        
        if mid not in task_remaining:
            task_remaining[mid] = [0] * n_goods
        
        # Dòng này bây giờ đã an toàn
        add_vector_inplace(task_remaining[mid], alloc_vec)
        processing_list.append((mid, alloc_vec))

    random.shuffle(processing_list)
    task_remaining_copy = {mid: vec[:] for mid, vec in task_remaining.items()}
    unassigned_tasks_temp = []

    while processing_list:
        (mid, alloc) = processing_list.pop(0)
        if sum(alloc) <= 0:
            continue

        best = None
        for r in routes:
            if not feasible_add_task_to_route(r, alloc):
                continue
            # *** SỬA LỖI: Bỏ '+ 1' ***
            for pos in range(1, len(r.path)):
                delta_d = delta_insert_edge(r, mid, dist_fn, pos)
                if delta_d >= HUGE:
                    continue
                delta_cost = delta_d * r.veh_cost
                if (best is None) or (delta_cost < best[0]):
                    best = (delta_cost, r, delta_d, pos)

        if best is not None:
            _, r, delta_d, pos = best
            apply_insert_at(r, mid, alloc, delta_d, pos)
        else:
            best_v, base_cap_vec, veh_cost, veh_idx = pick_cheapest_vehicle(vehicles_json, depot_internal, GOODS)
            if best_v is None: 
                unassigned_tasks_temp.append((mid, alloc))
                continue
            
            veh_type = best_v.get("vehicle_type", "type_1")
            new_r = Route.new(depot_node, veh_idx, veh_cost, base_cap_vec, GOODS, vehicle_type=veh_type)
            
            fit_vec, rem_vec = fit_alloc_into_cap(alloc, new_r.cap_vec)
            if sum(fit_vec) <= 0: 
                unassigned_tasks_temp.append((mid, alloc))
                continue
            if dist_fn(depot_node, mid) >= HUGE or dist_fn(mid, depot_node) >= HUGE:
                continue
            
            delta_d = delta_insert_edge(new_r, mid, dist_fn, insert_pos=1)
            if delta_d < HUGE:
                apply_insert_at(new_r, mid, fit_vec, delta_d, insert_pos=1)
                routes.append(new_r)
                if sum(rem_vec) > 0:
                    processing_list.insert(0, (mid, rem_vec))
            else:
                unassigned_tasks_temp.append((mid, alloc))

    for (mid, alloc) in unassigned_tasks_temp:
        if mid not in task_remaining_copy:
            task_remaining_copy[mid] = [0] * n_goods
        add_vector_inplace(task_remaining_copy[mid], alloc)

    final_task_remaining = task_remaining_copy
    for r in routes:
        for task in r.assigned_tasks:
            mid = task["node_id"]
            alloc = task["alloc"]
            if mid in final_task_remaining:
                sub_vector_inplace(final_task_remaining[mid], alloc)

    sol = Solution(routes, set(), final_task_remaining, n_goods)
    force_feasible_all(sol, vehicles_json, dist_fn, DEPOT_NODE_TO_INTERNAL, GOODS)
    recompute_all_cost(sol, dist_fn)
    return sol
def roulette_select(weights: List[float]) -> int:
    s = sum(weights)
    if s <= 0:
        return int(random.random() * len(weights))
    threshold = random.random() * s
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc >= threshold:
            return i
    return len(weights)-1

def update_weights(weights: List[float], scores: List[float], counts: List[int], rho: float = RHO) -> None:
    for i in range(len(weights)):
        avg = scores[i] / max(1, counts[i])
        weights[i] = (1 - rho)*weights[i] + rho*avg
        scores[i] = 0.0
        counts[i] = 0
def alns_optimize_cluster(
    cluster,
    vehicles_json,
    dist_fn,
    DEPOT_NODE_TO_INTERNAL: Dict[int, int],
    GOODS: List[str],
    initial_solution_builder: Callable[..., Solution],
    max_iter=8000,
    destroy_ratio=0.25,
    cooling=0.997
):
    depot_node = int(cluster["depot_node_id"])
    print(f"\n--- Bắt đầu tối ưu cho Depot {depot_node} ---")
    depot_internal = DEPOT_NODE_TO_INTERNAL.get(depot_node)
    if depot_internal is None:
        print(f"LỖI: Depot ID {depot_node} không có trong map.")
        return {"depot_node": depot_node, "routes": [], "distance_cost_sum": 0.0, "unassigned_nodes": []}

    vehicles = vehicles_of_depot(vehicles_json, depot_internal)
    if not vehicles:
        print(f"Info: Không có xe cho depot {depot_node}.")
        return {"depot_node": depot_node, "routes": [], "distance_cost_sum": 0.0, "unassigned_nodes": []}

    curr = initial_solution_builder()
    best = curr.clone()
    best_cost = recompute_all_cost(best, dist_fn)
    curr_cost = best_cost

    if best_cost >= HUGE:
        print(f"Info: Không thể tạo solution ban đầu cho depot {depot_node}. Chi phí: {best_cost}")
        return {"depot_node": depot_node, "routes": [], "distance_cost_sum": 0.0, "unassigned_nodes": list(best.unassigned)}

    print(f"Depot {depot_node}: Chi phí ban đầu = {best_cost:.2f} (Unreachable: {len(best.unassigned)})")

    destroy_ops = [
        ("random_removal", lambda s: random_removal(s, dist_fn, remove_ratio=destroy_ratio)),
        ("worst_removal",  lambda s: worst_removal(s, dist_fn, remove_ratio=destroy_ratio)),
        ("shaw_removal",   lambda s: shaw_removal(s, dist_fn, remove_ratio=destroy_ratio)),
    ]
    repair_ops = [
        ("greedy",   lambda s, rem: greedy_best_position_insertion(s, rem, dist_fn, GOODS, vehicles_json, DEPOT_NODE_TO_INTERNAL)),
        ("regret-2", lambda s, rem: regret2_best_position_insertion(s, rem, dist_fn, GOODS, vehicles_json, DEPOT_NODE_TO_INTERNAL)),
    ]
    w_d = [1.0]*len(destroy_ops)
    w_r = [1.0]*len(repair_ops)
    scr_d = [0.0]*len(destroy_ops)
    cnt_d = [0]*len(destroy_ops)
    scr_r = [0.0]*len(repair_ops)
    cnt_r = [0]*len(repair_ops)

    T = max(1.0, (best_cost % HUGE) * 0.05)
    iter_since_last_weight_update = 0
    best_iter = 0

    for it in range(max_iter):
        di = roulette_select(w_d)
        ri = roulette_select(w_r)

        trial = curr.clone()
        removed = destroy_ops[di][1](trial)
        if not removed:
            continue

        repair_ops[ri][1](trial, removed)
        new_cost = recompute_all_cost(trial, dist_fn)

        improved_curr = new_cost < curr_cost
        improved_best = new_cost < best_cost
        reward = REWARD_GLOBAL_BEST if improved_best else (REWARD_IMPROVED if improved_curr else REWARD_ACCEPTED)
        scr_d[di] += reward; cnt_d[di] += 1
        scr_r[ri] += reward; cnt_r[ri] += 1

        delta = new_cost - curr_cost
        accept = (delta < 0) or (random.random() < exp(-delta / max(T, 1e-9)))

        if accept:
            curr = trial
            curr_cost = new_cost
            if improved_best:
                best = trial.clone()
                best_cost = new_cost
                best_iter = it

        T *= cooling
        iter_since_last_weight_update += 1
        if iter_since_last_weight_update >= SEGMENT_LENGTH:
            update_weights(w_d, scr_d, cnt_d, rho=RHO)
            update_weights(w_r, scr_r, cnt_r, rho=RHO)
            iter_since_last_weight_update = 0

        if it - best_iter >= EARLY_STOP_PATIENCE:
            print(f"Depot {depot_node}: Dừng sớm tại iter {it} (không cải thiện từ iter {best_iter})")
            break

    force_feasible_all(best, vehicles_json, dist_fn, DEPOT_NODE_TO_INTERNAL, GOODS)
    best_cost = recompute_all_cost(best, dist_fn)

    routes_out, total = [], 0.0
    for r in best.routes:
        d = r.dist_cache
        if d <= 0 or d >= HUGE:
            continue
        dist_cost = d * r.veh_cost
        routes_out.append({
            "vehicle_id": f"depot{DEPOT_NODE_TO_INTERNAL[depot_node]}_veh{r.vehicle_idx}",
            "vehicle_cost_per_km": r.veh_cost,
            "vehicle_type": r.vehicle_type,
            "path_node_ids": r.path[:],
            "distance_raw": float(d),
            "route_cost": float(dist_cost)
        })
        total += dist_cost

    if best.unassigned:
        print(f" Depot {depot_node}: Không thể phục vụ (unreachable) {len(best.unassigned)} nodes: {sorted(list(best.unassigned))}")

    print(f"Depot {depot_node}: Hoàn thành. Chi phí tốt nhất = {total:.2f} (Unreachable: {len(best.unassigned)})")
    return {"depot_node": depot_node, "routes": routes_out, "distance_cost_sum": total, "unassigned_nodes": list(best.unassigned)}
def expand_route_path(path_ids: List[int], path_map: Dict[Tuple[int,int], List[int]]) -> List[int]:
    if not path_ids or len(path_ids) == 1:
        return path_ids[:]
    out = [path_ids[0]]
    for i in range(len(path_ids)-1):
        u, v = path_ids[i], path_ids[i+1]
        key = (u, v)
        if key in path_map and len(path_map[key]) >= 2:
            seg = path_map[key]
            out.extend(seg[1:])
        else:
            out.append(v)
    return out
def run_pickup_phase(args, base_dir, nodes, vehicles, dist_json, GOODS, DEPOT_NODE_TO_INTERNAL):
    print("\n" + "="*20 + " GIAI ĐOẠN 1: PICKUP (VENDOR -> DEPOT) " + "="*20)
    path_cluster = base_dir / args.cluster_file
    if not path_cluster.exists():
        print(f"LỖI: Không tìm thấy file cluster vendor: {path_cluster}")
        return

    clusters = load_json(path_cluster)
    
    floyd_path = path_cluster.parent / (path_cluster.stem + "_floyd_pickup.json")
    dist_num, path_map = _preprocess_blocked_pairs(
        nodes, clusters, dist_json, floyd_path,
        node_type="vendor", cluster_key="assigned_vendors_demand"
    )
    id2idx = build_id_to_idx(nodes)
    dist_fn = dist_accessor(dist_num, id2idx, path_map)
    
    vendor_demand_map = get_node_demand(nodes, GOODS, node_type="vendor")
    
    all_caps = []
    for d_internal in DEPOT_NODE_TO_INTERNAL.values():
        for v in vehicles_of_depot(vehicles, d_internal):
            all_caps.append(sum(vehicle_capacity_vec(v, GOODS)))
    if not all_caps:
        print("LỖI: Không tìm thấy vehicle nào trong file vehicles.")
        return
    min_cap_scaled = int(max(1, floor(min(all_caps))))

    total_cost = 0.0
    results, total_unassigned = [], []

    for c in clusters.get("clusters", []):
        c_legacy = c.copy()
        c_legacy["assigned_vendors"] = list(c.get("assigned_vendors_demand", {}).keys())

        builder_lambda = lambda: build_initial_solution_for_cluster(
            c_legacy, vehicles_json=vehicles, 
            vehicle_costs=vehicles["meta"]["vehicle_costs"],
            dist_fn=dist_fn, 
            demand_map=vendor_demand_map,
            min_cap_scaled=min_cap_scaled,
            DEPOT_NODE_TO_INTERNAL=DEPOT_NODE_TO_INTERNAL, 
            GOODS=GOODS
        )
        
        res = alns_optimize_cluster(
            c_legacy, vehicles_json=vehicles, dist_fn=dist_fn,
            DEPOT_NODE_TO_INTERNAL=DEPOT_NODE_TO_INTERNAL,
            GOODS=GOODS, 
            initial_solution_builder=builder_lambda,
            max_iter=args.max_iter, 
            destroy_ratio=args.destroy_ratio,
            cooling=args.cooling
        )
        results.append(res)
        total_cost += float(res.get("distance_cost_sum", 0.0))
        if res.get("unassigned_nodes"):
            total_unassigned.extend(res["unassigned_nodes"])

    print("\n============ SUMMARY VEHICLE TYPE ROUTES (PICKUP) ============")
    for cluster_res in results:
        depot = cluster_res["depot_node"]
        routes = cluster_res["routes"]
        print(f"\nDepot {depot}:")
        type_summary = {}

        for r in routes:
            r["path_node_ids"] = expand_route_path(r["path_node_ids"], path_map)
            veh_type = r.get("vehicle_type", "unknown")
            if veh_type not in type_summary:
                type_summary[veh_type] = []
            type_summary[veh_type].append(r["path_node_ids"])

        for vt, trips in type_summary.items():
            print(f"  {vt}:")
            for i, trip in enumerate(trips, 1):
                path_str = " → ".join(map(str, trip))
                print(f"    Trip {i} → {path_str}")
        cluster_res["vehicle_type_summary"] = type_summary
        cluster_res["unassigned_vendors"] = cluster_res.pop("unassigned_nodes")


    output_path = path_cluster.parent / f"alns_routes_pickup.json"
    out = {
        "objective": "sum(distance * vehicle_type_cost) over routes (Pickup Phase)",
        "total_cost": total_cost,
        "distance_unit": "raw_from_matrix",
        "seed": SEED,
        "clusters": results,
        "total_unassigned_vendors": sorted(list(set(total_unassigned))),
        "preprocessed_graph": str(floyd_path.name)
    }
    save_json(output_path, out)
    print("\n" + "="*50)
    print(f"Đã lưu kết quả ALNS pickup vào: {output_path}")
    print(f"Tổng chi phí (Pickup): {total_cost:.2f}")
def run_delivery_phase(args, base_dir, nodes, vehicles, dist_json, GOODS, DEPOT_NODE_TO_INTERNAL):
    print("\n" + "="*20 + " GIAI ĐOẠN 2: DELIVERY (DEPOT -> MARKET) " + "="*20)
    
    vendor_cluster_path = Path(args.cluster_file)
    market_cluster_name = vendor_cluster_path.name.replace("vendor_cluster", "market_cluster")
    path_cluster = base_dir / market_cluster_name
    
    if not path_cluster.exists():
        print(f"LỖI: Không tìm thấy file cluster market: {path_cluster}")
        print("Bỏ qua Giai đoạn 2.")
        return

    clusters = load_json(path_cluster)
    
    floyd_path = path_cluster.parent / (path_cluster.stem + "_floyd_delivery.json")
    dist_num, path_map = _preprocess_blocked_pairs(
        nodes, clusters, dist_json, floyd_path,
        node_type="market", cluster_key="assigned_markets_demand"
    )
    id2idx = build_id_to_idx(nodes)
    dist_fn = dist_accessor(dist_num, id2idx, path_map)
    
    total_cost = 0.0
    results, total_unassigned = [], []

    for c in clusters.get("clusters", []):
        builder_lambda = lambda: build_initial_solution_for_market_cluster(
            c, vehicles_json=vehicles, 
            vehicle_costs=vehicles["meta"]["vehicle_costs"],
            dist_fn=dist_fn, 
            DEPOT_NODE_TO_INTERNAL=DEPOT_NODE_TO_INTERNAL, 
            GOODS=GOODS
        )
        
        res = alns_optimize_cluster(
            c, vehicles_json=vehicles, dist_fn=dist_fn,
            DEPOT_NODE_TO_INTERNAL=DEPOT_NODE_TO_INTERNAL,
            GOODS=GOODS, 
            initial_solution_builder=builder_lambda,
            max_iter=args.max_iter, 
            destroy_ratio=args.destroy_ratio,
            cooling=args.cooling
        )
        results.append(res)
        total_cost += float(res.get("distance_cost_sum", 0.0))
        if res.get("unassigned_nodes"):
            total_unassigned.extend(res["unassigned_nodes"])

    print("\n============ SUMMARY VEHICLE TYPE ROUTES (DELIVERY) ============")
    for cluster_res in results:
        depot = cluster_res["depot_node"]
        routes = cluster_res["routes"]
        print(f"\nDepot {depot}:")
        type_summary = {}

        for r in routes:
            r["path_node_ids"] = expand_route_path(r["path_node_ids"], path_map)
            veh_type = r.get("vehicle_type", "unknown")
            if veh_type not in type_summary:
                type_summary[veh_type] = []
            type_summary[veh_type].append(r["path_node_ids"])

        for vt, trips in type_summary.items():
            print(f"  {vt}:")
            for i, trip in enumerate(trips, 1):
                path_str = " → ".join(map(str, trip))
                print(f"    Trip {i} → {path_str}")
        cluster_res["vehicle_type_summary"] = type_summary
        cluster_res["unassigned_markets"] = cluster_res.pop("unassigned_nodes")


    output_path = path_cluster.parent / f"alns_routes_delivery.json"
    out = {
        "objective": "sum(distance * vehicle_type_cost) over routes (Delivery Phase)",
        "total_cost": total_cost,
        "distance_unit": "raw_from_matrix",
        "seed": SEED,
        "clusters": results,
        "total_unassigned_markets": sorted(list(set(total_unassigned))),
        "preprocessed_graph": str(floyd_path.name)
    }
    save_json(output_path, out)
    print("\n" + "="*50)
    print(f"Đã lưu kết quả ALNS delivery vào: {output_path}")
    print(f"Tổng chi phí (Delivery): {total_cost:.2f}")