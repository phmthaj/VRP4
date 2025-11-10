README.md
# ALNS Solver for Loggi CVRP Instances

This project provides a lightweight and extensible Adaptive Large Neighborhood Search (ALNS) solver for the Capacitated Vehicle Routing Problem (CVRP), designed to run on realistic Loggi benchmark datasets.

The solver parses TSPLIB-like `.vrp` files and applies a minimal ALNS search strategy, making it simple, fast, and easy to modify.

---

## ✅ Features

### Problem characteristics
- Single depot
- Multiple customers
- Symmetric explicit distance matrix
- Customer demands included
- Vehicle capacity constraint
- Unlimited fleet (solver chooses number of routes)
- No time windows
- No split delivery

### ALNS operations
- Greedy initial construction
- Destroy operators:
  - random removal
  - worst removal
- Repair operator:
  - cheapest feasible insertion
- Simulated annealing acceptance
- Adaptive operator selection
- Removal rate: 5%–15%

---

##  Requirements

The solver uses Python built-in modules plus `numpy`.

Install the only required external library:

```bash
pip install numpy

 Directory Structure
CVRP4/
│
├── base_case/
│   ├── alns_loggi.py
│   ├── README.md
│   ├── RESULT.txt
│   └── data/
│       └── instances/
│           ├── Loggi-n401-k23.vrp
│           ├── Loggi-n501-k24.vrp
│           ├── Loggi-n601-k19.vrp
│           └── ...
│
└── advanced_case/

 Installation
Clone repository
git clone https://github.com/phmthaj/VRP4.git
cd VRP4/CVRP4/base_case

(Optional) Create Python environment
Windows
py -3.10 -m venv venv
venv\Scripts\activate
pip install numpy

Linux / macOS
python3 -m venv venv
source venv/bin/activate
pip install numpy

 How to Run
Run one instance
python alns_loggi.py --folder data/instances --instances Loggi-n401-k23.vrp --iters 10000 --seed 17

Run multiple instances
python alns_loggi.py --folder data/instances \
    --instances Loggi-n401-k23.vrp Loggi-n501-k24.vrp \
    --iters 10000 --seed 17

Run all instances (Windows PowerShell)
python alns_loggi.py --folder data/instances `
    --instances (Get-ChildItem data/instances/*.vrp).Name `
    --iters 10000 --seed 17

Run all instances (Linux/macOS)
python alns_loggi.py --folder data/instances --instances *.vrp --iters 10000 --seed 17

 Example Output
Loading: data/instances/Loggi-n401-k23.vrp
  n=401 cap=100 depot=1
  distance matrix loaded ((401, 401))
✔ Done Loggi-n401-k23.vrp | best=145016.00 | time=0.90s

 Notes

Solver automatically determines the number of routes.

Instances must contain standard TSPLIB sections:

DIMENSION

CAPACITY

EDGE_WEIGHT_SECTION

DEMAND_SECTION

DEPOT_SECTION

No split deliveries

No time windows
