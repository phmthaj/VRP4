# ALNS Solver for Loggi CVRP Instances

This project implements a lightweight Adaptive Large Neighborhood Search (ALNS) solver for the Capacitated Vehicle Routing Problem (CVRP) using Loggi benchmark data.

---

## ✅ Requirements

Install the only external dependency:

```
pip install numpy
```

---

## ✅ Directory Structure

```
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
```

---

## ✅ Installation

### Clone repository

```
git clone https://github.com/phmthaj/VRP4.git
cd VRP4/CVRP4/base_case
```

### (Optional) Create Python environment

#### Windows

```
py -3.11 -m venv venv
venv\Scripts\activate
pip install numpy
```

#### Linux / macOS

```
python3 -m venv venv
source venv/bin/activate
pip install numpy
```

---

## ✅ How to Run

### Run one instance

```
python alns_loggi.py --folder data/instances --instances Loggi-n401-k23.vrp --iters 10000 --seed 17
```

### Run multiple instances

```
python alns_loggi.py --folder data/instances \
    --instances Loggi-n401-k23.vrp Loggi-n501-k24.vrp \
    --iters 10000 --seed 17
```

### Run all instances (Windows PowerShell)

```
python alns_loggi.py --folder data/instances `
    --instances (Get-ChildItem data/instances/*.vrp).Name `
    --iters 10000 --seed 17
```

### Run all instances (Linux/macOS)

```
python alns_loggi.py --folder data/instances --instances *.vrp --iters 10000 --seed 17
```

---

## ✅ Example Output

```
Loading: data/instances/Loggi-n401-k23.vrp
  n=401 cap=100 depot=1
  distance matrix loaded ((401, 401))
✔ Done Loggi-n401-k23.vrp | best=145016.00 | time=0.90s
```

---

## ✅ Notes

* Solver automatically determines the number of routes.
* Instances must contain standard TSPLIB sections:

  * DIMENSION
  * CAPACITY
  * EDGE_WEIGHT_SECTION
  * DEMAND_SECTION
  * DEPOT_SECTION
* No split deliveries
* No time windows

---
