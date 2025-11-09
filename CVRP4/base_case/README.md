# ALNS Loggi CVRP Solver

This project provides a minimal ALNS solver for CVRP instances formatted in Loggi/TSPLIB style.

## Project Structure

```
project/
│
├── alns_loggi.py
└── data/
    └── instances/
        ├── Loggi-n401-k23.vrp
        ├── Loggi-n601-k19.vrp
        ├── Loggi-n701-k38.vrp
        ├── Loggi-n701-k41.vrp
        ├── Loggi-n1001-k52.vrp
        └── ... (other .vrp files)
```

Make sure `.vrp` files are placed in  
`data/instances/`.

---

## How to Run

### Run one instance

```bash
python alns_loggi.py --folder data/instances --instances Loggi-n401-k23.vrp --iters 10000 --seed 17
```

### Run multiple instances

```bash
python alns_loggi.py --folder data/instances     --instances Loggi-n401-k23.vrp Loggi-n601-k19.vrp Loggi-n701-k38.vrp     --iters 10000 --seed 17
```

### Run all `.vrp` files in the folder (PowerShell)

```powershell
python alns_loggi.py --folder data/instances --instances (Get-ChildItem data/instances/*.vrp).Name --iters 10000 --seed 17
```

### Run all `.vrp` files (Linux / Mac)

```bash
python alns_loggi.py --folder data/instances --instances *.vrp --iters 10000 --seed 17
```

---

## Output

For each instance, the solver prints:

```
Loading: data/instances/Loggi-n401-k23.vrp
  n=401 cap=100 depot=1
  distance matrix loaded ((401, 401))
✔ Done Loggi-n401-k23.vrp | best=145016.00 | time=0.90s
```

That’s it!
