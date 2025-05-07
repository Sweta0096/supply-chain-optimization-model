# plant_location.py
import pulp, numpy as np, matplotlib.pyplot as plt

class PlantLocation:
    """
    Capacitated Plant‑location MILP (open/close + shipments).
    """
    def __init__(self):
        self.model = self.status = self.results = None

    def solve(self, fixed, ship_cost, demand, capacity):
        F, C, D, K = map(np.array, (fixed, ship_cost, demand, capacity))
        m, n = len(F), len(D)

        mdl = pulp.LpProblem("CapacitatedPlantLocation", pulp.LpMinimize)
        y   = pulp.LpVariable.dicts("open", range(m), cat='Binary')
        x   = pulp.LpVariable.dicts("ship",
                                    [(i,j) for i in range(m) for j in range(n)],
                                    lowBound=0)

        mdl += (pulp.lpSum(F[i]*y[i] for i in range(m))
              + pulp.lpSum(C[i][j]*x[(i,j)] for (i,j) in x))

        for j in range(n):
            mdl += pulp.lpSum(x[(i,j)] for i in range(m)) == D[j]

        for i in range(m):
            mdl += pulp.lpSum(x[(i,j)] for j in range(n)) <= K[i] * y[i]

        mdl.solve(pulp.PULP_CBC_CMD(msg=False))
        self.model, self.status = mdl, pulp.LpStatus[mdl.status]
        self.results = {} if self.status!="Optimal" else self._extract(F, C, y, x)
        return self.results

    def _extract(self, F, C, y, x):
        opens  = [i for i in y if y[i].value() > 0.5]
        alloc  = {(i,j):v.value() for (i,j),v in x.items() if v.value() > 1e-6}
        fixed  = sum(F[i] for i in opens)
        trans  = sum(C[i][j]*alloc[(i,j)] for (i,j) in alloc)
        return {"opens":opens, "alloc":alloc,
                "fixed":fixed, "trans":trans,
                "total":self.model.objective.value()}

    def summarize(self):
        if not self.results:
            print("No result"); return
        r = self.results
        print(self.status,
              f"Total={r['total']:.2f}  Fixed={r['fixed']:.2f}  Trans={r['trans']:.2f}")
        for (i,j),v in r['alloc'].items():
            print(f"  Plant {i} → Region {j}: {v:.1f}")

    def plot(self, plants, regions):
        if not self.results:
            raise RuntimeError("No solution")
        plt.figure(figsize=(6,6))
        plt.scatter(*zip(*plants), s=100, marker='s', label='Plants')
        plt.scatter(*zip(*[plants[i] for i in self.results['opens']]),
                    s=100, color='r', label='Opened')
        plt.scatter(*zip(*regions), s=80, marker='o', label='Regions')
        for (i,j) in self.results['alloc']:
            plt.plot([plants[i][0], regions[j][0]],
                     [plants[i][1], regions[j][1]], 'k-', alpha=.3)
        plt.legend(); plt.grid(True); plt.show()
