# aggregate_planner.py
import pulp, numpy as np, matplotlib.pyplot as plt

class AggregatePlanner:
    """
    Multi‑period aggregate production‑inventory‑workforce MILP.
    """
    def __init__(self):
        self.model = self.status = self.results = None

    def solve(self, T, demand, reg_cost, hire_cost, fire_cost,
              hold_cost, back_cost=None,
              prod_rate=1, ot_rate=0.5, ot_cap=None,
              init_work=0, init_inv=0):
        D   = np.array(demand)
        RC  = np.array(reg_cost)
        HC  = np.array(hire_cost)
        FC  = np.array(fire_cost)
        HC_inv = np.array(hold_cost)
        BC  = np.array(back_cost) if back_cost is not None else None
        OTcap = np.array(ot_cap)  if ot_cap  is not None else None

        m = pulp.LpProblem("AggregatePlan", pulp.LpMinimize)
        W = pulp.LpVariable.dicts('W', range(T), lowBound=0)
        H = pulp.LpVariable.dicts('Hire', range(T), lowBound=0)
        F = pulp.LpVariable.dicts('Fire', range(T), lowBound=0)
        P = pulp.LpVariable.dicts('Prod', range(T), lowBound=0)
        I = pulp.LpVariable.dicts('Inv',  range(T), lowBound=0)
        B = pulp.LpVariable.dicts('Back', range(T), lowBound=0) if BC is not None else {}
        O = pulp.LpVariable.dicts('OT',   range(T), lowBound=0) if OTcap is not None else {}

        # ---- objective ----
        obj = pulp.lpSum(RC[t]*W[t] + HC[t]*H[t] + FC[t]*F[t] + HC_inv[t]*I[t]
                         for t in range(T))
        if BC is not None: obj += pulp.lpSum(BC[t]*B[t] for t in range(T))
        if O: obj += pulp.lpSum(RC[t]*ot_rate*O[t] for t in range(T))
        m += obj

        # ---- constraints ----
        for t in range(T):
            prevW = init_work if t==0 else W[t-1]
            m += W[t] == prevW + H[t] - F[t]                 # workforce balance

            reg_cap = prod_rate * W[t]
            ot_cap  = O[t] if O else 0
            m += P[t] <= reg_cap + ot_cap                    # capacity

            if OTcap is not None: m += O[t] <= OTcap[t]      # OT ceiling

            prevI = init_inv if t==0 else I[t-1]
            prevB = 0 if (t==0 or BC is None) else B[t-1]
            if BC is not None:
                m += I[t] - B[t] == prevI - prevB + P[t] - D[t]
            else:
                m += I[t] == prevI + P[t] - D[t]

        if BC is not None: m += B[T-1] == 0                  # no backlog at end

        # ---- solve ----
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        self.model, self.status = m, pulp.LpStatus[m.status]
        self.results = {} if self.status!='Optimal' else self._extract(T, W,H,F,P,I,B,O)
        return self.results

    def _extract(self, T, W,H,F,P,I,B,O):
        res = {'work':{t:W[t].value() for t in range(T)},
               'hire':{t:H[t].value() for t in range(T)},
               'fire':{t:F[t].value() for t in range(T)},
               'prod':{t:P[t].value() for t in range(T)},
               'inv' :{t:I[t].value() for t in range(T)},
               'cost':self.model.objective.value()}
        if B: res['back'] = {t:B[t].value() for t in range(T)}
        if O: res['ot']   = {t:O[t].value() for t in range(T)}
        return res

    def summarize(self):
        if not self.results: print("No result"); return
        print(self.status, f"Cost={self.results['cost']:.2f}")
        for k in ('work','hire','fire','prod','inv','back','ot'):
            if k in self.results:
                for t,val in self.results[k].items():
                    print(f"  {k}{t}={val:.1f}")

    def plot(self):
        if not self.results: raise RuntimeError("No solution")
        T = list(self.results['prod'])
        prod = [self.results['prod'][t] for t in T]
        work = [self.results['work'][t] for t in T]
        inv  = [self.results['inv'][t]  for t in T]
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.bar(T, prod, alpha=.6, label='Prod')
        ax1.plot(T, work, 's-', label='Work')
        ax1.set_ylabel('Units / Work')
        ax2 = ax1.twinx()
        ax2.plot(T, inv, 'o-', label='Inv')
        if 'back' in self.results:
            back = [self.results['back'][t] for t in T]
            ax2.plot(T, back, 'x--', label='Back')
        ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
        plt.show()
