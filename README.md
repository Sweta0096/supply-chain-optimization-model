# Supply‑Chain Optimization Model

**Supply‑Chain Optimization Model** is a lightweight, open‑source toolkit that lets you explore strategic **network design** and tactical **aggregate production planning** in a single code base.  
It contains two PuLP‑powered MILP solvers:

* **Capacitated Plant Location** – decides **where to open facilities and how to route demand** to minimize fixed + transport cost.  
* **Aggregate Planner** – balances **workforce, production, inventory, overtime, and backorders** over multiple periods to meet demand at least cost.

Both models are pure‑Python, dependency‑light (`pulp`, `numpy`, `matplotlib`), unit‑tested, and come with runnable demos so you can clone, install, and solve in minutes.

## Quick start

```bash
git clone https://github.com/Sweta0096/supply-chain-optimization-model.git
cd supply-chain-optimization-model
pip install -r requirements.txt          #on mac run with pip3
python demo/demo.py         # runs both models with sample data
