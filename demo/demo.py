# -*- coding: utf-8 -*-
"""Untitled28.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ePfBWCBK8szcvuroTWiPb7UWzcfIU6UE
"""

from plant_location import PlantLocation
from aggregate_planner import AggregatePlanner
import matplotlib.pyplot as plt
import numpy as np

# demo.py

def demo_plant():
    print("--- Running Plant Location Demo ---")
    fixed      = [4000, 5500, 4800]
    capacity   = [120, 130, 100]
    demand     = [40,60,70,45]
    ship_cost  = [[3,5,6,4],
                  [4,2,5,3],
                  [6,4,3,5]]
    plants = [(0,0),(10,0),(5,7)]
    regions= [(2,2),(8,1),(6,5),(3,6)]
    pl = PlantLocation()
    pl.solve(fixed, ship_cost, demand, capacity)
    pl.summarize()
    pl.plot(plants, regions)
    plt.show() # Call show after plotting

def demo_agg():
    print("\n--- Running Aggregate Planner Demo ---")
    T,demand = 6,[100,120,90,110,130,80]
    ap = AggregatePlanner()
    ap.solve(T,demand,
             reg_cost=[50]*T, hire_cost=[200]*T, fire_cost=[150]*T,
             hold_cost=[2]*T, back_cost=[10]*T,
             prod_rate=10, ot_rate=0.5, ot_cap=[20]*T, # Ensure ot_cap is used meaningfully in class
             init_work=12, init_inv=15)
    ap.summarize()
    ap.plot()
    plt.show() # Call show after plotting

if __name__ == "__main__":
    demo_plant()
    demo_agg()
