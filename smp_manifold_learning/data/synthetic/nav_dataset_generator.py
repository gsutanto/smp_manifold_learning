#!/usr/bin/env python3 

import numpy as np

# create a dataset of configurations that keep a robot within 1m of the origin

def is_within_radius(pt, coords=[0,0], radius=1): 
    # coords is list-like coordinates of a POI
    # if pt <= radius distance from coords return true
    
    # sqrt((x-a)^2 + (y-b)^2) <= radius 
    offset = np.array(pt) - np.array(coords)
    return np.linalg.norm(offset) <= radius


np.random.seed(38)
n_data = 1e5

tower_loc = (0, 0)
safe_radius = 1.0

x_lo, x_hi = -5, 5
y_lo, y_hi = -5, 5
range_lo = np.array([x_lo, y_lo])
range_hi = np.array([x_hi, y_hi])

on_dataset = []
off_dataset = []

while len(on_dataset) < n_data or len(off_dataset) < n_data:
    config = np.random.uniform(size=tuple(range_lo.shape))
    config = (range_hi - range_lo) * config + range_lo 
    # if config is within the safe zone, add to on dataset
#    if is_within_radius(config, tower_loc, safe_radius):
    if abs(np.linalg.norm(config - np.array(tower_loc)) - safe_radius) <= 0.01: 
        if len(on_dataset) < n_data:
            on_dataset.append(config)
    else: 
        if len(off_dataset) < n_data:
            off_dataset.append(config)

np.save('../nav_dataset_on_circle.npy', np.array(on_dataset))
np.save('../nav_dataset_off_circle.npy', np.array(off_dataset))
