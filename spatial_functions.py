#!/usr/bin/env python3

# functions for computing spatial relationships between points

import numpy as np

def haversine(loc1, loc2, r = 6371):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    arguments: loc1 - array of coordinates of first point
               loc2 - array of coordinates of second point
               r (float, optional) - Radius of earth in kilometers.
    """
    
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [loc1[0], loc1[1], loc2[0], loc2[1]])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    return c * r
