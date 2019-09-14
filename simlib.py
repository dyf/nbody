import numpy as np

def bounce():
    return {
        'N': 4,
        'D': 3,  
        'rules': [
            { 'rtype': 'avoidance', 'params': { 'dist': 1.1, 'k': 100.0 } },
            { 'rtype': 'cohesion', 'params': { 'dist': 10.0, 'k': 1000.0 } },
            { 'rtype': 'alignment', 'params': { 'dist': 1.1, 'k': 100.0 } },
            { 'rtype': 'collision' }
        ],
        'P': [[0,0,0],
              [1,0,0],
              [1,1,0],
              [0,1,0]],
        'R': [.1,.1,.1,.1],
        'V': [[0,1,0],
              [0,-1,0],
              [0,-1,0],
              [0,1,0]],
        'M': [100,100,100,100]
    }

def rand():
    return {
        'N': 100,
        'D': 3,  
        'rules': [
            { 'rtype': 'avoidance', 'params': { 'dist': 0.5, 'k': 100.0 } },
            { 'rtype': 'cohesion', 'params': { 'dist': 0.5, 'k': 16000.0 } },
            { 'rtype': 'alignment', 'params': { 'dist': 0.5, 'k': 2500.0 } },            
            { 'rtype': 'attractor', 'params': { 'point': [0.0,0.0,0.0], 'k': 100.0, 'atype': 'square' } }
        ],
        'R': np.random.random(100)*.05+.02
    }

def find(name, **params):
    sim = globals()[name]()
    sim.update(params)
    return sim




