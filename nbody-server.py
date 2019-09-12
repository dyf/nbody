from flask import Flask, jsonify, render_template, url_for, request
from nbody import NBody
from multiprocessing import Process, Array, Lock, Queue
import numpy as np
import queue
import rules as nbr
import argparse

SIMS = {
    'bounce': {
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
    },
    'rand': {
        'N': 100,
        'D': 3,  
        'rules': [
            { 'rtype': 'avoidance', 'params': { 'dist': 0.5, 'k': 100.0 } },
            { 'rtype': 'cohesion', 'params': { 'dist': 0.5, 'k': 16000.0 } },
            { 'rtype': 'alignment', 'params': { 'dist': 0.5, 'k': 2500.0 } },            
            { 'rtype': 'attractor', 'params': { 'point': [0.0,0.0,0.0], 'k': 100.0, 'atype': 'square' } }
        ],
        'R': np.random.random(100).astype(np.float32)*.05+.02
    }
}

SIM = {
    'dtype': np.float32,
    'integrator': 'rk4',
    'lock': Lock()
}

SIM.update(SIMS['rand'])

# TODO generalize for DTYPE
P = Array(np.ctypeslib.ctypes.c_float, SIM['N']*SIM['D'], lock=False)
R = Array(np.ctypeslib.ctypes.c_float, SIM['N'], lock=False)

def init_sim(parr, rarr):    
    SIM['rules'] = [ nbr.Rule.from_dict(r['rtype'], r['params']) for r in SIM['rules'] ]
    nb = NBody(**SIM)
    
    # use the shared array    
    p = np.frombuffer(parr, dtype=SIM['dtype']).reshape(nb.P.shape[0], nb.P.shape[1])    
    r = np.frombuffer(rarr, dtype=SIM['dtype']).reshape(nb.R.shape[0])
    p[:] = nb.P[:]
    r[:] = nb.R[:]
    nb.P = p
    nb.R = r



    return nb
    
def run_nbody(q, parr, rarr):
    running = False
    
    nb = init_sim(parr, rarr)
    
    dt = None

    while True:
        if running:
            nb.step(dt)

        try:
            command, cmd_dt = q.get(block=False)
        except queue.Empty:
            continue

        if command == 'step':
            dt = cmd_dt
            nb.step(dt)
        elif command == "toggle":
            dt = cmd_dt
            running = not running
        elif command == "set":
            dt = cmd_dt
        elif command == "reset":
            nb = init_sim(parr, rarr)
            running = False
            
    
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/bodies')
def bodies():    
    parr = np.frombuffer(P, dtype=SIM['dtype']).reshape(SIM['N'],SIM['D'])    
    rarr = np.frombuffer(R, dtype=SIM['dtype']).reshape(SIM['N'])
    return np.array([SIM['N'],SIM['D']], dtype=SIM['dtype']).tobytes() + parr.tobytes() + rarr.tobytes()
    
@app.route('/step')
def step():
    dt = float(request.args.get('dt'))
    Q.put(['step', dt])
    return jsonify({'msg':'success'})

@app.route('/set')
def set():
    dt = float(request.args.get('dt'))
    Q.put(['set', dt])
    return jsonify({'msg':'success'})

@app.route('/reset')
def reset():
    Q.put(['reset', None])
    return jsonify({'msg':'success'})

@app.route('/toggle')
def toggle():
    dt = float(request.args.get('dt'))
    Q.put(['toggle', dt])
    return jsonify({'msg':'success'})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging', action='store_true')
    args = parser.parse_args()

    if not args.logging:
        import logging
        app.logger.disabled = True
        log = logging.getLogger('werkzeug')
        log.disabled = True

    Q = Queue()
    PROC = Process(target=run_nbody, args=(Q,P,R))
    PROC.start()

    app.run()#threaded=True)
