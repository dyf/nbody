from flask import Flask, jsonify, render_template, url_for, request
from nbody import NBody
from multiprocessing import Process, Array, Lock, Queue
import numpy as np
import queue
import rules as nbr
import argparse
import simlib
import copy

SIM = None
P = None
R = None
Q = None

app = Flask(__name__)


def init_sim(parr, rarr):
    sim = copy.copy(SIM)

    sim['rules'] = [ nbr.Rule.from_dict(r['rtype'], r.get('params',{})) for r in SIM['rules'] ]
    nb = NBody(**sim)
    
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

def disable_flask_logging():
    import logging
    app.logger.disabled = True
    log = logging.getLogger('werkzeug')
    log.disabled = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--integrator', default='rk4')
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('sim', nargs='?', default='rand')

    args = parser.parse_args()

    if not args.logging:
        disable_flask_logging()

    np_dtype = np.dtype(args.dtype)
    np_ctype = np.ctypeslib._ctype_from_dtype(np_dtype)

    global SIM, P, R, Q
    
    SIM = simlib.find(args.sim, integrator=args.integrator, dtype=np_dtype, lock=Lock())
    P = Array(np_ctype, SIM['N']*SIM['D'], lock=False)
    R = Array(np_ctype, SIM['N'], lock=False)
    
    Q = Queue()
    PROC = Process(target=run_nbody, args=(Q,P,R))
    PROC.start()

    app.run()#threaded=True)

if __name__ == "__main__": main()    

    
