from flask import Flask, jsonify, render_template, url_for, request
from nbody import NBody
from multiprocessing import Process, RawArray, Lock, Queue
import numpy as np
import queue

N = 2
D = 3
P = RawArray(np.ctypeslib.ctypes.c_float, N*D)
R = RawArray(np.ctypeslib.ctypes.c_float, N)
INTEGRATOR='rk4'

LOCK = Lock()
DTYPE = np.float32
K = 0.7

def init_nb_bounce():
    nb = NBody(
        N,
        integrator=INTEGRATOR,
        D=D,
        P=[[0,0,.5],
           [1,1,.5]],
        R=[.1,.1],
        V=[[10,0,0],
           [0,-10,0]],
        M=[3,1],
        K=0,
        lock=LOCK,
        dtype=DTYPE
    )

    return nb

def init_nb_rand():
    nb = NBody(
        N,
        integrator=INTEGRATOR,
        D=D,
        K=K,
        R=np.ones(N, dtype=DTYPE)*.05,
        lock=LOCK,
        dtype=DTYPE
    )
    nb.M[0] = 1000.0
    nb.fix(0)

    return nb

def init_sim():
    nb = init_nb_rand()
    #nb = init_nb_bounce()
    
    # use the shared array
    #p = np.frombuffer(P, dtype=DTYPE).reshape(N,D)
    p = np.frombuffer(P, dtype=DTYPE).reshape(nb.P.shape[0], nb.P.shape[1])
    r = np.frombuffer(R, dtype=DTYPE).reshape(nb.R.shape[0])
    np.copyto(p, nb.P)
    np.copyto(r, nb.R)
    nb.P = p
    nb.R = r

    return nb
    
def run_nbody(q):
    running = False
    
    nb = init_sim()
    
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
            nb = init_sim()
            running = False
            
    
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/bodies')
def bodies():
    parr = np.frombuffer(P, dtype=DTYPE).reshape(N,D)
    rarr = np.frombuffer(R, dtype=DTYPE).reshape(N)
    return np.array([N,D], dtype=DTYPE).tobytes() + parr.tobytes() + rarr.tobytes()

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
    Q = Queue()
    PROC = Process(target=run_nbody, args=(Q,))
    PROC.start()

    app.run()#threaded=True)
