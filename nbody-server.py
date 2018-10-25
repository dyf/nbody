from flask import Flask, jsonify, render_template, url_for, request
from nbody import NBody
from multiprocessing import Process, RawArray, Lock, Queue
import numpy as np
import queue

N = 2
D = 3
P = RawArray(np.ctypeslib.ctypes.c_float, N*D)
R = RawArray(np.ctypeslib.ctypes.c_float, N)

LOCK = Lock()
DTYPE = np.float32
K = 0.7

def init_sim():
    nb = NBody(
        N,
        integrator='rk4',
        D=D,
        P=[[0,.5,.5],
           [1,.5,.5]],
        R=[.1,.1],
        V=[[1,0,0],
           [-1,0,0]],
        M=[1,1],
        K=0,
        lock=LOCK,
        dtype=DTYPE
    )
    
    #nb = NBody(
    #    N,
    #    integrator='rk4',
    #    D=D,
    #    K=K,
    #    lock=LOCK,
    #    dtype=DTYPE
    #)
    #nb.M[0] = 1000.0
    #nb.fix(0)
    #nb.V = 100

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
            print(command, cmd_dt)
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
