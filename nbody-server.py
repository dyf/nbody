from flask import Flask, jsonify, render_template, url_for, request
from nbody import NBody
from multiprocessing import Process, RawArray, Lock, Queue
import numpy as np
import queue

N = 20
D = 3
P = RawArray(np.ctypeslib.ctypes.c_float, N*D)
LOCK = Lock()
DTYPE = np.float32

def init_sim():
    nb = NBody(
        N,
        integrator='rk4',
        D=D,
        K=0.1,
        lock=LOCK,
        dtype=DTYPE
    )

    # use the shared array
    p = np.frombuffer(P, dtype=DTYPE).reshape(N,D)
    np.copyto(p, nb.P)
    nb.P = p

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
            print(command, dt)
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
    arr = np.frombuffer(P, dtype=DTYPE).reshape(N,D)
    return arr.tobytes()

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
