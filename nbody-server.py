from flask import Flask, jsonify, render_template, url_for, request
from nbody import NBody
from multiprocessing import Process, RawArray, Lock, Queue
import numpy as np
import queue

N = 4
D = 3
P = RawArray(np.ctypeslib.ctypes.c_float, N*D)
LOCK = Lock()
DTYPE = np.float32

def run_nbody(q):
    running = False
    
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
    dt = None

    while True:
        if running:
            nb.step(dt)

        try:
            command, cmd_dt = q.get(block=False)
            print(command, dt)
        except queue.Empty:
            continue

        if cmd_dt is not None:
            dt = cmd_dt
            
        if command == 'step':
            nb.step(dt)
        elif command == "stop":
            running = False
        elif command == "start":
            running = True
        elif command == "set_dt":
            pass
    
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

@app.route('/start')
def start():
    dt = float(request.args.get('dt'))
    Q.put(['start', dt])
    return jsonify({'msg':'success'})

@app.route('/set')
def set():
    dt = float(request.args.get('dt'))
    Q.put(['set_dt', dt])
    return jsonify({'msg':'success'})

@app.route('/stop')
def stop():
    Q.put(['stop', None])
    return jsonify({'msg':'success'})

if __name__ == "__main__":
    Q = Queue()
    PROC = Process(target=run_nbody, args=(Q,))
    PROC.start()

    app.run()#threaded=True)
