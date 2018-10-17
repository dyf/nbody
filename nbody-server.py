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

    while True:
        if running:
            nb.step(0.01)

        try:
            msg = q.get(block=False)
        except queue.Empty:
            continue
            

        if msg == 'step':
            nb.step(0.01)
        elif msg == "stop":
            running = False
        elif msg == "start":
            running = True
    
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
    Q.put('step')
    return jsonify({'msg':'success'})

@app.route('/start')
def start():
    Q.put('start')
    return jsonify({'msg':'success'})

@app.route('/stop')
def stop():
    Q.put('stop')
    return jsonify({'msg':'success'})

if __name__ == "__main__":
    Q = Queue()
    PROC = Process(target=run_nbody, args=(Q,))
    PROC.start()

    app.run()#threaded=True)
