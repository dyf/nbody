from flask import Flask, jsonify, render_template, url_for
from nbody import NBody

app = Flask(__name__)
nb = NBody(
    4,
    integrator='rk4',
    D=3,
    K=0.1,
    M=[1,1,1,.01]
)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/bodies')
def positions():
    return jsonify(nb.P.tolist())

app.run(threaded=True)
