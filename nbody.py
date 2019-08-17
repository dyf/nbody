import numpy as np
import scipy.spatial.distance as ssdist
import forces
from integrators import Integrator
import functools

np.random.seed(0)

class NBody:
    def __init__(self, N, G=100.0, K=0.1, D=3, collision=True, M=None, P=None, V=None, R=None, integrator='euler', dtype=np.float32, lock=None):
        self.forces = []
        self.corrective_forces = []

        if G is not None:
            self.forces.append(forces.Gravity(G))

        if K is not None:
            self.forces.append(forces.Drag(K))

        if collision:
            self.corrective_forces.append(forces.Collision())
            
        self.D = D
        self.lock = lock
        
        self.M = np.array(M).astype(dtype) if M is not None else np.ones(N, dtype=dtype)
        self.P = np.array(P).astype(dtype) if P is not None else np.random.random((N,D)).astype(dtype)
        self.V = np.array(V).astype(dtype) if V is not None else np.random.random((N,D)).astype(dtype)
        self.R = np.array(R).astype(dtype) if R is not None else np.ones(N, dtype=dtype)

        self.buf_pairs = np.zeros((N,N,D), dtype=dtype)
        self.buf_items = np.zeros((N,D), dtype=dtype)

        self.tidx = np.triu_indices(N, k=1)
        self.lidx = np.tril_indices(N, k=-1)


        self.integrator = Integrator.new(integrator, functools.partial(self.compute_derivatives))

        self.fixed = {}

    def fix(self, i):
        self.fixed[i] = self.P[i]
        
    def step(self, dt):
        dV, dP = self.integrator.step(dt, self.V, self.P)

        if self.lock:
            self.lock.acquire()

        self.V += dV
        self.P += dP

        for i,pi in self.fixed.items():
            self.P[i] = pi

        if self.lock:
            self.lock.release()

    def compute_derivatives(self, dt, V, P):
        N = len(self.M)

        dP = (P[:, np.newaxis] - P[np.newaxis,:])[self.tidx]
        r = ssdist.pdist(P)

        F = np.zeros_like(V)
        Poff = np.zeros_like(P)
        
        for f in self.forces:
            F += f.compute(dt, V, P, self.M, self.R, dP, r, self.tidx, self.lidx, self.buf_pairs, self.buf_items)

        for f in self.corrective_forces:
            Fcf, dPcf = f.compute(dt, V, P, self.M, self.R, dP, r, self.tidx, self.lidx, self.buf_pairs, self.buf_items)
            F += Fcf
            Poff += dPcf

        dV = dt * F / self.M[:,np.newaxis]
        dP = (V+dV) * dt + 0.5 * dV * dt * dt + Poff

        return dV, dP

def save(nb, file_name):
    import matplotlib
    matplotlib.use('agg')
    #from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    plt.scatter(x=nb.P[:,0],
                y=nb.P[:,1],
                marker='o',
                s=10)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    #ax.set_zlim([0,1])
    
    plt.savefig(file_name)
    plt.close()
    
def main():
    np.set_printoptions(precision=5, suppress=True)
    nb = NBody(2, integrator='rk2',
               D=2,
               K=0,
               M = [1,1],
               P = [[-1,0],[1,0]],
               R = [ 0.5, 0.5],
               V = [ [0,0], [0,0] ]
               #P = [1,-1]
               #P = [[1,.5],[0,.5]],
               #V = [[0,.1],[0,-.1]]
    )
    print(nb.P)
    for i in range(10):
        #save(nb, 'test%02d.jpg' % i)
        #print("P")

        nb.step(.01)
        print(nb.P)

if __name__ == "__main__": main()


    
        
        
