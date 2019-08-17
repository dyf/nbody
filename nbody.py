import numpy as np
import scipy.spatial.distance as ssdist
import forces
from integrators import Integrator
import functools

np.random.seed(0)

class NBody:
    def __init__(self, N, G=100.0, K=0.1, D=3, M=None, P=None, V=None, R=None, integrator='euler', dtype=np.float32, lock=None):
        self.D = D
        self.G = G
        self.K = K
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
        
    def step_euler(self, dt):
        return self.compute_derivatives(dt, self.V, self.P)

    def step_rk2(self, dt):
        dV1, dP1 = self.compute_derivatives(dt, self.V, self.P)
        return self.compute_derivatives(dt*0.5, self.V + dV1*dt*0.5, self.P + dP1*dt*0.5)

    def step_rk4(self, dt):
        dV1, dP1 = self.compute_derivatives(dt, self.V, self.P)
        dV2, dP2 = self.compute_derivatives(dt*0.5, self.V + dV1*dt*0.5, self.P + dP1*dt*0.5)
        dV3, dP3 = self.compute_derivatives(dt*0.5, self.V + dV2*dt*0.5, self.P + dP2*dt*0.5)
        dV4, dP4 = self.compute_derivatives(dt, self.V + dV3*dt, self.P + dP3*dt)
        return ( (dV1 + 2*dV2 + 2*dV3 + dV4) / 6.0,
                 (dP1 + 2*dP2 + 2*dP3 + dP4) / 6.0 )

    def compute_derivatives(self, dt, V, P):
        N = len(self.M)

        dP = (P[:, np.newaxis] - P[np.newaxis,:])[self.tidx]
        r = ssdist.pdist(P)

        F = np.zeros_like(V)
        F += forces.Gravity(self.G).compute(dt, V, P, self.M, self.R, dP, r, self.tidx, self.lidx, self.buf_pairs, self.buf_items)
        F += forces.Drag(self.K).compute(dt, V, P, self.M, self.R, dP, r, self.tidx, self.lidx, self.buf_pairs, self.buf_items)

        Fcoll, Poff = forces.Collision().compute(dt, V, P, self.M, self.R, dP, r, self.tidx, self.lidx, self.buf_pairs, self.buf_items)
        F += Fcoll

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


    
        
        
