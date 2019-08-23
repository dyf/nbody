import numpy as np
import scipy.spatial.distance as ssdist
import rules as nbr
from integrators import Integrator
import functools as ft

np.random.seed(0)

class NBody:
    def __init__(self, N, D, rules=None, M=None, P=None, V=None, R=None, integrator='euler', dtype=np.float32, lock=None):
        rules = [] if rules is None else rules
        self.forces = [ r for r in rules if isinstance(r, nbr.Force) ] 
        self.corrective_forces = [ r for r in rules if isinstance(r, nbr.CorrectiveForce) ] 

        self.D = D
        self.lock = lock
        
        self.M = np.array(M).astype(dtype) if M is not None else np.ones(N, dtype=dtype)
        self.P = np.array(P).astype(dtype) if P is not None else np.random.random((N,D)).astype(dtype)
        self.V = np.array(V).astype(dtype) if V is not None else np.random.random((N,D)).astype(dtype)
        self.R = np.array(R).astype(dtype) if R is not None else np.ones(N, dtype=dtype)

        self.buf_pairs = np.zeros((N,N,D), dtype=dtype)
        self.buf_items = np.zeros((N,D), dtype=dtype)

        self.tidx = np.triu_indices(N, k=1)
        self.lidx = ( self.tidx[1], self.tidx[0] )


        self.integrator = Integrator.new(integrator, ft.partial(self.compute_derivatives))

        self.fixed = {}

    def fix(self, i):
        self.fixed[i] = self.P[i]
        
    def step(self, dt):
        if dt == 0.0:
            return
        
        dV, dP = self.integrator.step(dt, self.V, self.P)

        if self.lock:
            self.lock.acquire()

        self.V += dV
        self.P += dP

        #print("KE",sum(self.M*(np.linalg.norm(self.V, axis=1)**2)*0.5))

        for i,pi in self.fixed.items():
            self.P[i] = pi

        if self.lock:
            self.lock.release()

    def compute_derivatives(self, dt, V, P):
        nbs = NBodyState(self, V=V, P=P)
        
        F = np.zeros_like(V)
        Poff = np.zeros_like(P)
        
        for f in self.forces:
            F += f.compute(nbs, dt, buf_pairs=self.buf_pairs, buf_items=self.buf_items)

        for f in self.corrective_forces:
            Fcf, dPcf = f.compute(nbs, dt, buf_pairs=self.buf_pairs, buf_items=self.buf_items)
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

def cached_property(fn):
    return property(ft.lru_cache(None)(fn))

class NBodyState:
    def __init__(self, nb, P=None, V=None, eps=0.001):
        self.M = nb.M
        self.R = nb.R
        self.P = P if P is not None else nb.P
        self.V = V if V is not None else nb.V
        self.eps = eps

        self.tidx = nb.tidx
        self.lidx = nb.lidx

    @cached_property
    def p1p2(self):
        return self.P[:, np.newaxis] - self.P[np.newaxis,:]
    
    @cached_property
    def p1p2_dense(self):
        return self.p1p2[self.tidx]

    @cached_property
    def unit_p1p2_dense(self):
        return self.p1p2_dense / self.pdist_dense[:, np.newaxis]
    
    @cached_property
    def pdist_dense(self):
        return ssdist.pdist(self.P)

    @cached_property
    def pdist2_dense(self):
        return np.square(self.pdist_dense)

    @cached_property
    def pdist2i_dense(self):
        p2 = self.pdist2_dense
        p2i = np.ones_like(p2) / self.eps

        far_enough = p2 > self.eps
        p2i[far_enough] = 1.0 / p2[far_enough]

        return p2i

    @cached_property
    def r1r2(self):
        return self.R[:, np.newaxis] + self.R[np.newaxis, :]

    @cached_property
    def r1r2_dense(self):
        return self.r1r2[self.tidx]
        
    @cached_property
    def overlapping_pairs(self):
        idx = np.where(self.pdist_dense <= self.r1r2_dense)[0]
        return idx, self.tidx[0][idx], self.tidx[1][idx]

    @ft.lru_cache(None)
    def near_pairs(self, dist):
        idx = np.where(self.pdist_dense < dist)[0]
        return idx, self.tidx[0][idx], self.tidx[1][idx]
    
def main():
    np.set_printoptions(precision=5, suppress=True)
    nb = NBody(2, integrator='rk2',
               D=2,
               rules=[
                   nbr.Gravity(100.0),
                   nbr.Drag(0.0)
               ],
               M = [1,1],
               P = [[-1,0],[1,0]],
               R = [ 0.5, 0.5],
               V = [ [0,0], [0,0] ]
               #P = [1,-1]
               #P = [[1,.5],[0,.5]],
               #V = [[0,.1],[0,-.1]]
    )

    nbs = NBodyState(nb)
    print("P", nbs.P)
    print("p1p2", nbs.p1p2)
    print("unit_p1p2", nbs.unit_p1p2_dense)
    
    for i in range(3):
        #save(nb, 'test%02d.jpg' % i)
        nb.step(.01)
        #print(nb.P)

if __name__ == "__main__": main()


    
        
        
