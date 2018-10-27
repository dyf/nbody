import numpy as np
import scipy.spatial.distance as ssdist
from numpy.core.umath_tests import inner1d

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
        self.Fbuf = np.zeros((N,N,D), dtype=dtype)

        self.tidx = np.triu_indices(N, k=1)
        self.lidx = np.tril_indices(N, k=-1)

        if integrator == 'euler':
            self.step_fn = self.step_euler
        elif integrator == 'rk2':
            self.step_fn = self.step_rk2
        elif integrator == 'rk4':
            self.step_fn = self.step_rk4

        self.fixed = {}

    def fix(self, i):
        self.fixed[i] = self.P[i]
        
    def step(self, dt):
        dV, dP = self.step_fn(dt)

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

    def compute_derivatives_full(self, dt, V, P):
        N = len(self.M)

        # pairwise distances
        dP = (P[:, np.newaxis] - P[np.newaxis,:])
        r = ssdist.squareform(ssdist.pdist(self.P))
        r3 = np.power(r,3)
        r3[r==0] = 1 

        # force due to gravity
        m1m2 = np.outer(self.M, self.M)[:,:,np.newaxis]
        Fg = self.G * m1m2 * (dP / r3[:,:,np.newaxis])
        F = Fg.sum(axis=0)

        # force due to drag
        if self.K != 0:
            F += - self.K * V

        dV = dt * F / self.M[:,np.newaxis]
        dP = (V+dV) * dt + 0.5 * dV * dt * dt

        return dV, dP
        
    def compute_derivatives(self, dt, V, P):
        N = len(self.M)

        # pairwise distances
        dP = (P[:, np.newaxis] - P[np.newaxis,:])[self.tidx]
        r = ssdist.pdist(self.P)
        r3 = np.power(r,3)
        r3[r==0] = 1 

        # force due to gravity
        m1m2 = np.outer(self.M, self.M)[self.tidx][:,np.newaxis]
        Fg = self.G * m1m2 * (dP / r3[:,np.newaxis])

        self.Fbuf[self.tidx] = Fg
        self.Fbuf[self.lidx] = -np.swapaxes(self.Fbuf,0,1)[self.lidx]
        F = self.Fbuf.sum(axis=0)
        
        # force due to drag
        if self.K != 0:
            F += - self.K * V

        # force due to collision
        r1r2 = (self.R[:, np.newaxis] + self.R[np.newaxis, :])[self.tidx]
        hits = np.where(r1r2 > r)[0]

        offsets = None
        if len(hits):
            dPh = dP[hits] / r[hits, np.newaxis] # unit vector normal to surface
            src_idx = self.tidx[0][hits]
            tgt_idx = self.tidx[1][hits]
            
            hdr = 0.5 * (r[hits] - r1r2[hits])

            offsets = np.zeros_like(self.P)
            offsets[src_idx,:] = -dPh * hdr[:,np.newaxis]
            offsets[tgt_idx,:] = dPh * hdr[:,np.newaxis]

            M1h = self.M[src_idx]
            M2h = self.M[tgt_idx]

            V1h = self.V[src_idx]
            V2h = self.V[tgt_idx]

            
            
            # project V1 and V2 onto dP
            bdb = inner1d(dPh, dPh)
            V1p = (inner1d(V1h, dPh) / bdb)[:, np.newaxis] * dPh
            V2p = (inner1d(V2h, dPh) / bdb)[:, np.newaxis] * dPh
            
            # orthogonal component sticks around
            V1o = V1h - V1p
            V2o = V2h - V2p

            # new velocities after collision
            V1f = ((M1h - M2h) / (M1h + M2h))[:, np.newaxis] * V1p  +  (2 * M2h / (M1h + M2h))[:, np.newaxis] * V2p
            V2f = (2 * M1h / (M1h + M2h))[:, np.newaxis] * V1p  -  ((M1h - M2h) / (M1h + M2h))[:, np.newaxis] * V2p 

            # f = m * dv / dt
            F1 = M1h[:, np.newaxis] * ((V1f + V1o) - V1h) / dt
            F2 = M2h[:, np.newaxis] * ((V2f + V2o) - V2h) / dt

            F[src_idx,:] += F1
            F[tgt_idx,:] += F2



        dV = dt * F / self.M[:,np.newaxis]


        dP = (V+dV) * dt + 0.5 * dV * dt * dt
        if offsets is not None:
            dP += offsets

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
    nb = NBody(2, integrator='euler',
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
    for i in range(1000):
        #save(nb, 'test%02d.jpg' % i)
        #print("P")
        #print(nb.P)
        nb.step(.01)

if __name__ == "__main__": main()


    
        
        
