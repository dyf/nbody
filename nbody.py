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
        
        self.M = np.array(M).astype(dtype) if M else np.ones(N, dtype=dtype)
        self.P = np.array(P).astype(dtype) if P else np.random.random((N,D)).astype(dtype)
        self.V = np.array(V).astype(dtype) if V else np.random.random((N,D)).astype(dtype)
        self.R = np.array(R).astype(dtype) if R else np.ones(N, dtype=dtype)
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

        #print("p",self.P)

        # force due to collision
        r1r2 = (self.R[:, np.newaxis] + self.R[np.newaxis, :])[self.tidx]
        hits = np.where(r1r2 > r)
        if len(hits[0]):
            dPh = dP[hits] / r[hits] # unit vector normal to surface

            # I want to apply a force on both bodies 
            # I know what direction the force will be applied
            # momentum is the same on either side of the collision
            # let's just give either side half

            # m1v1i + m2v2i = m1v1f + m2v2f
            # m2v2i = m1v1f + m1v2f # relative to body 1

            # Pt = m1v1f + m2v2f
            #print("****")
            M1h = self.M[self.tidx[0][hits]]
            M2h = self.M[self.tidx[1][hits]]

            V1h = self.V[self.tidx[0][hits]]
            V2h = self.V[self.tidx[1][hits]]

            Vh = V2h - V1h
            Vhn = Vh / np.linalg.norm(Vh)

            ptotal = np.linalg.norm(M2h * Vh, axis=1) * 2

            # need to scale by how lined up they are
            scale = inner1d(dPh, Vhn)

            p1h = 0.5 * ptotal * dPh * scale
            p2h = -0.5 * ptotal * dPh * scale

            #print("before", F)
            F[self.tidx[0][hits],:] += p1h / dt
            F[self.tidx[1][hits],:] += p2h / dt
            #print("fafter", F)

        dV = dt * F / self.M[:,np.newaxis]
        #print("dV",dV)
        dP = (V+dV) * dt + 0.5 * dV * dt * dt
        #print("dP",dP)

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


    
        
        
