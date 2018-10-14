import numpy as np
import scipy.spatial.distance as ssdist

class NBody:
    def __init__(self, N, G=1.0, K=0.1, D=3, M=None, P=None, V=None, integrator='euler'):
        self.D = D
        self.G = G
        self.K = K

        self.M = np.array(M).astype(float) if M else np.ones(N)
        self.P = np.array(P).astype(float) if P else np.random.random((N,D))
        self.V = np.array(V).astype(float) if V else np.zeros((N,D))

        if integrator == 'euler':
            self.step = self.step_euler
        elif integrator == 'rk2':
            self.step = self.step_rk2
        elif integrator == 'rk4':
            self.step = self.step_rk4

    def step_euler(self, dt):
        dV, dP = compute_derivatives(dt, self.G, self.K, self.M, self.V, self.P)
        self.V += dV
        self.P += dP

    def step_rk2(self, dt):
        dV1, dP1 = compute_derivatives(dt, self.G, self.K, self.M, self.V, self.P)
        dV2, dP2 = compute_derivatives(dt*0.5, self.G, self.K, self.M, self.V + dV1*dt*0.5, self.P + dP1*dt*0.5)

        self.V += dV2
        self.P += dP2
        
    def step_rk4(self, dt):
        dV1, dP1 = compute_derivatives(dt, self.G, self.K, self.M, self.V, self.P)
        dV2, dP2 = compute_derivatives(dt*0.5, self.G, self.K, self.M, self.V + dV1*dt*0.5, self.P + dP1*dt*0.5)
        dV3, dP3 = compute_derivatives(dt*0.5, self.G, self.K, self.M, self.V + dV2*dt*0.5, self.P + dP2*dt*0.5)
        dV4, dP4 = compute_derivatives(dt, self.G, self.K, self.M, self.V + dV3*dt, self.P + dP3*dt)

        self.V += (dV1 + 2*dV2 + 2*dV3 + dV4) / 6.0
        self.P += (dP1 + 2*dP2 + 2*dP3 + dP4) / 6.0

        
def compute_forces(G, K, M, V, P):
    N = len(M)

    dP = P[:, np.newaxis] - P[np.newaxis,:]
    r = ssdist.squareform(ssdist.pdist(P))
    r3 = np.power(r,2)
    r3[r==0] = 1 

    m1m2 = np.outer(M, M)[:,:,np.newaxis]
    Fg = G * m1m2 * (dP / r3[:,:,np.newaxis])
    Fd = - K * V

    return (Fg+Fd).sum(axis=0)

def compute_derivatives(dt, G, K, M, V, P):
    F = compute_forces(G, K, M, V, P)
    
    dV = dt * F / M[:,np.newaxis]
    dP = (V+dV) * dt + 0.5 * dV * dt * dt

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
    nb = NBody(2, integrator='rk4',
               D=2,
               K=0.1,
               #P = [1,-1]
               P = [[1,.5],[0,.5]],
               V = [[0,.1],[0,-.1]]
    )
    for i in range(50):
        save(nb, 'test%02d.jpg' % i)
        print("P")
        print(nb.P)
        nb.step(.1)

if __name__ == "__main__": main()


    
        
        
