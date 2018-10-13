import numpy as np

class NBody:
    def __init__(self, N, G=-1.0, D=3, M=None, P=None, V=None, integrator='euler'):
        self.D = 3
        self.G = G

        self.M = np.array(M).astype(float) if M else np.random.random(N)
        self.P = np.array(P).astype(float) if P else np.random.random((N,D))
        self.V = np.array(V).astype(float) if V else np.zeros((N,D))

        if integrator == 'euler':
            self.step = self.step_euler
        elif integrator == 'rk4':
            self.step = self.step_rk4

    def step_euler(self, dt):
        dV, dP = compute_derivatives(dt, self.G, self.M, self.V, self.P)
        self.V += dV
        self.P += dP

    def step_rk4(self, dt):
        dV1, dP1 = compute_derivatives(dt, self.G, self.M, self.V, self.P)

        dV2, dP2 = compute_derivatives(dt*0.5, self.G, self.M, self.V + dV1*dt*0.5, self.P + dP1*dt*0.5)

        dV3, dP3 = compute_derivatives(dt*0.5, self.G, self.M, self.V + dV2*dt*0.5, self.P + dP2*dt*0.5)

        dV4, dP4 = compute_derivatives(dt, self.G, self.M, self.V + dV3*dt, self.P + dP3*dt)

        self.V += (dV1 + 2*dV2 + 2*dV3 + dV4) / 6.0
        self.P += (dP1 + 2*dP2 + 2*dP3 + dP4) / 6.0

        
def compute_forces(G, M, P):
    N = len(M)

    dP = P[:, np.newaxis] - P[np.newaxis,:]
    
    r3 = np.abs(np.power(dP,3))
    r3[np.eye(N,dtype=np.uint8)] = 1
        
    m1m2 = np.outer(M, M)[:,:,np.newaxis]
    
    F = -G * m1m2 * dP / r3
    return F.sum(axis=0)

def compute_derivatives(dt, G, M, V, P):
    F = compute_forces(G, M, P)
    
    dV = dt * F / M[:,np.newaxis]
    dP = (V+dV) * dt + 0.5 * dV * dt * dt

    return dV, dP

def main():
    nb = NBody(2, P=[[1,0,0],[-1,0,0,]], M=[1,1], integrator='rk4')
    nb.step(.1)
    print(nb.P)
    nb.step(.1)
    print(nb.P)

if __name__ == "__main__": main()


    
        
        
