import numpy as np

class NBody:
    def __init__(self, N, G=-1.0, D=3, M=None, P=None, V=None):
        self.D = 3
        self.G = G

        self.M = np.array(M).astype(float) if M else np.random.random(N)
        self.P = np.array(P).astype(float) if P else np.random.random((N,D))
        self.V = np.array(V).astype(float) if V else np.zeros((N,D))

    def compute_forces(self):
        N = len(self.M)

        dP = self.P[:, np.newaxis] - self.P[np.newaxis,:]
        
        r3 = np.abs(np.power(dP,3))
        r3[np.eye(N,dtype=np.uint8)] = 1
        
        m1m2 = np.outer(self.M, self.M)[:,:,np.newaxis]
        
        F = -self.G * m1m2 * dP / r3
        return F.sum(axis=0)

    def advect(self, F, dt):
        dV = F * dt / self.M[:,np.newaxis]
        self.V += dV
        self.P += self.V * dt + 0.5 * dV * dt * dt
        
    def step(self, dt):
        F = self.compute_forces()
        self.advect(F, dt)

def main():
    nb = NBody(2, P=[[1,0,0],[-1,0,0,]], M=[1,1])
    nb.step(.1)
    print(nb.P)
    nb.step(.1)
    print(nb.P)

if __name__ == "__main__": main()


    
        
        
