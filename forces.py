import numpy as np
from numpy.core.umath_tests import inner1d

class Force: pass
class CorrectiveForce: pass

class Gravity(Force):
    def __init__(self, G):
        self.G = G
        
    def compute(self, dt, V, P, M, R, dP, r, tidx, lidx, buf_pairs, buf_items):
        r3 = np.power(r,3)
        r3[r==0] = 1
        
        m1m2 = np.outer(M, M)[tidx][:,np.newaxis]
        Fg = self.G * m1m2 * (dP / r3[:,np.newaxis])
    
        buf_pairs[tidx] = Fg
        buf_pairs[lidx] = -np.swapaxes(buf_pairs,0,1)[lidx]
        
        return buf_pairs.sum(axis=0)

class Drag(Force):
    def __init__(self, K):
        self.K = K

    def compute(self, dt, V, P, M, R, dP, r, tidx, lidx, buf_pairs, buf_items):
        if self.K != 0:
            return -self.K * V
        else:
            buf_items.fill(0)
            return buf_items

class Collision(CorrectiveForce):
    def compute(self, dt, V, P, M, R, dP, r, tidx, lidx, buf_pairs, buf_items):
        F = buf_items
        F.fill(0)
        
        dPcorr = np.zeros_like(F)
        
        # detect collisions
        r1r2 = (R[:, np.newaxis] + R[np.newaxis, :])[tidx]
        hits = np.where(r1r2 > r)[0]

        if len(hits):
            # upper-triangle indices of colliding pairs
            b1_idx = tidx[0][hits]
            b2_idx = tidx[1][hits]

            # unit collision vector
            dPh = dP[hits] / r[hits, np.newaxis] 

            # half of the overlapping distance, used for undoing overlap
            hdr = 0.5 * (r[hits] - r1r2[hits])

            # position correction vectors
            dPcorr[b1_idx,:] = -dPh * hdr[:,np.newaxis]
            dPcorr[b2_idx,:] = dPh * hdr[:,np.newaxis]

            # masses and velocities of colliding pairs
            M1h = M[b1_idx]
            M2h = M[b2_idx]
            V1h = V[b1_idx]
            V2h = V[b2_idx]

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

            F[b1_idx,:] += M1h[:, np.newaxis] * ((V1f + V1o) - V1h) / dt
            F[b2_idx,:] += M2h[:, np.newaxis] * ((V2f + V2o) - V2h) / dt

        return F, dPcorr

        
