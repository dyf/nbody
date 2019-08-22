import numpy as np
from numpy.core.umath_tests import inner1d

class Rule: pass
class Force(Rule): pass
class CorrectiveForce(Rule): pass

class Gravity(Force):
    def __init__(self, G):
        self.G = G
        
    def compute(self, nbs, dt, buf_items, buf_pairs):
        r2i = nbs.pdist2i_dense[:,np.newaxis]
        #r2 = nbs.pdist2_dense[:,np.newaxis]
        #r2[r2<0.001]=0.001

        m1m2 = np.outer(nbs.M, nbs.M)[nbs.tidx][:,np.newaxis]
        v = nbs.unit_p1p2_dense

        Fg = self.G * m1m2 * r2i * v

        #print(np.allclose((self.G * m1m2 / r2 * v), (self.G * m1m2 * r2i * v)))

        buf_pairs[nbs.tidx] = Fg
        buf_pairs[nbs.lidx] = -np.swapaxes(buf_pairs,0,1)[nbs.lidx]
        
        return buf_pairs.sum(axis=0)

class Drag(Force):
    def __init__(self, K):
        self.K = K

    def compute(self, nbs, dt, buf_items, buf_pairs):
        if self.K != 0:
            return -self.K * nbs.V
        else:
            buf_items.fill(0)
            return buf_items

class Collision(CorrectiveForce):
    def compute(self, nbs, dt, buf_items, buf_pairs):
        F = buf_items
        F.fill(0)
        
        dPcorr = np.zeros_like(F)

        # detect collisions
        overlaps, b1_idx, b2_idx = nbs.overlapping_pairs

        if len(overlaps):
            vdv = inner1d(nbs.V[b1_idx], nbs.V[b2_idx])
            approaching = vdv < -0.01
            
            if np.sum(approaching):
                hits = overlaps[approaching]
                b1h_idx = b1_idx[approaching]
                b2h_idx = b2_idx[approaching]

                # unit collision vector
                dPh = nbs.unit_p1p2_dense[hits]

                # masses and velocities of colliding pairs
                M1h = nbs.M[b1h_idx]
                M2h = nbs.M[b2h_idx]
                V1h = nbs.V[b1h_idx]
                V2h = nbs.V[b2h_idx]

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

                F[b1h_idx,:] += M1h[:, np.newaxis] * ((V1f + V1o) - V1h) / dt
                F[b2h_idx,:] += M2h[:, np.newaxis] * ((V2f + V2o) - V2h) / dt

            dPh = nbs.unit_p1p2_dense[overlaps]
            hdr = 0.5 * (nbs.pdist_dense[overlaps] - nbs.r1r2_dense[overlaps])[:,np.newaxis]
                
            Mr = (nbs.M[b1_idx] / (nbs.M[b1_idx] + nbs.M[b2_idx]))[:,np.newaxis]
            dPcorr[b1_idx,:] -= Mr * dPh * hdr
            dPcorr[b2_idx,:] += Mr * dPh * hdr

        return F, dPcorr

class Separation(Force):
    def __init__(self, dist, k):
        self.dist = dist
        self.k = k

    def compute(self, nbs, dt, buf_pairs, buf_items):
        F = buf_items
        F.fill(0)

        # if they've collided, bouncing will deal with that
        r1r2 = (R[:, np.newaxis] + R[np.newaxis, :])[tidx]
        near = np.where((r < self.dist) & (r>r1r2))[0]

        if len(near) > 0:
            # upper-triangle indices of near pairs
            b1_idx = tidx[0][near]
            b2_idx = tidx[1][near]

            # avoidance vector
            dPv = dP[near] / np.power(r[near, np.newaxis], 3) * self.k

            F[b1_idx,:] = dPv
            F[b2_idx,:] = -dPv

        return F
