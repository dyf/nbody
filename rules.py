import numpy as np
from numpy.core.umath_tests import inner1d
from collections import Counter

class Rule: pass
class Force(Rule): pass
class CorrectiveForce(Rule): pass

class Gravity(Force):
    def __init__(self, G):
        self.G = G
        
    def compute(self, nbs, dt, buf_items, buf_pairs):
        r2i = nbs.pdist2i_dense[:,np.newaxis]

        m1m2 = np.outer(nbs.M, nbs.M)[nbs.tidx][:,np.newaxis]
        v = nbs.unit_p1p2_dense

        Fg = self.G * m1m2 * r2i * v

        buf_pairs[nbs.tidx] = Fg
        buf_pairs[nbs.lidx] = -Fg

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

class Avoidance(Force):
    def __init__(self, dist, k):
        self.dist = dist
        self.k = k

    def compute(self, nbs, dt, buf_items, buf_pairs):
        F = buf_pairs
        F.fill(0)

        near, b1_idx, b2_idx = nbs.near_pairs(self.dist)

        if len(near) > 0:
            # avoidance vector
            dPv = nbs.unit_p1p2_dense[near] / nbs.pdist2_dense[near,np.newaxis] * self.k

            F[b1_idx,b2_idx,:] = -dPv
            F[b2_idx,b1_idx,:] = dPv

        return F.sum(axis=0)

class Cohesion(Force):
    def __init__(self, dist, k):
        self.dist = dist
        self.k = k

    def compute(self, nbs, dt, buf_items, buf_pairs):
        C = buf_items
        F = buf_pairs
        C.fill(0)
        F.fill(0)

        near, b1_idx, b2_idx = nbs.near_pairs(self.dist)

        if len(near) > 0:
            # centroids
            cts = Counter(np.append(b1_idx,b2_idx))
            for i1,i2 in zip(b1_idx, b2_idx):
                C[i1] += nbs.P[i2] / cts[i1]
                C[i2] += nbs.P[i1] / cts[i2]

            idx = np.array(list(cts.keys()))
            V = C[idx] - nbs.P[idx]
            V /= np.linalg.norm(V)

        return V * self.k
