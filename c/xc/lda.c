#include "xc_gpaw.h"


void fastlda(int N,
             double *e,
             const double *n,
             double *v)
{
    double A = GAMMA;
    double alpha1 = 0.21370;
    double beta1 = 7.5957;
    double beta2 = 3.5876;
    double beta3 = 1.6382;
    double beta4 = 0.49294;
    for (int r = 0; r < N; r++) {
        double d = n[r];
        if (d < 1e-15) {
            e[r] = 0.0;
            continue;
        }
        double rs = pow(C0I / d, THIRD);
        double rtrs = sqrt(rs);
        double Q0 = -2.0 * A * (1.0 + alpha1 * rtrs * rtrs);
        double Q1 = 2.0 * A * rtrs * (beta1 +
                                      rtrs * (beta2 +
                                              rtrs * (beta3 +
                                                      rtrs * beta4)));
        double ec = Q0 * log(1.0 + 1.0 / Q1);
        double dQ1drs = A * (beta1 / rtrs + 2.0 * beta2 +
                             rtrs * (3.0 * beta3 + 4.0 * beta4 * rtrs));
        double decdrs = (-2.0 * A * alpha1 * ec / Q0
                         - Q0 * dQ1drs / (Q1 * (Q1 + 1.0)));
        double ex = C1 / rs;
        double dexdrs = -ex / rs;
        e[r] = d * (ex + ec);
        v[r] += ex + ec - rs * (dexdrs + decdrs) / 3.0;
    }
}
