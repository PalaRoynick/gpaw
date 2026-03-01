#include "xc_hlmgga.h"
#include <math.h>

/* exchange */
double _mgga_x_thetapbe(const void *p, const double n, const double rs,
                        const double a2, const double mu, double *dedrs,
                        double *deda2, double *dedmu) {
  double e = C1 / rs;
  *dedrs = -e / rs;
  double c = C2 * rs / n;
  c *= c;
  double s2 = a2 * c;
  double dxdmu = s2 / KAPPA;
  double x = 1.0 + mu * dxdmu;
  double Fx = 1.0 + KAPPA - KAPPA / x;
  double i_xsquared = 1.0 / (x * x);
  double dFxds2 = mu * i_xsquared;
  double ds2drs = 8.0 * c * a2 / rs;
  *dedrs = *dedrs * Fx + e * dFxds2 * ds2drs;
  *deda2 = e * dFxds2 * c;
  *dedmu = e * KAPPA * i_xsquared * dxdmu;
  e *= Fx;
  return e;
}

double mgga_x_thetapbe(void *p, const double n, const double rs,
                       const double a2, const double g_, double *dedrs,
                       double *deda2, double *dedg_) {
  double mu = (g_ * MU_H + (1.0 - g_) * MU_GE);
  double e = _mgga_x_thetapbe(p, n, rs, a2, mu, dedrs, deda2, dedg_);
  *dedg_ *= (MU_H - MU_GE);
  return e;
}

// defined in pbe.c
double G(double rtrs, double A, double alpha1, double beta1, double beta2,
         double beta3, double beta4, double *dGdrs);

double _mgga_c_thetapbe(void *p, const double n, const double rs,
                        const double a2, const double beta, double *dedrs,
                        double *deda2, double *dedbeta) {

  double rtrs = sqrt(rs);
  double de0drs;
  double e0 = G(rtrs, GAMMA, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, &de0drs);
  double e;
  double xp = 117.0;
  double xm = 117.0;
  double n2 = n * n;
  double t2;
  double y;
  *dedrs = de0drs;
  e = e0;
  t2 = C3 * a2 * rs / n2;
  y = -e / GAMMA;
  double x = exp(y);
  double A;
  double dAdbeta;
  if (x != 1.0)
    dAdbeta = 1.0 / (GAMMA * (x - 1.0));
  else
    dAdbeta = 1.0 / (GAMMA * y);
  A = beta * dAdbeta;
  double At2 = A * t2;
  double nom = 1.0 + At2;
  double denom = nom + At2 * At2;
  double H = GAMMA * log(1.0 + beta * t2 * nom / (denom * GAMMA));
  double tmp = (GAMMA * beta / (denom * (beta * t2 * nom + GAMMA * denom)));
  double tmp2 = A * A * x / beta;
  double dAdrs = tmp2 * *dedrs;

  double dHdt2 = (1.0 + 2.0 * At2) * tmp;
  double dHdA = -At2 * t2 * t2 * (2.0 + At2) * tmp;
  *dedbeta = (t2 * nom * denom * tmp / beta + dHdA * dAdbeta);
  *dedrs += dHdt2 * 7 * t2 / rs + dHdA * dAdrs;
  *deda2 = dHdt2 * C3 * rs / n2;
  e += H;
  return e;
}

double mgga_c_thetapbe(void *p, const double n, const double rs,
                       const double a2, const double g_, double *dedrs,
                       double *deda2, double *dedg_) {
  double beta = g_ * BETA_H + (1.0 - g_) * BETA_GE;
  double e = _mgga_c_thetapbe(p, n, rs, a2, beta, dedrs, deda2, dedg_);
  *dedg_ *= (BETA_H - BETA_GE);
  return e;
}

double _mgga_c_thetapbe_sp(void *p, const double n, const double rs,
                           const double zeta, const double a2,
                           const double beta, double *dedrs, double *dedzeta,
                           double *deda2, double *dedbeta) {
  double rtrs = sqrt(rs);
  double de0drs;
  double e0 = G(rtrs, GAMMA, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, &de0drs);
  double e;
  double xp = 117.0;
  double xm = 117.0;
  double de1drs;
  double e1 =
      G(rtrs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, &de1drs);
  double dalphadrs;
  double alpha =
      -G(rtrs, 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, &dalphadrs);
  dalphadrs = -dalphadrs;
  double zp = 1.0 + zeta;
  double zm = 1.0 - zeta;
  xp = pow(zp, THIRD);
  xm = pow(zm, THIRD);
  double f = CC1 * (zp * xp + zm * xm - 2.0);
  double f1 = CC2 * (xp - xm);
  double zeta2 = zeta * zeta;
  double zeta3 = zeta2 * zeta;
  double zeta4 = zeta2 * zeta2;
  double xzeta = 1.0 - zeta4;
  *dedrs = (de0drs * (1.0 - f * zeta4) + de1drs * f * zeta4 +
            dalphadrs * f * xzeta * IF2);
  *dedzeta = (4.0 * zeta3 * f * (e1 - e0 - alpha * IF2) +
              f1 * (zeta4 * e1 - zeta4 * e0 + xzeta * alpha * IF2));
  e = e0 + alpha * IF2 * f * xzeta + (e1 - e0) * f * zeta4;
  double n2 = n * n;
  double phi = 0.5 * (xp * xp + xm * xm);
  double phi2 = phi * phi;
  double phi3 = phi * phi2;
  double t2 = C3 * a2 * rs / (n2 * phi2);
  double y = -e / (GAMMA * phi3);
  double x = exp(y);
  double A;
  double dAdbeta;
  if (x != 1.0)
    dAdbeta = 1.0 / (GAMMA * (x - 1.0));
  else
    dAdbeta = 1.0 / (GAMMA * y);
  A = beta * dAdbeta;
  double At2 = A * t2;
  double nom = 1.0 + At2;
  double denom = nom + At2 * At2;
  double H = GAMMA * log(1.0 + beta * t2 * nom / (denom * GAMMA));
  double tmp = (GAMMA * beta / (denom * (beta * t2 * nom + GAMMA * denom)));
  double tmp2 = A * A * x / beta;
  double dAdrs = tmp2 * *dedrs;
  H *= phi3;
  tmp *= phi3;
  dAdrs /= phi3;
  double dHdt2 = (1.0 + 2.0 * At2) * tmp;
  double dHdA = -At2 * t2 * t2 * (2.0 + At2) * tmp;
  *dedbeta = (t2 * nom * denom * tmp / beta + dHdA * dAdbeta);
  *dedrs += dHdt2 * 7 * t2 / rs + dHdA * dAdrs;
  *deda2 = dHdt2 * C3 * rs / n2;
  double dphidzeta = (1.0 / xp - 1.0 / xm) / 3.0;
  double dAdzeta = tmp2 * (*dedzeta - 3.0 * e * dphidzeta / phi) / phi3;
  *dedzeta +=
      ((3.0 * H / phi - dHdt2 * 2.0 * t2 / phi) * dphidzeta + dHdA * dAdzeta);
  *deda2 /= phi2;
  e += H;
  return e;
}

double mgga_c_thetapbe_sp(void *p, const double n, const double rs,
                          const double zeta, const double *sigma,
                          const double *g_, double *dedrs, double *dedzeta,
                          double *dedsigma, double *dedg_) {

  double g_a = g_[0];
  double g_b = g_[1];
  double dg_da = 0.5 * (1 + zeta);
  double dg_db = 0.5 * (1 - zeta);
  double g_zeta = dg_da * g_a + dg_db * g_b;
  double dg_dzeta = 0.5 * (g_a - g_b);
  double beta = g_zeta * BETA_H + (1.0 - g_zeta) * BETA_GE;
  double dedg_zeta;
  double a2 = fmax(sigma[0] + 2 * sigma[1] + sigma[2], MGGA_HESSIAN_MINSIGMA);
  double deda2;
  double e = _mgga_c_thetapbe_sp(p, n, rs, zeta, a2, beta, dedrs, dedzeta,
                                 &deda2, &dedg_zeta);
  dedg_zeta *= (BETA_H - BETA_GE);
  *dedzeta += dedg_zeta * dg_dzeta;
  dedg_[0] = dedg_zeta * dg_da;
  dedg_[1] = dedg_zeta * dg_db;
  dedsigma[0] = deda2;
  dedsigma[1] = 2 * deda2;
  dedsigma[2] = deda2;
  return e;
}

double mgga_null_thetapbe(void *p, const double n, const double rs,
                          const double a2, const double g_, double *dedrs,
                          double *deda2, double *dedg_) {

  *dedrs = 0.0;
  *deda2 = 0.0;
  *dedg_ = 0.0;
  return 0.0;
}

double mgga_null_thetapbe_sp(void *p, const double n, const double rs,
                             const double zeta, const double *sigma,
                             const double *g_, double *dedrs, double *dedzeta,
                             double *dedsigma, double *dedg_) {
  dedrs[0] = 0.0;
  dedrs[1] = 0.0;
  *dedzeta = 0.0;
  dedsigma[0] = 0.0;
  dedsigma[1] = 0.0;
  dedsigma[2] = 0.0;
  dedg_[0] = 0.0;
  dedg_[1] = 0.0;
  return 0.0;
}
