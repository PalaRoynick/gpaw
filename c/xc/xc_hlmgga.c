#include "xc_hlmgga.h"
#include <math.h>
#include <stdio.h>
#include <xc.h>

hlmgga_func_info thetapbe_info = {0, NULL, mgga_x_thetapbe, mgga_c_thetapbe,
                                  mgga_c_thetapbe_sp};

hlmgga_func_info thetapbex_info = {0, NULL, mgga_x_thetapbe, mgga_null_thetapbe,
                                   mgga_null_thetapbe_sp};

hlmgga_func_info thetapbec_info = {0, NULL, mgga_null_thetapbe, mgga_c_thetapbe,
                                   mgga_c_thetapbe_sp};

void init_hlmgga(void **params, int code) {
  const hlmgga_func_info *finfo;
  switch (code) {
  case 124:
    finfo = &thetapbe_info;
    break;
  case 125:
    finfo = &thetapbex_info;
    break;
  case 126:
    finfo = &thetapbec_info;
    break;
  default:
    finfo = NULL;
    assert(code >= 124 && code <= 126);
  }
  *params = malloc(sizeof(hlmgga_info));
  if (*params == NULL) {
    perror("init_mgga_hessian malloc");
    exit(1);
  }
  hlmgga_info *info = (hlmgga_info *)*params;
  info->nspin = 0;
  info->code = code;
  info->funcinfo = finfo;
}

void free_hlmgga(void **params) {
  hlmgga_info *info = (hlmgga_info *)*params;
  free(*params);
  *params = NULL;
}

void calc_hlmgga(void **params, int nspin, int ng, const double *n_g,
                 const double *sigma_g, const double *g_g, double *e_g,
                 double *v_g, double *dedsigma_g, double *dedg__g) {
  hlmgga_info *info = (hlmgga_info *)*params;
  if (info->nspin != nspin) {
    info->nspin = nspin;
  }
  void *p = (void *)info->funcinfo->params;
  if (nspin == 1) {
    for (int g = 0; g < ng; g++) {
      double n = n_g[g];
      if (n < NMIN)
        n = NMIN;
      double rs = pow(C0I / n, THIRD);
      double dexdrs = 0.0;
      double dexda2 = 0.0;
      double ex = 0.0;
      double decdrs = 0.0;
      double decda2 = 0.0;
      double ec = 0.0;
      double a2 = fmax(sigma_g[g], MGGA_HESSIAN_MINSIGMA);
      double g_ = g_g[g];
      double dexdg_ = 0.0;
      double decdg_ = 0.0;
      ex =
          info->funcinfo->exchange(p, n, rs, a2, g_, &dexdrs, &dexda2, &dexdg_);
      ec = info->funcinfo->correlation(p, n, rs, a2, g_, &decdrs, &decda2,
                                       &decdg_);
      dedsigma_g[g] = n * (dexda2 + decda2);
      dedg__g[g] = n * (dexdg_ + decdg_);
      e_g[g] = n * (ex + ec);
      v_g[g] += ex + ec - rs * (dexdrs + decdrs) * THIRD;
    }
  } else {
    const double *na_g = n_g;
    double *va_g = v_g;
    const double *nb_g = na_g + ng;
    double *vb_g = va_g + ng;

    const double *sigma0_g = 0;
    const double *sigma1_g = 0;
    const double *sigma2_g = 0;
    double *dedsigma0_g = 0;
    double *dedsigma1_g = 0;
    double *dedsigma2_g = 0;
    const double *g_a_g = 0;
    const double *g_b_g = 0;
    double *dedg_a_g = 0;
    double *dedg_b_g = 0;

    sigma0_g = sigma_g;
    sigma1_g = sigma0_g + ng;
    sigma2_g = sigma1_g + ng;
    dedsigma0_g = dedsigma_g;
    dedsigma1_g = dedsigma0_g + ng;
    dedsigma2_g = dedsigma1_g + ng;

    g_a_g = g_g;
    g_b_g = g_a_g + ng;
    dedg_a_g = dedg__g;
    dedg_b_g = dedg_a_g + ng;

    for (int g = 0; g < ng; g++) {
      double na = 2.0 * na_g[g];
      if (na < NMIN)
        na = NMIN;
      double rsa = pow(C0I / na, THIRD);
      double nb = 2.0 * nb_g[g];
      if (nb < NMIN)
        nb = NMIN;
      double rsb = pow(C0I / nb, THIRD);
      double n = 0.5 * (na + nb);
      double rs = pow(C0I / n, THIRD);
      double zeta = 0.5 * (na - nb) / n;
      double dexadrs;
      double dexada2;
      double exa;
      double dexbdrs;
      double dexbda2;
      double exb;
      double decdrs;
      double decdzeta;
      double decdsigma[3];
      double ec;
      double dexadg_, dexbdg_, decdg_[2];
      double g_a = g_a_g[g];
      double g_b = g_b_g[g];
      double g_[2] = {g_a, g_b};
      exa = info->funcinfo->exchange(
          p, na, rsa, 4.0 * fmax(sigma0_g[g], MGGA_HESSIAN_MINSIGMA), g_a,
          &dexadrs, &dexada2, &dexadg_);
      exb = info->funcinfo->exchange(
          p, nb, rsb, 4.0 * fmax(sigma2_g[g], MGGA_HESSIAN_MINSIGMA), g_b,
          &dexbdrs, &dexbda2, &dexbdg_);
      double sigma[3] = {sigma0_g[g], sigma1_g[g], sigma2_g[g]};
      ec = info->funcinfo->correlation_sp(p, n, rs, zeta, sigma, g_, &decdrs,
                                          &decdzeta, decdsigma, decdg_);
      dedsigma0_g[g] = 2 * na * dexada2 + n * decdsigma[0];
      dedsigma1_g[g] = n * decdsigma[1];
      dedsigma2_g[g] = 2 * nb * dexbda2 + n * decdsigma[2];
      dedg_a_g[g] = 0.5 * na * dexadg_ + n * decdg_[0];
      dedg_b_g[g] = 0.5 * nb * dexbdg_ + n * decdg_[1];
      e_g[g] = 0.5 * (na * exa + nb * exb) + n * ec;
      va_g[g] += (exa + ec - (rsa * dexadrs + rs * decdrs) * THIRD -
                  (zeta - 1.0) * decdzeta);
      vb_g[g] += (exb + ec - (rsb * dexbdrs + rs * decdrs) * THIRD -
                  (zeta + 1.0) * decdzeta);
    }
  }
}
