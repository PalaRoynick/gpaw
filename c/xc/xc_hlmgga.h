#ifndef XC_MGGA_HESSIAN_H
#define XC_MGGA_HESSIAN_H

#include "xc_gpaw.h"
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

#define MGGA_HESSIAN_MINSIGMA 1e-20
#define MU_GE 0.12345679012345678 // 10/81
#define MU_H 0.2758318388794276
#define MU_H_RPBE 0.2283481639697126
#define BETA_GE 0.03752636431197695 // 10/81 / (pi^2 / 3)
#define BETA_H 0.08384282520451862  // mu_h / (pi^2 / 3)
#define BETA_H_RPBE 0.06940951876789773
#define KAPPA 0.8040
#define C_rs3 4.1887902047863905   // 4/3 * pi
#define C_TUNIF 2.8712340001881915 // 3/10 * (3 * pi^2)^(2 / 3)

typedef struct hlmgga_func_info {
  int size_params;
  void *params;
  double (*exchange)(void *p, const double n, const double rs, const double a2,
                     const double g_, double *dexdrs, double *dex,
                     double *dedg_);
  double (*correlation)(void *p, const double n, const double rs,
                        const double a2, const double g_, double *dedrs,
                        double *decda2, double *dedg_);
  double (*correlation_sp)(void *p, const double n, const double rs,
                           const double zeta, const double *sigma,
                           const double *g_, double *decdrs, double *decdzeta,
                           double *decda2, double *dedg_);

} hlmgga_func_info;

typedef struct hlmgga_info {
  int nspin;
  int code;
  const hlmgga_func_info *funcinfo;
} hlmgga_info;

void init_hlmgga(void **funcinfo, int code);

void calc_hlmgga(void **funcinfo, int nspin, int ng, const double *n_g,
                 const double *sigma_g, const double *g_g, double *e_g,
                 double *v_g, double *dedsigma_g, double *dedg__g);

void free_mhlmgga(void **funcinfo);

double mgga_null_thetapbe(void *, const double n, const double rs,
                          const double a2, const double g_, double *dedrs,
                          double *deda2, double *dedg_);

double mgga_null_thetapbe_sp(void *, const double n, const double rs,
                             const double zeta, const double *sigma,
                             const double *g_, double *dedrs, double *dedzeta,
                             double *decdsigma, double *dedg_);

double mgga_x_thetapbe(void *, const double n, const double rs, const double a2,
                       const double g_, double *dexdrs, double *dexda2,
                       double *dedg_);

double mgga_c_thetapbe(void *, const double n, const double rs, const double a2,
                       const double g_, double *dedrs, double *decda2,
                       double *dedg_);

double mgga_c_thetapbe_sp(void *, const double n, const double rs,
                          const double zeta, const double *sigma,
                          const double *g_, double *decdrs, double *decdzeta,
                          double *decda2, double *decdg_);

#endif // XC_MGGA_HESSIAN_H
