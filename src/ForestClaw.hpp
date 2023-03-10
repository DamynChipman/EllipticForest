#ifndef FORESTCLAW_HPP_
#define FORESTCLAW_HPP_

#if USE_FORESTCLAW

#include "EllipticForest.hpp"

#include <p4est_wrap.h>
#include <fclaw2d_include_all.h>
#include <fclaw2d_elliptic_solver.h>
#include <fclaw2d_clawpatch.h>
#include <fclaw2d_clawpatch_options.h>
#include <fclaw2d_clawpatch_output_ascii.h> 
#include <fclaw2d_clawpatch_output_vtk.h>

// namespace EllipticForest {

// namespace ForestClaw {

/*************************************************/
// fc2d_hps
/*************************************************/

typedef void (*fc2d_hps_fort_qexact_t)(const int* example,
                                       const double* x,
                                       const double* y,
                                       const double* q,
                                       const double* qlap,
                                       const double* grad,
                                       const int* flag);

typedef  void (*fc2d_hps_fort_rhs_t)(const int* blockno, 
                                     const int* mbc,
                                     const int* mx, const int* my,
                                     const int* mfields,
                                     const double* xlower, const double* ylower,
                                     const double* dx, const double* dy,
                                     double rhs[]);

typedef  void (*fc2d_hps_fort_beta_t)(const double* x,
                                      const double* y,
                                      const double* beta,
                                      double grad[]);

typedef double (*fc2d_hps_fort_eval_bc_t)(const int *iface, const double *t, 
                                          const double *x, const double *y);

typedef void (*fc2d_hps_fort_apply_bc_t)(const int* blockno, const  int* mx, const  int* my, 
                                         const  int* mbc, const  int* meqn, 
                                         const double* xlower, const double* ylower,
                                         const double* dx, const double* dy, 
                                         const double *t, 
                                         int intersects_bc[], int mthbc[], 
                                         double rhs[], fc2d_hps_fort_eval_bc_t g_bc, 
                                         int* cons_check, double flux_sum[]);

typedef void (*fc2d_hps_patch_solver_t)(struct fclaw2d_global *glob);


typedef void (*fc2d_hps_fort_output_t)(const char* matname1,
                                       int* mx,        int* my,
                                       int* meqn,      int* mbc,
                                       double* xlower, double* ylower,
                                       double* dx,     double* dy,
                                       double q[],double soln[], double error[],
                                       int* patch_num, int* level,
                                       int* blockno,   int* mpirank);

typedef struct fc2d_hps_vtable
{
    /* Solver that defines patch solver and calls Hps solver */
    // fc2d_hps_patch_solver_t   patch_solver;  /* 'solver' is a keyword */

    /* Fortran routines */
    fc2d_hps_fort_rhs_t        fort_rhs;	
    fc2d_hps_fort_beta_t       fort_beta;	
    fc2d_hps_fort_apply_bc_t   fort_apply_bc;
    fc2d_hps_fort_eval_bc_t    fort_eval_bc;
    fc2d_hps_fort_qexact_t     fort_qexact;

    /* Allows us to output error and exact solution, along with computed solution */
    fc2d_hps_fort_output_t     fort_output;   

	  int is_set;
} fc2d_hps_vtable_t;

fc2d_hps_vtable_t* fc2d_hps_vt();
void fc2d_hps_solver_initialize(fclaw2d_global_t* glob);
void fc2d_hps_setprob(fclaw2d_global_t* glob);
void fc2d_hps_rhs(fclaw2d_global_t* glob, fclaw2d_patch_t *patch, int blockno, int patchno);
void fc2d_hps_heat_set_lambda(double lambda);
double fc2d_hps_heat_get_lambda();
void hps_setup_solver(fclaw2d_global_t *glob);
void hps_rhs(fclaw2d_global_t *glob, fclaw2d_patch_t *patch, int blockno, int patchno);
void hps_solve(fclaw2d_global_t *glob);
void hps_output(fclaw2d_global_t *glob, int iframe);
int hps_tag4refinement(fclaw2d_global_t *glob, fclaw2d_patch_t *this_patch, int blockno, int patchno, int initflag);
int hps_tag4coarsening(fclaw2d_global_t *glob, fclaw2d_patch_t *fine_patches, int blockno, int patchno, int initflag);
void hps_compute_error(fclaw2d_global_t *glob, fclaw2d_patch_t *patch, int blockno, int patchno, void *user);
void hps_conservation_check(fclaw2d_global_t *glob, fclaw2d_patch_t *patch, int blockno, int patchno, void *user);
fc2d_hps_vtable_t* hps_vt_init();

/*************************************************/
// fc2d_hps_physical_bc
/*************************************************/

struct fclaw2d_global;
struct fclaw2d_domain;
struct fclaw2d_patch;

typedef struct fc2d_hps_time_info
{
    double t;
} fc2d_hps_time_info_t;

void fc2d_hps_physical_bc(struct fclaw2d_global *glob);
void cb_fc2d_hps_physical_bc(fclaw2d_domain_t *domain, fclaw2d_patch_t *patch, int blockno, int patchno, void *user);
void fc2d_hps_physical_get_bc(fclaw2d_global_t *glob, int blockno, int patchno, int *intersects_bdry);

/*************************************************/
// fc2d_hps_output_ascii
/*************************************************/

void fc2d_hps_time_header_ascii(fclaw2d_global_t* glob, int iframe);
void cb_hps_output_ascii(fclaw2d_domain_t * domain, fclaw2d_patch_t * patch, int blockno, int patchno, void *user);

/*************************************************/
// fc2d_hps_options
/*************************************************/

typedef enum {
    LAPLACE = 0,      /* Laplacian (no beta) */    
    VARPOISSON,         /* Variable Poisson operator */
    HEAT,               /* Variable Poisson operator */
    USER_OPERATOR
} fc2d_hps_operator_types;

typedef enum {
    FFT = 0,    /* Must use fivepoint */
    FISHPACK,   /* Must use fivepoint */
    DST,        /* Must use fivepoint */
    BICG,       /* Can be used with variable coefficient */
    USER_SOLVER
} fc2d_hps_solver_types;

typedef struct fc2d_hps_options
{
    /* Boundary conditions */
    int *boundary_conditions;
    const char *bc_cond_string;

    /* Output */
    int ascii_out;
    int vtk_out;
    // int mmio_out;

    /* Caching */
    int cache_T;

    /* Timing */
    int time_setup;
    int time_build;
    int time_upwards;
    int time_solve;
    // int time_copy;

    /* Homogeneous vs. non-homogeneous */
    int nonhomogeneous_rhs;

    // Patch solver short-circuit
    int only_patch_solver;

    int operator_type;  /* laplace, varpoisson, etc ... */
    sc_keyvalue_t *kv_operator_type;

    int patch_solver;                /* FFT, DST, FISHPACK, ... */
    sc_keyvalue_t *kv_patch_solver;

    int is_registered;
} fc2d_hps_options_t;

fc2d_hps_options_t*  fc2d_hps_options_register (fclaw_app_t * app, const char *configfile);
void fc2d_hps_package_register(fclaw_app_t* app, fc2d_hps_options_t* mg_opt);
fc2d_hps_options_t* fc2d_hps_get_options(struct fclaw2d_global *glob);
void fc2d_hps_options_store (struct fclaw2d_global* glob, fc2d_hps_options_t* mg_opt);
void* hps_register (fc2d_hps_options_t* hps_opt, sc_options_t * opt);
fclaw_exit_type_t hps_postprocess (fc2d_hps_options_t * hps_opt);
fclaw_exit_type_t hps_check(fc2d_hps_options_t *hps_opt, fclaw2d_clawpatch_options_t *clawpatch_opt);
void hps_destroy (fc2d_hps_options_t * hps_opt);
void* options_register (fclaw_app_t * app, void *package, sc_options_t * opt);
fclaw_exit_type_t options_postprocess (fclaw_app_t * app, void *package, void *registered);
fclaw_exit_type_t options_check (fclaw_app_t * app, void *package, void *registered);
void options_destroy (fclaw_app_t * app, void *package, void *registered);

/*************************************************/
// fc2d_hps_diagnostics
/*************************************************/

typedef struct fc2d_hps_error_info {
    double* local_error;  /* meqn x 3 array of errors on a patch */
    double* global_error; /* meqn x 3 array of errors after gather */
    double *mass0;  /* Mass at initial time */
    double *mass;
    double area;
    double *rhs;       /* Sum of rhs hand side */
    double *boundary;  /* sum around boundary */
    double *c_kahan;  
} fc2d_hps_error_info_t;

void fc2d_hps_diagnostics_initialize(fclaw2d_global_t *glob, void **acc_patch);
void fc2d_hps_diagnostics_reset(fclaw2d_global_t *glob, void* patch_acc);
void fc2d_hps_compute(fclaw2d_domain_t *domain, fclaw2d_patch_t *patch, int blockno, int patchno, void* user);
void fc2d_hps_diagnostics_compute(fclaw2d_global_t* glob, void* patch_acc);
void fc2d_hps_diagnostics_gather(fclaw2d_global_t *glob, void* patch_acc, int init_flag);
void fc2d_hps_diagnostics_finalize(fclaw2d_global_t *glob, void** patch_acc);
void fc2d_hps_compute_diagnostics(fclaw2d_domain_t *domain, fclaw2d_patch_t *patch, int blockno, int patchno, void* user);

// } // NAMESPACE : ForestClaw

// } // NAMESPACE : EllipticForest

#endif // USE_FORESTCLAW

#endif // FORESTCLAW_HPP_