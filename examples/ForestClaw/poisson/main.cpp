/*
  Copyright (c) 2019-2022 Carsten Burstedde, Donna Calhoun, Scott Aiton, Grady Wright
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "poisson_user.h"

#include <ForestClaw.hpp>
    
#include <fclaw2d_include_all.h>

#include <fclaw2d_output.h>
#include <fclaw2d_diagnostics.h>

#include <fclaw2d_elliptic_solver.h>

#include <fclaw2d_clawpatch_options.h>
#include <fclaw2d_clawpatch.h>

#include <EllipticForestApp.hpp>

static
fclaw2d_domain_t* create_domain(sc_MPI_Comm mpicomm, fclaw_options_t* fclaw_opt)
{
    /* Mapped, multi-block domain */
    p4est_connectivity_t     *conn = NULL;
    fclaw2d_domain_t         *domain;
    fclaw2d_map_context_t    *cont = NULL, *brick = NULL;
 
    int mi = fclaw_opt->mi;
    int mj = fclaw_opt->mj;

    int a = fclaw_opt->periodic_x;
    int b = fclaw_opt->periodic_y;

    /* Map unit square to disk using mapc2m_disk.f */
    conn = p4est_connectivity_new_brick(mi,mj,a,b);
    brick = fclaw2d_map_new_brick_conn (conn,mi,mj);
    cont = fclaw2d_map_new_nomap_brick(brick);

    domain = fclaw2d_domain_new_conn_map (mpicomm, fclaw_opt->minlevel, conn, cont);
    fclaw2d_domain_list_levels(domain, FCLAW_VERBOSITY_ESSENTIAL);
    fclaw2d_domain_list_neighbors(domain, FCLAW_VERBOSITY_DEBUG);  
    return domain;
}

static
void run_program(fclaw2d_global_t* glob)
{
    // const poisson_options_t           *user_opt;

    // user_opt = poisson_get_options(glob);

    /* ---------------------------------------------------------------
       Set domain data.
       --------------------------------------------------------------- */
    fclaw2d_global_set_global(glob);
    fclaw2d_domain_data_new(glob->domain);

    /* Initialize virtual table for ForestClaw */
    fclaw2d_vtables_initialize(glob);

    /* Test HPS solver */
    fc2d_hps_solver_initialize(glob);

    /* set up elliptic solver to use the thunderegg solver */
    poisson_link_solvers(glob);

    /* ---------------------------------------------------------------
       Run
       --------------------------------------------------------------- */

    /* Set up grid and RHS */
    fclaw2d_initialize(glob);

    /* Compute sum of RHS; reset error accumulators */
    int init_flag = 1;  
    fclaw2d_diagnostics_gather(glob,init_flag);
    init_flag = 0;

    /* Output rhs */
    int Frame = 0;
    fclaw2d_output_frame(glob,Frame);
    fclaw2d_clawpatch_output_vtk(glob, Frame);
 
    /* Solve the elliptic problem */
    fclaw2d_elliptic_solve(glob);

    /* Compute error, compute conservation */
    fclaw2d_diagnostics_gather(glob, init_flag);                

    /* Output solution */
    Frame = 1;
    fclaw2d_output_frame(glob,Frame);

    /* ---------------------------------------------------------------
       Finalize
       --------------------------------------------------------------- */
    fclaw2d_finalize(glob);
}

int
main (int argc, char **argv)
{
    fclaw_app_t *fclaw_app;
    int first_arg;
    fclaw_exit_type_t vexit;

    /* Options */
    sc_options_t                *options;
    fclaw_options_t             *fclaw_opt;

    fclaw2d_clawpatch_options_t *clawpatch_opt;
    // fc2d_thunderegg_options_t    *mg_opt;
    poisson_options_t              *user_opt;

    fclaw2d_global_t            *glob;
    fclaw2d_domain_t            *domain;
    sc_MPI_Comm mpicomm;

    int retval;

    /* Initialize application */
    fclaw_app = fclaw_app_new (&argc, &argv, NULL);
    EllipticForest::EllipticForestApp ef_app(&argc, &argv);

    /* Create new options packages */
    fclaw_opt =                   fclaw_options_register(fclaw_app,  NULL,        "fclaw_options.ini");
    clawpatch_opt =   fclaw2d_clawpatch_options_register(fclaw_app, "clawpatch",  "fclaw_options.ini");
    // mg_opt =            fc2d_thunderegg_options_register(fclaw_app, "thunderegg", "fclaw_options.ini");
    user_opt =                  poisson_options_register(fclaw_app,               "fclaw_options.ini"); 

    /* Read configuration file(s) and command line, and process options */
    options = fclaw_app_get_options (fclaw_app);
    retval = fclaw_options_read_from_file(options);
    vexit =  fclaw_app_options_parse (fclaw_app, &first_arg,"fclaw_options.ini.used");
    
    // Configure EllipticForest options
    ef_app.options["homogeneous-rhs"] = false;
    ef_app.options["cache-operators"] = false;
    ef_app.options["min-level"] = fclaw_opt->minlevel;
    ef_app.options["max-level"] = fclaw_opt->maxlevel;
    ef_app.options["nx"] = clawpatch_opt->mx;
    ef_app.options["ny"] = clawpatch_opt->my;

    /* Run the program */
    if (!retval & !vexit)
    {
        /* Options have been checked and are valid */

        mpicomm = fclaw_app_get_mpi_size_rank (fclaw_app, NULL, NULL);
        domain = create_domain(mpicomm, fclaw_opt);
    
        /* Create global structure which stores the domain, timers, etc */
        glob = fclaw2d_global_new();
        fclaw2d_global_store_domain(glob, domain);

        /* Store option packages in glob */
        fclaw2d_options_store           (glob, fclaw_opt);
        fclaw2d_clawpatch_options_store (glob, clawpatch_opt);
        // fc2d_thunderegg_options_store    (glob, mg_opt);
        poisson_options_store            (glob, user_opt);

        run_program(glob);

        fclaw2d_global_destroy(glob);        
    }
    
    fclaw_app_destroy (fclaw_app);

    return 0;
}