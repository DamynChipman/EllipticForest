add_library(poisson_f OBJECT 
  hsmooth.f90 
  fortran/poisson_setprob.f90 
  fortran/poisson_fort_rhs.f90 
  fortran/poisson_compute_error.f 
  fortran/poisson_qexact.f90 
  fortran/poisson_apply_bc.f90 
  fortran/poisson_eval_bc.f90 
  fortran/poisson_output.f90
)

add_executable(forestclaw_poisson main.cpp poisson_user.cpp poisson_options.c poisson_diagnostics.c $<TARGET_OBJECTS:poisson_f>)
target_include_directories(forestclaw_poisson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EXTERNAL_PKGS_INCLUDE})
target_link_directories(forestclaw_poisson PUBLIC ${EXTERNAL_PKGS_LIB})
# target_link_libraries(forestclaw_poisson PUBLIC z sc p4est petsc forestclaw clawpatch EllipticForest)
target_link_libraries(forestclaw_poisson PUBLIC z EllipticForest forestclaw clawpatch)

configure_file(fclaw_options.ini fclaw_options.ini COPYONLY)