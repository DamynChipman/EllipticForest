#include "gtest/gtest.h"
#include <string>
#include <mpi.h>
#include <petsc.h>

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	PetscInitialize(&argc, &argv, NULL, NULL);
	::testing::InitGoogleTest(&argc, argv);
	int res = RUN_ALL_TESTS();
	PetscFinalize();
	MPI_Finalize();
	return res;
}