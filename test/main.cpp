#include "gtest/gtest.h"
#include <mpi.h>
#include <string>

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	::testing::InitGoogleTest(&argc, argv);
	int res = RUN_ALL_TESTS();
	MPI_Finalize();
	return res;
}