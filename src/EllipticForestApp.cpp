#include "EllipticForestApp.hpp"

namespace EllipticForest {

EllipticForestApp::EllipticForestApp()
    {}
    
EllipticForestApp::EllipticForestApp(int* argc, char*** argv) :
    argc_{argc},
    argv_{argv},
    parser{*argc, *argv} {
    
    addTimer("app-lifetime");
    timers["app-lifetime"].start();

    int isMPIIntialized;
    MPI_Initialized(&isMPIIntialized);
    if (!isMPIIntialized) MPI_Init(argc_, argv_);
#if USE_PETSC
    PetscInitialize(argc_, argv_, NULL, NULL);
    PetscGetArgs(argc_, argv_);
#endif
    
    int mpi_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (mpi_rank == 0) {
        std::cout << "[EllipticForest] Welcome to EllipticForest!" << std::endl;
    }
    this->actualClassPointer_ = this;
}

EllipticForestApp::~EllipticForestApp() {
    timers["app-lifetime"].stop();

    int mpi_rank = 0;
    int mpi_size = 0;
    int isMPIFinalized;
    MPI_Finalized(&isMPIFinalized);
    if (!isMPIFinalized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    }
    
    // std::map<std::string, std::vector<double>> global_timers;
    // for (auto& [key, value] : timers) {
    //     if (mpi_rank == 0) {
    //         global_timers[key] = std::vector<double>(mpi_size);
    //     }
    //     double time = value.time();
    //     MPI::gather<double>(time, global_timers[key], 0, MPI_COMM_WORLD);
    // }

    if (mpi_rank == 0) {
        // std::string filename = "ef-timers-" + getCurrentDateTimeString() + ".csv";
        // writeMapToCSV(global_timers, filename);

        std::cout << "[EllipticForest] End of app life cycle, finalizing..." << std::endl;
        std::cout << "[EllipticForest] Options:" << std::endl << options;
        
        std::cout << "[EllipticForest] Timers: " << std::endl;
        for (auto& [key, value] : timers) {
            std::cout << "[EllipticForest]   " << key << " : " << value.time() << " [sec]" << std::endl;
        }
        std::cout << "[EllipticForest] Done!" << std::endl;
    }

    if (!isMPIFinalized) {
#if USE_PETSC
        PetscFinalize();
#endif
        MPI_Finalize();
    }
}

void EllipticForestApp::addTimer(std::string name) {
    timers[name] = Timer();
}

void EllipticForestApp::sleep(int seconds) {
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

} // NAMESPACE : EllipticForest