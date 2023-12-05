#include "EllipticForestApp.hpp"

namespace EllipticForest {

EllipticForestApp::EllipticForestApp() :
    argc_(nullptr), argv_(nullptr)
        {}
    
EllipticForestApp::EllipticForestApp(int* argc, char*** argv) :
    argc_(argc), argv_(argv) {
    
    addTimer("app-lifetime");
    timers["app-lifetime"].start();

    int isMPIIntialized;
    MPI_Initialized(&isMPIIntialized);
    if (!isMPIIntialized) MPI_Init(argc_, argv_);
#if USE_PETSC
    PetscInitialize(argc_, argv_, NULL, NULL);
    PetscGetArgs(argc_, argv_);
#endif

    // Create options
    InputParser inputParser(*argc_, *argv_);
    inputParser.parse(options);
    
    int myRank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0) {
        std::cout << "[EllipticForest] Welcome to EllipticForest!" << std::endl;
    }
    this->actualClassPointer_ = this;
}

EllipticForestApp::~EllipticForestApp() {
    timers["app-lifetime"].stop();

    int myRank = 0;
    int isMPIFinalized;
    MPI_Finalized(&isMPIFinalized);
    if (!isMPIFinalized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    }
    
    if (myRank == 0) {
        std::cout << "[EllipticForest] End of app life cycle, finalizing..." << std::endl;
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