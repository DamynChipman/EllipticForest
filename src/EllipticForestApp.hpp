#ifndef ELLIPTIC_FOREST_APP_HPP_
#define ELLIPTIC_FOREST_APP_HPP_

#include <iostream>
#include <string>
#include <map>
#include <variant>
#include <cstdarg>

#include "GenericSingleton.hpp"
#include "Quadtree.hpp"
#include <mpi.h>
#include <petsc.h>
#include <p4est.h>

namespace EllipticForest {

struct Logger {

    Logger() {}
    ~Logger() {}

    template<class... Args>
    void log(std::string message, Args... args) {
    
        int myRank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::string toPrint = "[EllipticForest " + std::to_string(myRank) + "] " + message + "\n";
        printf(toPrint.c_str(), args...);

    }

};


struct Options {
    
    using OptionTypes = std::variant<std::string, bool, int, double>;
    std::map<std::string, OptionTypes> optionsMap;

    Options() {}
    Options(std::map<std::string, OptionTypes> map) : optionsMap(map) {}

    OptionTypes operator[](std::string const key) {
        return optionsMap[key];
    }

    void setOption(std::string const key, OptionTypes const value) {
        optionsMap[key] = value;
    }

    void setFromFile(std::string filename) {

    }

    friend std::ostream& operator<<(std::ostream& os, const Options& options) {
        os << "---===== HPS Options =====---" << std::endl;
        for (const auto& [key, value] : options.optionsMap) {
            os << key << " : ";
            std::visit([&] (auto&& v)
                {os << v << std::endl; }, value);
        }
        os << "---===== HPS Options =====---" << std::endl;
        return os;
    }

};


class EllipticForestApp : public GenericSingleton<EllipticForestApp> {

private:

    int* argc_;
    char*** argv_;

public:
    
    Logger logger{};
    Options options{};

    EllipticForestApp() : argc_(nullptr), argv_(nullptr) {}
    
    EllipticForestApp(int* argc, char*** argv) : argc_(argc), argv_(argv) {
        MPI_Init(argc_, argv_);
        PetscInitialize(argc_, argv_, NULL, NULL);
        
        int myRank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        if (myRank == 0) {
            std::cout << "[EllipticForest] Welcome to EllipticForest!" << std::endl;
        }
        this->actualClassPointer_ = this;
    }

    ~EllipticForestApp() {
        int myRank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        if (myRank == 0) {
            std::cout << "[EllipticForest] End of app life cycle, finalizing..." << std::endl;
        }
        PetscFinalize();
        MPI_Finalize();
    }

    template<class... Args>
    void log(std::string message, Args... args) {
        logger.log(message, args...);
    }

};


} // NAMESPACE : EllipticForest

#endif // ELLIPTIC_FOREST_APP_HPP_