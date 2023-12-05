#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include "MPI.hpp"

namespace EllipticForest {

class Logger {

public:

    /**
     * @brief Construct a new Logger object
     * 
     */
    Logger();

    /**
     * @brief Destroy the Logger object
     * 
     */
    ~Logger();

    /**
     * @brief Outputs a message to the log
     * 
     * @tparam Args Variable argument holder for variable formatting
     * @param message Message to log
     * @param args Variables to format into string
     */
    template<class... Args>
    void log(std::string message, Args... args) {
    
        int myRank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        std::string toPrint = "[EllipticForest " + std::to_string(myRank) + "] " + message + "\n";
        printf(toPrint.c_str(), args...);

    }

    /**
     * @brief Outputs a message to the head rank log
     * 
     * @tparam Args Variable argument holder for variable formatting
     * @param message Message to log
     * @param args Variables to format into string
     */
    template<class... Args>
    void logHead(std::string message, Args... args) {
    
        int myRank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        if (myRank == MPI::HEAD_RANK) {
            std::string toPrint = "[EllipticForest " + std::to_string(myRank) + "] " + message + "\n";
            printf(toPrint.c_str(), args...);
        }

    }

};

} // NAMESPACE : EllipticForest

#endif // LOGGER_HPP_