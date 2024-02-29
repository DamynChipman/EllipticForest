#ifndef ELLIPTIC_FOREST_APP_HPP_
#define ELLIPTIC_FOREST_APP_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <map>
#include <variant>
#include <algorithm>
#include <cstdarg>
#include <cctype>

#include <mpi.h>
#include <petsc.h>
#include <p4est.h>

#include "MPI.hpp"
#include "Logger.hpp"
#include "Options.hpp"
#include "InputParser.hpp"
#include "Timer.hpp"
#include "GenericSingleton.hpp"
#include "Helpers.hpp"

namespace EllipticForest {

/**
 * @brief The main, single instance of the EllipticForest app
 * 
 * This singleton class handles the options, logging, and timing of various components of an
 * EllipticForest runtime application. Upon construction, MPI and PETSc are initialized and input
 * arguments are read in from the command-line. When destructed, it handles the finalizing of MPI
 * and PETSc.
 * 
 * @note This should always be created in main.
 * 
 */
class EllipticForestApp : public GenericSingleton<EllipticForestApp> {

public:
    
    /**
     * @brief Logger instance for outputting messages to a log
     * 
     */
    Logger logger{};

    /**
     * @brief Options instance for storing app-global options
     * 
     */
    Options options{};

    InputParser parser{};

    /**
     * @brief Timer instance for managing timers
     * 
     */
    std::map<std::string, Timer> timers{};

    /**
     * @brief Construct a new Elliptic Forest App object
     * 
     */
    EllipticForestApp();
    
    /**
     * @brief Construct a new Elliptic Forest App object
     * 
     * @param argc Pointer to argc from main
     * @param argv Pointer to argv from main
     */
    EllipticForestApp(int* argc, char*** argv);

    /**
     * @brief Destroy the Elliptic Forest App object
     * 
     */
    ~EllipticForestApp();

    /**
     * @brief Logs a message to the log output
     * 
     * Wraps @sa Logger::log
     * 
     * @tparam Args Variable argument holder for variable formatting
     * @param message Message to log
     * @param args Variables to format into string
     */
    template<class... Args>
    void log(std::string message, Args... args) {
        logger.log(message, args...);
    }

    /**
     * @brief Logs a message to the head rank log output
     * 
     * Wraps @sa Logger::logHead
     * 
     * @tparam Args Variable argument holder for variable formatting
     * @param message Message to log
     * @param args Variables to format into string
     */
    template<class... Args>
    void logHead(std::string message, Args... args) {
        logger.logHead(message, args...);
    }

    /**
     * @brief Wraps `std::this_thread::sleep_for` in seconds
     * 
     * @param seconds Number of seconds to sleep
     */
    void sleep(int seconds);

    /**
     * @brief Adds a timer to the timer map
     * 
     * @param name Name of timer
     */
    void addTimer(std::string name);

private:

    /**
     * @brief Pointer to argc from main
     * 
     */
    int* argc_ = nullptr;

    /**
     * @brief Pointer to argv from main
     * 
     */
    char*** argv_ = nullptr;

};


} // NAMESPACE : EllipticForest

#endif // ELLIPTIC_FOREST_APP_HPP_