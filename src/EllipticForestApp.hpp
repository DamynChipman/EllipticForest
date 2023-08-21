#ifndef ELLIPTIC_FOREST_APP_HPP_
#define ELLIPTIC_FOREST_APP_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <variant>
#include <algorithm>
#include <cstdarg>
#include <cctype>

#include <mpi.h>

#if USE_PETSC
#include <petsc.h>
#endif

#include <p4est.h>
#include "GenericSingleton.hpp"

namespace EllipticForest {

static int HEAD_RANK = 0;

/**
 * @brief Structure for writing messages to a log file or the console
 * 
 * @TODO: Add ability to log according to the logging level
 * @TODO: Add ability to change logging location (file, console, other)
 * 
 */
struct Logger {

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
        if (myRank == HEAD_RANK) {
            std::string toPrint = "[EllipticForest " + std::to_string(myRank) + "] " + message + "\n";
            printf(toPrint.c_str(), args...);
        }

    }

};

/**
 * @brief Class for handling a list of options as key-value pairs in a map
 * 
 * Supports the following datatypes for options: {std::string, bool, int, double}
 * 
 */
class Options {

public:
    
    /**
     * @brief Acceptable types for options
     * 
     */
    using OptionTypes = std::variant<std::string, bool, int, double>;

    /**
     * @brief Map of options
     * 
     */
    std::map<std::string, OptionTypes> optionsMap;

    /**
     * @brief Construct a new Options object
     * 
     */
    Options();

    /**
     * @brief Construct a new Options object
     * 
     * @param map Map to copy into @sa optionsMap
     */
    Options(std::map<std::string, OptionTypes> map);

    /**
     * @brief Accesses an option from the option name
     * 
     * @param key Option name
     * @return OptionTypes& 
     */
    OptionTypes& operator[](std::string const key);

    /**
     * @brief Set an option by key-value pair
     * 
     * @param key Name of option
     * @param value Value of option
     */
    void setOption(std::string const key, OptionTypes const value);

    /**
     * @brief Sets a list of options from a supplied .in file
     * 
     * @param filename Name of .in file
     */
    void setFromFile(std::string filename);

    /**
     * @brief Checks if the option is in the map
     * 
     * @param optionName Name of option
     * @return true if option exists
     * @return false otherwise
     */
    bool optionExists(std::string optionName);

    // TODO: Add set from command line options here

    /**
     * @brief Checks and returns the type of the option
     * 
     * @param value Value of option
     * @return OptionTypes 
     */
    OptionTypes checkValue(std::string value);

    /**
     * @brief Checks if the string is a floating point number (i.e., contains a decimal)
     * 
     * @param str Number as string
     * @return true 
     * @return false 
     */
    bool isDouble(std::string& str);

    /**
     * @brief Checks if the string is an integer (i.e., does not have a decimal)
     * 
     * @param str Number as string
     * @return true 
     * @return false 
     */
    bool isInt(std::string& str);

    /**
     * @brief Writes the options to an ostream
     * 
     * @param os ostream to write to
     * @param options Options instance
     * @return std::ostream& 
     */
    friend std::ostream& operator<<(std::ostream& os, const Options& options);

private:

    /**
     * @brief Reads an .ini file to Options
     * 
     * The .ini file should have the `[EllipticForest]` header for the section corresponding to the
     * EllipticForest options
     * 
     * @param filename Name of .ini file
     * @param section Name of section to read
     */
    void readINIFile(std::string filename, std::string section);

    /**
     * @brief Removes whitespace from string
     * 
     * @param str String with whitespace
     */
    void stripWhitespace(std::string& str);

};

/**
 * @brief Handles the input parsing of command-line arguments
 * 
 * @TODO: Add ability for user to add arguments
 */
class InputParser {

public:

    /**
     * @brief List of command-line arguments
     * 
     */
    std::vector<std::string> args;

    /**
     * @brief Construct a new Input Parser object
     * 
     * @param argc Reference to argc from main
     * @param argv Reference to argv from main
     */
    InputParser(int& argc, char**& argv);

    /**
     * @brief Parses the input arguments into an Options instance
     * 
     * @param options 
     */
    void parse(Options& options);

};

/**
 * @brief Class for timing different pieces of execution
 * 
 */
class Timer {

public:

    /**
     * @brief Alias for clock backend
     * 
     */
    using Clock = std::chrono::steady_clock;

    /**
     * @brief Construct a new Timer object
     * 
     */
    Timer();

    /**
     * @brief Starts the timer
     * 
     */
    void start();

    /**
     * @brief Stops the timer and returns the elapsed time
     * 
     * @return double 
     */
    double stop();

    /**
     * @brief Returns the total accumulated time of this timer
     * 
     * @return double 
     */
    double time();

    /**
     * @brief Restarts the timer accumulated time
     * 
     */
    void restart();

private:

    /**
     * @brief Storage for the accumulated time of this timer
     * 
     */
    double accumulatedTime_ = 0;

    /**
     * @brief Start time of trigger
     * 
     */
    std::chrono::time_point<Clock> startTime_;

    /**
     * @brief End time of trigger
     * 
     */
    std::chrono::time_point<Clock> endTime_;

};

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
    int* argc_;

    /**
     * @brief Pointer to argv from main
     * 
     */
    char*** argv_;

};


} // NAMESPACE : EllipticForest

#endif // ELLIPTIC_FOREST_APP_HPP_