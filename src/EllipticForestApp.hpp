#ifndef ELLIPTIC_FOREST_APP_HPP_
#define ELLIPTIC_FOREST_APP_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <variant>
#include <cstdarg>
#include <cctype>

#include <mpi.h>
#include <petsc.h>
#include <p4est.h>
#include "GenericSingleton.hpp"

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

    Options() {
        setDefaultOptions();
    }
    Options(std::map<std::string, OptionTypes> map) : optionsMap(map) {
        setDefaultOptions();
    }

    OptionTypes& operator[](std::string const key) {
        return optionsMap[key];
    }

    void setOption(std::string const key, OptionTypes const value) {
        optionsMap[key] = value;
    }

    void setFromFile(std::string filename) {

    }

    void setDefaultOptions() {
        optionsMap["min-level"] = 0;
        optionsMap["max-level"] = 1;
        optionsMap["nx"] = 16;
        optionsMap["ny"] = 16;
        optionsMap["cache-operators"] = true;
        optionsMap["homogeneous-rhs"] = true;
        optionsMap["refinement-threshold"] = 0;
    }

    friend std::ostream& operator<<(std::ostream& os, const Options& options) {
        for (const auto& [key, value] : options.optionsMap) {
            os << "[EllipticForest]   " << key << " : ";
            std::visit([&] (auto&& v)
                {os << v << std::endl; }, value);
        }
        return os;
    }

};

class InputParser {

public:

    std::vector<std::string> args;

    InputParser(int& argc, char**& argv) :
        args{argv+1, argv + argc}
            {}

    void parse(Options& options) {
        for (const auto& arg : args) {
            if (arg[0] == '-') {
                if (arg[1] == '-') {
                    // Long option
                    std::string option = arg.substr(2);
                    if (option.find('=') != std::string::npos) {
                        std::string value = option.substr(option.find('=') + 1);
                        option = option.substr(0, option.find('='));
                        options[option] = checkValue(value);
                    }
                    else {
                        options[option] = "";
                    }
                }
                else {
                    // Short option
                    std::string option = arg.substr(1);
                    if (option.find('=') != std::string::npos) {
                        std::string value = option.substr(option.find('=') + 1);
                        option = option.substr(0, option.find('='));
                        options[option] = value;
                    }
                    else {
                        options[option] = "";
                    }
                }
            }
        }
    }

    Options::OptionTypes checkValue(std::string value) {
        if (value == "true") {
            return (bool) true;
        }
        else if (value == "false") {
            return (bool) false;
        }
        else if (isDouble(value)) {
            return (double) std::stod(value);
        }
        else if (isInt(value)) {
            return (int) std::stoi(value);
        }
    }

private:

    bool isDouble(std::string& str) {
        return str.find('.') != std::string::npos;
    }

    bool isInt(std::string& str) {
        for (auto c : str) {
            if (!std::isdigit(c)) return false;
        }
        return true;
    }

};


class Timer {

public:

    using Clock = std::chrono::steady_clock;

    Timer() {}

    void start() {
        startTime_ = Clock::now();
    }

    double stop() {
        endTime_ = Clock::now();
        std::chrono::duration<double> diff = endTime_ - startTime_;
        accumulatedTime_ += diff.count();
        return diff.count();
    }

    double time() { return accumulatedTime_; }

    void restart() {
        accumulatedTime_ = 0;
    }

private:

    double accumulatedTime_ = 0;
    std::chrono::time_point<Clock> startTime_;
    std::chrono::time_point<Clock> endTime_;

};


class EllipticForestApp : public GenericSingleton<EllipticForestApp> {

public:
    
    Logger logger{};
    Options options{};
    std::map<std::string, Timer> timers{};

    EllipticForestApp() : argc_(nullptr), argv_(nullptr) {}
    
    EllipticForestApp(int* argc, char*** argv) : argc_(argc), argv_(argv) {
        addTimer("app-lifetime");
        timers["app-lifetime"].start();
        MPI_Init(argc_, argv_);
        PetscInitialize(argc_, argv_, NULL, NULL);
        PetscGetArgs(argc_, argv_);

        // Create options
        InputParser inputParser(*argc_, *argv_);
        inputParser.parse(options);
        
        int myRank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        if (myRank == 0) {
            std::cout << "[EllipticForest] Welcome to EllipticForest!" << std::endl;
            std::cout << "[EllipticForest] Options:" << std::endl << options << std::endl;
        }
        this->actualClassPointer_ = this;
    }

    ~EllipticForestApp() {
        timers["app-lifetime"].stop();
        int myRank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        if (myRank == 0) {
            std::cout << "[EllipticForest] End of app life cycle, finalizing..." << std::endl;
            std::cout << "[EllipticForest] Timers: " << std::endl;
            for (auto& [key, value] : timers) {
                std::cout << "[EllipticForest]   " << key << " : " << value.time() << " [sec]" << std::endl;
            }
            std::cout << "[EllipticForest] Done!" << std::endl;
        }
        PetscFinalize();
        MPI_Finalize();
    }

    int argc() { return *argc_; }
    char** argv() { return *argv_; }

    template<class... Args>
    void log(std::string message, Args... args) {
        logger.log(message, args...);
    }

    void addTimer(std::string name) {
        timers[name] = Timer();
    }

private:

    int* argc_;
    char*** argv_;

};


} // NAMESPACE : EllipticForest

#endif // ELLIPTIC_FOREST_APP_HPP_