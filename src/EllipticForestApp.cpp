#include "EllipticForestApp.hpp"

namespace EllipticForest {

Logger::Logger()
    {}

Logger::~Logger()
    {}

Options::Options()
    {}

Options::Options(std::map<std::string, OptionTypes> map) :
    optionsMap(map)
        {}

Options::OptionTypes& Options::operator[](std::string const key) {
    return optionsMap[key];
}

void Options::setOption(std::string const key, OptionTypes const value) {
    optionsMap[key] = value;
}

void Options::setFromFile(std::string filename) {
    if (filename.substr(filename.length() - 3) == "ini") {
        // Read from .ini file
        readINIFile(filename, "EllipticForest");
    }
    else {
        throw std::invalid_argument("[EllipticForest] Invalid filename or file format to read in options. `filename` = " + filename);
    }
}

bool Options::optionExists(std::string optionName) {
    return optionsMap.find(optionName) != optionsMap.end();
}

std::ostream& operator<<(std::ostream& os, const Options& options) {
    for (const auto& [key, value] : options.optionsMap) {
        os << "[EllipticForest]   " << key << " : ";
        std::visit([&] (auto&& v)
            {os << v << std::endl; }, value);
    }
    return os;
}

void Options::readINIFile(std::string filename, std::string section) {
    std::ifstream infile(filename);

    std::string line;
    bool read_section = false;
    while (std::getline(infile, line)) {
        // Remove any leading and trailing whitespaces
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);

        // Ignore blank lines and comments
        if (line.length() == 0 || line[0] == ';') {
            continue;
        }

        // Parse section
        if (line[0] == '[' && line[line.length() - 1] == ']') {
            std::string current_section = line.substr(1, line.length() - 2);
            if (current_section == section) {
                read_section = true;
            }
            else {
                read_section = false;
            }
        }
        // Parse key-value pair
        else if (read_section) {
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);

                stripWhitespace(key);
                stripWhitespace(value);

                optionsMap[key] = checkValue(value);
            }
        }
    }

    infile.close();
}

void Options::stripWhitespace(std::string& str) {
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
}

Options::OptionTypes Options::checkValue(std::string value) {
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
    return {};
}

bool Options::isDouble(std::string& str) {
    return str.find('.') != std::string::npos;
}

bool Options::isInt(std::string& str) {
    for (auto c : str) {
        if (!std::isdigit(c)) return false;
    }
    return true;
}

InputParser::InputParser(int& argc, char**& argv) :
    args{argv+1, argv + argc}
        {}

void InputParser::parse(Options& options) {
    for (const auto& arg : args) {
        if (arg[0] == '-') {
            if (arg[1] == '-') {
                // Long option
                std::string option = arg.substr(2);
                if (option.find('=') != std::string::npos) {
                    std::string value = option.substr(option.find('=') + 1);
                    option = option.substr(0, option.find('='));
                    options[option] = options.checkValue(value);
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

Timer::Timer()
    {}

void Timer::start() {
    startTime_ = Clock::now();
}

double Timer::stop() {
    endTime_ = Clock::now();
    std::chrono::duration<double> diff = endTime_ - startTime_;
    accumulatedTime_ += diff.count();
    return diff.count();
}

double Timer::time() { return accumulatedTime_; }

void Timer::restart() {
    accumulatedTime_ = 0;
}

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

} // NAMESPACE : EllipticForest