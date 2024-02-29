#include "Options.hpp"

namespace EllipticForest {

Options::Options()
    {}

Options::Options(std::map<std::string, OptionTypes> map) :
    optionsMap(map)
        {}

Options::OptionTypes& Options::operator[](std::string const key) {
    return optionsMap[key];
}

void Options::addOption(std::string option_name, OptionTypes& value, std::string info) {

    // Set option in the map
    optionsMap[option_name] = value;

    // Check for command line override of option
    

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

} // NAMESPACE : EllipticForest