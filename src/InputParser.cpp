#include "InputParser.hpp"

namespace EllipticForest {

InputParser::InputParser()
    {}

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

} // NAMESPACE : EllipticForest