#ifndef INPUT_PARSER_HPP_
#define INPUT_PARSER_HPP_

#include <string>
#include <vector>

#include "Options.hpp"

namespace EllipticForest {

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

} // NAMESPACE : EllipticForest

#endif // INPUT_PARSER_HPP_