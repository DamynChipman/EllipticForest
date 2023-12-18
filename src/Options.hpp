#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include <iostream>
#include <string>
#include <variant>
#include <map>
#include <fstream>

namespace EllipticForest {

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

} // NAMESPACE : EllipticForest

#endif // OPTIONS_HPP_
