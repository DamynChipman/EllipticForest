#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <string>

namespace EllipticForest {

std::string getCurrentDateTimeString();

void writeMapToCSV(const std::map<std::string, std::vector<double>>& data, const std::string& filename);

} // NAMESPACE : EllipticForest

#endif // HELPERS_HPP_