#include "Helpers.hpp"

namespace EllipticForest {

std::string getCurrentDateTimeString() {
    std::time_t now = std::time(nullptr);
    std::tm timeinfo = *std::localtime(&now);

    std::ostringstream oss;
    oss << std::put_time(&timeinfo, "%Y%m%d%H%M%S");
    return oss.str();
}

void writeMapToCSV(const std::map<std::string, std::vector<double>>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }

    for (const auto& entry : data) {
        file << entry.first; // Writing the key to the file
        const auto& vec = entry.second;
        for (const auto& value : vec) {
            file << "," << value; // Writing values in the vector
        }
        file << std::endl; // End of row
    }

    file.close();
}

} // NAMESPACE : EllipticForest