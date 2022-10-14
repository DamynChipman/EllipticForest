#ifndef DATA_CACHE_HPP_
#define DATA_CACHE_HPP_

#include <map>
#include <string>

namespace EllipticForest {

template<typename T>
class DataCache {

public:

    DataCache() : map_{} {}
    DataCache(const std::map<std::string, T>& map) : map_(map) {}

    T& operator[](std::string key) {
        return map_[key];
    }

    bool contains(std::string key) {
        return map_.contains(key);
    }

private:

    std::map<std::string, T> map_;

};

} // NAMESPACE : EllipticForest

#endif // DATA_CACHE_HPP_