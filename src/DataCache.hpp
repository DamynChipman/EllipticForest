/**
 * @file DataCache.hpp
 * @author Damyn Chipman (DamynChipman@u.boisestate.edu)
 * @brief Data structure that wraps a map for a templated data cache
 */
#ifndef DATA_CACHE_HPP_
#define DATA_CACHE_HPP_

#include <map>
#include <string>

namespace EllipticForest {

template<typename T>
class DataCache {

public:

    /**
     * @brief Construct a new Data Cache object
     * 
     */
    DataCache() : map_{} {}

    /**
     * @brief Construct a new Data Cache object from an existing map
     * 
     * @param map 
     */
    DataCache(const std::map<std::string, T>& map) : map_(map) {}

    /**
     * @brief Accessor function
     * 
     * @param key Name of data
     * @return T& 
     */
    T& operator[](std::string key) {
        return map_[key];
    }

    /**
     * @brief Check if the DataCache contains the key
     * 
     * @param key Name of data
     * @return true 
     * @return false 
     */
    bool contains(std::string key) {
        return map_.contains(key);
    }

private:

    /**
     * @brief Underlying storage
     * 
     */
    std::map<std::string, T> map_;

};

} // NAMESPACE : EllipticForest

#endif // DATA_CACHE_HPP_