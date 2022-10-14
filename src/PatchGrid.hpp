#ifndef PATCH_GRID_HPP_
#define PATCH_GRID_HPP_

#include <string>

namespace EllipticForest {

#define XDIM 0
#define YDIM 1

template<typename FloatingPointType>
class PatchGridBase {

public:

    virtual std::string name() = 0;
    virtual std::size_t nPointsX() = 0;
    virtual std::size_t nPointsY() = 0;
    virtual FloatingPointType xLower() = 0;
    virtual FloatingPointType xUpper() = 0;
    virtual FloatingPointType yLower() = 0;
    virtual FloatingPointType yUpper() = 0;
    virtual FloatingPointType dx() = 0;
    virtual FloatingPointType dy() = 0;
    virtual FloatingPointType operator()(std::size_t DIM, std::size_t index) = 0;

};

} // NAMESPACE : EllipticForest

#endif // PATCH_GRID_HPP_