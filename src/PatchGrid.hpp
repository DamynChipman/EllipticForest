#ifndef PATCH_GRID_HPP_
#define PATCH_GRID_HPP_

#include <string>

namespace EllipticForest {

#define XDIM 0
#define YDIM 1

template<typename FloatingPointType>
class PatchGridBase {

public:

    /**
     * @brief Returns the name of patch
     * 
     * @return std::string 
     */
    virtual std::string name() = 0;

    /**
     * @brief Returns the number of points in the x-direction
     * 
     * @return std::size_t 
     */
    virtual std::size_t nPointsX() = 0;

    /**
     * @brief Returns the number of points in the y-direction
     * 
     * @return std::size_t 
     */
    virtual std::size_t nPointsY() = 0;

    /**
     * @brief Returns the x lower bound of the grid
     * 
     * @return FloatingPointType 
     */
    virtual FloatingPointType xLower() = 0;

    /**
     * @brief Returns the x upper bound of the grid
     * 
     * @return FloatingPointType 
     */
    virtual FloatingPointType xUpper() = 0;

    /**
     * @brief Returns the y lower bound of the grid
     * 
     * @return FloatingPointType 
     */
    virtual FloatingPointType yLower() = 0;

    /**
     * @brief Returns the y upper bound of the grid
     * 
     * @return FloatingPointType 
     */
    virtual FloatingPointType yUpper() = 0;

    /**
     * @brief Returns the spacing between points in the x-direction
     * 
     * @return FloatingPointType 
     */
    virtual FloatingPointType dx() = 0;

    /**
     * @brief Returns the spacing between points in the y-direction
     * 
     * @return FloatingPointType 
     */
    virtual FloatingPointType dy() = 0;

    /**
     * @brief Returns the coordinate (either `XDIM` or `YDIM`) at the supplied `index`
     * 
     * @param DIM Either `XDIM` or `YDIM` for the x- or y-coordinate
     * @param index Index of grid point
     * @return FloatingPointType 
     */
    virtual FloatingPointType operator()(std::size_t DIM, std::size_t index) = 0;

};

} // NAMESPACE : EllipticForest

#endif // PATCH_GRID_HPP_