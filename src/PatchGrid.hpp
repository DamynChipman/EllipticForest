#ifndef PATCH_GRID_HPP_
#define PATCH_GRID_HPP_

#include <string>

#include "MPI.hpp"

#if USE_MATPLOTLIBCPP
#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;
#endif

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

    std::string str() {
        std::string out = "";
        out += "X: [" + std::to_string(xLower()) + ":" + std::to_string(xUpper()) + "], nx = " + std::to_string(nPointsX()) + ", dx = " + std::to_string(dx()) + "  ";
        out += "Y: [" + std::to_string(yLower()) + ":" + std::to_string(yUpper()) + "], ny = " + std::to_string(nPointsY()) + ", dy = " + std::to_string(dy()) + "\n";
        return out;
    }

#if USE_MATPLOTLIBCPP
    void plot(std::string name = "", bool plotBox = true, bool plotPoints = true, bool plotEdges = true, bool plotName = true, std::string edgeColor = "b", std::string pointColor = "r") {

        std::vector<double> xPoints(nPointsX());
        for (auto i = 0; i < nPointsX(); i++) {
            xPoints[i] = operator()(XDIM, i);
        }

        std::vector<double> yPoints(nPointsY());
        for (auto j = 0; j < nPointsY(); j++) {
            yPoints[j] = operator()(YDIM, j);
        }

        if (plotBox) {
            std::vector<double> xCornerCoords = { xLower(), xUpper(), xUpper(), xLower(), xLower() };
            std::vector<double> yCornerCoords = { yLower(), yLower(), yUpper(), yUpper(), yLower() };
            plt::plot(xCornerCoords, yCornerCoords, "-k");
        }

        if (plotEdges) {
            plt::plot(std::vector(nPointsY(), xLower()), yPoints, "." + edgeColor);
            plt::plot(std::vector(nPointsY(), xUpper()), yPoints, "." + edgeColor);
            plt::plot(xPoints, std::vector(nPointsX(), yLower()), "." + edgeColor);
            plt::plot(xPoints, std::vector(nPointsX(), yUpper()), "." + edgeColor);
        }

        if (plotPoints) {
            for (auto j = 0; j < nPointsY(); j++) {
                std::vector<double> yLine(nPointsX(), yPoints[j]);
                plt::plot(xPoints, yLine, "." + pointColor);
            }
        }

        if (plotName) {
            plt::text((xLower() + xUpper()) / 2, (yLower() + yUpper()) / 2, name);
        }

    }
#endif

};

} // NAMESPACE : EllipticForest

#endif // PATCH_GRID_HPP_