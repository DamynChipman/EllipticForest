#ifndef PATCH_GRID_HPP_
#define PATCH_GRID_HPP_

#include <string>

#include "MPI.hpp"

#if USE_MATPLOTLIBCPP
#ifdef _GNU_SOURCE
#undef _GNU_SOURCE
#endif
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
    virtual std::size_t nx() = 0;

    /**
     * @brief Returns the number of points in the y-direction
     * 
     * @return std::size_t 
     */
    virtual std::size_t ny() = 0;

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

    /**
     * @brief Returns a string version of the grid
     * 
     * @return std::string 
     */
    std::string str() {
        std::string out = "";
        out += "X: [" + std::to_string(xLower()) + ":" + std::to_string(xUpper()) + "], nx = " + std::to_string(nx()) + ", dx = " + std::to_string(dx()) + "  ";
        out += "Y: [" + std::to_string(yLower()) + ":" + std::to_string(yUpper()) + "], ny = " + std::to_string(ny()) + ", dy = " + std::to_string(dy()) + "\n";
        return out;
    }

#if USE_MATPLOTLIBCPP
    /**
     * @brief Plots the patch with matplotlibcpp
     * 
     * @param name Title of plot
     * @param plotBox Flag to plot box around grid
     * @param plotPoints Flag to plot points in grid
     * @param plotEdges Flag to plot edges in grid
     * @param plotName Flag to display plot name
     * @param edgeColor String with color specifier for edges (standard matplotlib colors)
     * @param pointColor String with color specifier for points (standard matplotlib colors)
     */
    void plot(std::string name = "", bool plotBox = true, bool plotPoints = true, bool plotEdges = true, bool plotName = true, std::string edgeColor = "b", std::string pointColor = "r") {

        std::vector<double> xPoints(nx());
        for (auto i = 0; i < nx(); i++) {
            xPoints[i] = operator()(XDIM, i);
        }

        std::vector<double> yPoints(ny());
        for (auto j = 0; j < ny(); j++) {
            yPoints[j] = operator()(YDIM, j);
        }

        if (plotBox) {
            std::vector<double> xCornerCoords = { xLower(), xUpper(), xUpper(), xLower(), xLower() };
            std::vector<double> yCornerCoords = { yLower(), yLower(), yUpper(), yUpper(), yLower() };
            plt::plot(xCornerCoords, yCornerCoords, "-k");
        }

        if (plotEdges) {
            plt::plot(std::vector(ny(), xLower()), yPoints, "." + edgeColor);
            plt::plot(std::vector(ny(), xUpper()), yPoints, "." + edgeColor);
            plt::plot(xPoints, std::vector(nx(), yLower()), "." + edgeColor);
            plt::plot(xPoints, std::vector(nx(), yUpper()), "." + edgeColor);
        }

        if (plotPoints) {
            for (auto j = 0; j < ny(); j++) {
                std::vector<double> yLine(nx(), yPoints[j]);
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