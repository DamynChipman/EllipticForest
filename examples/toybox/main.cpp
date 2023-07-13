#include <cstdlib>
#include <cmath>
#include <iostream>
#include <utility>
#include <string>
#include <map>

#include <p4est_bits.h>

#include <PlotUtils.hpp>
#include <MPI.hpp>
#include <FISHPACK.hpp>
#include <P4est.hpp>
#include <PETSc.hpp>
#include <EllipticForest.hpp>
#include <QuadNode.hpp>
#include <Quadtree.hpp>
#include <Vector.hpp>
#include <Matrix.hpp>
#include <VTK.hpp>
#include <Mesh.hpp>

namespace ef = EllipticForest;

#ifdef USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

class Mesh : public ef::UnstructuredGridNodeBase {

public:

    ef::Vector<double> points;
    ef::Vector<int> connectivity;
    ef::Vector<int> offsets;
    ef::Vector<int> types;

    Mesh() {
        points = {
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            2.0, 0.0, 0.0,
            2.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            1.0, 1.0, 0.0,
            1.0, 2.0, 0.0,
            0.0, 2.0, 0.0,
            1.0, 1.0, 0.0,
            2.0, 1.0, 0.0,
            2.0, 2.0, 0.0,
            1.0, 2.0, 0.0
        };
        points.setNumberOfComponents("3");
        points.name() = "points";

        connectivity = {
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
            12, 13, 14, 15
        };
        connectivity.name() = "connectivity";
        connectivity.setType("Int64");

        offsets = {
            4, 8, 12, 16
        };
        offsets.name() = "offsets";
        offsets.setType("Int64");

        types = {
            9, 9, 9, 9
        };
        types.name() = "types";
        types.setType("Int64");
    }

    std::string getNumberOfPoints() { return "16"; }
    std::string getNumberOfCells() { return "4"; }
    ef::DataArrayNodeBase& getPoints() { return points; }
    ef::DataArrayNodeBase& getConnectivity() { return connectivity; }
    ef::DataArrayNodeBase& getOffsets() { return offsets; }
    ef::DataArrayNodeBase& getTypes() { return types; }

};

int main(int argc, char** argv) {

    ef::EllipticForestApp app(&argc, &argv);

    Mesh mesh{};

    ef::Vector<double> pressure = {10.0, 40.0, 80.0, 160.0};
    pressure.name() = "pressure";
    ef::Vector<double> temperature = {-0.2, 3.4, 9.3, 10.3};
    temperature.name() = "temperature";

    ef::UnstructuredGridVTK vtu{};
    vtu.buildMesh(mesh);
    vtu.addCellData(pressure);
    vtu.addCellData(temperature);
    vtu.toVTK("test.vtu");

    return EXIT_SUCCESS;
}