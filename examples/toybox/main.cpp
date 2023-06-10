#include <cstdlib>
#include <cmath>
#include <iostream>
#include <utility>
#include <string>
#include <map>

#include <p4est_bits.h>

#include <PlotUtils.hpp>
#include <P4est.hpp>
#include <PETSc.hpp>
#include <EllipticForest.hpp>
#include <QuadNode.hpp>
#include <Quadtree.hpp>
#include <MPI.hpp>

#ifdef USE_MATPLOTLIBCPP
namespace plt = matplotlibcpp;
#endif

class DoubleNodeFactory : public EllipticForest::AbstractNodeFactory<double> {
public:
    DoubleNodeFactory() {}
    ~DoubleNodeFactory() {}
    
    virtual EllipticForest::Node<double>* createNode(double data, std::string path, int level, int pfirst, int plast) {
        EllipticForest::Node<double>* node = new EllipticForest::Node<double>;
        node->data = data;
        node->path = path;
        node->level = level;
        node->pfirst = pfirst;
        node->plast = plast;
        return node;
    }

    virtual EllipticForest::Node<double>* createChildNode(EllipticForest::Node<double>* parentNode, int siblingID, int pfirst, int plast) {
        EllipticForest::Node<double>* node = new EllipticForest::Node<double>;
        node->data = parentNode->data / 4.0;
        node->path = parentNode->path + std::to_string(siblingID);
        node->level = parentNode->level + 1;
        node->pfirst = pfirst;
        node->plast = plast;
        return node;
    }

    virtual EllipticForest::Node<double>* createParentNode(std::vector<EllipticForest::Node<double>*> childrenNodes, int pfirst, int plast) {

        throw std::runtime_error("Virtual function `createParentNode` is NOT IMPLEMENTED!");
        return nullptr;

    }

};

// template<typename T>
// struct Node {
// // public:

//     int pfirst = -1;
//     int plast = -1;
//     T data;

//     std::string str() {
//         return std::to_string(data);
//     }

//     friend std::ostream& operator<<(std::ostream& os, const Node<T>& node) {
//         os << "node: pfrist = " << node.pfirst << ", plast = " << node.plast << ", data = " << node.data << std::endl;
//         return os; 
//     }

//     Node<T> fromParent(int siblingID, int pfirst, int plast) {
//         Node<T> node;
//         node.pfirst = pfirst;
//         node.plast = plast;
//         node.data = data / 4;
//         return node;
//     }

// };

std::string p4est_quadrant_to_string(p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* q) {
    double xyz_lower[3];
    p4est_qcoord_to_vertex(p4est->connectivity, which_tree, q->x, q->y, xyz_lower);

    double xyz_upper[3];
    p4est_qcoord_to_vertex(p4est->connectivity, which_tree, q->x + P4EST_QUADRANT_LEN(q->level), q->y + P4EST_QUADRANT_LEN(q->level), xyz_upper);

    std::string res = "";
    res += "x = [" + std::to_string(xyz_lower[0]) + ", " + std::to_string(xyz_upper[0]) + "], ";
    res += "y = [" + std::to_string(xyz_lower[1]) + ", " + std::to_string(xyz_upper[1]) + "], ";
    res += "l = " + std::to_string(q->level);
    return res;
}

std::string p4est_quadrant_path(const p4est_quadrant_t* q) {
    std::string path = "";
    for (int l = q->level; l != 0; l--) {
        path += std::to_string(p4est_quadrant_ancestor_id(q, l));
    }
    std::reverse(path.begin(), path.end());
    path = "0" + path;
    return path;
}

// std::string p4est_quadrant_path_add(const p4est_quadrant_t* q, std::string p) {
//     return p + std::to_string(p4est_quadrant_child_id);
// }

// struct QCoordKey {
//     p4est_qcoord_t xyz_lower[3];
//     p4est_qcoord_t xyz_upper[3];
//     // friend bool operator<(QCoordKey const& l, QCoordKey const& r) {
//     //     for (int i = 0; i < 3; i++) {
//     //         if (l.xyz_lower[i] < )
//     //     }
//     // }
// };

// struct QCoordKeyHash {
//     std::size_t operator()(const QCoordKey& key) const {
//         std::hash<p4est_qcoord_t> hasher;
//         std::size_t hash = 6;
//         for (int i = 0; i < 3; i++) {
//             hash ^= hasher(key.xyz_lower[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
//         }
//         for (int i = 0; i < 3; i++) {
//             hash ^= hasher(key.xyz_upper[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
//         }
//         return hash;
//     }
// };

// struct QCoordKeyEqual {
//     bool operator()(const QCoordKey& lhs, const QCoordKey& rhs) const {
//         return  lhs.xyz_lower[0] == rhs.xyz_lower[0] &&
//                 lhs.xyz_lower[1] == rhs.xyz_lower[1] &&
//                 lhs.xyz_lower[2] == rhs.xyz_lower[2] &&
//                 lhs.xyz_upper[0] == rhs.xyz_upper[0] &&
//                 lhs.xyz_upper[1] == rhs.xyz_upper[1] &&
//                 lhs.xyz_upper[2] == rhs.xyz_upper[2];
//     }
// };

// using QCoordMap = std::map<QCoordKey, Node*, QCoordKeyHash, QCoordKeyEqual>;

double besselJ(int n, double x) {
    return jn(n, x);
}

const double kappa = 80.0;

double uExact(double x, double y) {
    // Laplace 1
    // return sin(2.0*M_PI*x) * sinh(2.0*M_PI*y);

    // Poisson 1
    // return sin(2.0*M_PI*x) + cos(2.0*M_PI*y);

    // Helmholtz 1
    double x0 = -2;
    double y0 = 0;
    return besselJ(0, kappa*sqrt(pow(x - x0, 2) + pow(y - y0, 2)));

    // Variable Poisson 1 (just set BC)
    // return 4.0;
}

double fRHS(double x, double y) {
    // Laplace 1
    // return 0.0;

    // Poisson 1
    // return -4.0*pow(M_PI, 2)*uExact(x, y);

    // Helmholtz 1
    // return 0.0;

    // Variable Poisson 1
    if (-0.5 < x && x < 1 && -0.5 < y && y < 1) {
        return 10.;
    }
    else {
        return 0.;
    }
}

int main(int argc, char** argv) {

    EllipticForest::EllipticForestApp app(&argc, &argv);
    app.logHead("Starting toybox...");

    std::vector<int> ns = {16, 32, 64, 128, 256, 512, 1024};
    std::vector<double> errors;
    EllipticForest::Vector<double> u_exact, u_petsc;
    int nx, ny;

    for (auto n : ns) {

        nx = n;
        ny = n;
        double x_lower = -1;
        double x_upper = 1;
        double y_lower = -1;
        double y_upper = 1;
        EllipticForest::Petsc::PetscGrid grid(nx, ny, x_lower, x_upper, y_lower, y_upper);

        EllipticForest::Vector<double> g_west(nx);
        EllipticForest::Vector<double> g_east(nx);
        EllipticForest::Vector<double> g_south(nx);
        EllipticForest::Vector<double> g_north(nx);
        for (int j = 0; j < ny; j++) {
            double y = grid(1, j);
            g_west[j] = uExact(x_lower, y);
            g_east[j] = uExact(x_upper, y);
        }
        for (int i = 0; i < nx; i++) {
            double x = grid(0, i);
            g_south[i] = uExact(x, y_lower);
            g_north[i] = uExact(x, y_upper);
        }
        EllipticForest::Vector<double> g = EllipticForest::concatenate({g_west, g_east, g_south, g_north});

        EllipticForest::Vector<double> f(nx*ny);
        u_exact = EllipticForest::Vector<double>(nx*ny);
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                double x = grid(0, i);
                double y = grid(1, j);
                int I = i + j*nx;
                f[I] = fRHS(x, y);
                u_exact[I] = uExact(x, y);
            }
        }

        EllipticForest::Petsc::PetscPatchSolver solver;
        solver.setAlphaFunction([&](double x, double y){
            return x + y + 4.0;;
        });
        solver.setBetaFunction([&](double x, double y){
            if (x < 0) {
                return 1.0;
            }
            else if (0 < x && x < 0.2) {
                return 2.0;
            }
            else if (0.2 < x && x < 0.4) {
                return 4.0;
            }
            else if (0.4 < x && x < 0.8) {
                return 8.0;
            }
            else {
                return 10.0;
            }
        });
        solver.setLambdaFunction([&](double x, double y){
            return x;
        });
        u_petsc = solver.solve(grid, g, f);

        double error = EllipticForest::vectorInfNorm(u_exact, u_petsc);
        errors.push_back(error);
        app.log("N = %4i, Error = %.8e", n, error);

    }

    EllipticForest::Vector<float> u_petsc_float(u_petsc.size());
    for (int i = 0; i < u_petsc.size(); i++) {
        u_petsc_float[i] = static_cast<float>(u_petsc[i]);
    }

    PyObject* py_obj;
    plt::imshow(u_petsc_float.dataPointer(), nx, ny, 1, {}, &py_obj);
    plt::colorbar(py_obj);
    plt::title("u_petsc");
    plt::show();


#if 0

    // Create app
    // EllipticForest::EllipticForestApp app(&argc, &argv);
    // app.logHead("Starting toybox...");

    // std::map<std::string, int> map;

    // map["0"] = 0;
    // map["00"] = 1;
    // map["000"] = 2;
    // map["0000"] = 3;
    // map["0001"] = 4;
    // map["0002"] = 5;
    // map["0003"] = 6;
    // map["001"] = 7;
    // map["002"] = 8;
    // map["003"] = 9;
    // map["01"] = 10;
    // map["02"] = 11;
    // map["03"] = 12;
    // map["030"] = 13;
    // map["031"] = 14;
    // map["032"] = 15;
    // map["033"] = 16;
    // map["0330"] = 17;
    // map["0331"] = 18;
    // map["0332"] = 19;
    // map["0333"] = 20;

    // for (std::map<std::string, int>::iterator iter = map.begin(); iter != map.end(); iter++) {
    //     std::cout << iter->second << std::endl;
    // }

    MPI_Init(&argc, &argv);
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    
    EllipticForest::MPI::MPIObject mpi(MPI_COMM_WORLD);
    // printf("I am rank [%i / %i]\n", mpi.getRank(), mpi.getSize());
#if 0
    EllipticForest::Node<double> node;
    if (mpi.getRank() == 0) {
        // Head rank create node
        node.data = 3.14;
        node.path = "001";
        node.level = 2;
        node.pfirst = 0;
        node.plast = mpi.getSize();
    }
    // std::string node;
    // std::vector<std::string> node;
    // std::vector<double> node;
    // if (mpi.getRank() == 0) {
    //     // node = "asdf";
    //     // node = {"1.", "2.", "3.", "4.", "6.", ""};
    //     // node = {1., 2., 3., 4.};
    // }

    std::cout << "[RANK " << mpi.getRank() << " / " << mpi.getSize() << "] " << node << std::endl;
    // std::cout << "[RANK " << mpi.getRank() << " / " << mpi.getSize() << "] ";
    // for (auto& i : node) std::cout << i << ", ";
    // std::cout << std::endl;

    // Broadcast node
    EllipticForest::MPI::broadcast(node, 0, MPI_COMM_WORLD);

    // if (mpi.getRank() == 0) {
    //     // Head rank send
    //     for (int i = 1; i < mpi.getSize(); i++)
    //         EllipticForest::MPI::send(node, i, 0, MPI_COMM_WORLD);
    // }
    // else {
    //     // Other ranks recieve
    //     MPI_Status status;
    //     EllipticForest::MPI::receive(node, 0, 0, MPI_COMM_WORLD, &status);
    // }

    // Print stuff
    std::cout << "[RANK " << mpi.getRank() << " / " << mpi.getSize() << "] " << node << std::endl;
    // std::cout << "[RANK " << mpi.getRank() << " / " << mpi.getSize() << "] ";
    // for (auto& i : node) std::cout << i << ", ";
    // std::cout << std::endl;
#endif
    // Create p4est
    int fillUniform = 1;
    int refineRecursive = 1;
    int minQuadrants = 0;
    int minLevel = 0;
    std::size_t dataSize = 0;
    p4est_connectivity_t* conn = EllipticForest::p4est::p4est_connectivity_new_square_domain(-1, 1, -1, 1);
    p4est_t* p4est = p4est_new_ext(MPI_COMM_WORLD, conn, minQuadrants, minLevel, fillUniform, dataSize, NULL, NULL);

    // Refine the p4est according to the RHS up to the max level
    p4est_refine(p4est, 1,
    [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){

        if (quadrant->level > 4) {
            return 0;
        }

        return p4est_quadrant_touches_corner(quadrant, 0, 1);

    },
    NULL);
    
    p4est_refine(p4est, 1,
    [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){

        if (quadrant->level > 4) {
            return 0;
        }

        return p4est_quadrant_touches_corner(quadrant, 3, 1);

    },
    NULL);

    // p4est_refine(p4est, 0, [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
    //     return 1;
    // },
    // NULL);

    // p4est_refine(p4est, 0, [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
    //     int id = p4est_quadrant_child_id(quadrant);
    //     if (id == 0 || id == 3) {
    //         return 1;
    //     }
    //     else {
    //         return 0;
    //     }
    // },
    // NULL);

    // p4est_refine(p4est, 0, [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){
    //     int id = p4est_quadrant_child_id(quadrant);
    //     p4est_quadrant_t* parent;
    //     p4est_quadrant_parent(quadrant, parent);
    //     if (id == 0 && )
    // },
    // NULL);

    // Balance the p4est
    p4est_balance(p4est, P4EST_CONNECT_CORNER, NULL);
    p4est_partition(p4est, 0, NULL);

    // Save initial mesh
    bool vtkFlag = true;
    if (vtkFlag) {
        std::string VTKFilename = "toybox_mesh";
        p4est_vtk_write_file(p4est, NULL, VTKFilename.c_str());
    }

    // EllipticForest::Quadtree<double> quadtree{};
    // quadtree.buildFromP4est(p4est, 10.0, [&](double& parentData, int childIndex){
    //     return parentData / 4.0;
    // });

    // MPI_Barrier(MPI_COMM_WORLD);
    // std::this_thread::sleep_for(std::chrono::seconds(mpi.getRank()));

    DoubleNodeFactory factory{};

    double rootData = 10.0;
    EllipticForest::Quadtree<double> quadtree(MPI_COMM_WORLD, p4est, rootData, factory);

    // quadtree.traversePreOrder([&](EllipticForest::Node<double>* node){
    //    std::cout << "PRE: " << *node << "\t" << node->data << std::endl;
    //     return 1;
    // });

    // quadtree.traversePostOrder([&](EllipticForest::Node<double>* node){
    //     std::cout << "POST: " << *node << std::endl;
    //     return 1;
    // });

    // std::cout << "--- MERGING ---" << std::endl;
    quadtree.merge(
        [&](EllipticForest::Node<double>* leafNode) {
            // std::cout << "Calling leaf callback" << std::endl;
            return 1;
        },
        [&](EllipticForest::Node<double>* parentNode, std::vector<EllipticForest::Node<double>*> childrenNodes) {
            // std::cout << "Calling family callback" << std::endl;
            // std::cout << "PARENT: " << *parentNode << std::endl;
            // for (int i = 0; i < 4; i++) {
            //     std::cout << "CHILD " << i << ": " << *childrenNodes[i] << std::endl;
            // }
            double sum = 0;
            for (auto* child : childrenNodes) sum += child->data;
            parentNode->data = 10.0*sum;
            return 1;
        }
    );

    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for(std::chrono::seconds(mpi.getRank()));

    quadtree.traversePreOrder([&](EllipticForest::Node<double>* node){
        printf("[RANK %i / %i] ", mpi.getRank(), mpi.getSize());
        std::cout << "PRE: " << *node << "\t" << node->data << std::endl;
        return 1;
    });

    quadtree.split(
        [&](EllipticForest::Node<double>* leafNode) {
            return 1;
        },
        [&](EllipticForest::Node<double>* parentNode, std::vector<EllipticForest::Node<double>*> childrenNodes) {
            for (auto* child : childrenNodes) child->data = parentNode->data / 4.0;
            return 1;
        }
    );

    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for(std::chrono::seconds(mpi.getRank()));

    quadtree.traversePreOrder([&](EllipticForest::Node<double>* node){
        printf("[RANK %i / %i] ", mpi.getRank(), mpi.getSize());
        std::cout << "PRE: " << *node << "\t" << node->data << std::endl;
        return 1;
    });


#if 0
    // printf("[RANK %i / %i] Quadtree:\n", mpi.getRank(), mpi.getSize());
    // std::cout << quadtree << std::endl;

    //
    // int skipLevels = 0;
    // p4est_search_reorder(p4est, skipLevels, NULL,
    //     [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, p4est_locidx_t local_num, void* point){
    //         int rank;
    //         int size;
    //         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //         MPI_Comm_size(MPI_COMM_WORLD, &size);

    //         double xyz_lower[3];
    //         p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x, quadrant->y, xyz_lower);

    //         double xyz_upper[3];
    //         p4est_qcoord_to_vertex(p4est->connectivity, which_tree, quadrant->x + P4EST_QUADRANT_LEN(quadrant->level), quadrant->y + P4EST_QUADRANT_LEN(quadrant->level), xyz_upper);

    //         Node* node;
    //         if (local_num < 0) {
    //             node = new Node(20);
    //         }
    //         else {
    //             node = (Node*) quadrant->p.user_data;
    //         }
            
    //         printf("[RANK %i / %i] quadrant: num = %i, x = [%10.4f, %10.4f], y = [%10.4f, %10.4f], l = %i, data = %s\n", rank, size, (int) local_num, xyz_lower[0], xyz_upper[0], xyz_lower[1], xyz_upper[1], quadrant->level, node->str().c_str());

    //         return 1;
    //     },
    //     NULL,
    //     NULL, NULL
    // );

    // QCoordMap map;
    // p4est->user_pointer = &map;



    // std::map<std::string, Node<double>*> nodeMap;
    // p4est->user_pointer = &nodeMap;

    // int callPost = 0;
    // p4est_search_all(
    //     p4est,
    //     callPost,
    //     [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, int pfirst, int plast, p4est_locidx_t local_num, void* point) {
    //         // auto& map = *(QCoordMap*) p4est->user_pointer;
    //         auto& map = *(std::map<std::string, Node<double>*>*) p4est->user_pointer;

    //         int rank;
    //         int size;
    //         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //         MPI_Comm_size(MPI_COMM_WORLD, &size);

    //         // Compute unique path
    //         std::string path = p4est_quadrant_path(quadrant);

    //         // Check if quadrant is owned by this rank
    //         bool owned = pfirst <= rank && rank <= plast;

    //         // If owned, create Node in map
    //         if (owned) {
    //             Node<double>* node = new Node<double>;
    //             if (quadrant->level == 0) {
    //                 node->pfirst = pfirst;
    //                 node->plast = plast;
    //                 node->data = 10.0;
    //             }
    //             else {
    //                 std::string parentPath = path.substr(0, path.length()-1);
    //                 Node<double>* parentNode = map[parentPath];
    //                 int siblingID = p4est_quadrant_child_id(quadrant);
    //                 *node = parentNode->fromParent(siblingID, pfirst, plast);
    //             }
    //             map[path] = node;

    //             // Node<double>* node = new Node<double>;
    //             // map.insert({path, node});
    //         }
    //         else {
    //             map[path] = nullptr;
    //         }

    //         std::string q_str = p4est_quadrant_to_string(p4est, which_tree, quadrant);
    //         printf("[RANK %i / %i]:\n\twhich_tree = %i, num = %i\n\tquadrant: %s\n\tpfirst = %i, plast = %i, path = %s\n",
    //                 rank, size-1, which_tree, local_num, q_str.c_str(), pfirst, plast, path.c_str());
    //         if (owned) {
    //             std::cout << *map[path] << std::endl;
    //         }

    //         return 1;
    //     },
    //     NULL,
    //     NULL
    //     // [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant, int pfirst, int plast, p4est_locidx_t local_num, void* point) {
    //     //     int rank;
    //     //     int size;
    //     //     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //     //     MPI_Comm_size(MPI_COMM_WORLD, &size);

    //     //     double* p = (double*) point;

    //     //     printf("[RANK %i / %i]:\n\tpoint = %10.4f\n\n",
    //     //             rank, size-1, *p);

    //     //     return 1;
    //     // },
    //     // points
    // );

    // for (std::map<std::string, Node<double>*>::iterator iter = nodeMap.begin(); iter != nodeMap.end(); iter++) {
    //     if (iter->second != nullptr)
    //         std::cout << (iter->first) << "->" << *(iter->second) << std::endl;
    // }
#endif
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;
}