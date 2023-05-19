#include <cstdlib>
#include <cmath>
#include <iostream>
#include <utility>
#include <string>
#include <map>

#include <p4est_bits.h>

#include <PlotUtils.hpp>
#include <P4est.hpp>
#include <EllipticForest.hpp>
#include <Quadtree.hpp>
#include <MPI.hpp>

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

template<typename T>
struct Node {
// public:

    int pfirst = -1;
    int plast = -1;
    T data;

    std::string str() {
        return std::to_string(data);
    }

    friend std::ostream& operator<<(std::ostream& os, const Node<T>& node) {
        os << "node: pfrist = " << node.pfirst << ", plast = " << node.plast << ", data = " << node.data << std::endl;
        return os; 
    }

    Node<T> fromParent(int siblingID, int pfirst, int plast) {
        Node<T> node;
        node.pfirst = pfirst;
        node.plast = plast;
        node.data = data / 4;
        return node;
    }

};

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

int main(int argc, char** argv) {

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
    
    EllipticForest::MPIObject mpi(MPI_COMM_WORLD);
    printf("I am rank [%i / %i]\n", mpi.getRank(), mpi.getSize());

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

        if (quadrant->level > 2) {
            return 0;
        }

        return p4est_quadrant_touches_corner(quadrant, 0, 1);

    },
    NULL);
    
    p4est_refine(p4est, 1,
    [](p4est_t* p4est, p4est_topidx_t which_tree, p4est_quadrant_t* quadrant){

        if (quadrant->level > 2) {
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

    MPI_Barrier(MPI_COMM_WORLD);
    std::this_thread::sleep_for(std::chrono::seconds(mpi.getRank()));

    DoubleNodeFactory factory{};

    double rootData = 10.0;
    EllipticForest::Quadtree<double> quadtree(MPI_COMM_WORLD, p4est, rootData, factory);

    quadtree.traversePreOrder([&](EllipticForest::Node<double>* node){
        std::cout << "node data = " << node->data << std::endl;
        return 1;
    });

    std::cout << std::endl;

    quadtree.traversePostOrder([&](EllipticForest::Node<double>* node){
        std::cout << "node data = " << node->data << std::endl;
        return 1;
    });

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

    MPI_Finalize();

    return EXIT_SUCCESS;
}