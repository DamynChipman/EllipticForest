#include "P4est.hpp"

namespace EllipticForest {

namespace p4est {

p4est_connectivity_t* p4est_connectivity_new_square_domain(double x_lower, double x_upper, double y_lower, double y_upper) {
  
    const p4est_topidx_t num_vertices = 4;
    const p4est_topidx_t num_trees = 1;
    const p4est_topidx_t num_ctt = 0;
    const double        vertices[4 * 3] = {
        x_lower, y_lower, 0,
        x_upper, y_lower, 0,
        x_lower, y_upper, 0,
        x_upper, y_upper, 0,
    };
    const p4est_topidx_t tree_to_vertex[1 * 4] = {
        0, 1, 2, 3,
    };
    const p4est_topidx_t tree_to_tree[1 * 4] = {
        0, 0, 0, 0,
    };
    const int8_t        tree_to_face[1 * 4] = {
        0, 1, 2, 3,
    };

    return p4est_connectivity_new_copy (num_vertices, num_trees, 0,
                                        vertices, tree_to_vertex,
                                        tree_to_tree, tree_to_face,
                                        NULL, &num_ctt, NULL, NULL);

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

} // NAMESPACE : p4est

} // NAMESPACE : EllipticForest