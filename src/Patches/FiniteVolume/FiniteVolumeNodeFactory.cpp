#include "FiniteVolumeNodeFactory.hpp"
#include "../../PlotUtils.hpp"

namespace EllipticForest {

FiniteVolumeNodeFactory::FiniteVolumeNodeFactory(FiniteVolumeSolver& solver) :
    solver(solver)
        {}

FiniteVolumeNodeFactory::FiniteVolumeNodeFactory(MPI::Communicator comm, FiniteVolumeSolver& solver) :
    MPIObject(comm),
    solver(solver)
        {}

Node<FiniteVolumePatch>* FiniteVolumeNodeFactory::createNode(FiniteVolumePatch data, std::string path, int level, int pfirst, int plast) {
    return new Node<FiniteVolumePatch>(this->getComm(), data, path, level, pfirst, plast);
}

Node<FiniteVolumePatch>* FiniteVolumeNodeFactory::createChildNode(Node<FiniteVolumePatch>* parent_node, int sibling_id, int pfirst, int plast) {
    
    auto& app = EllipticForest::EllipticForestApp::getInstance();
    // app.log("[createChildNode] parent_node = " + parent_node->path + ", sibling_id = " + std::to_string(sibling_id));
    // Get parent grid info
    auto& parent_patch = parent_node->data;
    auto& parent_grid = parent_node->data.grid();
    int nx = parent_grid.nx();
    int ny = parent_grid.ny();
    double x_lower = parent_grid.xLower();
    double x_upper = parent_grid.xUpper();
    double x_mid = (x_lower + x_upper) / 2.0;
    double y_lower = parent_grid.yLower();
    double y_upper = parent_grid.yUpper();
    double y_mid = (y_lower + y_upper) / 2.0;

    // Create child grid
    FiniteVolumeGrid child_grid;
    switch (sibling_id) {
        case 0:
            // Lower left
            child_grid = FiniteVolumeGrid(parent_node->data.getComm(), nx, x_lower, x_mid, ny, y_lower, y_mid);
            break;
        case 1:
            // Lower right
            child_grid = FiniteVolumeGrid(parent_node->data.getComm(), nx, x_mid, x_upper, ny, y_lower, y_mid);
            break;
        case 2:
            // Upper left
            child_grid = FiniteVolumeGrid(parent_node->data.getComm(), nx, x_lower, x_mid, ny, y_mid, y_upper);
            break;
        case 3:
            // Upper right
            child_grid = FiniteVolumeGrid(parent_node->data.getComm(), nx, x_mid, x_upper, ny, y_mid, y_upper);
            break;
        default:
            break;
    }

    // Create child patch
    FiniteVolumePatch child_patch(parent_node->data.getComm(), child_grid);

    // Set child path data
    bool tagged_for_refinement = false;
    if (parent_patch.matrixT().nrows() != 0 && parent_patch.matrixT().ncols() != 0) {
        // app.log("[createChildNode] creating child T");
        child_patch.matrixT() = solver.buildD2N(child_patch.grid());
        tagged_for_refinement = true;
    }

//     if (parent_patch.vectorG().size() != 0) {
//         // app.log("[createChildNode] creating child g");
        
//         int n_parent = parent_grid.nx();
//         double h_child = child_grid.dx();
//         int n_child = child_grid.nx();
//         double h_parent = parent_grid.dx();

//         auto parent_y = linspace(parent_grid.yLower() + h_parent/2, parent_grid.yUpper() - h_parent/2, n_parent);
//         auto parent_x = linspace(parent_grid.xLower() + h_parent/2, parent_grid.xUpper() - h_parent/2, n_parent);

//         // auto parent_x_W = linspace(parent_grid(1, 0), parent_grid(1, parent_grid.ny()-1), parent_grid.ny());
//         // auto parent_x_E = linspace(parent_grid(1, 0), parent_grid(1, parent_grid.ny()-1), parent_grid.ny());
//         // auto parent_x_S = linspace(parent_grid(0, 0), parent_grid(0, parent_grid.nx()-1), parent_grid.nx());
//         // auto parent_x_N = linspace(parent_grid(0, 0), parent_grid(0, parent_grid.nx()-1), parent_grid.nx());

//         // auto parent_I_W = vectorRange(0, parent_grid.ny() - 1);
//         // auto parent_I_E = vectorRange(parent_grid.ny(), 2*parent_grid.ny() - 1);
//         // auto parent_I_S = vectorRange(2*parent_grid.ny(), (2*parent_grid.ny() + parent_grid.nx()) - 1);
//         // auto parent_I_N = vectorRange(2*parent_grid.ny() + parent_grid.nx(), (2*parent_grid.ny() + 2*parent_grid.nx()) - 1);

//         auto parent_g_W = parent_patch.vectorG().getSegment(0*n_parent, n_parent);
//         auto parent_g_E = parent_patch.vectorG().getSegment(1*n_parent, n_parent);
//         auto parent_g_S = parent_patch.vectorG().getSegment(2*n_parent, n_parent);
//         auto parent_g_N = parent_patch.vectorG().getSegment(3*n_parent, n_parent);

//         // auto parent_g_W = parent_patch.vectorG()(parent_I_W);
//         // auto parent_g_E = parent_patch.vectorG()(parent_I_E);
//         // auto parent_g_S = parent_patch.vectorG()(parent_I_S);
//         // auto parent_g_N = parent_patch.vectorG()(parent_I_N);

//         PolynomialInterpolant interpolant_W(parent_y, parent_g_W, 4);
//         PolynomialInterpolant interpolant_E(parent_y, parent_g_E, 4);
//         PolynomialInterpolant interpolant_S(parent_x, parent_g_S, 4);
//         PolynomialInterpolant interpolant_N(parent_x, parent_g_N, 4);

//         // PolynomialInterpolant interpolant_W(parent_x_W, parent_g_W, 2);
//         // PolynomialInterpolant interpolant_E(parent_x_E, parent_g_E, 2);
//         // PolynomialInterpolant interpolant_S(parent_x_S, parent_g_S, 2);
//         // PolynomialInterpolant interpolant_N(parent_x_N, parent_g_N, 2);

//         auto child_y = linspace(child_grid.yLower() + h_child/2, child_grid.yUpper() - h_child/2, n_child);
//         auto child_x = linspace(child_grid.xLower() + h_child/2, child_grid.xUpper() - h_child/2, n_child);

//         // auto child_x_W = linspace(child_grid(1, 0), child_grid(1, child_grid.ny()-1), child_grid.ny());
//         // auto child_x_E = linspace(child_grid(1, 0), child_grid(1, child_grid.ny()-1), child_grid.ny());
//         // auto child_x_S = linspace(child_grid(0, 0), child_grid(0, child_grid.nx()-1), child_grid.nx());
//         // auto child_x_N = linspace(child_grid(0, 0), child_grid(0, child_grid.nx()-1), child_grid.nx());

//         auto child_g_W = interpolant_W(child_y);
//         auto child_g_E = interpolant_E(child_y);
//         auto child_g_S = interpolant_S(child_x);
//         auto child_g_N = interpolant_N(child_x);

//         // auto child_g_W = interpolant_W(child_x_W);
//         // auto child_g_E = interpolant_E(child_x_E);
//         // auto child_g_S = interpolant_S(child_x_S);
//         // auto child_g_N = interpolant_N(child_x_N);

// // #if USE_MATPLOTLIBCPP

// //         plt::named_plot("parent_x_W", parent_x_W.data(), parent_g_W.data(), "-ro");
// //         plt::named_plot("parent_x_E", parent_x_E.data(), parent_g_E.data(), "--ro");
// //         plt::named_plot("parent_x_S", parent_x_S.data(), parent_g_S.data(), "-.ro");
// //         plt::named_plot("parent_x_N", parent_x_N.data(), parent_g_N.data(), ":ro");
// //         plt::named_plot("child_x_W", child_x_W.data(), child_g_W.data(), "-bs");
// //         plt::named_plot("child_x_E", child_x_E.data(), child_g_E.data(), "--bs");
// //         plt::named_plot("child_x_S", child_x_S.data(), child_g_S.data(), "-.bs");
// //         plt::named_plot("child_x_N", child_x_N.data(), child_g_N.data(), ":bs");
// //         plt::legend({{"loc", "upper right"}});
// //         plt::grid(true);
// //         plt::show();

// // #endif

//         child_patch.vectorG() = concatenate({child_g_W, child_g_E, child_g_S, child_g_N});
//         tagged_for_refinement = true;
//     }
    
    if (parent_patch.vectorF().size() != 0) {
        // app.log("[createChildNode] creating child f");
        // TODO: Add try-catch for if the RHS function is set in the solver
        // auto x1_parent = linspace(parent_grid(0, 0), parent_grid(0, parent_grid.nx()-1), parent_grid.nx());
        // auto x2_parent = linspace(parent_grid(1, 0), parent_grid(1, parent_grid.ny()-1), parent_grid.ny());
        // auto& f_parent = parent_patch.vectorF();

        // BilinearInterpolant interpolant(x1_parent, x2_parent, f_parent);

        // auto x1_child = linspace(child_grid(0, 0), child_grid(0, child_grid.nx()-1), child_grid.nx());
        // auto x2_child = linspace(child_grid(1, 0), child_grid(1, child_grid.ny()-1), child_grid.ny());
        // auto f_child = interpolant(x1_child, x2_child);

        // child_patch.vectorF() = f_child;

        // SANITY CHECKS: Plug in exact RHS function here instead of interpolation
        child_patch.vectorF() = Vector<double>(child_grid.nx()*child_grid.ny(), 0);
        for (auto i = 0; i < child_grid.nx(); i++) {
            for (auto j = 0; j < child_grid.ny(); j++) {
                double x = child_grid(0, i);
                double y = child_grid(1, j);
                int index = j + i*child_grid.ny();
                child_patch.vectorF()[index] = -(sin(x) + sin(y));
            }
        }

        tagged_for_refinement = true;
    }

//     if (parent_patch.vectorU().size() != 0) {
//         // app.log("[createChildNode] creating child u");
//         auto x1_parent = linspace(parent_grid(0, 0), parent_grid(0, parent_grid.nx()-1), parent_grid.nx());
//         auto x2_parent = linspace(parent_grid(1, 0), parent_grid(1, parent_grid.ny()-1), parent_grid.ny());
//         auto& u_parent = parent_patch.vectorU();

//         BilinearInterpolant interpolant(x1_parent, x2_parent, u_parent);

//         auto x1_child = linspace(child_grid(0, 0), child_grid(0, child_grid.nx()-1), child_grid.nx());
//         auto x2_child = linspace(child_grid(1, 0), child_grid(1, child_grid.ny()-1), child_grid.ny());
//         child_patch.vectorU() = interpolant(x1_child, x2_child);

//         // child_patch.vectorU() = solver.solve(child_patch.grid(), child_patch.vectorG(), child_patch.vectorF());
//         tagged_for_refinement = true;
//     }

    if (parent_patch.vectorH().size() != 0) {
        // app.log("[createChildNode] creating child h");
        child_patch.vectorH() = solver.particularNeumannData(child_patch.grid(), child_patch.vectorF());
        tagged_for_refinement = true;
    }

    // Create child node
    std::string path = parent_node->path + std::to_string(sibling_id);
    int level = parent_node->level + 1;
    Node<FiniteVolumePatch>* child_node = new Node<FiniteVolumePatch>(this->getComm(), child_patch, path, level, pfirst, plast);

    // Store siblings in order to update parent data
    siblings[sibling_id] = &child_node->data;

    // Update parent patch data
    if (tagged_for_refinement && sibling_id == 3) {
        // app.log("[createChildNode] merging parent...");
        auto& tau = parent_patch;
        auto& alpha = *siblings[0];
        auto& beta = *siblings[1];
        auto& gamma = *siblings[2];
        auto& omega = *siblings[3];
        FiniteVolumeHPS::merge4to1(tau, alpha, beta, gamma, omega, solver);
        FiniteVolumeHPS::upwards4to1(tau, alpha, beta, gamma, omega);
        // FiniteVolumeHPS::mergeX(tau, alpha, beta, gamma, omega);
        // FiniteVolumeHPS::mergeS(tau, alpha, beta, gamma, omega);
        // FiniteVolumeHPS::mergeT(tau, alpha, beta, gamma, omega);
        // FiniteVolumeHPS::reorderOperators(tau, alpha, beta, gamma, omega);
        // FiniteVolumeHPS::coarsen(tau, alpha, beta, gamma, omega);
    }

    return child_node;
    
}

Node<FiniteVolumePatch>* FiniteVolumeNodeFactory::createParentNode(std::vector<Node<FiniteVolumePatch>*> child_nodes, int pfirst, int plast) {

    // std::cout << child_nodes[0]->data.str() << std::endl;

    // Create parent grid
    int nx = child_nodes[0]->data.grid().nx();
    int ny = child_nodes[0]->data.grid().ny();
    double x_lower = child_nodes[0]->data.grid().xLower();
    double x_upper = child_nodes[1]->data.grid().xUpper();
    double y_lower = child_nodes[0]->data.grid().yLower();
    double y_upper = child_nodes[2]->data.grid().yUpper();
    FiniteVolumeGrid parent_grid(this->getComm(), nx, x_lower, x_upper, ny, y_lower, y_upper);

    // Create parent patch
    FiniteVolumePatch parent_patch(this->getComm(), parent_grid); // TODO: Switch MPI_COMM_WORLD to patch communicator

    // Create parent patch data
    bool needs_data = false;
    bool needs_rhs_data = false;
    bool needs_T_data = (child_nodes[0]->data.matrixT().nrows() != 0 && child_nodes[0]->data.matrixT().ncols() != 0) ||
                        (child_nodes[1]->data.matrixT().nrows() != 0 && child_nodes[1]->data.matrixT().ncols() != 0) ||
                        (child_nodes[2]->data.matrixT().nrows() != 0 && child_nodes[2]->data.matrixT().ncols() != 0) ||
                        (child_nodes[3]->data.matrixT().nrows() != 0 && child_nodes[3]->data.matrixT().ncols() != 0);
    bool needs_g_data = child_nodes[0]->data.vectorG().size() ||
                        child_nodes[1]->data.vectorG().size() ||
                        child_nodes[2]->data.vectorG().size() ||
                        child_nodes[3]->data.vectorG().size();
    bool needs_u_data = child_nodes[0]->data.vectorU().size() ||
                        child_nodes[1]->data.vectorU().size() ||
                        child_nodes[2]->data.vectorU().size() ||
                        child_nodes[3]->data.vectorU().size();
    bool needs_f_data = child_nodes[0]->data.vectorF().size() ||
                        child_nodes[1]->data.vectorF().size() ||
                        child_nodes[2]->data.vectorF().size() ||
                        child_nodes[3]->data.vectorF().size();
    bool needs_h_data = child_nodes[0]->data.vectorH().size() ||
                        child_nodes[1]->data.vectorH().size() ||
                        child_nodes[2]->data.vectorH().size() ||
                        child_nodes[3]->data.vectorH().size();

    if (needs_T_data) {
        parent_patch.matrixT() = solver.buildD2N(parent_patch.grid());
    }
    if (needs_g_data) {
        auto& alpha = child_nodes[0]->data;
        auto& beta = child_nodes[1]->data;
        auto& gamma = child_nodes[2]->data;
        auto& omega = child_nodes[3]->data;
        auto& tau = parent_patch;

        auto child_x_W = linspace(alpha.grid()(1, 0), gamma.grid()(1, gamma.grid().ny()-1), alpha.grid().ny() + gamma.grid().ny());
        auto child_x_E = linspace(beta.grid()(1, 0), omega.grid()(1, omega.grid().ny()-1), beta.grid().ny() + omega.grid().ny());
        auto child_x_S = linspace(alpha.grid()(0, 0), beta.grid()(0, beta.grid().nx()-1), alpha.grid().nx() + beta.grid().nx());
        auto child_x_N = linspace(gamma.grid()(0, 0), omega.grid()(0, omega.grid().nx()-1), gamma.grid().nx() + omega.grid().nx());
        
        auto I_W = vectorRange(0, alpha.grid().ny()-1);
        auto I_E = vectorRange(alpha.grid().ny(), 2*alpha.grid().ny()-1);
        auto I_S = vectorRange(2*alpha.grid().ny(), (2*alpha.grid().ny() + alpha.grid().nx())-1);
        auto I_N = vectorRange((2*alpha.grid().ny() + alpha.grid().nx()), (2*alpha.grid().ny() + 2*alpha.grid().nx())-1);

        auto child_g_W = concatenate({alpha.vectorG()(I_W), gamma.vectorG()(I_W)});
        auto child_g_E = concatenate({beta.vectorG()(I_E), omega.vectorG()(I_E)});
        auto child_g_S = concatenate({alpha.vectorG()(I_S), beta.vectorG()(I_S)});
        auto child_g_N = concatenate({gamma.vectorG()(I_N), omega.vectorG()(I_N)});
        
        LinearInterpolant interpolant_W(child_x_W, child_g_W);
        LinearInterpolant interpolant_E(child_x_E, child_g_E);
        LinearInterpolant interpolant_S(child_x_S, child_g_S);
        LinearInterpolant interpolant_N(child_x_N, child_g_N);

        auto parent_x_W = linspace(tau.grid()(1, 0), tau.grid()(1, tau.grid().ny()-1), tau.grid().ny());
        auto parent_x_E = linspace(tau.grid()(1, 0), tau.grid()(1, tau.grid().ny()-1), tau.grid().ny());
        auto parent_x_S = linspace(tau.grid()(0, 0), tau.grid()(0, tau.grid().nx()-1), tau.grid().nx());
        auto parent_x_N = linspace(tau.grid()(0, 0), tau.grid()(0, tau.grid().nx()-1), tau.grid().nx());

        auto parent_g_W = interpolant_W(parent_x_W);
        auto parent_g_E = interpolant_E(parent_x_E);
        auto parent_g_S = interpolant_S(parent_x_S);
        auto parent_g_N = interpolant_N(parent_x_N);

        parent_patch.vectorG() = concatenate({parent_g_W, parent_g_E, parent_g_S, parent_g_N});
    }
    if (needs_u_data) {
        auto& alpha = child_nodes[0]->data;
        auto& beta = child_nodes[1]->data;
        auto& gamma = child_nodes[2]->data;
        auto& omega = child_nodes[3]->data;
        auto& tau = parent_patch;

        auto x1_child = linspace(alpha.grid()(0, 0), beta.grid()(0, beta.grid().nx()-1), alpha.grid().nx() + beta.grid().nx());
        auto x2_child = linspace(alpha.grid()(1, 0), gamma.grid()(1, gamma.grid().ny()-1), alpha.grid().ny() + gamma.grid().ny());
        
        Vector<int> block_sizes(4*alpha.grid().ny(), alpha.grid().nx());
        Vector<int> pi_patch_order(4*alpha.grid().ny());
        for (int j = 0; j < alpha.grid().ny(); j++) {
            pi_patch_order[2*j] = j;
            pi_patch_order[2*j+1] = alpha.grid().ny() + j;
            pi_patch_order[2*j + 2*alpha.grid().ny()] = j + 2*alpha.grid().ny();
            pi_patch_order[2*j + 2*alpha.grid().ny() + 1] = alpha.grid().ny() + j + 2*alpha.grid().ny();
        }
        
        Vector<double> u_child = concatenate({alpha.vectorU(), beta.vectorU(), gamma.vectorU(), omega.vectorU()});
        Vector<double> u_child_patch_order = u_child.blockPermute(pi_patch_order, block_sizes);

        BilinearInterpolant interpolant(x1_child, x2_child, u_child_patch_order);

        auto x1_parent = linspace(tau.grid()(0, 0), tau.grid()(0, tau.grid().nx()-1), tau.grid().nx());
        auto x2_parent = linspace(tau.grid()(1, 0), tau.grid()(1, tau.grid().ny()-1), tau.grid().ny());
        parent_patch.vectorU() = interpolant(x1_parent, x2_parent);
    }
    if (needs_f_data) {
        auto& alpha = child_nodes[0]->data;
        auto& beta = child_nodes[1]->data;
        auto& gamma = child_nodes[2]->data;
        auto& omega = child_nodes[3]->data;
        auto& tau = parent_patch;

        auto x1_child = linspace(alpha.grid()(0, 0), beta.grid()(0, beta.grid().nx()-1), alpha.grid().nx() + beta.grid().nx());
        auto x2_child = linspace(alpha.grid()(1, 0), gamma.grid()(1, gamma.grid().ny()-1), alpha.grid().ny() + gamma.grid().ny());
        
        Vector<int> block_sizes(4*alpha.grid().ny(), alpha.grid().nx());
        Vector<int> pi_patch_order(4*alpha.grid().ny());
        for (int j = 0; j < alpha.grid().ny(); j++) {
            pi_patch_order[2*j] = j;
            pi_patch_order[2*j+1] = alpha.grid().ny() + j;
            pi_patch_order[2*j + 2*alpha.grid().ny()] = j + 2*alpha.grid().ny();
            pi_patch_order[2*j + 2*alpha.grid().ny() + 1] = alpha.grid().ny() + j + 2*alpha.grid().ny();
        }
        
        Vector<double> u_child = concatenate({alpha.vectorF(), beta.vectorF(), gamma.vectorF(), omega.vectorF()});
        Vector<double> u_child_patch_order = u_child.blockPermute(pi_patch_order, block_sizes);

        BilinearInterpolant interpolant(x1_child, x2_child, u_child_patch_order);

        auto x1_parent = linspace(tau.grid()(0, 0), tau.grid()(0, tau.grid().nx()-1), tau.grid().nx());
        auto x2_parent = linspace(tau.grid()(1, 0), tau.grid()(1, tau.grid().ny()-1), tau.grid().ny());
        parent_patch.vectorF() = interpolant(x1_parent, x2_parent);
    }
    if (needs_h_data) {
        parent_patch.vectorH() = solver.particularNeumannData(parent_patch.grid(), parent_patch.vectorF());
    }
    
    // Create parent node
    std::string path = child_nodes[0]->path.substr(0, child_nodes[0]->path.length()-1);
    int level = child_nodes[0]->level - 1;
    return new Node<FiniteVolumePatch>(this->getComm(), parent_patch, path, level, pfirst, plast);

}

} // NAMESPACE : EllipticForest