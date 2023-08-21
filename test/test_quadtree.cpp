#include "gtest/gtest.h"
#include <EllipticForestApp.hpp>
#include <Quadtree.hpp>

using namespace EllipticForest;

// Commenting out becuase parallel quadtree doesn't work with adaptive rebuild yet...
#if 0
std::vector<double> refineFunction(double& parentData) {
    std::vector<double> childrenData = {parentData/4.0, parentData/4.0, parentData/4.0, parentData/4.0};
    return childrenData;
}

double coarsenFunction(double& c0, double& c1, double& c2, double& c3) {
    return c0 + c1 + c2 + c3;
}

TEST(Quadtree, refine_coarsen) {

    // Create variables for level arrays to test
    Quadtree<double>::LevelArray gID;
    Quadtree<double>::LevelArray pID;
    Quadtree<double>::LevelArray cID;
    Quadtree<double>::LevelArray lID;
    std::vector<double> data;

    // Create quadtree with root
    double root = 10.0;
    Quadtree<double> quadtree;
    quadtree.buildFromRoot(root);

    gID = {{0}};
    pID = {{-1}};
    cID = {{-1}};
    lID = {{0}};
    data = {10.0};

    for (int l = 0; l < gID.size(); l++) {
        for (int i = 0; i < gID[l].size(); i++) {
            EXPECT_EQ(quadtree.globalIndices()[l][i], gID[l][i]);
            EXPECT_EQ(quadtree.parentIndices()[l][i], pID[l][i]);
            EXPECT_EQ(quadtree.childIndices()[l][i], cID[l][i]);
            EXPECT_EQ(quadtree.leafIndices()[l][i], lID[l][i]);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(quadtree.data()[i], data[i]);
    }

    // Refine node 0
    quadtree.refineNode(0, refineFunction);

    gID = {{0}, {1, 2, 3, 4}};
    pID = {{-1}, {0, 0, 0, 0}};
    cID = {{1}, {-1, -1, -1, -1}};
    lID = {{-1}, {0, 1, 2, 3}};
    data = {10.0, 2.5, 2.5, 2.5, 2.5};

    for (int l = 0; l < gID.size(); l++) {
        for (int i = 0; i < gID[l].size(); i++) {
            EXPECT_EQ(quadtree.globalIndices()[l][i], gID[l][i]);
            EXPECT_EQ(quadtree.parentIndices()[l][i], pID[l][i]);
            EXPECT_EQ(quadtree.childIndices()[l][i], cID[l][i]);
            EXPECT_EQ(quadtree.leafIndices()[l][i], lID[l][i]);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(quadtree.data()[i], data[i]);
    }

    // Refine node 3
    quadtree.refineNode(3, refineFunction);

    gID = {{0}, {1, 2, 3, 8}, {4, 5, 6, 7}};
    pID = {{-1}, {0, 0, 0, 0}, {3, 3, 3, 3}};
    cID = {{1}, {-1, -1, 4, -1}, {-1, -1, -1, -1}};
    lID = {{-1}, {0, 1, -1, 6}, {2, 3, 4, 5}};
    data = {10.0, 2.5, 2.5, 2.5, 0.625, 0.625, 0.625, 0.625, 2.5};

    for (int l = 0; l < gID.size(); l++) {
        for (int i = 0; i < gID[l].size(); i++) {
            EXPECT_EQ(quadtree.globalIndices()[l][i], gID[l][i]);
            EXPECT_EQ(quadtree.parentIndices()[l][i], pID[l][i]);
            EXPECT_EQ(quadtree.childIndices()[l][i], cID[l][i]);
            EXPECT_EQ(quadtree.leafIndices()[l][i], lID[l][i]);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(quadtree.data()[i], data[i]);
    }

    // Refine node 1
    quadtree.refineNode(1, refineFunction);

    gID = {{0}, {1, 6, 7, 12}, {2, 3, 4, 5, 8, 9, 10, 11}};
    pID = {{-1}, {0, 0, 0, 0}, {1, 1, 1, 1, 7, 7, 7, 7}};
    cID = {{1}, {2, -1, 8, -1}, {-1, -1, -1, -1, -1, -1, -1, -1}};
    lID = {{-1}, {-1, 4, -1, 9}, {0, 1, 2, 3, 5, 6, 7, 8}};
    data = {10.0, 2.5, 0.625, 0.625, 0.625, 0.625, 2.5, 2.5, 0.625, 0.625, 0.625, 0.625, 2.5};

    for (int l = 0; l < gID.size(); l++) {
        for (int i = 0; i < gID[l].size(); i++) {
            EXPECT_EQ(quadtree.globalIndices()[l][i], gID[l][i]);
            EXPECT_EQ(quadtree.parentIndices()[l][i], pID[l][i]);
            EXPECT_EQ(quadtree.childIndices()[l][i], cID[l][i]);
            EXPECT_EQ(quadtree.leafIndices()[l][i], lID[l][i]);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(quadtree.data()[i], data[i]);
    }

    // Refine node 10
    quadtree.refineNode(10, refineFunction);

    gID = {{0}, {1, 6, 7, 16}, {2, 3, 4, 5, 8, 9, 10, 15}, {11, 12, 13, 14}};
    pID = {{-1}, {0, 0, 0, 0}, {1, 1, 1, 1, 7, 7, 7, 7}, {10, 10, 10, 10}};
    cID = {{1}, {2, -1, 8, -1}, {-1, -1, -1, -1, -1, -1, 11, -1}, {-1, -1, -1, -1}};
    lID = {{-1}, {-1, 4, -1, 12}, {0, 1, 2, 3, 5, 6, -1, 11}, {7, 8, 9, 10}};
    data = {10.0, 2.5, 0.625, 0.625, 0.625, 0.625, 2.5, 2.5, 0.625, 0.625, 0.625, 0.15625, 0.15625, 0.15625, 0.15625, 0.625, 2.5};

    for (int l = 0; l < gID.size(); l++) {
        for (int i = 0; i < gID[l].size(); i++) {
            EXPECT_EQ(quadtree.globalIndices()[l][i], gID[l][i]);
            EXPECT_EQ(quadtree.parentIndices()[l][i], pID[l][i]);
            EXPECT_EQ(quadtree.childIndices()[l][i], cID[l][i]);
            EXPECT_EQ(quadtree.leafIndices()[l][i], lID[l][i]);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(quadtree.data()[i], data[i]);
    }

    // Refine node 2
    quadtree.refineNode(2, refineFunction);

    gID = {{0}, {1, 10, 11, 20}, {2, 7, 8, 9, 12, 13, 14, 19}, {3, 4, 5, 6, 15, 16, 17, 18}};
    pID = {{-1}, {0, 0, 0, 0}, {1, 1, 1, 1, 11, 11, 11, 11}, {2, 2, 2, 2, 14, 14, 14, 14}};
    cID = {{1}, {2, -1, 12, -1}, {3, -1, -1, -1, -1, -1, 15, -1}, {-1, -1, -1, -1, -1, -1, -1, -1}};
    lID = {{-1}, {-1, 7, -1, 15}, {-1, 4, 5, 6, 8, 9, -1, 14}, {0, 1, 2, 3, 10, 11, 12, 13}};
    data = {10.0, 2.5, 0.625, 0.15625, 0.15625, 0.15625, 0.15625, 0.625, 0.625, 0.625, 2.5, 2.5, 0.625, 0.625, 0.625, 0.15625, 0.15625, 0.15625, 0.15625, 0.625, 2.5};

    for (int l = 0; l < gID.size(); l++) {
        for (int i = 0; i < gID[l].size(); i++) {
            EXPECT_EQ(quadtree.globalIndices()[l][i], gID[l][i]);
            EXPECT_EQ(quadtree.parentIndices()[l][i], pID[l][i]);
            EXPECT_EQ(quadtree.childIndices()[l][i], cID[l][i]);
            EXPECT_EQ(quadtree.leafIndices()[l][i], lID[l][i]);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(quadtree.data()[i], data[i]);
    }

    // Coarsen node 14
    quadtree.coarsenNode(14, coarsenFunction);

    gID = {{0}, {1, 10, 11, 16}, {2, 7, 8, 9, 12, 13, 14, 15}, {3, 4, 5, 6}};
    pID = {{-1}, {0, 0, 0, 0}, {1, 1, 1, 1, 11, 11, 11, 11}, {2, 2, 2, 2}};
    cID = {{1}, {2, -1, 12, -1}, {3, -1, -1, -1, -1, -1, -1, -1}, {-1, -1, -1, -1}};
    lID = {{-1}, {-1, 7, -1, 12}, {-1, 4, 5, 6, 8, 9, 10, 11}, {0, 1, 2, 3}};
    data = {10.0, 2.5, 0.625, 0.15625, 0.15625, 0.15625, 0.15625, 0.625, 0.625, 0.625, 2.5, 2.5, 0.625, 0.625, 0.625, 0.625, 2.5};

    for (int l = 0; l < gID.size(); l++) {
        for (int i = 0; i < gID[l].size(); i++) {
            EXPECT_EQ(quadtree.globalIndices()[l][i], gID[l][i]);
            EXPECT_EQ(quadtree.parentIndices()[l][i], pID[l][i]);
            EXPECT_EQ(quadtree.childIndices()[l][i], cID[l][i]);
            EXPECT_EQ(quadtree.leafIndices()[l][i], lID[l][i]);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(quadtree.data()[i], data[i]);
    }

    // Coarsen node 11
    quadtree.coarsenNode(11, coarsenFunction);

    gID = {{0}, {1, 10, 11, 12}, {2, 7, 8, 9}, {3, 4, 5, 6}};
    pID = {{-1}, {0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}};
    cID = {{1}, {2, -1, -1, -1}, {3, -1, -1, -1}, {-1, -1, -1, -1}};
    lID = {{-1}, {-1, 7, 8, 9}, {-1, 4, 5, 6,}, {0, 1, 2, 3}};
    data = {10.0, 2.5, 0.625, 0.15625, 0.15625, 0.15625, 0.15625, 0.625, 0.625, 0.625, 2.5, 2.5, 2.5};

    for (int l = 0; l < gID.size(); l++) {
        for (int i = 0; i < gID[l].size(); i++) {
            EXPECT_EQ(quadtree.globalIndices()[l][i], gID[l][i]);
            EXPECT_EQ(quadtree.parentIndices()[l][i], pID[l][i]);
            EXPECT_EQ(quadtree.childIndices()[l][i], cID[l][i]);
            EXPECT_EQ(quadtree.leafIndices()[l][i], lID[l][i]);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(quadtree.data()[i], data[i]);
    }

    // Coarsen node 2
    quadtree.coarsenNode(2, coarsenFunction);

    gID = {{0}, {1, 6, 7, 8}, {2, 3, 4, 5}};
    pID = {{-1}, {0, 0, 0, 0}, {1, 1, 1, 1}};
    cID = {{1}, {2, -1, -1, -1}, {-1, -1, -1, -1}};
    lID = {{-1}, {-1, 4, 5, 6}, {0, 1, 2, 3}};
    data = {10.0, 2.5, 0.625, 0.625, 0.625, 0.625, 2.5, 2.5, 2.5};

    for (int l = 0; l < gID.size(); l++) {
        for (int i = 0; i < gID[l].size(); i++) {
            EXPECT_EQ(quadtree.globalIndices()[l][i], gID[l][i]);
            EXPECT_EQ(quadtree.parentIndices()[l][i], pID[l][i]);
            EXPECT_EQ(quadtree.childIndices()[l][i], cID[l][i]);
            EXPECT_EQ(quadtree.leafIndices()[l][i], lID[l][i]);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(quadtree.data()[i], data[i]);
    }

    // Coarsen node 1
    quadtree.coarsenNode(1, coarsenFunction);

    gID = {{0}, {1, 2, 3, 4}};
    pID = {{-1}, {0, 0, 0, 0}};
    cID = {{1}, {-1, -1, -1, -1}};
    lID = {{-1}, {0, 1, 2, 3}};
    data = {10.0, 2.5, 2.5, 2.5, 2.5};

    for (int l = 0; l < gID.size(); l++) {
        for (int i = 0; i < gID[l].size(); i++) {
            EXPECT_EQ(quadtree.globalIndices()[l][i], gID[l][i]);
            EXPECT_EQ(quadtree.parentIndices()[l][i], pID[l][i]);
            EXPECT_EQ(quadtree.childIndices()[l][i], cID[l][i]);
            EXPECT_EQ(quadtree.leafIndices()[l][i], lID[l][i]);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(quadtree.data()[i], data[i]);
    }

    // Coarsen node 0
    quadtree.coarsenNode(0, coarsenFunction);

    gID = {{0}};
    pID = {{-1}};
    cID = {{-1}};
    lID = {{0}};
    data = {10.0};

    for (int l = 0; l < gID.size(); l++) {
        for (int i = 0; i < gID[l].size(); i++) {
            EXPECT_EQ(quadtree.globalIndices()[l][i], gID[l][i]);
            EXPECT_EQ(quadtree.parentIndices()[l][i], pID[l][i]);
            EXPECT_EQ(quadtree.childIndices()[l][i], cID[l][i]);
            EXPECT_EQ(quadtree.leafIndices()[l][i], lID[l][i]);
        }
    }
    for (int i = 0; i < data.size(); i++) {
        EXPECT_FLOAT_EQ(quadtree.data()[i], data[i]);
    }

}

TEST(Quadtree, traverse_preorder) {

    // Create quadtree with root
    double root = 10.0;
    Quadtree<double> quadtree;
    quadtree.buildFromRoot(root);

    // Refine to test case
    quadtree.refineNode(0, refineFunction);
    quadtree.refineNode(3, refineFunction);
    quadtree.refineNode(1, refineFunction);
    quadtree.refineNode(10, refineFunction);
    quadtree.refineNode(2, refineFunction);

    // Perform traversal and check for correct order
    int globalIDCounter = 0;
    std::vector<double> data = {10.0, 2.5, 0.625, 0.15625, 0.15625, 0.15625, 0.15625, 0.625, 0.625, 0.625, 2.5, 2.5, 0.625, 0.625, 0.625, 0.15625, 0.15625, 0.15625, 0.15625, 0.625, 2.5};
    quadtree.traversePreOrder([&](Quadtree<double>::QuadtreeNode node){
        EXPECT_EQ(node.globalID, globalIDCounter);
        EXPECT_FLOAT_EQ(*node.data, data[globalIDCounter]);
        globalIDCounter++;
        return true;
    });

}
#endif