#ifndef FINITE_VOLUME_NODE_FACTORY_HPP_
#define FINITE_VOLUME_NODE_FACTORY_HPP_

#include "../../QuadNode.hpp"
#include "../../MPI.hpp"
#include "FiniteVolumePatch.hpp"

namespace EllipticForest {

using FiniteVolumeHPS = HPSAlgorithm<FiniteVolumeGrid, FiniteVolumeSolver, FiniteVolumePatch, double>;

class FiniteVolumeNodeFactory : public MPI::MPIObject, public AbstractNodeFactory<FiniteVolumePatch> {

public:

    /**
     * @brief Reference to finite volume solver
     * 
     */
    FiniteVolumeSolver& solver;

    std::vector<FiniteVolumePatch*> siblings{4};

    /**
     * @brief Construct a new FiniteVolumeNodeFactory object (default)
     * 
     */
    FiniteVolumeNodeFactory(FiniteVolumeSolver& solver);

    /**
     * @brief Construct a new FiniteVolumeNodeFactory object on a communicator
     * 
     * @param comm MPI communicator
     */
    FiniteVolumeNodeFactory(MPI::Communicator comm, FiniteVolumeSolver& solver);

    /**
     * @brief Create a node from provided data
     * 
     * @param data Patch to put into node
     * @param path Path of node
     * @param level Level of refinement
     * @param pfirst First rank that owns this node
     * @param plast Last rank that owns this node
     * @return Node<FiniteVolumePatch>* 
     */
    Node<FiniteVolumePatch>* createNode(FiniteVolumePatch data, std::string path, int level, int pfirst, int plast);

    /**
     * @brief Create a child node from a parent node and a sibling index
     * 
     * @param parent_node Parent patch
     * @param sibling_id Sibling index
     * @param pfirst First rank that owns this node
     * @param plast Last rank that owns this node
     * @return Node<FiniteVolumePatch>* 
     */
    Node<FiniteVolumePatch>* createChildNode(Node<FiniteVolumePatch>* parent_node, int sibling_id, int pfirst, int plast);

    /**
     * @brief Create a parent node from children nodes
     * 
     * @param child_nodes Children nodes
     * @param pfirst First rank that owns this node
     * @param plast Last rank that owns this node
     * @return Node<FiniteVolumePatch>* 
     */
    Node<FiniteVolumePatch>* createParentNode(std::vector<Node<FiniteVolumePatch>*> child_nodes, int pfirst, int plast);

};
    
} // NAMESPACE : EllipticForest

#endif // FINITE_VOLUME_NODE_FACTORY_HPP_