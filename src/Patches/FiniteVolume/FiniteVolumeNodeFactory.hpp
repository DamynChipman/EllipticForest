#ifndef FINITE_VOLUME_NODE_FACTORY_HPP_
#define FINITE_VOLUME_NODE_FACTORY_HPP_

#include "../../QuadNode.hpp"
#include "../../MPI.hpp"
#include "FiniteVolumePatch.hpp"

namespace EllipticForest {

class FiniteVolumeNodeFactory : public MPI::MPIObject, public AbstractNodeFactory<FiniteVolumePatch> {

public:

    FiniteVolumeNodeFactory();
    FiniteVolumeNodeFactory(MPI::Communicator comm);

    Node<FiniteVolumePatch>* createNode(FiniteVolumePatch data, std::string path, int level, int pfirst, int plast);
    Node<FiniteVolumePatch>* createChildNode(Node<FiniteVolumePatch>* parentNode, int siblingID, int pfirst, int plast);
    Node<FiniteVolumePatch>* createParentNode(std::vector<Node<FiniteVolumePatch>*> childNodes, int pfirst, int plast);

};
    
} // NAMESPACE : EllipticForest

#endif // FINITE_VOLUME_NODE_FACTORY_HPP_