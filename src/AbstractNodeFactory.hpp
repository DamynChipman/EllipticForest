#ifndef ABSTRACT_NODE_FACTORY_HPP_
#define ABSTRACT_NODE_FACTORY_HPP_

#include "QuadNode.hpp"

namespace EllipticForest {

template<typename T>
class AbstractNodeFactory {
public:

    /**
     * @brief Create and allocate storage for a new Node instance from the data, path, and metadata
     * 
     * @param data The data to store at the created node
     * @param path The path of the node
     * @param level The level of the node
     * @param pfirst The first rank this node lives on
     * @param plast The last rank this node lives on
     * @return Node<T>* 
     */
	virtual Node<T>* createNode(T data, std::string path, int level, int pfirst, int plast) = 0;

    /**
     * @brief Create and allocate storage for a child node based on the parent node and sibling ID
     * 
     * @param parent_node Pointer to the parent node
     * @param sibling_id Sibling index for node to create
     * @param pfirst The first rank this node lives on
     * @param plast The last rank this node lives on
     * @return Node<T>* 
     */
	virtual Node<T>* createChildNode(Node<T>* parent_node, int sibling_id, int pfirst, int plast) = 0;

    /**
     * @brief Create and allocate storage for a parent node based on the children nodes
     * 
     * @param children_nodes Vector of pointers to children nodes
     * @param pfirst The first rank this node lives on
     * @param plast The last rank this node lives on
     * @return Node<T>* 
     */
	virtual Node<T>* createParentNode(std::vector<Node<T>*> children_nodes, int pfirst, int plast) = 0;

};

} // NAMESPACE : EllipticForest

#endif // ABSTRACT_NODE_FACTORY_HPP_