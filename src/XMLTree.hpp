#ifndef XML_TREE_HPP_
#define XML_TREE_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace EllipticForest {

/**
 * @brief XML node with storage for name, attributes, and data
 * 
 * An XML node has the following structure:
 * 
 * <Node attribute1="value1" attribute2="value2">
 *   000 111 222
 * </Node>
 * 
 * where:
 * 
 * name = "Node"
 * attributes = { {"attribute1", "value1"}, {"attribute2", "value2"} }
 * data = "000 111 222"
 * 
 * Any children are also nodes and come before the end of the node (right after the data block).
 */
struct XMLNode {

    /**
     * @brief Name of node
     * 
     */
    std::string name;

    /**
     * @brief Vector of pairs of attribute names and info
     * 
     */
    std::vector<std::pair<std::string, std::string>> attributes;

    /**
     * @brief Data block
     * 
     */
    std::string data;

    /**
     * @brief 
     * 
     */
    std::vector<XMLNode*> children;

    /**
     * @brief Construct an empty XMLNode object
     * 
     */
    XMLNode();

    /**
     * @brief Construct a new XMLNode object
     * 
     * @param nodeName Name of node
     */
    XMLNode(std::string node_name);

    /**
     * @brief Add attribute pair to node
     * 
     * @param key Name of attribute
     * @param value Value of attribute
     */
    void addAttribute(std::string key, std::string value);

    /**
     * @brief Add child node
     * 
     * @param child Reference to child node
     */
    void addChild(XMLNode& child);

    /**
     * @brief Add child noe
     * 
     * @param child Pointer to child node
     */
    void addChild(XMLNode* child);

};

/**
 * @brief A tree structure for building an XML file
 * 
 */
class XMLTree {

public:

    /**
     * @brief Construct an empty XMLTree object
     * 
     */
    XMLTree();

    /**
     * @brief Construct a new XMLTree object
     * 
     * @param root Reference to root node
     */
    XMLTree(XMLNode& root);

    /**
     * @brief Write the contents of the tree to file
     * 
     * @param filename Name of file
     */
    void write(std::string filename);

private:

    /**
     * @brief Address of root node
     * 
     */
    XMLNode* root_;

    /**
     * @brief Write the contents of the tree recursively
     * 
     * @param file 
     * @param node 
     * @param prefix 
     */
    void write_(std::ofstream& file, XMLNode& node, std::string prefix);

};


} // NAMESPACE: EllipticForest

#endif // XML_TREE_HPP_