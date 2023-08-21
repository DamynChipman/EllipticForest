#include "XMLTree.hpp"

namespace EllipticForest {

XMLNode::XMLNode() :
    name("")
        {}

XMLNode::XMLNode(std::string node_name) :
    name(node_name)
        {}

void XMLNode::addAttribute(std::string key, std::string value) {
    attributes.push_back({key, value});
}

void XMLNode::addChild(XMLNode& child) {
    children.push_back(&child);
}

void XMLNode::addChild(XMLNode* child) {
    children.push_back(child);
}

XMLTree::XMLTree() :
    root_(nullptr)
        {}

XMLTree::XMLTree(XMLNode& root) :
    root_(&root)
        {}

void XMLTree::write(std::string filename) {
    std::ofstream file;
    file.open(filename);
    file << "<?xml version=\"1.0\"?>" << std::endl;
    write_(file, *root_, "");
    file.close();
}

void XMLTree::write_(std::ofstream& file, XMLNode& node, std::string prefix) {

    // Write header
    std::string header = prefix;
    header += "<" + node.name + " ";
    for (auto& attributePair : node.attributes) {
        header += attributePair.first + "=\"" + attributePair.second + "\" ";
    }
    header.pop_back(); // Remove last space
    header += ">";
    file << header << std::endl;

    // Write data
    if (node.data != "") {
        file << node.data << std::endl;
    }

    // Write children
    for (auto& child : node.children) {
        write_(file, *child, prefix + "  ");
    }

    // Write footer
    file << prefix << "</" << node.name << ">" << std::endl;

    return;
}

} // NAMESPACE : EllipticForest