#ifndef XML_TREE_HPP_
#define XML_TREE_HPP_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace EllipticForest {

struct XMLNode {

    std::string name;
    std::vector<std::pair<std::string, std::string>> attributes;
    std::string data;
    std::vector<XMLNode*> children;

    XMLNode(std::string nodeName) : name(nodeName) {}

    void addAttribute(std::string key, std::string value) {
        attributes.push_back({key, value});
    }

    void addChild(XMLNode& child) {
        children.push_back(&child);
    }

    void addChild(XMLNode* child) {
        children.push_back(child);
    }

};

class XMLTree {

public:

    XMLTree() : root_(nullptr) {}
    XMLTree(XMLNode& root) : root_(&root) {}

    void write(std::string filename) {
        std::ofstream file;
        file.open(filename);
        file << "<?xml version=\"1.0\"?>" << std::endl;
        write_(file, *root_, "");
        file.close();
    }

private:

    XMLNode* root_;

    void write_(std::ofstream& file, XMLNode& node, std::string prefix) {

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

};


} // NAMESPACE: EllipticForest

#endif // XML_TREE_HPP_