#include <fstream>
#include "gtest/gtest.h"
#include "XMLTree.hpp"

using namespace EllipticForest;

TEST(XMLTree, add_attribute) {
    XMLNode node("TestNode");
    node.addAttribute("attr1", "value1");
    node.addAttribute("attr2", "value2");
    EXPECT_EQ(node.attributes.size(), 2);
    EXPECT_EQ(node.attributes[0].first, "attr1");
    EXPECT_EQ(node.attributes[0].second, "value1");
    EXPECT_EQ(node.attributes[1].first, "attr2");
    EXPECT_EQ(node.attributes[1].second, "value2");
}

TEST(XMLTree, add_child) {
    XMLNode parent("ParentNode");
    XMLNode child1("ChildNode1");
    XMLNode child2("ChildNode2");
    
    parent.addChild(child1);
    parent.addChild(&child2);
    
    EXPECT_EQ(parent.children.size(), 2);
    EXPECT_EQ(parent.children[0]->name, "ChildNode1");
    EXPECT_EQ(parent.children[1]->name, "ChildNode2");
}

TEST(XMLTree, write) {
    XMLNode root("RootNode");
    XMLTree tree(root);

    root.addAttribute("version", "1.0");
    root.addAttribute("encoding", "UTF-8");

    XMLNode child1("ChildNode1");
    child1.addAttribute("attr1", "value1");
    child1.addAttribute("attr2", "value2");
    child1.data = "Child data for node 1";
    root.addChild(child1);

    XMLNode child2("ChildNode2");
    child2.data = "Child data for node 2";
    root.addChild(child2);

    tree.write("test.xml");

    // Read and validate the written file
    std::ifstream file("test.xml");
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    file.close();

    // Validate that the content matches the expected XML structure
    std::string expectedContent = "<?xml version=\"1.0\"?>\n"
                                  "<RootNode version=\"1.0\" encoding=\"UTF-8\">\n"
                                  "  <ChildNode1 attr1=\"value1\" attr2=\"value2\">\n"
                                  "Child data for node 1\n"
                                  "  </ChildNode1>\n"
                                  "  <ChildNode2>\n"
                                  "Child data for node 2\n"
                                  "  </ChildNode2>\n"
                                  "</RootNode>\n";

    EXPECT_EQ(content, expectedContent);
}