#include <gtest/gtest.h>
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "graph/tensor.hpp"

using namespace tc;


// graph: a, b -> Add -> out
static std::shared_ptr<Graph> createSimpleAddGraph()
{
    auto graph = std::make_shared<Graph>("test_add");

    // tensors
    auto t_a = std::make_shared<Tensor>("a", DataType::FLOAT, TensorShape{{1}});
    auto t_b = std::make_shared<Tensor>("b", DataType::FLOAT, TensorShape{{1}});
    auto t_out = std::make_shared<Tensor>("out", DataType::FLOAT, TensorShape{{1}});
    graph->addTensor(t_a);
    graph->addTensor(t_b);
    graph->addTensor(t_out);

    // inputs
    graph->addInput("a");
    graph->addInput("b");
    graph->addOutput("out");

    // Add
    auto node = std::make_shared<Node>("add", OpType::Add, "Add",
                                    std::vector<std::string>{"a", "b"},
                                    std::vector<std::string>{"out"},
                                    Node::AttributeMap{});
    graph->addNode(node);
    return graph;
}

// graph: a -> Mul -> Add -> out
static std::shared_ptr<Graph> createChainGraph()
{
    auto graph = std::make_shared<Graph>("chain");

    
    graph->addTensor(std::make_shared<Tensor>("a", DataType::FLOAT, TensorShape{{1}}));
    graph->addTensor(std::make_shared<Tensor>("t1", DataType::FLOAT, TensorShape{{1}}));
    graph->addTensor(std::make_shared<Tensor>("out", DataType::FLOAT, TensorShape{{1}}));

    graph->addInput("a");
    graph->addOutput("out");


    auto node1 = std::make_shared<Node>("mul", OpType::Mul, "Mul",
                                        std::vector<std::string>{"a"},
                                        std::vector<std::string>{"t1"},
                                        Node::AttributeMap{});
    graph->addNode(node1);


    auto node2 = std::make_shared<Node>("add", OpType::Add, "Add",
                                        std::vector<std::string>{"t1"},
                                        std::vector<std::string>{"out"},
                                        Node::AttributeMap{});
    graph->addNode(node2);
    return graph;
}

TEST(GraphTest, ConstructionAndBasics)
{
    auto graph = std::make_shared<Graph>("my_graph");
    EXPECT_EQ(graph->getName(), "my_graph");
    graph->setName("name123");
    EXPECT_EQ(graph->getName(), "name123");
    EXPECT_TRUE(graph->getNodes().empty());
    EXPECT_TRUE(graph->getTensors().empty());
    EXPECT_TRUE(graph->getInputs().empty());
    EXPECT_TRUE(graph->getOutputs().empty());
}

TEST(GraphTest, AddAndFindNodes)
{
    auto graph = std::make_shared<Graph>();
    auto node1 = std::make_shared<Node>("n1", OpType::Add, "Add", std::vector<std::string>{},
        std::vector<std::string>{});
    auto node2 = std::make_shared<Node>("n2", OpType::Mul, "Mul", std::vector<std::string>{},
        std::vector<std::string>{});

    graph->addNode(node1);
    graph->addNode(node2);

    const auto& nodes = graph->getNodes();
    ASSERT_EQ(nodes.size(), 2);
    EXPECT_EQ(nodes[0], node1);
    EXPECT_EQ(nodes[1], node2);

    auto found = graph->findNode("n1");
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found.value(), node1);

    found = graph->findNode("n3");
    EXPECT_FALSE(found.has_value());
}

TEST(GraphTest, AddAndFindTensors)
{
    auto graph = std::make_shared<Graph>();
    auto t1 = std::make_shared<Tensor>("x", DataType::FLOAT, TensorShape{{2,2}});
    auto t2 = std::make_shared<Tensor>("y", DataType::INT32, TensorShape{{}});
    graph->addTensor(t1);
    graph->addTensor(t2);

    const auto& tensors = graph->getTensors();
    ASSERT_EQ(tensors.size(), 2);

    auto it = tensors.find("x");
    ASSERT_NE(it, tensors.end());
    EXPECT_EQ(it->second, t1);

    it = tensors.find("y");
    ASSERT_NE(it, tensors.end());
    EXPECT_EQ(it->second, t2);

    auto found = graph->findTensor("x");
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found.value(), t1);
    found = graph->findTensor("z");

    EXPECT_FALSE(found.has_value());
}

TEST(GraphTest, InputsOutputs)
{
    auto graph = std::make_shared<Graph>();

    graph->addInput("in1");
    graph->addInput("in2");
    graph->addOutput("out1");
    graph->addOutput("out2");

    EXPECT_EQ(graph->getInputs().size(), 2);
    EXPECT_EQ(graph->getInputs()[0], "in1");
    EXPECT_EQ(graph->getInputs()[1], "in2");
    EXPECT_EQ(graph->getOutputs().size(), 2);
    EXPECT_EQ(graph->getOutputs()[0], "out1");
    EXPECT_EQ(graph->getOutputs()[1], "out2");
}

TEST(GraphTest, TopologicalSortSimple)
{
    auto graph = createSimpleAddGraph();
    auto sorted = graph->topologicalSort();
    ASSERT_EQ(sorted.size(), 1);
    EXPECT_EQ(sorted[0]->getName(), "add");
}

TEST(GraphTest, TopologicalSortChain)
{
    auto graph = createChainGraph();
    auto sorted = graph->topologicalSort();
    ASSERT_EQ(sorted.size(), 2);

    // mul must be before add
    EXPECT_EQ(sorted[0]->getName(), "mul");
    EXPECT_EQ(sorted[1]->getName(), "add");
}

TEST(GraphTest, TopologicalSortWithDataAvailability)
{

    auto graph = std::make_shared<Graph>("test");
    graph->addInput("a");
    graph->addInput("b");

    graph->addTensor(std::make_shared<Tensor>("a", DataType::FLOAT, TensorShape{}));
    graph->addTensor(std::make_shared<Tensor>("b", DataType::FLOAT, TensorShape{}));
    graph->addTensor(std::make_shared<Tensor>("c", DataType::FLOAT, TensorShape{}));
    graph->addTensor(std::make_shared<Tensor>("d", DataType::FLOAT, TensorShape{}));

    auto node1 = std::make_shared<Node>("add1", OpType::Add, "Add",
                                        std::vector<std::string>{"a", "b"},
                                        std::vector<std::string>{"c"},
                                        Node::AttributeMap{});
    auto node2 = std::make_shared<Node>("add2", OpType::Add, "Add",
                                        std::vector<std::string>{"a", "b"},
                                        std::vector<std::string>{"d"},
                                        Node::AttributeMap{});
    graph->addNode(node1);
    graph->addNode(node2);

    auto sorted = graph->topologicalSort();
    ASSERT_EQ(sorted.size(), 2);

    // oreder not defined, but both nodes are present
    EXPECT_TRUE((sorted[0] == node1 && sorted[1] == node2) ||
                (sorted[0] == node2 && sorted[1] == node1));
}

TEST(GraphTest, SummaryDoesNotCrash)
{
    auto graph = createSimpleAddGraph();
    std::string sum = graph->summary();
    
    EXPECT_NE(sum.find("test_add"), std::string::npos);
    EXPECT_NE(sum.find("Add"), std::string::npos);
}


