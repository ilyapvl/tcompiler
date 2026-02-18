#include <iostream>
#include <graph/graph.h>
#include <graph/exceptions.h>
#include <fstream>
#include <reader/reader.h>
#include <graph/types.h>
#include <vector>



int main()
{
    
    try
    {
        std::string filename = {};
        std::cout << "Enter file name (must be located in models folder): ";
        std::getline(std::cin, filename);
        
        graph::Graph g = {};
        g.load_from_onnx("../models/" + filename);

        std::ofstream file("../output/graph.dot");
        g.dump_dot(file);
        file.close();
    }
    catch (const graph::graph_exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    catch (std::exception& e)
    {
        std::cerr << "Unexpected exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Граф модели построен" << std::endl;



    return 0;
}
