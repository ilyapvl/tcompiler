#ifndef TEST_DIMENSIONS_HPP
#define TEST_DIMENSIONS_HPP

#include <vector>
#include <cstdint>
#include "mlir/IR/BuiltinTypes.h"

constexpr int64_t N = mlir::ShapedType::kDynamic;

namespace tc::test
{


    static const std::vector<std::vector<int64_t>> unaryShapes = {
        {2, 3},
        {3},
        {},
        {N, 3},
        {2, N},
        {N},
        {N, N},
        {N, 7, N}
    };







    struct CompatiblePair
    {
        std::vector<int64_t> lhs;
        std::vector<int64_t> rhs;
        std::vector<int64_t> expected;
    };


    inline const std::vector<CompatiblePair>& compatiblePairs()
    {
        static const std::vector<CompatiblePair> pairs = {
            // { shape1 }, { shape2 }, { result }
            {{2, 3}, {2, 3}, {2, 3}},
            {{},     {},     {}},

            {{1, 3}, {2, 3}, {2, 3}},
            {{2, 1}, {2, 3}, {2, 3}},
            {{1, 3}, {2, 1}, {2, 3}},
            {{3},    {2, 3}, {2, 3}},
            {{2, 3}, {},     {2, 3}},
            {{1, 1, 4}, {2, 3, 1}, {2, 3, 4}},
            {{2, 1, 4}, {1, 3, 1}, {2, 3, 4}},


            {{N, 3}, {N, 3}, {N, 3}},
            {{2, N}, {2, N}, {2, N}},
            {{N},    {N},    {N}},

            {{N, 3}, {2, 3}, {N, 3}},
            {{2, N}, {2, 3}, {2, N}},
            {{N, 3}, {1, 3}, {N, 3}},


            {{}, {N, 3}, {N, 3}},
            {{N, 3}, {}, {N, 3}},
            
            
            {
                {N, 1},
                {2, N},
                {N, N}
            },



            {
                {N, 2, N, 1},
                {1, 2, 3},
                {N, 2, N, 3}
            },
        };


        return pairs;
    }







    struct IncompatiblePair
    {
        std::vector<int64_t> lhs;
        std::vector<int64_t> rhs;
    };

    inline const std::vector<IncompatiblePair>& incompatiblePairs()
    {
        static const std::vector<IncompatiblePair> pairs = {

            {{2, 3}, {4, 5}},
            {{3},    {2, 4}},
            {{2, 3}, {1, 4}}

        };

        return pairs;
    }













    struct MatMulCase
    {
        std::vector<int64_t> a_shape;
        std::vector<int64_t> b_shape;
        std::vector<int64_t> expected_shape;
        bool transA;
        bool transB;
    };

    static const std::vector<MatMulCase> MatMulTests =
    {
        {{2, 3}, {3, 4}, {2, 4}, false, false},
        {{5, 2, 3}, {5, 3, 4}, {5, 2, 4}, false, false},
        {{3, 4, 5, 6}, {1, 6, 7}, {3, 4, 5, 7}, false, false},
        {{2, 1, 3}, {2, 3, 4}, {2, 1, 4}, false, false},


        {{3, 2}, {3, 4}, {2, 4}, true, false},
        {{2, 3}, {4, 3}, {2, 4}, false, true},
        {{3, 2}, {4, 3}, {2, 4}, true, true}, 


        {{N, 3}, {3, 4}, {N, 4}, false, false},
        {{2, N}, {N, 4}, {2, 4}, false, false},
        {{N, N}, {N, N}, {N, N}, false, false},
        {{N, 2, 3}, {N, 3, 4}, {N, 2, 4}, false, false},
        {{2, N, 3}, {2, 3, N}, {2, N, N}, false, false},
        {{1, N, 3, 1}, {5, N, 3, 4}, {5, N, 1, 4}, true, false},
    };









    

    struct ShapeCase
    {
        std::vector<int64_t> input_shape;
        int64_t start;
        int64_t end;
        std::vector<int64_t> expected_output_shape;
    };


    static const std::vector<ShapeCase> shapeTests = {

        {{2, 3, 4}, 0, std::numeric_limits<int64_t>::max(), {3}},
        {{2, N, 4}, 0, std::numeric_limits<int64_t>::max(), {3}},
        {{N, N, N}, 0, std::numeric_limits<int64_t>::max(), {3}},

        {{2, 3, 4}, 1, 3, {2}},
        {{2, 3, 4}, 0, 2, {2}},

        {{2, 3, 4}, -2, std::numeric_limits<int64_t>::max(), {2}},
        {{2, 3, 4}, 0, -1, {2}},

        {{2, 3, 4}, 0, 0, {0}},
        {{2, 3, 4}, 2, 2, {0}},

        {{N, 3, N}, -2, -1, {1}}, 
    };






    struct ReshapeCase
    {
        std::vector<int64_t> input_shape;
        std::vector<int64_t> shape_tensor;
        bool allowzero;
        std::vector<int64_t> expected_output_shape;
    };

    static const std::vector<ReshapeCase> reshapeTests = {

        {{2, 3}, {3, 2}, false, {3, 2}},
        {{2, 3}, {6}, false, {6}},
        {{2, 3}, {1, 6}, false, {1, 6}},

        {{2, 3}, {-1}, false, {6}},
        {{2, 3}, {2, -1}, false, {2, 3}},
        {{2, 3}, {-1, 2}, false, {3, 2}},
        {{2, 3, 4}, {2, -1, 2}, false, {2, 6, 2}},

        {{2, 3}, {0, 3}, false, {2, 3}},
        {{2, 3}, {2, 0}, false, {2, 3}},
        {{2, 3, 4}, {0, -1, 4}, false, {2, 3, 4}},







        {{0}, {0, 0}, true, {0, 0}},
       

        {{N, 3}, {2, -1}, false, {2, N}},
        {{2, N}, {-1, 3}, false, {N, 3}},
        {{N, N}, {4, -1}, false, {4, N}},
        {{N, 3, 32, 32}, {0, -1}, false, {N, 3072}},


    };



    static const std::vector<ReshapeCase> reshapeErrorTests = {
        {{2, 3}, {-1, -1}, false, {}},
        {{2, 3, 4}, {0, -1, 4}, true, {}},
    };













    struct ConcatCase
    {
        std::vector<std::vector<int64_t>> input_shapes;
        int64_t axis;
        std::vector<int64_t> expected_output_shape;
    };

    static const std::vector<ConcatCase> concatTests = {

        { {{2, 3}, {3, 3}}, 0, {5, 3} }, 
        { {{1, 3}, {4, 3}}, 0, {5, 3} }, 

        { {{2, 2}, {2, 3}}, 1, {2, 5} }, 
        { {{2, 1}, {2, 4}}, 1, {2, 5} }, 

        { {{2, 3, 4}, {2, 3, 5}}, 2, {2, 3, 9} }, 
        { {{2, 3, 4}, {2, 3, 4}}, 1, {2, 6, 4} }, 

        { {{N, 3}, {N, 3}}, 0, {N, 3} }, 
        { {{2, N}, {2, N}}, 1, {2, N} }, 
        { {{N, 3, 4}, {N, 3, 5}}, 2, {N, 3, 9} }, 

        { {{2, 3}, {N, 3}}, 0, {N, 3} }, 
        { {{2, N}, {2, 3}}, 1, {2, N} }, 

        { {{2, 3}, {3, 3}}, -2, {5, 3} },
        { {{2, 3}, {2, 4}}, -1, {2, 7} },

        { {{2, 3}, {1, 3}, {2, 3}}, 0, {5, 3} }, 
        { {{2, 2}, {2, 1}, {2, 2}}, 1, {2, 5} }, 
    };





} // namespace tc::test





#endif // TEST_DIMENSIONS_HPP
