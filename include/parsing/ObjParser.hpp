/*
** RAYCASTING LIBRARY
** ObjParser.hpp
** Created by marton on 01/07/24.
*/

#ifndef OBJPARSER_HPP
#define OBJPARSER_HPP

#include "shapes/ObjShape.hpp"
#include <string>
#include "math/Vec3.cuh"

template <typename T>
struct vec2 {
    T x;
    T y;
};

namespace rcr {

    class ObjParser {
        std::vector<vec3<float>> v_{};
        std::vector<vec2<float>> vt_{};
        std::vector<vec3<float>> vn_{};
        ObjShape shape_{};

        static bool isVertex(const std::string& line);
        static bool isTextureCoord(const std::string& line);
        static bool isNormalVertex(const std::string& line);
        static bool isFace(const std::string& line);

        void parseVertex(const std::string& line);
        void parseTextureCoord(const std::string& line);
        void parseNormalVertex(const std::string& line);
        void parseFace(const std::string& line);

    public:
        ObjParser() = default;
        ~ObjParser() = default;

        void parseLine(const std::string& line);
        [[nodiscard]] ObjShape getObj() const { return shape_; }
    };

} // rcr

#endif //OBJPARSER_HPP
