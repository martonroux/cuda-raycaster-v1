/*
** RAYCASTING LIBRARY
** ObjParser.cpp
** Created by marton on 01/07/24.
*/

#include "parsing/ObjParser.hpp"
#include <regex>

namespace rcr {

    bool ObjParser::isVertex(const std::string &line) {
        std::regex pattern(R"(^v ([-]?\d+([.]\d+)?) ([-]?\d+([.]\d+)?) ([-]?\d+([.]\d+)?)$)");
        std::smatch match;

        if (std::regex_match(line, match, pattern))
            return true;
        return false;
    }

    bool ObjParser::isTextureCoord(const std::string &line) {
        std::regex pattern(R"(^vt (\d+([.]\d+)?) (\d+([.]\d+)?)$)");
        std::smatch match;

        if (std::regex_match(line, match, pattern))
            return true;
        return false;
    }

    bool ObjParser::isNormalVertex(const std::string &line) {
        std::regex pattern(R"(^vn ([-]?\d+([.]\d+)?) ([-]?\d+([.]\d+)?) ([-]?\d+([.]\d+)?)$)");
        std::smatch match;

        if (std::regex_match(line, match, pattern))
            return true;
        return false;
    }

    bool ObjParser::isFace(const std::string &line) {
        std::regex pattern(R"(^f\s*)");
        std::smatch match;

        if (std::regex_search(line, match, pattern))
            return true;
        return false;
    }

    void ObjParser::parseVertex(const std::string &line) {
        std::regex pattern(R"(^v ([-]?\d+([.]\d+)?) ([-]?\d+([.]\d+)?) ([-]?\d+([.]\d+)?)$)");
        std::smatch match;
        vec3<float> vertex{};

        if (std::regex_match(line, match, pattern)) {
            vertex.x = std::stof(match[1].str());
            vertex.y = std::stof(match[3].str());
            vertex.z = std::stof(match[5].str());
        }
        v_.push_back(vertex);
    }

    void ObjParser::parseTextureCoord(const std::string &line) {
        std::regex pattern(R"(^vt (\d+([.]\d+)?) (\d+([.]\d+)?)$)");
        std::smatch match;
        vec2<float> vertex{};

        if (std::regex_match(line, match, pattern)) {
            vertex.x = std::stof(match[1].str());
            vertex.y = std::stof(match[3].str());
        }
        vt_.push_back(vertex);
    }

    void ObjParser::parseNormalVertex(const std::string &line) {
        std::regex pattern(R"(^vn ([-]?\d+([.]\d+)?) ([-]?\d+([.]\d+)?) ([-]?\d+([.]\d+)?)$)");
        std::smatch match;
        vec3<float> vertex{};

        if (std::regex_match(line, match, pattern)) {
            vertex.x = std::stof(match[1].str());
            vertex.y = std::stof(match[3].str());
            vertex.z = std::stof(match[5].str());
        }
        vn_.push_back(vertex);
    }

    void ObjParser::parseFace(const std::string &line) {
        std::regex face_pattern(R"(^f\s*)");
        std::regex vertex_pattern(R"(\s*((\d+)([\/](\d+)?)([\/](\d+)?)))");
        std::smatch match;
        std::vector<vec3<int>> indexes{};

        if (std::regex_search(line, match, face_pattern)) {
            auto search_start = line.cbegin();

            while (std::regex_search(search_start, line.cend(), match, vertex_pattern)) {
                vec3<int> vec{};

                vec.x = std::stoi(match[2].str());
                vec.y = std::stoi(match[4].str());
                vec.z = std::stoi(match[6].str());

                indexes.push_back(vec);

                search_start = match.suffix().first;
            }

            for (int i = 0; i < static_cast<int>(indexes.size()) - 2; i++) {
                shape_.addTriangle(Triangle{v_[indexes[0].x], v_[indexes[i + 1].x], v_[indexes[i + 2].x]});
            }
        }
    }

    void ObjParser::parseLine(const std::string &line) {
        if (isVertex(line)) return parseVertex(line);
        if (isTextureCoord(line)) return parseTextureCoord(line);
        if (isNormalVertex(line)) return parseNormalVertex(line);
        if (isFace(line)) return parseFace(line);
    }

} // rcr
