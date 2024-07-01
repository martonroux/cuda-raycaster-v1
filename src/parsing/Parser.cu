/*
** RAYCASTING LIBRARY
** Parser.cpp
** Created by marton on 01/07/24.
*/

#include "parsing/Parser.hpp"
#include "Error.hpp"

namespace rcr {

    ObjShape Parser::parseOBJFile(const std::string& filePath) {
        ObjParser parser{};
        std::ifstream file(filePath);

        if (!file.is_open())
            throw ParserError("Tried to open OBJ file that doesn't exist.", "Parser.cpp | ::parseOBJFile()");

        std::string line;
        while (std::getline(file, line)) {
            parser.parseLine(line);
        }

        file.close();
        return parser.getObj();
    }

} // rcr
