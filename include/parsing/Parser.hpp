/*
** RAYCASTING LIBRARY
** Parser.hpp
** Created by marton on 01/07/24.
*/

#ifndef PARSER_HPP
#define PARSER_HPP

#include "shapes/ObjShape.hpp"
#include "parsing/ObjParser.hpp"
#include <iostream>
#include <fstream>
#include <string>

namespace rcr {

    class Parser {
    public:
        Parser() = default;
        ~Parser() = default;

        [[nodiscard]] static ObjShape parseOBJFile(const std::string& filePath);
    };

} // rcr

#endif //PARSER_HPP
