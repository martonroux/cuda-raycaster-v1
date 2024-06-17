/*
** RAYCASTING LIBRARY
** Error.hpp
** Created by marton on 17/06/24.
*/

#ifndef ERROR_HPP
#define ERROR_HPP

#include <exception>
#include <string>

namespace rcr {

    class HostError : public std::exception {
        std::string what_;
        std::string where_;

    public:
        HostError(std::string what, std::string where) : what_(what), where_(where) {}

        const char * what() const noexcept { return what_.c_str(); }
        const char * where() const noexcept { return where_.c_str(); }
    };

    class MatrixError : public HostError {
    public:
        MatrixError(std::string what, std::string where) : HostError(what, where) {}
    };

}


#endif //ERROR_HPP
