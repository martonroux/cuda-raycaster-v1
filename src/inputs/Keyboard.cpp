/*
** RAYCASTING LIBRARY
** Keyboard.cpp
** Created by marton on 29/06/24.
*/

#include "inputs/Keyboard.hpp"

namespace rcr {

    void Keyboard::setKeyPressed(Keys key, bool pressed) {
        presses_[key] = pressed;
    }

    void Keyboard::resetPresses() {
        presses_.clear();
    }

    bool Keyboard::isKeyPressed(Keys key) const {
        if (presses_.find(key) == presses_.end())
            return false;
        return presses_.at(key);
    }

} // rcr
