/*
** RAYCASTING LIBRARY
** Mouse.cpp
** Created by marton on 29/06/24.
*/

#include "inputs/Mouse.hpp"

namespace rcr {

    void Mouse::resetPresses() {
        presses_.clear();
    }

    void Mouse::onMouse(int event, int x, int y) {
        if (event == cv::EVENT_MOUSEMOVE) {
            pos_.first = x;
            pos_.second = y;
        } else if (event == cv::EVENT_LBUTTONDOWN) {
            presses_[MouseKeys::MOUSE_LEFT] = true;
        } else if (event == cv::EVENT_RBUTTONDOWN) {
            presses_[MouseKeys::MOUSE_RIGHT] = true;
        } else if (event == cv::EVENT_MBUTTONDOWN) {
            presses_[MouseKeys::MOUSE_MIDDLE] = true;
        }
    }

    std::pair<int, int> Mouse::getMousePos() const {
        return pos_;
    }

    bool Mouse::isKeyPressed(MouseKeys key) const {
        if (presses_.find(key) == presses_.end())
            return false;
        return presses_.at(key);
    }

} // rcr
