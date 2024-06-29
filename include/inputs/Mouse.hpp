/*
** RAYCASTING LIBRARY
** Mouse.hpp
** Created by marton on 29/06/24.
*/

#ifndef MOUSE_HPP
#define MOUSE_HPP

#include <utility>
#include <unordered_map>
#include <opencv2/opencv.hpp>

namespace rcr {

    enum class MouseKeys {
        MOUSE_LEFT,
        MOUSE_RIGHT,
        MOUSE_MIDDLE
    };

    class Mouse {
        std::pair<int, int> pos_{};
        std::unordered_map<MouseKeys, bool> presses_{};

    public:
        Mouse() = default;
        ~Mouse() = default;

        void resetPresses();
        void onMouse(int event, int x, int y);

        [[nodiscard]] std::pair<int, int> getMousePos() const;
        [[nodiscard]] bool isKeyPressed(MouseKeys key) const;
    };

    inline void onMouseCallback(int event, int x, int y, int flags, void* userdata) {
        auto* handler = static_cast<Mouse*>(userdata);

        handler->onMouse(event, x, y);
    }

} // rcr

#endif //MOUSE_HPP
