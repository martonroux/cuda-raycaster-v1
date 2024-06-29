/*
** RAYCASTING LIBRARY
** Keyboard.hpp
** Created by marton on 29/06/24.
*/

#ifndef KEYBOARD_HPP
#define KEYBOARD_HPP

#include <unordered_map>

namespace rcr {

    // THIS IS CHATGPT CODE. May have problems.
    enum class Keys {
        // Alphabet keys
        KEY_A = 'a',
        KEY_B = 'b',
        KEY_C = 'c',
        KEY_D = 'd',
        KEY_E = 'e',
        KEY_F = 'f',
        KEY_G = 'g',
        KEY_H = 'h',
        KEY_I = 'i',
        KEY_J = 'j',
        KEY_K = 'k',
        KEY_L = 'l',
        KEY_M = 'm',
        KEY_N = 'n',
        KEY_O = 'o',
        KEY_P = 'p',
        KEY_Q = 'q',
        KEY_R = 'r',
        KEY_S = 's',
        KEY_T = 't',
        KEY_U = 'u',
        KEY_V = 'v',
        KEY_W = 'w',
        KEY_X = 'x',
        KEY_Y = 'y',
        KEY_Z = 'z',

        // Number keys
        KEY_0 = '0',
        KEY_1 = '1',
        KEY_2 = '2',
        KEY_3 = '3',
        KEY_4 = '4',
        KEY_5 = '5',
        KEY_6 = '6',
        KEY_7 = '7',
        KEY_8 = '8',
        KEY_9 = '9',

        // Special keys
        KEY_ENTER = 13,
        KEY_ESC = 27,
        KEY_SPACE = 32,
        KEY_BACKSPACE = 8,
        KEY_TAB = 9,

        // Arrow keys (Note: cv::waitKey may not distinguish between left/right and up/down without additional handling)
        KEY_LEFT = 81,
        KEY_UP = 82,
        KEY_RIGHT = 83,
        KEY_DOWN = 84,

        // Function keys (F1-F12)
        KEY_F1 = 0x70,
        KEY_F2 = 0x71,
        KEY_F3 = 0x72,
        KEY_F4 = 0x73,
        KEY_F5 = 0x74,
        KEY_F6 = 0x75,
        KEY_F7 = 0x76,
        KEY_F8 = 0x77,
        KEY_F9 = 0x78,
        KEY_F10 = 0x79,
        KEY_F11 = 0x7A,
        KEY_F12 = 0x7B,

        // Modifier keys
        KEY_SHIFT = 0x10,
        KEY_CTRL = 0x11,
        KEY_ALT = 0x12,

        // Additional common keys
        KEY_HOME = 0x24,
        KEY_END = 0x23,
        KEY_PAGEUP = 0x21,
        KEY_PAGEDOWN = 0x22,
        KEY_INSERT = 0x2D,
        KEY_DELETE = 0x2E
    };

    class Keyboard {
        std::unordered_map<Keys, bool> presses_{};

    public:
        Keyboard() = default;
        ~Keyboard() = default;

        void setKeyPressed(Keys key, bool pressed);
        void resetPresses();

        [[nodiscard]] bool isKeyPressed(Keys key) const;
    };

} // rcr

#endif //KEYBOARD_HPP
