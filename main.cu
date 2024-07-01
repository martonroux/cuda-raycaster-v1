#include "Displayer.hpp"
#include <chrono>
#include "math/Matrix2.cuh"

// Device 0: NVIDIA GeForce GTX 1080 Ti
// Max threads per block: 1024
// Max threads dimension (x, y, z): (1024, 1024, 64)
// Max grid size (x, y, z): (2147483647, 65535, 65535)

#include <iostream>
#include <chrono> // For time measurement

int main() {
    rcr::screenData screen = {{-4, -4, 0}, {8, 0, 0}, {0, 8, 0}};
    rcr::rendererData data = {{0, 0, -5}, screen};
    rcr::Displayer displayer{1920, 1080, 9999, data};

    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    const int updateInterval = 30; // Update FPS display every 30 frames

    while (true) {
        displayer.addShape(rcr::Triangle{{0, 2, 5}, {-2, -4, 5}, {2, -4, 5}, {255, 255, 255}});
        displayer.addShape(rcr::Triangle{{0, 6, 3}, {-2, -4, 7}, {2, -4, 7}, {255, 0, 0}});
        // displayer.addShape(rcr::Triangle{{2, -4, 5}, {0, -5.5, 5}, {0, -4, 5}});

        displayer.render();

        rcr::Keyboard keyboard = displayer.getKeyboardFrame();
        rcr::Mouse mouse = displayer.getMouseFrame();

        if (keyboard.isKeyPressed(rcr::Keys::KEY_Q))
            break;

        frameCount++;
        if (frameCount == updateInterval) {
            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsedTime = endTime - lastTime;
            double fps = frameCount / elapsedTime.count();
            std::cout << "FPS: " << fps << std::endl;
            lastTime = endTime;
            frameCount = 0;
        }

        displayer.clear();
    }

    return 0;
}
