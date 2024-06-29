#include "Displayer.hpp"
#include <chrono>
#include <thread>

// Device 0: NVIDIA GeForce GTX 1080 Ti
// Max threads per block: 1024
// Max threads dimension (x, y, z): (1024, 1024, 64)
// Max grid size (x, y, z): (2147483647, 65535, 65535)

#include <iostream>
#include <chrono> // For time measurement

int main() {
    rcr::screenData screen = {{-4, -4, 0}, {8, 0, 0}, {0, 8, 0}};
    rcr::rendererData data = {{0, 0, -5}, screen};
    rcr::Displayer displayer{500, 500, 9999, data};

    displayer.addShape(rcr::Triangle{{0, 2, 5}, {-2, -4, 5}, {2, -4, 5}});
    displayer.addShape(rcr::Triangle{{-2, -4, 5}, {0, -5.5, 5}, {0, -4, 5}});
    displayer.addShape(rcr::Triangle{{2, -4, 5}, {0, -5.5, 5}, {0, -4, 5}});

    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    const int updateInterval = 30; // Update FPS display every 30 frames

    while (true) {
        auto startTime = std::chrono::high_resolution_clock::now();

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

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> frameTime = endTime - startTime;
    }

    return 0;
}
