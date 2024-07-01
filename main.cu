/*
** RAYCASTING LIBRARY
** main.cuh
** Created by marton on 20/05/24.
*/

#include "Displayer.hpp"
#include <chrono>

// Device 0: NVIDIA GeForce GTX 1080 Ti
// Max threads per block: 1024
// Max threads dimension (x, y, z): (1024, 1024, 64)
// Max grid size (x, y, z): (2147483647, 65535, 65535)

#include <iostream>
#include <chrono> // For time measurement
#include "parsing/Parser.hpp"
#include "math/Ray.cuh"

int main() {
    std::cout << "Size: " << sizeof(rcr::rgb) << std::endl;
    rcr::ObjShape shape = rcr::Parser::parseOBJFile("./assets/Leaf/PUSHILIN_leaf.obj");

    std::cout << "Number of triangles: " << shape.getTriangles().size() << std::endl;
    rcr::screenData screen = {{-4, -4, 0}, {8, 0, 0}, {0, 8, 0}};
    rcr::rendererData data = {{0, 0, -5}, {255, 0, 0}, screen};
    rcr::Displayer displayer{500, 500, 9999, data};

    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    const int updateInterval = 30; // Update FPS display every 30 frames

    displayer.addShape(shape);

    while (true) {

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
    }

    return 0;
}
