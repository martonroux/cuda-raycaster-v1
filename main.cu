#include "Displayer.hpp"

// Device 0: NVIDIA GeForce GTX 1080 Ti
// Max threads per block: 1024
// Max threads dimension (x, y, z): (1024, 1024, 64)
// Max grid size (x, y, z): (2147483647, 65535, 65535)

int main()
{
    rcr::screenData screen = {{-4, -4, 0}, {8, 0, 0}, {0, 8, 0}};
    rcr::rendererData data = {{0, 0, -5}, screen};
    rcr::Displayer displayer{1920, 1080, 60, data};

    displayer.addShape(rcr::Triangle{{0, 2, 5}, {-2, -4, 5}, {2, -4, 5}});
    displayer.addShape(rcr::Triangle{{-2, -4, 5}, {0, -5.5, 5}, {0, -4, 5}});
    displayer.addShape(rcr::Triangle{{2, -4, 5}, {0, -5.5, 5}, {0, -4, 5}});

    displayer.render();
}
