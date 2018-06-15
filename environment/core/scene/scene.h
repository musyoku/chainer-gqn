#include "object.h"
#include <memory>
#include <vector>

namespace environment {
namespace scene {
    class Scene {
    private:
        std::vector<std::unique_ptr<Object>> _objects;
    };
}
}