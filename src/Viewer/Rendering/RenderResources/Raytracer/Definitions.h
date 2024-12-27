//
// Created by magnus-desktop on 12/8/24.
//

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <glm/glm.hpp>

namespace VkRender::RT{
    struct InputAssembly {
        glm::vec3 position;
        glm::vec3 color;
        glm::vec3 normal;
    };

    struct GaussianInputAssembly {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 scale;

        float intensity;         // Emissive power
        float diffuse;           // Diffuse coefficient
        float specular;          // Specular coefficient
        float phongExponent;     // Shininess exponent
    };

    struct GaussianScene {
        GaussianInputAssembly* inputAssembly;
    };

    struct GPUData {
        InputAssembly* vertices = nullptr;
        uint32_t*  indices = nullptr;  // e.g., {0, 1, 2, 2, 3, 0, ...}
        size_t numVertices;
        size_t numIndices;
        // GS
        GaussianInputAssembly* gaussianInputAssembly = nullptr;
        size_t numGaussians;

        // uint8_t
        uint8_t* imageMemory;
    };


}
#endif //DEFINITIONS_H
