//
// Created by magnus-desktop on 12/8/24.
//

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <glm/glm.hpp>

#include "Viewer/Rendering/Components/MaterialComponent.h"
#include "Viewer/Rendering/Components/Components.h"

namespace VkRender::RT {
    struct InputAssembly {
        glm::vec3 position;
        glm::vec3 color;
        glm::vec3 normal;
    };

    struct GaussianInputAssembly {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 scale;

        float emission;         // Emissive power
        float color;             // Albedo
        float diffuse;           // Diffuse coefficient
        float specular;          // Specular coefficient
        float phongExponent;     // Shininess exponent
    };


    struct GPUData {
        InputAssembly *vertices = nullptr;
        uint32_t *indices = nullptr;  // e.g., {0, 1, 2, 2, 3, 0, ...}
        uint32_t *vertexOffsets = nullptr;
        uint32_t *indexOffsets = nullptr;
        TransformComponent *transforms = nullptr;
        MaterialComponent* materials = nullptr;
        TagComponent* tagComponents = nullptr;
        uint32_t numEntities = 0;

        uint32_t totalVertices;
        uint32_t totalIndices;

        // GS
        GaussianInputAssembly *gaussianInputAssembly = nullptr;
        size_t numGaussians;

        // uint8_t
        uint8_t *imageMemory;
    };


}
#endif //DEFINITIONS_H
