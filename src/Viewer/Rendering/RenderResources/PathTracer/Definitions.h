//
// Created by magnus-desktop on 12/8/24.
//

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <glm/glm.hpp>

#include "Viewer/Rendering/Components/MaterialComponent.h"
#include "Viewer/Rendering/Components/Components.h"

namespace VkRender::PathTracer {
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

    struct RenderInformation {
        unsigned long int photonsAccumulated = 0;
        unsigned long int photonsAccumulatedDirect = 0;
        uint64_t frameID = 0;
        uint64_t totalPhotons = 0;

        float gamma = 2.2f;
    };
    struct GPUData {
        InputAssembly *vertices = nullptr;
        uint32_t *indices = nullptr;  // e.g., {0, 1, 2, 2, 3, 0, ...}
        uint32_t *vertexOffsets = nullptr;
        uint32_t *indexOffsets = nullptr;
        TransformComponent *transforms = nullptr;
        MaterialComponent *materials = nullptr;
        TagComponent *tagComponents = nullptr;
        uint32_t numEntities = 0;

        uint32_t totalVertices = 0;
        uint32_t totalIndices = 0;

        // GS
        GaussianInputAssembly *gaussianInputAssembly = nullptr;

        size_t numGaussians = 0;

        float *imageMemory = nullptr;
        float *contribution = nullptr;

        RenderInformation *renderInformation = nullptr;
    };

    struct PCG32 {
        uint64_t state{};
        uint64_t inc{};

        // Initialize the RNG with a seed and sequence
        void init(uint64_t seed, uint64_t sequence = 1) {
            state = 0;
            inc = (sequence << 1u) | 1u; // Increment must be odd
            nextUInt(); // Advance state
            state += seed;
            nextUInt(); // Advance state again
        }

        // Generate the next uint32_t random number
        uint32_t nextUInt() {
            uint64_t old_state = state;
            state = old_state * 6364136223846793005ULL + inc;
            uint32_t xorshifted = static_cast<uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
            uint32_t rot = static_cast<uint32_t>(old_state >> 59u);
            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }

        // Generate a random float in [0, 1)
        float nextFloat() {
            return nextUInt() / static_cast<float>(UINT32_MAX);
        }
    };

}
#endif //DEFINITIONS_H
