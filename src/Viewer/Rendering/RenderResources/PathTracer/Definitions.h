//
// Created by magnus-desktop on 12/8/24.
//

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <glm/glm.hpp>

#include "Viewer/Rendering/Components/MaterialComponent.h"
#include "Viewer/Rendering/Components/Components.h"

#ifdef DIFF_RENDERER_ENABLED
#include "torch/torch.h"
#endif

namespace VkRender::PathTracer {
#ifdef DIFF_RENDERER_ENABLED

    struct GPUDataTensors {
        torch::Tensor positions;
        torch::Tensor scales;
        torch::Tensor normals;

        // properties
        torch::Tensor emissions;
        torch::Tensor colors;
        torch::Tensor specular;
        torch::Tensor diffuse;
    };
#else
    struct GPUDataTensors;
#endif


    // Enum for kernel types
    typedef enum KernelType {
        KERNEL_PATH_TRACER_MESH,
        KERNEL_PATH_TRACER_2DGS,
        KERNEL_TYPE_COUNT // To count the number of kernels
    } KernelType;

    // Function to map KernelType to a string
    static const char* KernelTypeToString(KernelType kernel) {
        switch (kernel) {
        case KERNEL_PATH_TRACER_MESH: return "Path Tracer: Mesh";
        case KERNEL_PATH_TRACER_2DGS: return "Path Tracer: 2DGS";
        default: return "Unknown";
        }
    }

    // Function to map string to KernelType
    static KernelType StringToKernelType(const char* str) {
        if (strcmp(str, "Path Tracer: Mesh") == 0) return KERNEL_PATH_TRACER_MESH;
        if (strcmp(str, "Path Tracer: 2DGS") == 0) return KERNEL_PATH_TRACER_2DGS;
        return KERNEL_TYPE_COUNT; // Invalid
    }


    struct InputAssembly {
        glm::vec3 position;
        glm::vec3 color;
        glm::vec3 normal;
    };

    struct GaussianInputAssembly {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 scale;

        float emission; // Emissive power
        float color; // Albedo
        float diffuse; // Diffuse coefficient
        float specular; // Specular coefficient
        float phongExponent; // Shininess exponent
    };

    struct RenderInformation {
        uint64_t photonsAccumulated = 0;
        uint64_t totalPhotons = 0;

        float gamma = 2.2f;

        uint32_t numBounces = 32;
        uint32_t frameID = 0;

    };

    struct GPUData {
        InputAssembly* vertices = nullptr;
        uint32_t* indices = nullptr; // e.g., {0, 1, 2, 2, 3, 0, ...}
        uint32_t* vertexOffsets = nullptr;
        uint32_t* indexOffsets = nullptr;
        TransformComponent* transforms = nullptr;
        MaterialComponent* materials = nullptr;
        TagComponent* tagComponents = nullptr;
        uint32_t numEntities = 0;

        uint32_t totalVertices = 0;
        uint32_t totalIndices = 0;

        // GS
        GaussianInputAssembly* gaussianInputAssembly = nullptr;

        size_t numGaussians = 0;

        float* imageMemory = nullptr;
        float* contribution = nullptr;

        RenderInformation* renderInformation = nullptr;
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
