/**
 * @file: MultiSense-Viewer/include/Viewer/Rendering/Core/Definitions.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2021-05-15, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_DEFINITIONS_H
#define MULTISENSE_DEFINITIONS_H

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "Viewer/Application/pch.h"


#ifdef APIENTRY
#undef APIENTRY
#endif

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "VulkanGraphicsPipeline.h"


namespace VkRender {

    /**
 * @brief GLFW and Vulkan combination to create a SwapChain
 */
    typedef struct SwapChainCreateInfo {
        GLFWwindow *pWindow{};
        bool vsync = true;
        VkInstance instance{};
        VkPhysicalDevice physicalDevice{};
        VkDevice device{};
    } SwapChainCreateInfo;

    /**
     * @brief Default Vertex information
     */
    struct alignas(16) Vertex {
        glm::vec3 pos;      // 12 bytes + 4 bytes padding
        glm::vec3 normal;   // 12 bytes + 4 bytes padding
        glm::vec2 uv0;      // 8 bytes + 8 bytes padding
        glm::vec2 uv1;      // 8 bytes + 8 bytes padding
        glm::vec4 color;    // 16 bytes
        bool operator==(const Vertex &other) const {
            return pos == other.pos && color == other.color && uv0 == other.uv0;
        }
    };

    struct ImageVertex {
        glm::vec2 pos{};
        glm::vec2 uv0{};
    };

    /**
     * @brief MouseButtons user input
     */
    struct MouseButtons {
        bool left = false;
        bool right = false;
        bool middle = false;
        int action = 0;
        float wheel = 0.0f; // to initialize arcball zoom
        bool insideApp = false;

        union {
            struct {
                float dx;
                float dy;
            };

            glm::vec2 d;
        };

        union {
            struct {
                float x;
                float y;
            };

            glm::vec2 pos;
        };

        MouseButtons() : dx(0), dy(0), x(0), y(0) {
        }
    };

    /**
     * @brief Default MVP matrices
     */
    struct UBOMatrix {
        glm::mat4 projection{};
        glm::mat4 view{};
        glm::mat4 model{};
        glm::vec3 camPos{};
    };
    /**
     * @brief Basic lighting params for simple light calculation
     */
    struct FragShaderParams {
        glm::vec4 lightDir{};
        glm::vec4 zoomCenter{};
        glm::vec4 zoomTranslate{};
        float exposure = 4.5f;
        float gamma = 2.2f;
        float prefilteredCubeMipLevels = 0.0f;
        float scaleIBLAmbient = 1.0f;
        float debugViewInputs = 0.0f;
        float lod = 0.0f;
        glm::vec2 pad{};
        glm::vec4 disparityNormalizer; // (0: should normalize?, 1: min value, 2: max value, 3 pad)
        glm::vec4 kernelFilters; // 0 Sobel/Edge kernel, Blur kernel,
        float dt = 0.0f;
    };

    struct GlobalUniformBufferObject {
        glm::mat4 projection;
        glm::mat4 view;
        glm::vec3 cameraPosition;
    };

    struct PointCloudUBO {
        glm::mat4 Q;
        glm::mat4 intrinsics;
        glm::mat4 extrinsics;
        float width;
        float height;
        float disparity;
        float focalLength;
        float scale;
        float pointSize;
        float useColor;
        float hasSampler;
    };


    struct MaterialBufferObject {
        glm::vec4 baseColor;       // 16 bytes (aligned to 16)
        float specular;            // 4 bytes
        float diffuse;             // 4 bytes
        glm::vec2 _pad0;           // 8 bytes of padding to align to 16 bytes
        glm::vec4 emissiveFactor;  // 16 bytes (aligned to 16)
        float numLightSources;       // 4 bytes
        glm::vec3 _pad1;           // 12 bytes of padding for alignment
        glm::vec4 lightPosition[32]; // 32 * 16 bytes (vec3 expanded to vec4 for alignment)
        glm::vec4 lightNormal[32];   // 32 * 16 bytes (vec3 expanded to vec4 for alignment)
    };

    struct RenderPassInfo { // TODO move somewhere else
        VkSampleCountFlagBits sampleCount;
        VkRenderPass renderPass;
        uint32_t swapchainImageCount = 1;
        std::string debugName;
    };
}

namespace std {
    template <>
    struct hash<glm::vec3> {
        size_t operator()(const glm::vec3& v) const noexcept {
            size_t h1 = std::hash<float>{}(v.x);
            size_t h2 = std::hash<float>{}(v.y);
            size_t h3 = std::hash<float>{}(v.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2); // Combine the hashes
        }
    };

    template <>
    struct hash<glm::vec2> {
        size_t operator()(const glm::vec2& v) const noexcept {
            size_t h1 = std::hash<float>{}(v.x);
            size_t h2 = std::hash<float>{}(v.y);
            return h1 ^ (h2 << 1); // Combine the hashes
        }
    };

    template<>
    struct hash<VkRender::Vertex> {
        size_t operator()(VkRender::Vertex const &vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                     (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.uv0) << 1);
        }
    };
};

#endif //MULTISENSE_DEFINITIONS_H
