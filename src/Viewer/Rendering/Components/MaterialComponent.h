//
// Created by mgjer on 04/10/2024.
//

#ifndef MATERIALCOMPONENT_H
#define MATERIALCOMPONENT_H

#include <glm/glm.hpp>
#include <filesystem>

#include "Viewer/Rendering/Core/PipelineKey.h"
#include "Viewer/Rendering/Core/VulkanTexture.h"

namespace VkRender {
    enum class AlphaMode {
        Opaque,
        Mask,
        Blend
    };
    struct MaterialComponent {
        glm::vec4 albedo = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);    // Base color (could be an albedo color)
        float emission = 0.0f;         // Emissive power
        float diffuse = 0.5f;           // Diffuse coefficient
        float specular = 0.5f;          // Specular coefficient
        float phongExponent = 32.0f;     // Shininess exponent

        bool isSensor = false;

        bool reloadShader = false;

        std::filesystem::path vertexShaderName = "defaultBasic.vert";
        std::filesystem::path fragmentShaderName = "defaultPhongLight.frag";
        std::filesystem::path albedoTexturePath = "default.png";
    };

    struct MaterialInstance {
        RenderMode renderMode = RenderMode::Opaque;
        std::shared_ptr<VulkanTexture2D> baseColorTexture;
        // Rendering properties
        AlphaMode alphaMode = AlphaMode::Opaque;
        float alphaCutoff = 0.5f;  // Used if alphaMode is Mask
        bool doubleSided = false;
    };

}

#endif //MATERIALCOMPONENT_H
