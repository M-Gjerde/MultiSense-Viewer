//
// Created by magnus-desktop on 11/17/24.
//

#ifndef DESCRIPTORMANAGERREGISTRY_H
#define DESCRIPTORMANAGERREGISTRY_H

#include <vector>
#include <Viewer/Rendering/Editors/RenderCommand.h>

#include "Viewer/Rendering/Core/DescriptorSetManager.h"


namespace VkRender {
    enum class DescriptorManagerType : uint32_t {
        MVP = 0,
        Material = 1,
        DynamicCameraGizmo = 2,
        Viewport3DTexture = 4,
        // TODO This only works as long as we're not confusing Viewport3DTexture with SceneRenderer
    };

    static std::string descriptorManagerTypeToString(DescriptorManagerType type){
        switch (type) {
            case DescriptorManagerType::MVP:
                return "MVP";
            case DescriptorManagerType::Material:
                return "Material";
            case DescriptorManagerType::DynamicCameraGizmo:
                return "DynamicCameraGizmo";
            case DescriptorManagerType::Viewport3DTexture:
                return "Viewport3DTexture";
        default:
            throw std::runtime_error("Unknown Descriptor Manager Type");
        }
    }

    class DescriptorRegistry {
    public:
        DescriptorSetManager &getManager(DescriptorManagerType type) {
            if (!m_managers[type]) {
                throw std::runtime_error("Tried to get DescriptorManager that was not initialized");
            }
            return *m_managers[type];
        }

        void createManager(
                DescriptorManagerType type,
                VulkanDevice &device
        ) {
            std::vector<VkDescriptorSetLayoutBinding> bindings;


            switch (type) {
                case DescriptorManagerType::MVP:
                    bindings = {
                            {
                                    0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                                                             VK_SHADER_STAGE_VERTEX_BIT |
                                                                             VK_SHADER_STAGE_FRAGMENT_BIT,
                                    nullptr
                            },
                            {
                                    1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,
                                    nullptr
                            },
                    };

                    break;
                case DescriptorManagerType::Material:
                    bindings = {
                            {
                                    0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                    nullptr
                            },
                            {
                                    1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                    nullptr
                            },
                    };
                    break;
                case DescriptorManagerType::DynamicCameraGizmo:
                    bindings = {
                            {
                                    0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,
                                    nullptr
                            },
                            {
                                    1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,
                                    nullptr
                            }
                    };
                    break;
                case DescriptorManagerType::Viewport3DTexture:
                    bindings = {
                            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                            {
                             1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT |
                                                                              VK_SHADER_STAGE_FRAGMENT_BIT,
                                                                                                            nullptr
                            }
                    };
                    break;
            }


            m_managers[type] =
                    std::make_unique<DescriptorSetManager>(device, bindings, type);
        }

    private:
        std::unordered_map<DescriptorManagerType, std::unique_ptr<DescriptorSetManager>> m_managers;
    };
}
#endif //DESCRIPTORMANAGERREGISTRY_H
