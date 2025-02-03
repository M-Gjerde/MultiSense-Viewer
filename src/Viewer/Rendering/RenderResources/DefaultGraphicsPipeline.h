//
// Created by magnus on 4/20/24.
//

#ifndef MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H
#define MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H


#include <Viewer/Rendering/Core/PipelineKey.h>

#include "Viewer/Rendering/Core/RenderDefinitions.h"
#include "Viewer/Rendering/Components/Components.h"


namespace VkRender {
    class Application;

    class DefaultGraphicsPipeline {
    public:
        DefaultGraphicsPipeline() = delete;
        /** @brief
        // Delete copy constructors, we dont want to perform shallow copied of vulkan resources leading to double deletion.
        // If copy is necessary define custom copy constructor and use move semantics or references
        */
        DefaultGraphicsPipeline(const DefaultGraphicsPipeline &) = delete;

        DefaultGraphicsPipeline &operator=(const DefaultGraphicsPipeline &) = delete;

        ~DefaultGraphicsPipeline() = default;

        explicit DefaultGraphicsPipeline(Application &m_context, const RenderPassInfo &renderPassInfo,
                                         const PipelineKey &key);

        void bind(CommandBuffer &commandBuffer) const;

        std::shared_ptr<VulkanGraphicsPipeline> pipeline() { return m_graphicsPipeline; }

    private:
        VulkanDevice &m_vulkanDevice;
        RenderPassInfo m_renderPassInfo{};
        uint32_t m_numSwapChainImages = 0;
        std::filesystem::path m_vertexShader;
        std::filesystem::path m_fragmentShader;
        std::shared_ptr<VulkanGraphicsPipeline> m_graphicsPipeline;
    };
};
#endif //MULTISENSE_VIEWER_DEFAULTGRAPHICSPIPELINECOMPONENT2_H
