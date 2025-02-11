//
// Created by magnus on 7/30/24.
//

#include "VulkanGraphicsPipeline.h"

#include <Viewer/Tools/Macros.h>

#include "VulkanResourceManager.h"

namespace VkRender {
    VulkanGraphicsPipeline::VulkanGraphicsPipeline(const VulkanGraphicsPipelineCreateInfo& createInfo) : m_vulkanDevice(
        createInfo.vulkanDevice) {
        // Pipeline cache
        VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
        pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        if (vkCreatePipelineCache(m_vulkanDevice.m_LogicalDevice, &pipelineCacheCreateInfo, nullptr,
                                  &m_pipelineCache) !=
            VK_SUCCESS)
            throw std::runtime_error("Failed to create Pipeline Cache");


        // Pipeline layout
        // Push constants for UI rendering parameters
        VkPushConstantRange pushConstantRange = Populate::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT,
                                                                            createInfo.pushConstBlockSize, 0);
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = Populate::pipelineLayoutCreateInfo(
            createInfo.descriptorSetLayouts.data(), createInfo.descriptorSetLayouts.size());

        pipelineLayoutCreateInfo.pushConstantRangeCount = createInfo.pushConstBlockSize ? 1 : 0;
        pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
        if (
            vkCreatePipelineLayout(m_vulkanDevice.m_LogicalDevice, &pipelineLayoutCreateInfo, nullptr,
                                   &m_pipelineLayout) !=
            VK_SUCCESS)
            throw std::runtime_error("Failed to create m_Pipeline layout");

        VALIDATION_DEBUG_NAME(m_vulkanDevice.m_LogicalDevice,
                              reinterpret_cast<uint64_t>(m_pipelineLayout), VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              createInfo.debugInfo + ":PipelineLayout");

        // Setup graphics pipeline for UI rendering
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
            Populate::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0,
                                                           VK_FALSE);

        // Enable blending
        VkPipelineColorBlendAttachmentState blendAttachmentState{};
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlendState =
            Populate::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

        VkPipelineDepthStencilStateCreateInfo depthStencilState = Populate::pipelineDepthStencilStateCreateInfo(
            createInfo.depthTesting,
            createInfo.depthTesting,
            VK_COMPARE_OP_LESS_OR_EQUAL);

        VkPipelineViewportStateCreateInfo viewportState =
            Populate
            ::pipelineViewportStateCreateInfo(1, 1, 0);

        VkPipelineMultisampleStateCreateInfo multisampleState =
            Populate
            ::pipelineMultisampleStateCreateInfo(createInfo.msaaSamples);

        std::vector<VkDynamicState> dynamicStateEnables = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState =
            Populate
            ::pipelineDynamicStateCreateInfo(dynamicStateEnables);

        VkGraphicsPipelineCreateInfo pipelineCreateInfo = Populate::pipelineCreateInfo(m_pipelineLayout,
            createInfo.renderPass);


        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &createInfo.rasterizationStateCreateInfo;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(createInfo.shaders.size());
        pipelineCreateInfo.pStages = createInfo.shaders.data();


        // TODO InputBinding and Attributes should be part of CreateInfo
        pipelineCreateInfo.pVertexInputState = &createInfo.vertexInputState;

        if (vkCreateGraphicsPipelines(m_vulkanDevice.m_LogicalDevice, m_pipelineCache, 1, &pipelineCreateInfo,
                                      nullptr,
                                      &m_pipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics m_Pipeline");

        VALIDATION_DEBUG_NAME(m_vulkanDevice.m_LogicalDevice,
                      reinterpret_cast<uint64_t>(m_pipeline), VK_OBJECT_TYPE_PIPELINE,
                      createInfo.debugInfo + ":Pipeline");
    }

    VulkanGraphicsPipeline::~VulkanGraphicsPipeline() {
        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(m_vulkanDevice.m_LogicalDevice, &fenceCreateInfo, nullptr, &fence);

        // Capture all necessary members by value
        auto pipeline = m_pipeline;
        auto pipelinCache = m_pipelineCache;
        auto pipelineLayout = m_pipelineLayout;
        auto logicalDevice = m_vulkanDevice.m_LogicalDevice;


        VulkanResourceManager::getInstance().deferDeletion(
            [logicalDevice, pipeline, pipelinCache, pipelineLayout]() {
                Log::Logger::getInstance()->trace("Cleaning Up Vulkan GraphicsPipeline Resource");

                vkDestroyPipeline(logicalDevice, pipeline, nullptr);
                vkDestroyPipelineCache(logicalDevice, pipelinCache, nullptr);
                vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
            },
            fence, "Cleaning up GraphicsPipeline");
    }
}
