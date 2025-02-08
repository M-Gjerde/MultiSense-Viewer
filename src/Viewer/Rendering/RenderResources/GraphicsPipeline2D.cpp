//
// Created by magnus on 8/15/24.
//

#include "Viewer/Rendering/RenderResources/GraphicsPipeline2D.h"

#include "Viewer/Application/Application.h"

#include "Viewer/Tools/Utils.h"
#include "Viewer/Rendering/Components/MeshComponent.h"
#include "Viewer/Rendering/Core/VulkanResourceManager.h"


namespace VkRender {
    GraphicsPipeline2D::GraphicsPipeline2D(Application &m_context, const RenderPassInfo &renderPassInfo)
        : m_vulkanDevice(m_context.vkDevice()),
          m_renderPassInfo(renderPassInfo) {
        m_numSwapChainImages = m_context.swapChainBuffers().size();
        m_vulkanDevice = m_context.vkDevice();
        m_vertexShader = "default2D.vert";
        m_fragmentShader = "default2D.frag";
        m_renderData.resize(m_numSwapChainImages);
        std::vector<VkRender::ImageVertex> vertices = {
            // Bottom-left corner
            {glm::vec2{-1.0f, -1.0f}, glm::vec2{0.0f, 0.0f}},
            // Bottom-right corner
            {glm::vec2{1.0f, -1.0f}, glm::vec2{1.0f, 0.0f}},
            // Top-right corner
            {glm::vec2{1.0f, 1.0f}, glm::vec2{1.0f, 1.0f}},
            // Top-left corner
            {glm::vec2{-1.0f, 1.0f}, glm::vec2{0.0f, 1.0f}}
        };
        // Define the indices for two triangles that make up the quad
        std::vector<uint32_t> indices = {
            0, 1, 2, // First triangle (bottom-left to top-right)
            2, 3, 0 // Second triangle (top-right to bottom-left)
        };

        m_indices.indexCount = indices.size();
        size_t vertexBufferSize = vertices.size() * sizeof(ImageVertex);
        size_t indexBufferSize = indices.size() * sizeof(uint32_t);

        assert(vertexBufferSize > 0);

        struct StagingBuffer {
            VkBuffer buffer;
            VkDeviceMemory memory;
        } vertexStaging{}, indexStaging{};

        // Create staging buffers
        // Vertex data
        CHECK_RESULT(m_vulkanDevice.createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBufferSize,
            &vertexStaging.buffer,
            &vertexStaging.memory,
            vertices.data()));
        // Index data
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBufferSize,
                &indexStaging.buffer,
                &indexStaging.memory,
                indices.data()));
        }

        // Create m_vulkanDevice local buffers
        // Vertex buffer
        CHECK_RESULT(m_vulkanDevice.createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            vertexBufferSize,
            &m_vertices.buffer,
            &m_vertices.memory));
        // Index buffer
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                indexBufferSize,
                &m_indices.buffer,
                &m_indices.memory));
        }

        // Copy from staging buffers
        VkCommandBuffer copyCmd = m_vulkanDevice.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        VkBufferCopy copyRegion = {};

        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, m_vertices.buffer, 1, &copyRegion);

        if (indexBufferSize > 0) {
            copyRegion.size = indexBufferSize;
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, m_indices.buffer, 1, &copyRegion);
        }

        m_vulkanDevice.flushCommandBuffer(copyCmd, m_vulkanDevice.m_TransferQueue, true);

        vkDestroyBuffer(m_vulkanDevice.m_LogicalDevice, vertexStaging.buffer, nullptr);
        vkFreeMemory(m_vulkanDevice.m_LogicalDevice, vertexStaging.memory, nullptr);
        if (indexBufferSize > 0) {
            vkDestroyBuffer(m_vulkanDevice.m_LogicalDevice, indexStaging.buffer, nullptr);
            vkFreeMemory(m_vulkanDevice.m_LogicalDevice, indexStaging.memory, nullptr);
        }

        // Setup default texture
        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageCI.extent = {static_cast<uint32_t>(1280), static_cast<uint32_t>(720), 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
        imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        VulkanImageCreateInfo vulkanImageCreateInfo(m_context.vkDevice(), m_context.allocator(), imageCI,
                                                    imageViewCI);
        vulkanImageCreateInfo.setLayout = true;
        vulkanImageCreateInfo.srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkanImageCreateInfo.dstLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vulkanImageCreateInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vulkanImageCreateInfo.debugInfo = "Image Editor:Default Texture";
        m_defaultImage = std::make_shared<VulkanImage>(vulkanImageCreateInfo);

        VulkanTexture2DCreateInfo textureCreateInfo(m_context.vkDevice());
        textureCreateInfo.image = m_defaultImage;
        m_defaultTexture = std::make_shared<VulkanTexture2D>(textureCreateInfo);
        bindTexture(m_defaultTexture);
    }

    void GraphicsPipeline2D::bindTexture(const std::shared_ptr<VulkanTexture>& texture) {
        setupDescriptors(texture);
        setupPipeline();
    }

    GraphicsPipeline2D::~GraphicsPipeline2D() {
        auto logicalDevice = m_vulkanDevice.m_LogicalDevice;
        VkFence fence;
        VkFenceCreateInfo fenceInfo = Populate::fenceCreateInfo(0);
        vkCreateFence(logicalDevice, &fenceInfo, nullptr, &fence);

        Indices &indices = this->m_indices;
        Vertices &vertices = this->m_vertices;
        VkDescriptorSetLayout layout = m_descriptorSetLayout;
        VkDescriptorPool pool = m_descriptorPool;

        VulkanResourceManager::getInstance().deferDeletion(
            [logicalDevice, indices, vertices, layout, pool]() {
                Log::Logger::getInstance()->trace("Cleaning Up Vulkan GraphicsPipeline2D Resource");

                vkDestroyDescriptorSetLayout(logicalDevice, layout, nullptr);
                vkDestroyDescriptorPool(logicalDevice, pool, nullptr);

                if (vertices.buffer != VK_NULL_HANDLE) {
                    vkDestroyBuffer(logicalDevice, vertices.buffer, nullptr);
                }
                if (vertices.memory != VK_NULL_HANDLE) {
                    vkFreeMemory(logicalDevice, vertices.memory, nullptr);
                }
                if (indices.buffer != VK_NULL_HANDLE) {
                    vkDestroyBuffer(logicalDevice, indices.buffer, nullptr);
                }
                if (indices.memory != VK_NULL_HANDLE) {
                    vkFreeMemory(logicalDevice, indices.memory, nullptr);
                }
            },
            fence);
    }


    void GraphicsPipeline2D::draw(CommandBuffer &cmdBuffers) {
        if (m_graphicsPipeline) {
            vkCmdBindPipeline(cmdBuffers.getActiveBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                              m_graphicsPipeline->getPipeline());
            vkCmdBindDescriptorSets(cmdBuffers.getActiveBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_graphicsPipeline->getPipelineLayout(), 0, static_cast<uint32_t>(1),
                                    &m_renderData[cmdBuffers.frameIndex].descriptorSet, 0, nullptr);
            VkDeviceSize offsets[1] = {0};
            vkCmdBindVertexBuffers(cmdBuffers.getActiveBuffer(), 0, 1, &m_vertices.buffer, offsets);
            vkCmdBindIndexBuffer(cmdBuffers.getActiveBuffer(), m_indices.buffer, 0,
                                 VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(cmdBuffers.getActiveBuffer(), m_indices.indexCount, 1,
                             0, 0, 0);
        }
    }

    void GraphicsPipeline2D::setTexture(const VkDescriptorImageInfo *info) {
        VkWriteDescriptorSet writeDescriptorSets{};
        for (const auto &data: m_renderData) {
            writeDescriptorSets.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets.descriptorCount = 1;
            writeDescriptorSets.dstSet = data.descriptorSet;
            writeDescriptorSets.dstBinding = 0;
            writeDescriptorSets.pImageInfo = info;
            vkUpdateDescriptorSets(m_vulkanDevice.m_LogicalDevice, 1, &writeDescriptorSets, 0, nullptr);
        }
    }

    void GraphicsPipeline2D::setupPipeline() {
        // Vertex bindings an attributes
        VkVertexInputBindingDescription vertexInputBinding = {
            0, sizeof(VkRender::ImageVertex),
            VK_VERTEX_INPUT_RATE_VERTEX
        };
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
            {1, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 2},
        };
        VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
        vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputStateCI.vertexBindingDescriptionCount = 1;
        vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
        vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
        vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages(2);
        VkShaderModule vertModule{};
        VkShaderModule fragModule{};
        shaderStages[0] = Utils::loadShader(m_vulkanDevice.m_LogicalDevice, "spv/" + m_vertexShader,
                                            VK_SHADER_STAGE_VERTEX_BIT, &vertModule);
        shaderStages[1] = Utils::loadShader(m_vulkanDevice.m_LogicalDevice, "spv/" + m_fragmentShader,
                                            VK_SHADER_STAGE_FRAGMENT_BIT, &fragModule);

        VulkanGraphicsPipelineCreateInfo createInfo(m_renderPassInfo.renderPass, m_vulkanDevice);
        createInfo.rasterizationStateCreateInfo = Populate::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            VK_FRONT_FACE_COUNTER_CLOCKWISE);
        createInfo.msaaSamples = m_renderPassInfo.sampleCount;
        createInfo.shaders = shaderStages;
        createInfo.descriptorSetLayouts.push_back(m_descriptorSetLayout);
        createInfo.vertexInputState = vertexInputStateCI;

        m_graphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(createInfo);

        for (auto shaderStage: shaderStages) {
            vkDestroyShaderModule(m_vulkanDevice.m_LogicalDevice, shaderStage.module, nullptr);
        }
    }

    void GraphicsPipeline2D::setupDescriptors(const std::shared_ptr<VulkanTexture> &texture) {
        std::vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_numSwapChainImages * 2},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, m_numSwapChainImages * 2},
        };

        VkDescriptorPoolCreateInfo descriptorPoolCI{};
        descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        descriptorPoolCI.pPoolSizes = poolSizes.data();
        descriptorPoolCI.maxSets = m_numSwapChainImages * static_cast<uint32_t>(poolSizes.size());
        CHECK_RESULT(
            vkCreateDescriptorPool(m_vulkanDevice.m_LogicalDevice, &descriptorPoolCI, nullptr,
                &m_descriptorPool));


        // Scene (matrices and environment maps)
        {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {
                    1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT |
                                                             VK_SHADER_STAGE_FRAGMENT_BIT,
                    nullptr
                },
            };
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
            descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
            descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
            CHECK_RESULT(
                vkCreateDescriptorSetLayout(m_vulkanDevice.m_LogicalDevice, &descriptorSetLayoutCI,
                    nullptr,
                    &m_descriptorSetLayout));

            for (auto &resource: m_renderData) {
                VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocInfo.descriptorPool = m_descriptorPool;
                descriptorSetAllocInfo.pSetLayouts = &m_descriptorSetLayout;
                descriptorSetAllocInfo.descriptorSetCount = 1;
                VkResult res = vkAllocateDescriptorSets(m_vulkanDevice.m_LogicalDevice, &descriptorSetAllocInfo,
                                                        &resource.descriptorSet);
                if (res != VK_SUCCESS)
                    throw std::runtime_error("Failed to allocate descriptor sets");

                std::array<VkWriteDescriptorSet, 1> writeDescriptorSets{};
                writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[0].descriptorCount = 1;
                writeDescriptorSets[0].dstSet = resource.descriptorSet;
                writeDescriptorSets[0].dstBinding = 0;
                writeDescriptorSets[0].pImageInfo = &texture->getDescriptorInfo();

                vkUpdateDescriptorSets(m_vulkanDevice.m_LogicalDevice,
                                       static_cast<uint32_t>(writeDescriptorSets.size()),
                                       writeDescriptorSets.data(), 0, nullptr);
            }
        }
    }
}
