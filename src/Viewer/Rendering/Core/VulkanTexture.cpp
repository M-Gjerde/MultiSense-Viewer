//
// Created by mgjer on 24/08/2024.
//

#include <stb_image.h>
#include "VulkanTexture.h"
#include "Viewer/Tools/Utils.h"
#include "VulkanResourceManager.h"

namespace VkRender {


    VulkanTexture::VulkanTexture(VulkanTexture2DCreateInfo &createInfo) : m_vulkanDevice(createInfo.vulkanDevice) {
        m_image = createInfo.image;
        // Create m_Sampler
        VkSamplerCreateInfo samplerCreateInfo = {};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.mipLodBias = 0.0f;
        samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 0.0f;
        samplerCreateInfo.maxAnisotropy = 1.0f;

        CHECK_RESULT(vkCreateSampler(m_vulkanDevice.m_LogicalDevice, &samplerCreateInfo, nullptr, &m_sampler))


        m_imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        m_imageInfo.imageView = m_image->view(); // Your depth image view
        m_imageInfo.sampler = m_sampler; // Sampler you've created for depth texture
    }

    VulkanTexture::~VulkanTexture() {
        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(m_vulkanDevice.m_LogicalDevice, &fenceCreateInfo, nullptr, &fence);
        // Capture all necessary members by value
        auto sampler = m_sampler;
        auto logicalDevice = m_vulkanDevice.m_LogicalDevice;
        VulkanResourceManager::getInstance().deferDeletion(
                [logicalDevice, sampler]() {
                    Log::Logger::getInstance()->trace("Cleaning Up Vulkan Texture Resource");

                    vkDestroySampler(logicalDevice, sampler, nullptr);
                },
                fence);
    }


    void VulkanTexture2D::loadImage(void* data, uint32_t size) {
        if (size == 0) // TODO this does not seem like good code design..
            size = getSize();
        // Create a staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        // Create staging buffers
        // Vertex data
        CHECK_RESULT(m_vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                size,
                &stagingBuffer,
                &stagingBufferMemory,
                data));

        VkCommandBuffer copyCmd = m_vulkanDevice.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        // Transition the image layout to TRANSFER_DST_OPTIMAL
        Utils::setImageLayout(
                copyCmd,
                m_image->image(),
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);

        // Copy the staging buffer data to the Vulkan image
        VkBufferImageCopy bufferCopyRegion = {};
        bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bufferCopyRegion.imageSubresource.mipLevel = 0;
        bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
        bufferCopyRegion.imageSubresource.layerCount = 1;
        bufferCopyRegion.imageExtent.width = m_image->width();
        bufferCopyRegion.imageExtent.height =  m_image->height();
        bufferCopyRegion.imageExtent.depth = 1;
        bufferCopyRegion.bufferOffset = 0;

        vkCmdCopyBufferToImage(
                copyCmd,
                stagingBuffer,
                m_image->image(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &bufferCopyRegion
        );

        Utils::setImageLayout(
                copyCmd,
                m_image->image(),
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        Log::Logger::getInstance()->trace("Flushing Command Buffer for Texture Transfer");
        m_vulkanDevice.flushCommandBuffer(copyCmd, m_vulkanDevice.m_TransferQueue);
        // Clean up the staging buffer
        vkDestroyBuffer(m_vulkanDevice.m_LogicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(m_vulkanDevice.m_LogicalDevice, stagingBufferMemory, nullptr);
    }

}