//
// Created by mgjer on 20/08/2024.
//

#include "VulkanImage.h"
#include "vk_mem_alloc.h"
#include "VulkanResourceManager.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender {
    VulkanImage::VulkanImage(VkRender::VulkanImageCreateInfo &createInfo) : m_vulkanDevice(
            createInfo.vulkanDevice), m_allocator(createInfo.allocator) {
        // TODO add warning about adding a image with size of zero. Maybe
        std::string description = createInfo.debugInfo;
        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.usage = createInfo.usage;

        VkResult result = vmaCreateImage(m_allocator, &createInfo.imageCreateInfo, &allocInfo, &m_image,
                                         &m_allocation, nullptr);
        if (result != VK_SUCCESS)
            throw std::runtime_error("Failed to create image");
        vmaSetAllocationName(m_allocator, m_allocation, (description + "Image").c_str());
        VALIDATION_DEBUG_NAME(m_vulkanDevice.m_LogicalDevice,
                              reinterpret_cast<uint64_t>(m_image), VK_OBJECT_TYPE_IMAGE,
                              description + "Image");
        createInfo.imageViewCreateInfo.image = m_image;
        result = vkCreateImageView(m_vulkanDevice.m_LogicalDevice, &createInfo.imageViewCreateInfo, nullptr,
                                   &m_view);
        if (result != VK_SUCCESS) throw std::runtime_error("Failed to create image view");
        VALIDATION_DEBUG_NAME(m_vulkanDevice.m_LogicalDevice,
                              reinterpret_cast<uint64_t>(m_view), VK_OBJECT_TYPE_IMAGE_VIEW,
                              description + "View");

        if (createInfo.setLayout) {
            VkCommandBuffer copyCmd = m_vulkanDevice.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            VkImageSubresourceRange subresourceRange = {};
            subresourceRange.aspectMask = createInfo.aspectMask;
            subresourceRange.levelCount = 1;
            subresourceRange.layerCount = 1;
            Utils::setImageLayout(copyCmd, m_image, createInfo.srcLayout,
                                  createInfo.dstLayout, subresourceRange,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            m_vulkanDevice.flushCommandBuffer(copyCmd, m_vulkanDevice.m_TransferQueue, true);
        }

        // Image Size:
        m_imageSize = createInfo.imageCreateInfo.extent.width * createInfo.imageCreateInfo.extent.height * Utils::getBytesPerPixelFromVkFormat(createInfo.imageCreateInfo.format);
        m_width = createInfo.imageCreateInfo.extent.width;
        m_height = createInfo.imageCreateInfo.extent.height;

        Log::Logger::getInstance()->info("Created Image {}, size: {}x{}", createInfo.debugInfo, m_width, m_height);
    }

    VulkanImage::~VulkanImage() {

        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(m_vulkanDevice.m_LogicalDevice, &fenceCreateInfo, nullptr, &fence);

        // Capture all necessary members by value
        auto view = m_view;
        auto image = m_image;
        auto logicalDevice = m_vulkanDevice.m_LogicalDevice;
        auto allocator = m_allocator;
        auto allocation = m_allocation;

        VulkanResourceManager::getInstance().deferDeletion(
                [logicalDevice, allocator, allocation, view, image]() {
                    Log::Logger::getInstance()->trace("Cleaning Up Vulkan Image Resource");

                    vkDestroyImageView(logicalDevice, view, nullptr);
                    vmaDestroyImage(allocator, image, allocation);
                },
                fence);


    }

}

