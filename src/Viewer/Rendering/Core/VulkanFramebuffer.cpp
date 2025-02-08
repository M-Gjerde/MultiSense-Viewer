//
// Created by magnus on 8/22/24.
//

#include "VulkanFramebuffer.h"
#include "VulkanResourceManager.h"

namespace VkRender {


    VulkanFramebuffer::VulkanFramebuffer(VulkanFramebufferCreateInfo &createInfo) : m_vulkanDevice(
            createInfo.vulkanDevice) {

        VkFramebufferCreateInfo frameBufferCreateInfo = Populate::framebufferCreateInfo(
                static_cast<uint32_t>(createInfo.width), static_cast<uint32_t>(createInfo.height),
                createInfo.frameBufferAttachments.data(),
                createInfo.frameBufferAttachments.size(),
                createInfo.renderPass);
        VkResult result = vkCreateFramebuffer(m_vulkanDevice.m_LogicalDevice, &frameBufferCreateInfo, nullptr,
                                              &m_framebuffer);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create framebuffer");

        }
    }

    VulkanFramebuffer::~VulkanFramebuffer() {
        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(m_vulkanDevice.m_LogicalDevice, &fenceCreateInfo, nullptr, &fence);

        // Capture all necessary members by value
        auto framebuffer = m_framebuffer;
        auto logicalDevice = m_vulkanDevice.m_LogicalDevice;

        VulkanResourceManager::getInstance().deferDeletion(
                [logicalDevice, framebuffer]() {
                    Log::Logger::getInstance()->trace("Cleaning Up Vulkan FrameBuffer Resource");

                    vkDestroyFramebuffer(logicalDevice, framebuffer, nullptr);
                },
                fence);

    }
}