//
// Created by magnus on 7/29/24.
//


#include "VulkanResourceManager.h"
#include "Viewer/Tools/Logger.h"

namespace VkRender {
    VulkanResourceManager *VulkanResourceManager::instance = nullptr;

    void VulkanResourceManager::cleanup(bool onExit) {
        auto functionStart = std::chrono::high_resolution_clock::now();

        std::lock_guard<std::mutex> lock(resourceMutex);

        // TODO Its possible to make batch submits, which could speed up a little bit
        // TODO Is m_TransferQueue the correct queue to use? Maybe use graphics queue as we are dealing with render resources
        // Iterate through deferred tasks
        bool somethingToClean = false;
        for (auto it = m_deferredCleanupFunctions.begin(); it != m_deferredCleanupFunctions.end(); ) {
            // Only increment the frame counter if not forcing cleanup on exit.
            if (!onExit) {
                it->framesWaited++;
            }

            // If the task hasn't been in the queue for at least 10 frames, skip it.
            if (it->framesWaited < 120 && !onExit) {
                ++it;
                continue;
            }
            Log::Logger::getInstance()->trace("Deferred housekeeping: {} (waited {} frames)",it->debugString, it->framesWaited);

            // Now that the task has aged N frames (or we're exiting), flush the command buffer.¨
            vkDeviceWaitIdle(m_vulkanDevice->m_LogicalDevice);
            vkQueueWaitIdle(m_queue);
            VkCommandBuffer commandBuffer = m_vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
            m_vulkanDevice->flushCommandBuffer(commandBuffer,
                                                 m_queue,
                                                 m_vulkanDevice->m_CommandPool,
                                                 true,
                                                 it->fence);

            // Check if the fence is signaled (or if we're forcing cleanup on exit).
            VkResult result = vkGetFenceStatus(m_vulkanDevice->m_LogicalDevice, it->fence);
            if (result == VK_SUCCESS || onExit) {
                it->cleanupFunction();
                vkDestroyFence(m_vulkanDevice->m_LogicalDevice, it->fence, nullptr);
                it = m_deferredCleanupFunctions.erase(it);
                somethingToClean = true;
                vkDeviceWaitIdle(m_vulkanDevice->m_LogicalDevice);
                vkQueueWaitIdle(m_queue);
            } else {
                ++it;
            }
        }

        if (somethingToClean) {
            auto functionEnd = std::chrono::high_resolution_clock::now();
            auto functionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(functionEnd - functionStart).count();
            Log::Logger::getInstance()->trace("cleanup function took {} microseconds to execute", functionDuration);
        }
    }

    void VulkanResourceManager::submitCommandBuffers() {


    }

    std::once_flag VulkanResourceManager::initInstanceFlag;
    std::once_flag VulkanResourceManager::destroyInstanceFlag;

}
