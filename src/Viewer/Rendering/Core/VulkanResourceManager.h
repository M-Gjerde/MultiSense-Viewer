//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_VULKANRESOURCEMANAGER_H
#define MULTISENSE_VIEWER_VULKANRESOURCEMANAGER_H



#include <mutex>
#include <deque>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <functional>
#include <utility>
#include "VulkanDevice.h"

namespace VkRender {

    class VulkanResourceManager {
    public:
        static VulkanResourceManager& getInstance(VulkanDevice* device = VK_NULL_HANDLE, VmaAllocator allocator = VK_NULL_HANDLE) {
            std::call_once(initInstanceFlag, &VulkanResourceManager::initSingleton, device, allocator);
            return *instance;
        }
        static void destroyInstance() {
            std::call_once(destroyInstanceFlag, &VulkanResourceManager::cleanupSingleton);
        }

        ~VulkanResourceManager() {
            cleanup(true);
        }

        using CleanupFunction = std::function<void()>;

        void deferDeletion(CleanupFunction cleanupFunction, VkFence& fence) {
            std::lock_guard<std::mutex> lock(resourceMutex);
            m_deferredCleanupFunctions.push_back({std::move(cleanupFunction), fence});
        }

        void cleanup(bool onExit = false) ;

    private:
        VulkanResourceManager(VulkanDevice* device, VmaAllocator allocator)
                : m_vulkanDevice(device), m_allocator(allocator) {}

        static void initSingleton(VulkanDevice* device, VmaAllocator allocator) {
            instance = new VulkanResourceManager(device, allocator);
        }

        static void cleanupSingleton() {
            delete instance;
            instance = nullptr;
        }

        void submitCommandBuffers();

        static VulkanResourceManager* instance;
        static std::once_flag initInstanceFlag;
        static std::once_flag destroyInstanceFlag;
        std::mutex resourceMutex;

        VulkanDevice* m_vulkanDevice;
        VmaAllocator m_allocator = VK_NULL_HANDLE;

        struct DeferredCleanup {
            CleanupFunction cleanupFunction;
            VkFence fence;
        };

        std::deque<DeferredCleanup> m_deferredCleanupFunctions;
    };

}

#endif //MULTISENSE_VIEWER_VULKANRESOURCEMANAGER_H
