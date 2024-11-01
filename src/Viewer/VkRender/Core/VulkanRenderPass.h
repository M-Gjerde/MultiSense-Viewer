//
// Created by magnus on 7/17/24.
//

#ifndef MULTISENSE_VIEWER_VULKANRENDERPASS_H
#define MULTISENSE_VIEWER_VULKANRENDERPASS_H

#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>
#include <string>
#include <memory>
#include <utility>

#include "Viewer/VkRender/ImGui/GuiManager.h"
#include "Viewer/VkRender/Editors/EditorDefinitions.h"

namespace VkRender {
    class Application;

    typedef enum VulkanRenderPassType{
        DEFAULT,
        DEPTH_RESOLVE_RENDER_PASS,
    }VulkanRenderPassType;
    /**@brief Very early iteration of a editor create info which also includes renderpass create info. TODO They should be separated into EditorCreateInfo and RenderPass even though they share a lot of information*/
    struct VulkanRenderPassCreateInfo {
        VulkanDevice *vulkanDevice = nullptr;
        VmaAllocator *allocator = nullptr;

        VulkanRenderPassType type = VulkanRenderPassType::DEFAULT;
        std::string debugInfo = "Unnamed";

        int32_t width = 10;
        int32_t height = 10;

        uint32_t swapchainImageCount = 0;
        VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
        VkFormat swapchainColorFormat;
        VkFormat depthFormat;

        VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        VkImageLayout initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkImageLayout finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentStoreOp storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        static void copy(VulkanRenderPassCreateInfo *dst, VulkanRenderPassCreateInfo *src) {
            dst->vulkanDevice = src->vulkanDevice;
            dst->allocator = src->allocator;

            dst->height = src->height;
            dst->width = src->width;
            dst->swapchainImageCount = src->swapchainImageCount;
            dst->msaaSamples = src->msaaSamples;
            dst->swapchainColorFormat = src->swapchainColorFormat;
            dst->depthFormat = src->depthFormat;
            dst->loadOp = src->loadOp;
            dst->initialLayout = src->initialLayout;
            dst->storeOp = src->storeOp;
            dst->finalLayout = src->finalLayout;
        }


        VulkanRenderPassCreateInfo(VulkanDevice *dev, VmaAllocator *alloc)
                : vulkanDevice(dev), allocator(alloc) {

        }

        VulkanRenderPassCreateInfo() = default;

    };

    struct EditorCreateInfo {
        VulkanRenderPassCreateInfo pPassCreateInfo{};
        VulkanDevice *vulkanDevice = nullptr;
        VmaAllocator *allocator = nullptr;
        Application* context;
        VkFramebuffer *frameBuffers{};

        int32_t x = 0;
        int32_t y = 0;
        int32_t borderSize = 5;
        int32_t width = 10;
        int32_t height = 10;

        EditorType editorTypeDescription = EditorType::TestWindow;
        bool resizeable = true;
        size_t editorIndex = 0;
        std::vector<std::string> uiLayers;
        ImGuiContext *uiContext = nullptr;

        std::shared_ptr<GuiAssets> guiResources;

        EditorCreateInfo(std::shared_ptr<GuiAssets> guiRes, Application* ctx,
                         VulkanDevice *dev, VmaAllocator *alloc, VkFramebuffer* fbs)
                : vulkanDevice(dev), allocator(alloc), guiResources(std::move(guiRes)), context(ctx), frameBuffers(fbs) {
        }


        static void copy(EditorCreateInfo *dst, EditorCreateInfo *src) {
            dst->pPassCreateInfo = src->pPassCreateInfo;
            dst->vulkanDevice = src->vulkanDevice;
            dst->allocator = src->allocator;
            dst->frameBuffers = src->frameBuffers;
            dst->width = src->width;
            dst->height = src->height;

            dst->x = src->x;
            dst->y = src->y;
            dst->borderSize = src->borderSize;
            dst->editorTypeDescription = src->editorTypeDescription;
            dst->resizeable = src->resizeable;
            dst->editorIndex = src->editorIndex;
            dst->uiContext = src->uiContext;
            dst->uiLayers = src->uiLayers;
        };
    };

    struct VulkanRenderPass {
    public:

        VulkanRenderPass() = delete;

        explicit VulkanRenderPass(const VulkanRenderPassCreateInfo *createInfo);

        // Implement move constructor
        VulkanRenderPass(VulkanRenderPass &&other) noexcept: m_logicalDevice(other.m_logicalDevice) {
            std::swap(this->m_renderPass, other.m_renderPass);
        }

        // and move assignment operator
        VulkanRenderPass &operator=(VulkanRenderPass &&other) noexcept {
            if (this != &other) { // Check for self-assignment
                std::swap(this->m_logicalDevice, other.m_logicalDevice);
                std::swap(this->m_renderPass, other.m_renderPass);
            }
            return *this;
        }

        // No copying allowed
        VulkanRenderPass(const VulkanRenderPass &) = delete;

        VulkanRenderPass &operator=(const VulkanRenderPass &) = delete;

        ~VulkanRenderPass();

        VkRenderPass &getRenderPass() {
            return m_renderPass;
        }

    private:
        VkRenderPass m_renderPass = VK_NULL_HANDLE;

        VkDevice &m_logicalDevice;

        void setupDepthOnlyRenderPass(const VulkanRenderPassCreateInfo *createInfo);
        void setupDefaultRenderPass(const VulkanRenderPassCreateInfo *createInfo);
    };
}

#endif //MULTISENSE_VIEWER_VULKANRENDERPASS_H
