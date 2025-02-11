//
// Created by magnus on 7/17/24.
//

#include <array>
#include "VulkanRenderPass.h"

#include "Viewer/Tools/Populate.h"
#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Application/Application.h"
#include "VulkanResourceManager.h"

namespace VkRender {


    VulkanRenderPass::VulkanRenderPass(const VulkanRenderPassCreateInfo *createInfo) : m_logicalDevice(
            createInfo->vulkanDevice->m_LogicalDevice){
        switch (createInfo->type) {
            case DEFAULT:
                setupDefaultRenderPass(createInfo);
                break;
            case DEPTH_RESOLVE_RENDER_PASS:
                setupDepthOnlyRenderPass(createInfo);
                break;
        }
    }


    VulkanRenderPass::~VulkanRenderPass() {
        VkFence fence;
        VkFenceCreateInfo fenceInfo = Populate::fenceCreateInfo(0);
        vkCreateFence(m_logicalDevice, &fenceInfo, nullptr, &fence);
        // Capture all necessary members by value
        auto logicalDevice = m_logicalDevice;
        auto renderPass = m_renderPass;
        VulkanResourceManager::getInstance().deferDeletion(
                [logicalDevice, renderPass]() {
                    // Cleanup logic with captured values
                        Log::Logger::getInstance()->trace("Cleaning Up Vulkan Renderpass Resource");

                    vkDestroyRenderPass(logicalDevice, renderPass, nullptr);
                },
                fence, "Cleaning Up Render Pass Resource");

    }

    void VulkanRenderPass::setupDefaultRenderPass(const VulkanRenderPassCreateInfo *createInfo) {
        VkSampleCountFlagBits sampleCount = createInfo->msaaSamples;
        uint32_t width = std::max(createInfo->width, 1);
        uint32_t height = std::max(createInfo->height, 1);

        VkAttachmentDescription uiColorAttachment = {};
        uiColorAttachment.format = createInfo->swapchainColorFormat;
        uiColorAttachment.samples = createInfo->msaaSamples;
        uiColorAttachment.loadOp = createInfo->loadOp; // Load since it follows the main pass
        uiColorAttachment.storeOp = createInfo->storeOp;
        uiColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        uiColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        uiColorAttachment.initialLayout = createInfo->initialLayout;
        uiColorAttachment.finalLayout = createInfo->finalLayout;
        VkAttachmentReference uiColorAttachmentRef = {};
        uiColorAttachmentRef.attachment = 0;
        uiColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription uiResolveAttachment = {};
        uiResolveAttachment.format = createInfo->swapchainColorFormat;
        uiResolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        uiResolveAttachment.loadOp = createInfo->loadOp;
        uiResolveAttachment.storeOp = createInfo->storeOp;
        uiResolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        uiResolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        uiResolveAttachment.initialLayout = createInfo->initialLayout;
        uiResolveAttachment.finalLayout = createInfo->finalLayout;
        VkAttachmentReference uiResolveAttachmentRef = {};
        uiResolveAttachmentRef.attachment = 2;
        uiResolveAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depthAttachment = {};
        depthAttachment.format = createInfo->depthFormat;
        depthAttachment.samples = createInfo->msaaSamples;
        depthAttachment.loadOp = createInfo->loadOp;
        depthAttachment.storeOp = createInfo->storeOp;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef = {};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription uiSubpass = {};
        uiSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        uiSubpass.colorAttachmentCount = 1;
        uiSubpass.pColorAttachments = &uiColorAttachmentRef;
        uiSubpass.pResolveAttachments = &uiResolveAttachmentRef;
        uiSubpass.pDepthStencilAttachment = &depthAttachmentRef;

        std::array<VkSubpassDependency, 2> uiDependencies = {};
        uiDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        uiDependencies[0].dstSubpass = 0;
        uiDependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        uiDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        uiDependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        uiDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        uiDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        uiDependencies[1].srcSubpass = 0;
        uiDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        uiDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        uiDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        uiDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        uiDependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        uiDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        std::array<VkAttachmentDescription, 3> uiAttachments = {uiColorAttachment, depthAttachment,
                                                                uiResolveAttachment};
        VkRenderPassCreateInfo uiRenderPassInfo = {};
        uiRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        uiRenderPassInfo.attachmentCount = static_cast<uint32_t>(uiAttachments.size());
        uiRenderPassInfo.pAttachments = uiAttachments.data();
        uiRenderPassInfo.subpassCount = 1;
        uiRenderPassInfo.pSubpasses = &uiSubpass;
        uiRenderPassInfo.dependencyCount = static_cast<uint32_t>(uiDependencies.size());
        uiRenderPassInfo.pDependencies = uiDependencies.data();

        if (vkCreateRenderPass(m_logicalDevice, &uiRenderPassInfo, nullptr,
                               &m_renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create UI render pass!");
        }
        VALIDATION_DEBUG_NAME(m_logicalDevice,
                              reinterpret_cast<uint64_t>(m_renderPass), VK_OBJECT_TYPE_RENDER_PASS,
                              (createInfo->debugInfo + ":UIRenderPass").c_str());
    }
    void VulkanRenderPass::setupDepthOnlyRenderPass(const VulkanRenderPassCreateInfo *createInfo) {
        VkSampleCountFlagBits sampleCount = createInfo->msaaSamples;
        uint32_t width = std::max(createInfo->width, 1);
        uint32_t height = std::max(createInfo->height, 1);

        VkAttachmentDescription2 uiColorAttachment = {};
        uiColorAttachment.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        uiColorAttachment.format = createInfo->swapchainColorFormat;
        uiColorAttachment.samples = createInfo->msaaSamples;
        uiColorAttachment.loadOp = createInfo->loadOp; // Load since it follows the main pass
        uiColorAttachment.storeOp = createInfo->storeOp;
        uiColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        uiColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        uiColorAttachment.initialLayout = createInfo->initialLayout;
        uiColorAttachment.finalLayout = createInfo->finalLayout;

        VkAttachmentReference2 uiColorAttachmentRef = {};
        uiColorAttachmentRef.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        uiColorAttachmentRef.attachment = 0;
        uiColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription2 uiResolveAttachment = {};
        uiResolveAttachment.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        uiResolveAttachment.format = createInfo->swapchainColorFormat;
        uiResolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        uiResolveAttachment.loadOp = createInfo->loadOp;
        uiResolveAttachment.storeOp = createInfo->storeOp;
        uiResolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        uiResolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        uiResolveAttachment.initialLayout = createInfo->initialLayout;
        uiResolveAttachment.finalLayout = createInfo->finalLayout;

        VkAttachmentReference2 uiResolveAttachmentRef = {};
        uiResolveAttachmentRef.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        uiResolveAttachmentRef.attachment = 2;
        uiResolveAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription2 depthAttachment = {};
        depthAttachment.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        depthAttachment.format = createInfo->depthFormat;
        depthAttachment.samples = createInfo->msaaSamples;
        depthAttachment.loadOp = createInfo->loadOp;
        depthAttachment.storeOp = createInfo->storeOp;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference2 depthAttachmentRef = {};
        depthAttachmentRef.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription2 depthResolveAttachment = {};
        depthResolveAttachment.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        depthResolveAttachment.format = createInfo->depthFormat;
        depthResolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthResolveAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthResolveAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthResolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthResolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthResolveAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthResolveAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference2 depthResolveAttachmentRef = {};
        depthResolveAttachmentRef.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        depthResolveAttachmentRef.attachment = 3;
        depthResolveAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthResolveAttachmentRef.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

// Depth stencil resolve structure
        VkSubpassDescriptionDepthStencilResolveKHR depthStencilResolve = {};
        depthStencilResolve.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE_KHR;
        depthStencilResolve.depthResolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT; // or VK_RESOLVE_MODE_AVERAGE_BIT
        depthStencilResolve.stencilResolveMode = VK_RESOLVE_MODE_NONE;          // For depth-only
        depthStencilResolve.pDepthStencilResolveAttachment = &depthResolveAttachmentRef;

        VkSubpassDescription2 uiSubpass = {};
        uiSubpass.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
        uiSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        uiSubpass.colorAttachmentCount = 1;
        uiSubpass.pColorAttachments = &uiColorAttachmentRef;
        uiSubpass.pResolveAttachments = &uiResolveAttachmentRef;
        uiSubpass.pDepthStencilAttachment = &depthAttachmentRef;
        uiSubpass.pNext = &depthStencilResolve; // Attach resolve structure

        std::array<VkSubpassDependency2, 2> uiDependencies = {};
        uiDependencies[0].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
        uiDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        uiDependencies[0].dstSubpass = 0;
        uiDependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        uiDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        uiDependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        uiDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        uiDependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        uiDependencies[1].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
        uiDependencies[1].srcSubpass = 0;
        uiDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        uiDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        uiDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        uiDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        uiDependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        uiDependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

// Include depth resolve attachment in the attachments array
        std::array<VkAttachmentDescription2, 4> uiAttachments = {uiColorAttachment, depthAttachment,
                                                                 uiResolveAttachment, depthResolveAttachment};

        VkRenderPassCreateInfo2 uiRenderPassInfo = {};
        uiRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2;
        uiRenderPassInfo.attachmentCount = static_cast<uint32_t>(uiAttachments.size());
        uiRenderPassInfo.pAttachments = uiAttachments.data();
        uiRenderPassInfo.subpassCount = 1;
        uiRenderPassInfo.pSubpasses = &uiSubpass;
        uiRenderPassInfo.dependencyCount = static_cast<uint32_t>(uiDependencies.size());
        uiRenderPassInfo.pDependencies = uiDependencies.data();

        if (vkCreateRenderPass2(m_logicalDevice, &uiRenderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create UI render pass!");
        }
        VALIDATION_DEBUG_NAME(m_logicalDevice,
                              reinterpret_cast<uint64_t>(m_renderPass), VK_OBJECT_TYPE_RENDER_PASS,
                              createInfo->debugInfo + ":UIRenderPass");
    }
}