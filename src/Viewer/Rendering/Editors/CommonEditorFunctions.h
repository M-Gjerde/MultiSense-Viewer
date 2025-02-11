//
// Created by mgjer on 12/10/2024.
//

#ifndef COMMONEDITORFUNCTIONS_H
#define COMMONEDITORFUNCTIONS_H

#include "Viewer/Application/Application.h"
#include "Viewer/Rendering/ImGui/LayerUtils.h"

#include <stb_image.h>

namespace VkRender::EditorUtils {

    static std::shared_ptr<VulkanTexture2D> createEmptyTexture(uint32_t width, uint32_t height, VkFormat format,
                                                               Application* context, VmaMemoryUsage usageAllocation = VMA_MEMORY_USAGE_GPU_ONLY, bool setValues = false) {
        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = format;
        imageCI.extent = {width, height, 1};
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
        imageViewCI.format = format;
        imageViewCI.subresourceRange.baseMipLevel = 0;
        imageViewCI.subresourceRange.levelCount = 1;
        imageViewCI.subresourceRange.baseArrayLayer = 0;
        imageViewCI.subresourceRange.layerCount = 1;
        imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

        VulkanImageCreateInfo vulkanImageCreateInfo(context->vkDevice(), context->allocator(), imageCI,
                                                    imageViewCI);
        vulkanImageCreateInfo.setLayout = true;
        vulkanImageCreateInfo.srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkanImageCreateInfo.dstLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vulkanImageCreateInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vulkanImageCreateInfo.debugInfo = "Emtpy texture:" + std::to_string(width) + "x" + std::to_string(height);
        vulkanImageCreateInfo.usage = usageAllocation;
        VulkanTexture2DCreateInfo textureCreateInfo(context->vkDevice());

        textureCreateInfo.image = std::make_shared<VulkanImage>(vulkanImageCreateInfo);
        auto texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);

        if (setValues) {
            uint32_t imageSize = width * height * Utils::getBytesPerPixelFromVkFormat(format);

            void* imageMemory = malloc(imageSize);
            std::fill(static_cast<uint8_t*>(imageMemory), static_cast<uint8_t*>(imageMemory) + imageSize, 200);
            texture->loadImage(imageMemory, imageSize);
            free(imageMemory);
        }

        return texture;
    }

    static std::shared_ptr<VulkanTexture2D>
    createTextureFromFile(std::filesystem::path filePath, Application* context) {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(filePath.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4; // Assuming STBI_rgb_alpha gives us 4 channels per pixel
        if (!pixels) {
            if (!pixels) {
                // load empty texture
                Log::Logger::getInstance()->error("Failed to load texture image: {}. Reverting to empty",
                                                  filePath.string());
                pixels = stbi_load((Utils::getTexturePath() / "moon.png").string().c_str(), &texWidth, &texHeight,
                                   &texChannels, STBI_rgb_alpha);
                imageSize = static_cast<VkDeviceSize>(texWidth * texHeight * texChannels);
                if (!pixels) {
                    throw std::runtime_error("Failed to load backup texture image");
                }
            }
        }

        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageCI.extent = {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1};
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

        VulkanImageCreateInfo vulkanImageCreateInfo(context->vkDevice(), context->allocator(), imageCI,
                                                    imageViewCI);
        vulkanImageCreateInfo.debugInfo = "Color texture: Image Editor";
        VulkanTexture2DCreateInfo textureCreateInfo(context->vkDevice());
        textureCreateInfo.image = std::make_shared<VulkanImage>(vulkanImageCreateInfo);
        auto texture = std::make_shared<VulkanTexture2D>(textureCreateInfo);

        // Copy data to texturere
        texture->loadImage(pixels, imageSize);
        // Free the image data
        stbi_image_free(pixels);
        return texture;
    }

    static void openImportFileDialog(const std::string& fileDescription,
                                     const std::vector<std::string>& type,
                                     LayerUtils::FileTypeLoadFlow flow,
                                     std::future<LayerUtils::LoadFileInfo>* loadFileFuture,
                                     std::filesystem::path openLocation = Utils::getSystemHomePath()) {
        if (!loadFileFuture->valid()) {
            auto& opts = ApplicationConfig::getInstance().getUserSetting();
            if (!opts.lastOpenedImportModelFolderPath.empty()) {
                openLocation = opts.lastOpenedImportModelFolderPath.remove_filename().string();
            }
            *loadFileFuture = std::async(VkRender::LayerUtils::selectFile, "Select " + fileDescription + " file",
                                         type, openLocation, flow);
        }
    }

    static void openImportFolderDialog(const std::string& dialogName,const std::filesystem::path& openLocation, LayerUtils::FileTypeLoadFlow flow,
                                       std::future<LayerUtils::LoadFileInfo>* loadFolderFuture) {
        if (!loadFolderFuture->valid()) {
            *loadFolderFuture = std::async(VkRender::LayerUtils::selectFolder, dialogName, openLocation, flow);
        }
    }

    static void saveFileDialog(const std::string& fileDescription,
                               const std::vector<std::string>& type,
                               LayerUtils::FileTypeLoadFlow flow,
                               std::future<LayerUtils::LoadFileInfo>* loadFolderFuture,
                               std::filesystem::path openLocation = Utils::getSystemHomePath()) {
        if (!loadFolderFuture->valid()) {
            auto& opts = ApplicationConfig::getInstance().getUserSetting();
            if (!opts.lastOpenedImportModelFolderPath.empty()) {
                openLocation = opts.lastOpenedImportModelFolderPath.remove_filename().string();
            }

            *loadFolderFuture = std::async(VkRender::LayerUtils::saveFile, "Save as", type, openLocation, flow);
        }
    }

    static std::shared_ptr<MeshInstance> setupMesh(Application* ctx, float scaleX = 1.0f, float scaleY = 1.0f) {
        std::vector<VkRender::ImageVertex> vertices = {
            // Bottom-left corner
            {glm::vec2{-1.0f * scaleX, -1.0f * scaleY}, glm::vec2{0.0f, 0.0f}},
            // Bottom-right corner
            {glm::vec2{1.0f * scaleX, -1.0f * scaleY}, glm::vec2{1.0f, 0.0f}},
            // Top-right corner
            {glm::vec2{1.0f * scaleX, 1.0f * scaleY}, glm::vec2{1.0f, 1.0f}},
            // Top-left corner
            {glm::vec2{-1.0f * scaleX, 1.0f * scaleY}, glm::vec2{0.0f, 1.0f}}
        };

        // Define the indices for two triangles that make up the quad
        std::vector<uint32_t> indices = {
            0, 1, 2, // First triangle (bottom-left to top-right)
            2, 3, 0 // Second triangle (top-right to bottom-left)
        };

        auto meshInstance = std::make_shared<MeshInstance>();

        meshInstance->vertexCount = vertices.size();
        meshInstance->indexCount = indices.size();
        VkDeviceSize vertexBufferSize = vertices.size() * sizeof(VkRender::ImageVertex);
        VkDeviceSize indexBufferSize = indices.size() * sizeof(uint32_t);

        struct StagingBuffer {
            VkBuffer buffer;
            VkDeviceMemory memory;
        } vertexStaging{}, indexStaging{};

        // Create staging buffers
        // Vertex m_DataPtr
        CHECK_RESULT(ctx->vkDevice().createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            vertexBufferSize,
            &vertexStaging.buffer,
            &vertexStaging.memory,
            vertices.data()))
        // Index m_DataPtr
        if (indexBufferSize > 0) {
            CHECK_RESULT(ctx->vkDevice().createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indexBufferSize,
                &indexStaging.buffer,
                &indexStaging.memory,
                indices.data()))
        }

        CHECK_RESULT(ctx->vkDevice().createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            meshInstance->vertexBuffer, vertexBufferSize, nullptr, "SceneRenderer:InitializeMesh:Vertex",
            ctx->getDebugUtilsObjectNameFunction()));
        // Index buffer
        if (indexBufferSize > 0) {
            CHECK_RESULT(ctx->vkDevice().createBuffer(
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                meshInstance->indexBuffer,
                indexBufferSize, nullptr, "SceneRenderer:InitializeMesh:Index",
                ctx->getDebugUtilsObjectNameFunction()));
        }

        // Copy from staging buffers
        VkCommandBuffer copyCmd = ctx->vkDevice().createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkBufferCopy copyRegion = {};
        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, meshInstance->vertexBuffer->m_buffer, 1, &copyRegion);
        if (indexBufferSize > 0) {
            copyRegion.size = indexBufferSize;
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, meshInstance->indexBuffer->m_buffer, 1, &copyRegion);
        }
        ctx->vkDevice().flushCommandBuffer(copyCmd, ctx->vkDevice().m_TransferQueue, true);

        vkDestroyBuffer(ctx->vkDevice().m_LogicalDevice, vertexStaging.buffer, nullptr);
        vkFreeMemory(ctx->vkDevice().m_LogicalDevice, vertexStaging.memory, nullptr);

        if (indexBufferSize > 0) {
            vkDestroyBuffer(ctx->vkDevice().m_LogicalDevice, indexStaging.buffer, nullptr);
            vkFreeMemory(ctx->vkDevice().m_LogicalDevice, indexStaging.memory, nullptr);
        }

        return meshInstance;
    }
}

#endif //COMMONEDITORFUNCTIONS_H
