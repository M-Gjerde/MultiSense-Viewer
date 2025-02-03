//
// Created by mgjer on 04/08/2024.
//

#include "Viewer/Rendering/Editors/Video/VideoPlaybackSystem.h"

#include <Viewer/Rendering/Editors/CommonEditorFunctions.h>

#include "Viewer/Application/Application.h"


namespace VkRender {
    VideoPlaybackSystem::VideoPlaybackSystem(Application *context) : m_context(context) {
    }

    size_t VideoPlaybackSystem::addVideoSource(
            const std::filesystem::path &folderPath,
            VideoSource::ImageType type,
            float fps,
            bool loop, UUID ownerUUID) {
        auto videoSource = std::make_unique<VideoSource>(folderPath, type, fps, loop);

        // Determine the Vulkan format based on the image type
        VkFormat format;
        switch (type) {
            case VideoSource::ImageType::RGB:
                format = VK_FORMAT_R8G8B8A8_UNORM;
                break;
            case VideoSource::ImageType::Grayscale:
                format = VK_FORMAT_R8_UNORM;
                break;
            case VideoSource::ImageType::Disparity16Bit:
                format = VK_FORMAT_R16_UNORM;
                break;
            default:
                format = VK_FORMAT_R8G8B8A8_UNORM;
                break;
        }

        // Wait for the loading thread to load the first frame
        {
            std::unique_lock<std::mutex> lock(videoSource->bufferMutex);
            videoSource->bufferCV.wait(lock, [&videoSource]() {
                return !videoSource->frameBuffer.empty() || videoSource->stopLoading;
            });
        }

        if (!videoSource->frameBuffer.empty()) {
            // Get the first frame to determine dimensions
            const auto &frameData = videoSource->frameBuffer.front();

            // Create texture with correct dimensions
            videoSource->texture = createEmptyTexture(format, frameData.width, frameData.height);
        } else {
            // Handle error: no frames loaded
            throw std::runtime_error("Failed to load first frame");
        }

        // Add the unique_ptr to the vector
        m_videoSources[ownerUUID] = std::move(videoSource);
        return m_videoSources.size() - 1; // Return the index of the added video source
    }

    void VideoPlaybackSystem::update(float deltaTime) {
        for (auto &videoPtr: m_videoSources) {
            auto &video = *videoPtr.second;

            video.timeAccumulator += deltaTime;

            // Check if it's time to advance to the next frame
            if (video.timeAccumulator >= video.frameDuration) {
                video.timeAccumulator -= video.frameDuration;

                // Get the next frame from the buffer
                VideoSource::FrameData frameData;

                {
                    std::unique_lock<std::mutex> lock(video.bufferMutex);

                    if (video.frameBuffer.empty()) {
                        // Buffer is empty, cannot proceed
                        continue;
                    }

                    // Get the frame data
                    frameData = std::move(video.frameBuffer.front());
                    video.frameBuffer.pop_front();

                    // Notify the loading thread that space is available
                    video.bufferCV.notify_one();
                }

                // Update the texture with the new frame data
                VkDeviceSize imageSize = frameData.pixels.size();

                // Update the texture
                video.texture->loadImage(frameData.pixels.data(), imageSize);

                // Update currentFrameIndex
                video.currentFrameIndex = frameData.frameIndex;
            }
        }
    }

    std::shared_ptr<VulkanTexture2D> VideoPlaybackSystem::getTexture(UUID id) {
        if (m_videoSources.contains(id))
            return m_videoSources[id]->texture;

        return nullptr;
    }

    void VideoPlaybackSystem::resetAllSourcesPlayback() {
        for (auto &videoPtr: m_videoSources) {
            videoPtr.second->resetPlayback();
        }
    }

    std::shared_ptr<VulkanTexture2D> VideoPlaybackSystem::createEmptyTexture(
            VkFormat format, uint32_t width, uint32_t height) const {
        // Create an empty VulkanTexture2D with the specified format and dimensions

        VkImageCreateInfo imageCI = Populate::imageCreateInfo();
        imageCI.imageType = VK_IMAGE_TYPE_2D;
        imageCI.format = format;
        imageCI.extent = {width, height, 1};
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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

        VulkanImageCreateInfo
                vulkanImageCreateInfo(m_context->vkDevice(), m_context->allocator(), imageCI, imageViewCI);
        vulkanImageCreateInfo.debugInfo = "VideoPlaybackSystem:Empty:Texture";
        vulkanImageCreateInfo.setLayout = true;
        vulkanImageCreateInfo.srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkanImageCreateInfo.dstLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vulkanImageCreateInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        VulkanTexture2DCreateInfo textureCreateInfo(m_context->vkDevice());
        textureCreateInfo.image = std::make_shared<VulkanImage>(vulkanImageCreateInfo);

        return std::make_shared<VulkanTexture2D>(textureCreateInfo);
    }
}
