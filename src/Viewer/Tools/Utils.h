/**
 * @file: MultiSense-Viewer/include/Viewer/Tools/Utils.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2021-09-4, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_VIEWER_UTILS_H
#define MULTISENSE_VIEWER_UTILS_H

#include <fstream>
#include <vector>
#include <map>
#include <cassert>
#include <iostream>
#include <vulkan/vulkan_core.h>
#include <regex>
#include <algorithm>
#include <tiffio.h>

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <direct.h>
#include <windows.h>
#include <UserEnv.h>
#else

#include <sys/stat.h>

#endif

#include "Viewer/Tools/Macros.h"
#include "Viewer/Rendering/Core/RenderDefinitions.h"
#include "Viewer/Tools/Logger.h"

namespace Utils {
    DISABLE_WARNING_PUSH
    DISABLE_WARNING_UNREFERENCED_FUNCTION

    static std::filesystem::path getShadersPath() {
        return {"Resources/Assets/Shaders"};
    }

    static std::filesystem::path getFontsPath() {
        return {"Resources/Assets/Fonts"};
    }

    static std::filesystem::path getAssetsPath() {
        return {"Resources/Assets"};
    }

    static std::filesystem::path getTexturePath() {
        return {"Resources/Assets/Textures"};
    }

    static std::filesystem::path getModelsPath() {
        return {"Resources/Assets/Models"};
    }
    static std::filesystem::path getProjectsPath() {
        return {"Resources/Assets/Projects"};
    }

    static std::filesystem::path getEditorProjectPath() {
        return {"Resources/Assets/Projects/Editor.project"};
    }

    DISABLE_WARNING_POP

    /**@brief small utility function. Usage of this makes other code more readable */
    inline bool isInVector(const std::vector<std::string> &v, const std::string &str) {
        if (std::find(v.begin(), v.end(), str) != v.end())
            return true;
        return false;

    }

    /**@brief small utility function. Usage of this makes other code more readable */
    inline bool removeFromVector(std::vector<std::string> *v, const std::string &str) {
        auto it = std::find(v->begin(), v->end(), str);
        if (it == v->end())
            return false;
        v->erase(it);

        return true;
    }


    inline VkFormat
    findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat> &candidates, VkImageTiling tiling,
                        VkFormatFeatureFlags features) {
        for (VkFormat format: candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported m_Format!");
    }

    inline VkFormat findDepthFormat(VkPhysicalDevice physicalDevice) {
        return findSupportedFormat(physicalDevice,
                                   {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                                   VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }
    // Create an m_Image memory barrier for changing the layout of
    // an m_Image and put it into an active command buffer
    // See chapter 11.4 "Image Layout" for details

    inline void setImageLayout(
            VkCommandBuffer cmdbuffer,
            VkImage image,
            VkImageLayout oldImageLayout,
            VkImageLayout newImageLayout,
            VkImageSubresourceRange subresourceRange,
            VkPipelineStageFlags srcStageMask,
            VkPipelineStageFlags dstStageMask) {
        // Create an m_Image barrier object
        VkImageMemoryBarrier imageMemoryBarrier{};
        imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        imageMemoryBarrier.oldLayout = oldImageLayout;
        imageMemoryBarrier.newLayout = newImageLayout;
        imageMemoryBarrier.image = image;
        imageMemoryBarrier.subresourceRange = subresourceRange;

        // Source layouts (old)
        // Source access mask controls actions that have to be finished on the old layout
        // before it will be transitioned to the new layout
        switch (oldImageLayout) {
            case VK_IMAGE_LAYOUT_UNDEFINED:
                // Image layout is undefined (or does not matter)
                // Only valid as initial layout
                // No flags required, listed only for completeness
                imageMemoryBarrier.srcAccessMask = 0;
                break;

            case VK_IMAGE_LAYOUT_PREINITIALIZED:
                // Image is preinitialized
                // Only valid as initial layout for linear images, preserves memory contents
                // Make sure host writes have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                // Image is a color attachment
                // Make sure any writes to the color buffer have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                // Image is a depth/stencil attachment
                // Make sure any writes to the depth/stencil buffer have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                // Image is a transfer source
                // Make sure any reads from the m_Image have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                break;

            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                // Image is a transfer destination
                // Make sure any writes to the m_Image have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                // Image is read by a shader
                // Make sure any shader reads from the m_Image have been finished
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                break;
            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                imageMemoryBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
                break;
            default:
                // Other source layouts aren't firstUpdate (yet)
                Log::Logger::getInstance()->error("Source layouts arent updated yet. No Image transition completed");
                break;
        }

        // Target layouts (new)
        // Destination access mask controls the dependency for the new m_Image layout
        switch (newImageLayout) {
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                // Image will be used as a transfer destination
                // Make sure any writes to the m_Image have been finished
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                // Image will be used as a transfer source
                // Make sure any reads from the m_Image have been finished
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                break;

            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                // Image will be used as a color attachment
                // Make sure any writes to the color buffer have been finished
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                break;

            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                // Image layout will be used as a depth/stencil attachment
                // Make sure any writes to depth/stencil buffer have been finished
                imageMemoryBarrier.dstAccessMask =
                        imageMemoryBarrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                break;
            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                // Image layout will be used as a depth/stencil attachment
                // Make sure any writes to depth/stencil buffer have been finished
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
                break;

            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                // Image will be read in a shader (m_Sampler, input attachment)
                // Make sure any writes to the m_Image have been finished
                if (imageMemoryBarrier.srcAccessMask == 0) {
                    imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                }
                imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                break;
            default:
                Log::Logger::getInstance()->error("Missing Source layouts implementation. No Image transition completed");
                break;
        }

        // Put barrier inside setup command buffer
        vkCmdPipelineBarrier(
                cmdbuffer,
                srcStageMask,
                dstStageMask,
                0,
                0, nullptr,
                0, nullptr,
                1, &imageMemoryBarrier);
    }

    // Fixed sub resource on first mip level and layer
    inline void setImageLayout(
            VkCommandBuffer cmdbuffer,
            VkImage image,
            VkImageAspectFlags aspectMask,
            VkImageLayout oldImageLayout,
            VkImageLayout newImageLayout,
            VkPipelineStageFlags srcStageMask,
            VkPipelineStageFlags dstStageMask) {
        VkImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = aspectMask;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.layerCount = 1;
        setImageLayout(cmdbuffer, image, oldImageLayout, newImageLayout, subresourceRange, srcStageMask, dstStageMask);
    }

    inline void
    copyBufferToImage(VkCommandBuffer cmdBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height,
                      VkImageAspectFlagBits aspectFlagBits) {
        VkBufferImageCopy bufferCopyRegion = {};
        bufferCopyRegion.imageSubresource.aspectMask = aspectFlagBits;
        bufferCopyRegion.imageSubresource.layerCount = 1;
        bufferCopyRegion.imageSubresource.mipLevel = 0;
        bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
        bufferCopyRegion.imageExtent.width = width;
        bufferCopyRegion.imageExtent.height = height;
        bufferCopyRegion.imageExtent.depth = 1;

        vkCmdCopyBufferToImage(
                cmdBuffer,
                buffer,
                image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &bufferCopyRegion
        );

    }


    inline void loadShader(const char *fileName, const VkDevice &device, VkShaderModule *module) {
        std::ifstream is(fileName, std::ios::binary | std::ios::ate);
        if (is.is_open()) {
            std::streamsize size = is.tellg();
            is.seekg(0, std::ios::beg);
            std::vector<char> shaderCode(size);
            is.read(shaderCode.data(), size);
            is.close();
            assert(size > 0);
            VkShaderModuleCreateInfo moduleCreateInfo{};
            moduleCreateInfo.
                    sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            moduleCreateInfo.
                    codeSize = size;
            moduleCreateInfo.
                    pCode = reinterpret_cast<const uint32_t *>(shaderCode.data());
            VkResult res = vkCreateShaderModule(device, &moduleCreateInfo, nullptr,
                                                module);
            if (res != VK_SUCCESS)
                throw std::runtime_error("Failed to create shader module");
        } else {
            Log::Logger::getInstance()->error("Failed to open shader file {}", fileName);
        }
    }

    inline VkPipelineShaderStageCreateInfo
    loadShader(VkDevice device, std::string fileName, VkShaderStageFlagBits stage, VkShaderModule *module) {
        // Check if we have .spv extensions. If not then add it.
        std::size_t extension = fileName.find(".spv");
        if (extension == std::string::npos)
            fileName.append(".spv");
        Utils::loadShader((Utils::getShadersPath().append(fileName)).string().c_str(),
                          device, module);
        assert(module != VK_NULL_HANDLE);

        VkPipelineShaderStageCreateInfo shaderStage = {};
        shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStage.stage = stage;
        shaderStage.module = *module;
        shaderStage.pName = "main";


        Log::Logger::getInstance()->info("Loaded shader {} for stage {}", fileName, static_cast<uint32_t>(stage));
        return shaderStage;
    }

    template<typename T>
    size_t getIndexOf(const std::vector<T> &vecOfElements, const T &element) {
        // Find given element in vector
        auto it = std::find(vecOfElements.begin(), vecOfElements.end(), element);
        if (it != vecOfElements.end())
            return std::distance(vecOfElements.begin(), it);
        else
            return 0;
    }


    static bool stringToBool(const std::string &str) {
        std::string lowerStr;
        std::transform(str.begin(), str.end(), std::back_inserter(lowerStr),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (lowerStr == "true") {
            return true;
        } else if (lowerStr == "false") {
            return false;
        } else {
            // Handle invalid input string, e.g., throw an exception or return a default value
            throw std::invalid_argument("Invalid input string for boolean conversion");
        }
    }

    static std::string boolToString(bool value) {
        return value ? "true" : "false";
    }

/**
 * Returns the systemCache path for Windows/Ubuntu. If it doesn't exist it is created
 * @return path to cache folder
 */
    static std::filesystem::path getSystemCachePath() {
        // ON windows this file should be places in the user cache folder //
#ifdef WIN32
        const char *envVarName = "APPDATA";
        char *envVarValue = nullptr;
        size_t envVarValueSize = 0;
        _dupenv_s(&envVarValue, &envVarValueSize, envVarName);

        std::filesystem::path cachePath = envVarValue;
        std::filesystem::path multiSenseCachePath = cachePath / "multisense";
#else
        std::filesystem::path cachePath = std::getenv("HOME");
        cachePath /= ".cache";
        std::filesystem::path multiSenseCachePath = cachePath / "multisense";
#endif

        if (!std::filesystem::exists(multiSenseCachePath)) {
            std::error_code ec;
            if (std::filesystem::create_directories(multiSenseCachePath, ec)) {
                Log::Logger::getInstance((multiSenseCachePath / "logger.log").string())->info(
                        "Created cache directory {}", multiSenseCachePath.string());
            } else {
                Log::Logger::getInstance()->error("Failed to create cache directory {}. Error Code: {}",
                                                  multiSenseCachePath.string(), ec.value());
            }
        }
        return multiSenseCachePath;
    }

    static std::filesystem::path getSystemHomePath() {
#ifdef WIN32
        char path[MAX_PATH];
        HANDLE hToken;
        DWORD bufferSize = sizeof(path);

        if (OpenProcessToken(GetCurrentProcess(), TOKEN_READ, &hToken)) {
            if (!GetUserProfileDirectory(hToken, path, &bufferSize)) {
                CloseHandle(hToken);
                return ""; // Failed to get home directory
            }
            CloseHandle(hToken);
        } else {
            return ""; // Failed to open process token
        }

        return std::string(path);
#else
        const char *homeDir = getenv("HOME");
        if (homeDir) {
            return std::string(homeDir);
        } else {
            return getSystemCachePath();
        }
#endif
    }


    static std::filesystem::path getRuntimeConfigFilePath(){
        return getSystemCachePath() / "AppRuntimeConfig.yaml";
    }

    static bool checkRegexMatch(const std::string &str, const std::string &expression) {
        std::string lowered_str = str;
        std::transform(str.begin(), str.end(), lowered_str.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        std::regex pattern(expression, std::regex_constants::icase);  // icase for case-insensitive matching
        return std::regex_search(lowered_str, pattern);
    }


    static bool isLocalVersionLess(const std::string &localVersion, const std::string &remoteVersion) {
        int localMajor, localMinor, localPatch;
        int remoteMajor, remoteMinor, remotePatch;

        // Parse local version
        std::istringstream localStream(localVersion);
        localStream >> localMajor;
        localStream.ignore(1, '.'); // Skip the dot
        localStream >> localMinor;
        localStream.ignore(1, '-'); // Skip the dot
        localStream >> localPatch;

        // Parse remote version
        std::istringstream remoteStream(remoteVersion);
        remoteStream >> remoteMajor;
        remoteStream.ignore(1, '.'); // Skip the dot
        remoteStream >> remoteMinor;
        remoteStream.ignore(1, '-'); // Skip the dot
        remoteStream >> remotePatch;


        // Compare
        if (localMajor < remoteMajor) {
            return true;
        }
        if (localMajor == remoteMajor && localMinor < remoteMinor) {
            return true;
        }
        if (localMajor == remoteMajor && localMinor == remoteMinor && localPatch < remotePatch) {
            return true;
        }
        return false;
    }


    static void
    writeTIFFImage(const std::filesystem::path &fileName, uint32_t width, uint32_t height, float *data) {
        int samplesPerPixel = 1;
        TIFF *out = TIFFOpen(fileName.string().c_str(), "w");
        if (!out) {
            throw std::runtime_error("Failed to open TIFF file for writing.");
        }
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);  // set the width of the image
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);    // set the height of the image
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, samplesPerPixel);   // set number of channels per pixel
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 32);    // set the size of the channels
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
        //   Some other essential fields to set that you do not have to understand for now.
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);

        // We set the strip size of the file to be size of one row of pixels
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, 0));
        //Now writing image to the file one strip at a time
        uint32_t row;
        for (row = 0; row < height; row++) {
            // Write each row as a strip
            if (TIFFWriteScanline(out, &data[row * width], row, 0) < 0) {
                TIFFClose(out);
                if (std::filesystem::exists(fileName)) {
                    std::filesystem::remove(fileName);
                }
            }
        }
        TIFFClose(out);
    }

    // TODO dont know where to put this utility function
    static uint32_t getBytesPerPixelFromVkFormat(VkFormat format) {
        switch (format) {
            case VK_FORMAT_R8G8B8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB:
            case VK_FORMAT_B8G8R8A8_UNORM:
            case VK_FORMAT_B8G8R8A8_SRGB:
            case VK_FORMAT_R8G8B8A8_UINT:
                return 4; // 4 bytes per pixel (8 bits per channel, 4 channels)
            case VK_FORMAT_R8G8B8_UNORM:
            case VK_FORMAT_R8G8B8_SRGB:
                return 3; // 3 bytes per pixel (8 bits per channel, 3 channels)
            case VK_FORMAT_R8_UNORM:
                return 1; // 1 byte per pixel (8 bits per channel)
            case VK_FORMAT_R16G16B16A16_SFLOAT:
                return 8; // 8 bytes per pixel (16 bits per channel, 4 channels)
            case VK_FORMAT_R32G32B32A32_SFLOAT:
                return 16; // 16 bytes per pixel (32 bits per channel, 4 channels)
            case VK_FORMAT_D32_SFLOAT:
            case VK_FORMAT_R32_SFLOAT:
            case VK_FORMAT_R32_UINT:
            case VK_FORMAT_D24_UNORM_S8_UINT:
                return 4; // Approximation: 24 bits depth + 8 bits stencil = 4 bytes
            default:
                throw std::runtime_error("Unsupported VkFormat for size calculation");
        }
    }
}

#endif //MULTISENSE_VIEWER_UTILS_H
