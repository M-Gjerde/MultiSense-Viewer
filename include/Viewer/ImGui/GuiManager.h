/**
 * @file: MultiSense-Viewer/include/Viewer/ImGui/GuiManager.h
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
 *   2022-4-19, mgjerde@carnegierobotics.com, Created file.
 **/

#ifndef MULTISENSE_GUIMANAGER_H
#define MULTISENSE_GUIMANAGER_H


#include <type_traits>
#include <memory>
#include <vector>
#include <functional>
#include <glm/vec2.hpp>

#include "Viewer/Tools/Macros.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_ALL
#include <imgui_internal.h>
DISABLE_WARNING_POP

#include "Viewer/Tools/Utils.h"
#include "Viewer/Tools/Populate.h"
#include "Viewer/Core/Texture.h"
#include "Viewer/Core/VulkanDevice.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/ImGui/Layer.h"
#include "Viewer/ImGui/Widgets.h"
#include "Viewer/ImGui/LayerFactory.h"

namespace VkRender {
    class GuiManager {
    public:
        GuiObjectHandles handles{};

        GuiManager(VulkanDevice *vulkanDevice, VkRenderPass const &renderPass, const uint32_t &width,
                   const uint32_t &height, VkSampleCountFlagBits msaaSamples, uint32_t imageCount);

        ~GuiManager() {
            Log::Logger::getInstance()->info("Saving ImGui file: {}",
                                             (Utils::getSystemCachePath() / "imgui.ini").string().c_str());
            ImGui::SaveIniSettingsToDisk((Utils::getSystemCachePath() / "imgui.ini").string().c_str());
            for (const auto &layerStack: m_LayerStack)
                layerStack->onDetach();
            vkDestroyPipeline(device->m_LogicalDevice, pipeline, nullptr);
            vkDestroyPipelineCache(device->m_LogicalDevice, pipelineCache, nullptr);
            vkDestroyPipelineLayout(device->m_LogicalDevice, pipelineLayout, nullptr);
            vkDestroyDescriptorPool(device->m_LogicalDevice, descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(device->m_LogicalDevice, descriptorSetLayout, nullptr);
            for (auto &shaderModule: shaderModules) {
                vkDestroyShaderModule(device->m_LogicalDevice, shaderModule, nullptr);
            }

        };

        /**@brief Update function called from renderer. Function calls each layer in order to generate buffers for draw commands*/
        void update(bool updateFrameGraph, float frameTimer, uint32_t width, uint32_t height, const Input *pInput);

        /**@brief setup function called once vulkan renderer is setup. Function calls each layer in order to generate buffers for draw commands*/
        void setup(const uint32_t &width, const uint32_t &height, VkRenderPass const &renderPass, VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT);

        /**@brief Draw command called once per command buffer recording*/
        void drawFrame(VkCommandBuffer commandBuffer, uint32_t imageIndex);

        /**@brief ReCreate buffers if they have changed in size*/
        bool updateBuffers(uint32_t imageIndex);

        /** @brief Push a new GUI Layer to the layer stack. Layers are drawn in the order they are pushed. **/
        template<typename T>
        void pushLayer() {
            static_assert(std::is_base_of<Layer, T>::value, "Pushed type does not inherit Layer class!");
            m_LayerStack.emplace_back(std::make_shared<T>())->onAttach();
        }

        void pushLayer(const std::string& layerName) {
            auto layer = LayerFactory::createLayer(layerName);
            if(layer) {
                m_LayerStack.emplace_back(layer)->onAttach();
            } else {
                // Handle unknown layer case, e.g., throw an exception or log an error
            }
        }

    private:
        // UI params are set via push constants
        struct PushConstBlock {
            glm::vec2 scale;
            glm::vec2 translate;
        } pushConstBlock{};

        // GuiManager framework
        std::vector<std::shared_ptr<Layer>> m_LayerStack{};

        // Textures
        std::vector<Texture2D> iconTextures;
        std::vector<Texture2D> fontTexture;
        std::unique_ptr<Texture2D> gifTexture[99];

        // Vulkan resources for rendering the UI
        //Buffer vertexBuffer;
        //Buffer indexBuffer;
        // Vulkan resources for rendering the UI
        std::vector<Buffer> vertexBuffer;
        std::vector<Buffer> indexBuffer;

        std::vector<int32_t> vertexCount{};
        std::vector<int32_t> indexCount{};

        uint32_t updatedBufferIndex = 0;
        //int32_t vertexCount = 0;
        //int32_t indexCount = 0;

        VkPipelineCache pipelineCache{};
        VkPipelineLayout pipelineLayout{};
        VkPipeline pipeline{};
        VkDescriptorPool descriptorPool{};
        VkDescriptorSetLayout descriptorSetLayout{};
        std::vector<VkDescriptorSet> fontDescriptors{};
        std::vector<VkDescriptorSet> imageIconDescriptors{};
        std::vector<VkDescriptorSet> gifImageDescriptors{};

        std::vector<VkShaderModule> shaderModules{};
        VulkanDevice *device = nullptr;
        std::shared_ptr<VkRender::ThreadPool> pool;
        std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float>> saveSettingsTimer;

        // Initialization functions
        void initializeFonts();

        ImFont *loadFontFromFileName(std::string file, float fontSize);

        void loadImGuiTextureFromFileName(const std::string &file, uint32_t i);

        void loadAnimatedGif(const std::string &file);
    };
}
#endif //MULTISENSE_GUIMANAGER_H
