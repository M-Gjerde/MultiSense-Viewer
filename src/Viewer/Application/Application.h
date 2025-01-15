/**
 * @file: MultiSense-Viewer/include/Viewer/Rendering/Renderer.h
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
 *   2021-9-4, mgjerde@carnegierobotics.com, Created file.
 **/
#ifndef MULTISENSE_RENDERER_H
#define MULTISENSE_RENDERER_H

#include "pch.h"

#include <GLFW/glfw3.h>
#include <multisense_viewer/external/entt/include/entt/entt.hpp>


#include "Viewer/Application/Project.h"
#include "Viewer/Application/UsageMonitor.h"
#include "Viewer/Rendering/Core/VulkanRenderer.h"
#include "Viewer/Rendering/Core/UUID.h"
#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Rendering/Editors/EditorFactory.h"
#include "Viewer/Rendering/Editors/SceneRenderer.h"
#include "Viewer/Scenes/Scene.h"
#include "Viewer/Modules/MultiSense/MultiSenseRendererBridge.h"

namespace VkRender {
    class Entity;

    class Application : VulkanRenderer {
    public:
        /**
         * @brief Default constructor for renderer
         * @param title Title of application
         */
        explicit Application(const std::string &title);

        ~Application() override = default;

        /**
         * @brief runs the renderer loop
         */
        void run() {
            VulkanRenderer::renderLoop();
        }

        /**
         * @brief cleans up resources on application exist
         */
        void cleanUp();

        void closeApplication() override {
            VulkanRenderer::closeApplication();
        }

        std::shared_ptr<Scene> newScene();

        std::unique_ptr<Editor> createEditor(EditorCreateInfo &createInfo);

        std::unique_ptr<Editor> createEditorWithUUID(UUID uuid, EditorCreateInfo &createInfo);

        std::vector<SwapChainBuffer> &swapChainBuffers() { return swapchain->buffers; }

        VmaAllocator &allocator() { return m_allocator; }

        VulkanDevice &vkDevice() { return *m_vulkanDevice; }

        std::shared_ptr<MultiSense::MultiSenseRendererBridge> multiSense() { return m_multiSense; }
        std::shared_ptr<UsageMonitor> usageMonitor() { return m_usageMonitor; }

        // TODO we should collect per frame info like this somewhere
        float deltaTime() { return frameTimer; }

        SceneRenderer *getSceneRendererByUUID(const UUID &uuid);
        SceneRenderer* getOrAddSceneRendererByUUID(const UUID &uuid,const EditorCreateInfo& ownerCreateInfo);
        SceneRenderer * addSceneRendererWithUUID(const UUID &uuid, const EditorCreateInfo& ownerCreateInfo);

        ImGuiContext *getMainUIContext() { return m_mainEditor->guiContext(); }
        Entity& getSelectedEntity() {return m_selectedEntity;}
        void setSelectedEntity(Entity entity){m_selectedEntity = entity;}
        Editor3DViewport* getViewport();

        uint32_t currentFrameIndex() { return currentFrame; }
        uint32_t getFrameId() { return frameID; }

        void loadProject(const Project& project);
        Project getCurrentProject();

        bool isCurrentProject(std::string projectName);

        std::shared_ptr<Scene> activeScene();

        std::function<VkResult(VkDevice, const VkDebugUtilsObjectNameInfoEXT *)> getDebugUtilsObjectNameFunction() {
            return m_setDebugUtilsObjectNameEXT;
        }


    private:
        void keyboardCallback(GLFWwindow *window, int key, int scancode, int action, int mods) override;

        void onRender() override;

        void updateUniformBuffers() override;

        void windowResized(int32_t dx, int32_t dy, double widthScale, double heightScale) override;

        void addDeviceFeatures() override;

        void mouseMoved(float x, float y) override;

        void mouseScroll(float change) override;

        void postRenderActions() override;

        void onFileDrop(const std::filesystem::path &path) override;

        void onCharInput(unsigned int codepoint) override;

        std::vector<std::unique_ptr<Editor> > m_editors;

        std::shared_ptr<Scene> m_activeScene;
        std::string m_selectedCameraTag = "Default Camera";
        std::unique_ptr<EditorFactory> m_editorFactory;

        std::unique_ptr<Editor> m_mainEditor;
        std::unordered_map<UUID, std::shared_ptr<SceneRenderer> > m_sceneRenderers;
        std::shared_ptr<GuiAssets> m_guiResources;

        std::shared_ptr<UsageMonitor> m_usageMonitor;
        Entity m_selectedEntity;

        std::shared_ptr<MultiSense::MultiSenseRendererBridge> m_multiSense;

        friend class ApplicationConfig;
        friend class Scene;

        void updateEditors();

        EditorCreateInfo getNewEditorCreateInfo(std::unique_ptr<Editor> &editor);

        void resizeEditors(bool anyCornerClicked);

        void splitEditor(uint32_t splitEditorIndex);

        void handleEditorResize();

        void recreateEditor(std::unique_ptr<Editor> &editor, EditorCreateInfo &createInfo);

        void mergeEditors(const std::array<UUID, 2> &mergeEditorIndices);

        std::unique_ptr<Editor> &findEditorByUUID(const UUID &uuid);

    };
}


#endif // MultiS
