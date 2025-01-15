//
// Created by magnus on 7/16/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
#define MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H

#include <multisense_viewer/src/Viewer/Rendering/Core/DescriptorSetManager.h>
#include <multisense_viewer/src/Viewer/Rendering/Core/DescriptorRegistry.h>
#include <multisense_viewer/src/Viewer/Rendering/Editors/Video/VideoPlaybackSystem.h>
#include <multisense_viewer/src/Viewer/Rendering/RenderResources/GraphicsPipeline2D.h>
#include <multisense_viewer/src/Viewer/Rendering/Editors/PinholeCamera.h>
#include <Viewer/Rendering/Editors/ArcballCamera.h>

#include "Viewer/Rendering/Core/PipelineManager.h"
#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Rendering/RenderResources/DefaultGraphicsPipeline.h"
#include "Viewer/Rendering/Editors/SceneRenderer.h"

#include "Viewer/Rendering/Editors/RenderCommand.h"

namespace VkRender {
    class Editor3DViewport : public Editor {
    public:
        Editor3DViewport() = delete;

        explicit Editor3DViewport(EditorCreateInfo& createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer& drawCmdBuffers) override;


        void onSceneLoad(std::shared_ptr<Scene> scene) override;
        void onEditorResize() override;

        void onMouseMove(const MouseButtons& mouse) override;

        void onMouseScroll(float change) override;
        void onKeyCallback(const Input& input) override;
        std::shared_ptr<BaseCamera> getCamera() { return m_editorCamera; }

        void onRenderSettingsChanged();

        void bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand& command);

        void collectRenderCommands(
            std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>>& renderGroups,
            uint32_t
            frameIndex);

    private:

        std::shared_ptr<ArcballCamera> m_editorCamera;


        //std::shared_ptr<ArcballCamera> m_editorCamera;
        CameraComponent* m_lastActiveCamera = nullptr;
        bool m_wasSceneCameraActive = false;
        std::shared_ptr<Scene> m_activeScene;

        SceneRenderer* m_sceneRenderer;
        std::shared_ptr<VulkanTexture2D> m_colorTexture;

        std::vector<std::unique_ptr<Buffer>> m_shaderSelectionBuffer;
        // Quad and descriptor setup
        PipelineManager m_pipelineManager;
        DescriptorRegistry m_descriptorRegistry;
        std::shared_ptr<MeshInstance> m_meshInstances;

        void updateActiveCamera();
    };
}

#endif //MULTISENSE_VIEWER_EDITOR3DVIEWPORT_H
