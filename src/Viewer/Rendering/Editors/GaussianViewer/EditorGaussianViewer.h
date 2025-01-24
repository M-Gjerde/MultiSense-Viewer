//
// Created by magnus on 8/15/24.
//

#ifndef MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H
#define MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H

#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Scenes/Scene.h"
#include "Viewer/Tools/SyclDeviceSelector.h"
#include "Viewer/Rendering/Core/DescriptorSetManager.h"
#include "Viewer/Rendering/Core/PipelineManager.h"
#include "Viewer/Rendering/Core/DescriptorRegistry.h"

#include "Viewer/Rendering/RenderResources/2DGS/SYCLGaussian2D.h"
#include "Viewer/Rendering/RenderResources/3DGS/SYCLGaussian3D.h"
#include "Viewer/Rendering/Editors/RenderCommand.h"

namespace VkRender {


    class EditorGaussianViewer : public Editor {
    public:
        EditorGaussianViewer() = delete;

        explicit EditorGaussianViewer(EditorCreateInfo &createInfo, UUID uuid = UUID());


        void onRender(CommandBuffer &drawCmdBuffers) override;

        void onUpdate() override;

        ~EditorGaussianViewer() override {

        }

        void onMouseMove(const MouseButtons &mouse) override;

        void onMouseScroll(float change) override;
        void onEditorResize() override;
        void onSceneLoad(std::shared_ptr<Scene> scene) override;

        void onKeyCallback(const Input& input) override;

    private:
        std::shared_ptr<Camera> m_editorCamera;
        std::shared_ptr<Scene> m_activeScene;
        CameraComponent* m_lastActiveCamera = nullptr;

        SyclDeviceSelector m_deviceSelector = SyclDeviceSelector(SyclDeviceSelector::DeviceType::GPU);
        SYCLGaussian2D gaussianRenderer2D;
        SYCLGaussian3D gaussianRenderer3D;

        std::vector<std::unique_ptr<Buffer>> m_shaderSelectionBuffer;

        std::shared_ptr<VulkanTexture2D> m_colorTexture;
        PipelineManager m_pipelineManager;
        DescriptorRegistry m_descriptorRegistry;
        std::shared_ptr<MeshInstance> m_meshInstances;

        void bindResourcesAndDraw(const CommandBuffer &commandBuffer, RenderCommand &command);

        void collectRenderCommands(
                std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>> &renderGroups,
                uint32_t frameIndex);

        std::shared_ptr<MeshInstance> setupMesh();
    };
}

#endif //MULTISENSE_VIEWER_EDITORGAUSSIANVIEWER_H
