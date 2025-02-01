//
// Created by magnus on 1/15/25.
//

#ifndef MULTISENSE_VIEWER_EDITORPATHTRACER_H
#define MULTISENSE_VIEWER_EDITORPATHTRACER_H

#include "Viewer/Rendering/Core/DescriptorRegistry.h"
#include "Viewer/Rendering/Core/PipelineManager.h"

#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Scenes/Scene.h"
#include "Viewer/Rendering/Core/VulkanTexture.h"
#include "Viewer/Rendering/RenderResources/GraphicsPipeline2D.h"

#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer.h"
#include "Viewer/Rendering/Editors/RenderCommand.h"
#include "Viewer/Rendering/Editors/ArcballCamera.h"

namespace VkRender {


    class EditorPathTracer : public Editor {
    public:
        EditorPathTracer() = delete;
        ~EditorPathTracer() {
            // Make sure path tracer gets destroyed before the SYCL device
            m_pathTracer.reset();
            m_syclDevice.reset();
        }

        explicit EditorPathTracer(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;
        void collectRenderCommands(
                std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>>& renderGroups,
                uint32_t frameIndex);
        void bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand& command);
        void saveImage();

        void onSceneLoad(std::shared_ptr<Scene> scene) override;

        void onMouseMove(const MouseButtons &mouse) override;
        void onPipelineReload() override;
        void denoiseImage(float* singleChannelImage, uint32_t width, uint32_t height, std::vector<float>& output);

        void onFileDrop(const std::filesystem::path &path) override;

        void onMouseScroll(float change) override;
        void onKeyCallback(const Input& input) override;

        void onEditorResize() override;

    private:
        std::vector<std::unique_ptr<Buffer>> m_shaderSelectionBuffer;
        PipelineManager m_pipelineManager;
        DescriptorRegistry m_descriptorRegistry;
        std::shared_ptr<MeshInstance> m_meshInstances;
        std::shared_ptr<VulkanTexture2D> m_colorTexture;

        std::unique_ptr<PathTracer::PhotonTracer> m_pathTracer;
        std::unique_ptr<SyclDeviceSelector> m_syclDevice;

        std::shared_ptr<ArcballCamera> m_editorCamera;
        CameraComponent* m_lastActiveCamera = nullptr;
        bool m_wasSceneCameraActive = false;
        bool m_movedCamera = false;
        std::shared_ptr<Scene> m_activeScene;
        void updateActiveCamera();
    };
}
#endif //MULTISENSE_VIEWER_EDITORPATHTRACER_H
