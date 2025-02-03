//
// Created by magnus on 1/15/25.
//

#ifndef MULTISENSE_VIEWER_EDITOR_DIFFRENTIABLE_RENDERER
#define MULTISENSE_VIEWER_EDITOR_DIFFRENTIABLE_RENDERER

#include "Viewer/Rendering/Core/DescriptorRegistry.h"
#include "Viewer/Rendering/Core/PipelineManager.h"

#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Scenes/Scene.h"
#include "Viewer/Rendering/Core/VulkanTexture.h"
#include "Viewer/Rendering/RenderResources/GraphicsPipeline2D.h"

#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer.h"
#include "Viewer/Rendering/Editors/RenderCommand.h"
#include "Viewer/Rendering/Editors/ArcballCamera.h"

#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildModule.h"


namespace VkRender {


    class EditorDifferentiableRenderer : public Editor {
    public:
        EditorDifferentiableRenderer() = delete;
        ~EditorDifferentiableRenderer() {
            // Make sure path tracer gets destroyed before the SYCL device
            m_pathTracer.reset();
            m_syclDevice.reset();
        }

        explicit EditorDifferentiableRenderer(EditorCreateInfo &createInfo, UUID uuid);

        void onUpdate() override;

        void onRender(CommandBuffer &drawCmdBuffers) override;
        void collectRenderCommands(
                std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>>& renderGroups,
                uint32_t frameIndex);
        void bindResourcesAndDraw(const CommandBuffer& commandBuffer, RenderCommand& command);
        void initializeDifferentiableRenderer();
        torch::Tensor loadPFM(const std::string& filename, int expectedWidth, int expectedHeight);

        void onSceneLoad(std::shared_ptr<Scene> scene) override;

        void onEditorResize() override;

        struct DebugData {
            glm::vec3 positionGradient = glm::vec3(0.0f);
        }m_lastIteration;

    private:
        std::vector<std::unique_ptr<Buffer>> m_shaderSelectionBuffer;
        PipelineManager m_pipelineManager;
        DescriptorRegistry m_descriptorRegistry;
        std::shared_ptr<MeshInstance> m_meshInstances;
        std::shared_ptr<VulkanTexture2D> m_colorTexture;

        std::unique_ptr<PathTracer::PhotonTracer> m_pathTracer;

        std::shared_ptr<Scene> m_activeScene;
        std::shared_ptr<ArcballCamera> m_activeSceneCamera;

        // Diff Renderer stuff
        std::unique_ptr<PathTracer::PhotonRebuildModule> m_photonRebuildModule = nullptr;
        std::unique_ptr<torch::optim::Adam> m_optimizer;  // Or any other optimizer in <torch/optim.h>
        torch::Tensor m_accumulatedTensor = torch::Tensor();
        uint32_t m_numAccumulated = 0;
        std::unique_ptr<SyclDeviceSelector> m_syclDevice;

        PathTracer::PhotonTracer::RenderSettings m_renderSettings;
        uint32_t m_currentPipelineWidth = 0;
        uint32_t m_currentPipelineHeight = 0;
        CameraComponent* m_previousSceneCamera = nullptr;
        uint32_t m_stepIteration = 0;

    };
}
#endif //MULTISENSE_VIEWER_EDITOR_DIFFRENTIABLE_RENDERER
