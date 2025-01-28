//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RAYTRACER_H
#define MULTISENSE_VIEWER_RAYTRACER_H

#include "Viewer/Scenes/Scene.h"
#include "Viewer/Rendering/MeshManager.h"

#ifdef SYCL_ENABLED
#include "Viewer/Tools/SyclDeviceSelector.h"
#include "Viewer/Rendering/RenderResources/PathTracer/Definitions.h"
#endif

#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracerLayerUI.h"

namespace VkRender::PathTracer {
#ifdef SYCL_ENABLED

    class PhotonTracer {
    public:
        PhotonTracer(Application* context, std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height);

        struct Settings
        {
            // Device and execution settings
            bool clearImageMemory = false;
            std::string kernelDevice = "GPU";

            // Execution Flow:
            KernelType kernelType = KERNEL_PATH_TRACER_2DGS;

            // Render settings
            uint64_t photonCount = 1e4;
            int numBounces = 32;
            float gammaCorrection = 2.2f;

        };

        struct BackwardInfo {
            glm::vec3* gradients = nullptr;
            glm::vec3* sumGradients = nullptr;
            float* gradientImage = nullptr;
        };

        void setExecutionDevice(Settings& settings);
        RenderInformation getRenderInfo();

        void update(Settings& editorImageUI);
        BackwardInfo backward(Settings& settings);

        float* getImage() {return m_imageMemory;}
        void upload(std::weak_ptr<Scene> ptr);

        ~PhotonTracer();

        void setActiveCamera(const TransformComponent &transformComponent, float w, float h);
        void setActiveCamera(const std::shared_ptr<PinholeCamera>& camera, const TransformComponent *cameraTransform);
        void uploadGaussiansFromTensors(GPUDataTensors& data);


        uint32_t m_width = 0, m_height = 0;
        PinholeCamera m_camera{};
        TransformComponent m_cameraTransform{};
        std::unique_ptr<RenderInformation> m_renderInformation;
        BackwardInfo m_backwardInfo;

    private:


        Application* m_context;
        SyclDeviceSelector m_selector{};

        float* m_imageMemory = nullptr;

        GPUData m_gpu;
        MeshManager m_meshManager;


        void saveAsPPM(const std::filesystem::path& filename) const;

        void saveAsPFM(const std::filesystem::path &filename) const;

        void freeResources();

        void resetState();
        void prepareImageAndInfoBuffers();
        void uploadGaussianData(std::weak_ptr<Scene>& scene);
        void uploadVertexData(std::weak_ptr<Scene>& scene);

    };

#else

    class PhotonRebuild {
    public:
        PhotonRebuild(Application* context, std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height) {}
        void uploadGaussianData(std::shared_ptr<Scene>& scene) {}
        void uploadVertexData(std::shared_ptr<Scene>& scene) {}
        void update(const EditorPathTracerLayerUI& editorImageUI, std::shared_ptr<Scene> scene) {}
        float* getImage() {return nullptr;}
        ~PhotonRebuild() {}
        void upload(std::shared_ptr<Scene> ptr) {}
        void setActiveCamera(const TransformComponent &transformComponent, float w, float h){}
        void setActiveCamera(const std::shared_ptr<PinholeCamera>& camera, const TransformComponent *cameraTransform){}
    };
#endif

}


#endif //MULTISENSE_VIEWER_RAYTRACER_H
