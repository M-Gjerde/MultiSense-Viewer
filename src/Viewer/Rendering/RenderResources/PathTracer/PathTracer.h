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

    class PhotonRebuild {
    public:
        PhotonRebuild(Application* context, std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height);
        void uploadGaussianData(std::weak_ptr<Scene>& scene);
        void uploadVertexData(std::weak_ptr<Scene>& scene);

        void update(EditorPathTracerLayerUI& editorImageUI, std::shared_ptr<Scene> scene);


        float* getImage() {return m_imageMemory;}

        ~PhotonRebuild();


        void upload(std::weak_ptr<Scene> ptr);

        void setActiveCamera(const TransformComponent &transformComponent, float w, float h);
        void setActiveCamera(const std::shared_ptr<PinholeCamera>& camera, const TransformComponent *cameraTransform);

    private:
        PinholeCamera m_camera{};
        TransformComponent m_cameraTransform{};

        Application* m_context;
        SyclDeviceSelector m_selector{};

        float* m_imageMemory = nullptr;

        uint32_t m_width = 0, m_height = 0;

        GPUData m_gpu;
        std::unique_ptr<RenderInformation> m_renderInformation;
        MeshManager m_meshManager;


        void saveAsPPM(const std::filesystem::path& filename) const;

        uint32_t m_frameID = 0;

        void saveAsPFM(const std::filesystem::path &filename) const;

        void freeResources();

        void resetState();

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
