//
// Created by magnus on 11/27/24.
//

#ifndef MULTISENSE_VIEWER_RAYTRACER_H
#define MULTISENSE_VIEWER_RAYTRACER_H

#include "Viewer/Scenes/Scene.h"
#include "Viewer/Rendering/RenderResources/Raytracer/Definitions.h"
#include "Viewer/Rendering/MeshManager.h"

#ifdef SYCL_ENABLED
#include "Viewer/Tools/SyclDeviceSelector.h"
#endif

#include "Viewer/Rendering/Editors/ImageEditor/EditorImageUI.h"

namespace VkRender::RT {
#ifdef SYCL_ENABLED

    class RayTracer {
    public:
        RayTracer(Application* context, std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height);
        void uploadGaussianData(std::shared_ptr<Scene>& scene);
        void uploadVertexData(std::shared_ptr<Scene>& scene);

        void update(const EditorImageUI& editorImageUI);


        uint8_t* getImage() {return m_imageMemory;}

        ~RayTracer();


        void upload(std::shared_ptr<Scene> ptr);

    private:
        BaseCamera m_camera;
        Application* m_context;
        SyclDeviceSelector m_selector = SyclDeviceSelector(SyclDeviceSelector::DeviceType::CPU);

        std::shared_ptr<Scene> m_scene;
        uint8_t* m_imageMemory = nullptr;

        uint32_t m_width = 0, m_height = 0;

        GPUData m_gpu;
        MeshManager m_meshManager;


        void saveAsPPM(const std::filesystem::path& filename) const;

        uint32_t m_frameID = 0;
    };

#else

    class RayTracer {
    public:
        RayTracer(Application* context, std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height) {}
        void uploadGaussianData(std::shared_ptr<Scene>& scene) {}
        void uploadVertexData(std::shared_ptr<Scene>& scene) {}
        void update(const EditorImageUI& editorImageUI) {}
        uint8_t* getImage() {return nullptr;}
        ~RayTracer() {}
        void upload(std::shared_ptr<Scene> ptr) {}
    };
#endif

}


#endif //MULTISENSE_VIEWER_RAYTRACER_H
