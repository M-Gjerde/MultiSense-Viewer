//
// Created by magnus on 1/24/25.
//

#include "Viewer/Rendering/RenderResources/PathTracer/libtorch/PhotonRebuildModule.h"

#include <utility>

#include "Viewer/Rendering/Components/GaussianComponent.h"
#include "Viewer/Scenes/Entity.h"

namespace VkRender::PathTracer {
    PhotonRebuildModule::PhotonRebuildModule(PhotonTracer* rt, std::weak_ptr<Scene> scene)
        : m_photonRebuild(rt) {
        // Optionally register parameters or buffers if needed
        uploadFromScene(std::move(scene));
    }

    PhotonRebuildModule::~PhotonRebuildModule() {
        freeData();
    }

    torch::Tensor
    PhotonRebuildModule::forward(PhotonTracer::RenderSettings& settings) {
        // 1) Call PhotonTracer::update(...) or any function you want to do the actual path tracing
        //    In your case:  rt_->update(...);


        // Simply call the custom autograd function
        auto result = PhotonRebuildFunction::apply(
            settings,
            m_photonRebuild,
            m_tensorData.positions,
            m_tensorData.scales,
            m_tensorData.normals,
            m_tensorData.emissions,
            m_tensorData.colors,
            m_tensorData.specular,
            m_tensorData.diffuse
        );
        return result;
    }

    float* PhotonRebuildModule::getRenderedImage() {
        return m_photonRebuild->getImage();
    }

    void PhotonRebuildModule::freeData() {
        if (m_data.gaussianInputAssembly) {
            free(m_data.gaussianInputAssembly);
            m_data.gaussianInputAssembly = nullptr;
        }
    }

    void PhotonRebuildModule::uploadPathTracerFromTensor() {
        m_photonRebuild->uploadGaussiansFromTensors(m_tensorData);
    }
    void PhotonRebuildModule::uploadFromScene(std::weak_ptr<Scene> scene) {
        freeData();
        auto scenePtr = scene.lock();
        std::vector<GaussianInputAssembly> gaussianInputAssembly;
        std::vector<TransformComponent> transformMatrices; // Transformation matrices for entities
        auto& registry = scenePtr->getRegistry();
        // Find all entities with GaussianComponent
        auto view = registry.view<GaussianComponent2DGS>();
        for (auto e : view) {
            auto& component = Entity(e, scenePtr.get()).getComponent<GaussianComponent2DGS>();
            for (size_t i = 0; i < component.size(); ++i) {
                GaussianInputAssembly point{};
                point.position = component.positions[i];
                point.scale = component.scales[i];
                point.normal = component.normals[i];

                point.emission = component.emissions[i];
                point.color = component.colors[i];
                point.diffuse = component.diffuse[i];
                point.specular = component.specular[i];
                point.phongExponent = component.phongExponents[i];
                gaussianInputAssembly.push_back(point);
            }
            auto& transform = Entity(e, scenePtr.get()).getComponent<TransformComponent>();
            transformMatrices.emplace_back(transform);
        }

        m_data.gaussianInputAssembly = static_cast<GaussianInputAssembly*>(malloc(
            sizeof(GaussianComponent2DGS) * gaussianInputAssembly.size()));
        memcpy(m_data.gaussianInputAssembly, gaussianInputAssembly.data(),
               sizeof(GaussianComponent2DGS) * gaussianInputAssembly.size());

        m_data.numGaussians = gaussianInputAssembly.size(); // Number of entities for rendering

        Log::Logger::getInstance()->info("Uploaded  {} Gaussians from scene to PhotonRebuildModule",
                                         m_data.numGaussians);

        // Now we have gaussianInputAssembly filled. Suppose we want to create separate PyTorch Tensors:
        auto device = torch::kCPU; // or torch::kCPU, depends on your use-case

        // 1) Convert positions to a tensor of shape [N, 3]
        std::vector<float> hostPositions;
        hostPositions.reserve(gaussianInputAssembly.size() * 3);

        for (auto& item : gaussianInputAssembly) {
            hostPositions.push_back(item.position.x);
            hostPositions.push_back(item.position.y);
            hostPositions.push_back(item.position.z);
        }

        // from_blob does not copy by default. Once we go out of scope, hostPositions might be freed.
        // Usually, we wrap it in a clone() call to own the data inside a Torch tensor:
        m_tensorData.positions = torch::from_blob(
            hostPositions.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(true);

        // 2) Convert scales to a tensor of shape [N, 1]
        std::vector<glm::vec2> hostScales;
        hostScales.reserve(gaussianInputAssembly.size());
        for (auto& item : gaussianInputAssembly) {
            hostScales.push_back(item.scale);
        }

        m_tensorData.scales = torch::from_blob(
            hostScales.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 2},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        // ... do the same for normals, emissions, colors, etc. ...

        // Example for normals:
        std::vector<float> hostNormals;
        hostNormals.reserve(gaussianInputAssembly.size() * 3);
        for (auto& item : gaussianInputAssembly) {
            hostNormals.push_back(item.normal.x);
            hostNormals.push_back(item.normal.y);
            hostNormals.push_back(item.normal.z);
        }

        m_tensorData.normals = torch::from_blob(
            hostNormals.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 3},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        /**  Appearance properties //// **/
        // Example for normals:
        std::vector<float> emissions;
        std::vector<float> colors;
        std::vector<float> specular;
        std::vector<float> diffuse;
        emissions.reserve(gaussianInputAssembly.size());
        for (auto& item : gaussianInputAssembly) {
            emissions.push_back(item.emission);
            colors.push_back(item.color);
            specular.push_back(item.specular);
            diffuse.push_back(item.diffuse);
        }

        m_tensorData.emissions = torch::from_blob(
            emissions.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        m_tensorData.colors = torch::from_blob(
            colors.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        m_tensorData.specular = torch::from_blob(
            specular.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);

        m_tensorData.diffuse = torch::from_blob(
            diffuse.data(),
            {static_cast<long>(gaussianInputAssembly.size()), 1},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).clone().to(device).set_requires_grad(false);


        register_parameter("positions", m_tensorData.positions);
        register_parameter("scales", m_tensorData.scales);
        register_parameter("normals", m_tensorData.normals);

        register_parameter("emissions", m_tensorData.emissions);
        register_parameter("colors", m_tensorData.colors);
        register_parameter("specular", m_tensorData.scales);
        register_parameter("diffuse", m_tensorData.diffuse);
    }
}
