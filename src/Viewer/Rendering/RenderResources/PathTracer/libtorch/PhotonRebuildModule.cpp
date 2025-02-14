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
    PhotonRebuildModule::forward(IterationInfo info) {
        // 1) Call PhotonTracer::update(...) or any function you want to do the actual path tracing
        //    In your case:  rt_->update(...);


        // Simply call the custom autograd function
        auto result = PhotonRebuildFunction::apply(
            info,
            m_photonRebuild,
            m_tensorData.positions,
            m_tensorData.scales,
            m_tensorData.normals,
            m_tensorData.emissions,
            m_tensorData.colors,
            m_tensorData.specular,
            m_tensorData.diffuse
        );

        m_outputTensor = result.clone();  // Clone to ensure ownership

        return result;
    }

    float* PhotonRebuildModule::getRenderedImage() {
        if (m_outputTensor.defined()) {
            return m_outputTensor.data_ptr<float>();  // Get a float pointer to the tensor data
        }
        return nullptr;
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

void PhotonRebuildModule::uploadSceneFromTensor(std::shared_ptr<Scene> scene) {
    // Get views of all the 2DGS Gaussian components in the scene.
    auto gaussianView = scene->getRegistry().view<GaussianComponent2DGS>();

    // Get CPU copies of our tensors (if they aren’t already on CPU)
    auto positionsTensor = m_tensorData.positions.cpu();
    auto scalesTensor    = m_tensorData.scales.cpu();
    auto normalsTensor   = m_tensorData.normals.cpu();
    auto emissionsTensor = m_tensorData.emissions.cpu();
    auto colorsTensor    = m_tensorData.colors.cpu();
    auto specularTensor  = m_tensorData.specular.cpu();
    auto diffuseTensor   = m_tensorData.diffuse.cpu();

    // Assume that the first dimension of each tensor is the number of gaussians.
    int numGaussians = positionsTensor.size(0);

    // Pointers to the raw data.
    // (These assume that the tensors are contiguous and of type float.)
    float* posPtr    = positionsTensor.data_ptr<float>();   // shape: [numGaussians, 3]
    float* scalePtr  = scalesTensor.data_ptr<float>();        // shape: [numGaussians, 2]
    float* normPtr   = normalsTensor.data_ptr<float>();       // shape: [numGaussians, 3]
    float* emissPtr  = emissionsTensor.data_ptr<float>();     // shape: [numGaussians]
    float* colorPtr  = colorsTensor.data_ptr<float>();        // shape: [numGaussians]
    float* specPtr   = specularTensor.data_ptr<float>();      // shape: [numGaussians]
    float* diffPtr   = diffuseTensor.data_ptr<float>();       // shape: [numGaussians]

    // For each GaussianComponent2DGS in our scene, update its vectors with the tensor data.
    // (Often in an ECS there is only one global component of a given type,
    //  but if there are multiple, they will all be updated identically.)
    for (auto entityID : gaussianView) {
        auto entity = Entity(entityID, scene.get());
        auto &comp = entity.getComponent<GaussianComponent2DGS>();

        // Resize the vectors to hold data for all gaussians.
        comp.positions.resize(numGaussians);
        comp.scales.resize(numGaussians);
        comp.normals.resize(numGaussians);
        comp.emissions.resize(numGaussians);
        comp.colors.resize(numGaussians);
        comp.specular.resize(numGaussians);
        comp.diffuse.resize(numGaussians);
        // Note: If opacities or phongExponents should also be updated,
        // add them here and ensure the corresponding tensors exist.

        // Copy the data from the tensors to the component's vectors.
        for (int i = 0; i < numGaussians; i++) {
            // Update positions (each is 3 floats)
            comp.positions[i] = glm::vec3(
                posPtr[i * 3 + 0],
                posPtr[i * 3 + 1],
                posPtr[i * 3 + 2]
            );
            // Update scales (each is 2 floats)
            comp.scales[i] = glm::vec2(
                scalePtr[i * 2 + 0],
                scalePtr[i * 2 + 1]
            );
            // Update normals (each is 3 floats)
            comp.normals[i] = glm::vec3(
                normPtr[i * 3 + 0],
                normPtr[i * 3 + 1],
                normPtr[i * 3 + 2]
            );
            // Update colors (each is 4 floats)
            comp.colors[i] = glm::vec4(
                colorPtr[i * 4 + 0],
                colorPtr[i * 4 + 1],
                colorPtr[i * 4 + 2],
                colorPtr[i * 4 + 3]
            );

            // Update the other properties (each assumed to be a single float per gaussian)
            comp.emissions[i] = emissPtr[i];
            comp.specular[i]  = specPtr[i];
            comp.diffuse[i]   = diffPtr[i];
        }
    }
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
        std::vector<glm::vec4> colors;
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
            {static_cast<long>(gaussianInputAssembly.size()), 4},
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

        Log::Logger::getInstance()->info("Registrered and Uploaded {} Gaussians from scene to Tensors",
                                         m_data.numGaussians);
    }
}
