//
// Created by magnus on 11/27/24.
//

#include <utility>

#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer.h"
#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/Components/GaussianComponent.h"
#include "Viewer/Tools/SyclDeviceSelector.h"

#include "Viewer/Rendering/RenderResources/PathTracer/PathTracerMeshKernels.h"
#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer2DGSKernel.h"

namespace VkRender::PathTracer {
    PhotonTracer::PhotonTracer(Application* ctx, std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height) :
        m_context(ctx), m_selector(SyclDeviceSelector(SyclDeviceSelector::DeviceType::CPU)) {
        m_width = width;
        m_height = height;
        // Load the scene into gpu memory
        // Create image memory
        // Allocate host memory for RGBA image (4 floats per pixel)
        m_imageMemory = new float[width * height * 4];
        m_renderInformation = std::make_unique<RenderInformation>();
    }

    void PhotonTracer::setExecutionDevice(Settings& settings) {
        freeResources();
        if (settings.kernelDevice == "CPU") {
            m_selector = SyclDeviceSelector(SyclDeviceSelector::DeviceType::CPU);
        }
        else if (settings.kernelDevice == "GPU") {
            m_selector = SyclDeviceSelector(SyclDeviceSelector::DeviceType::GPU);
        }
        m_selector.getQueue().wait();
    }

    RenderInformation PhotonTracer::getRenderInfo() {

        return *m_renderInformation;
    }

    void PhotonTracer::update(Settings& settings) {
        try {
            if (m_cameraTransform.moved())
                resetState();
            if (settings.clearImageMemory) {
                resetState();
            }

            auto& queue = m_selector.getQueue();

            m_renderInformation->frameID++;
            uint64_t simulatePhotonCount = settings.photonCount;

            std::vector<PCG32> rng(simulatePhotonCount);
            for (int i = 0; i < simulatePhotonCount; ++i)
                rng[i].init(m_renderInformation->frameID, i * m_renderInformation->frameID);

            auto rngGPU = sycl::malloc_device<PCG32>(rng.size(), queue);
            queue.memcpy(rngGPU, rng.data(), sizeof(PCG32) * rng.size());

            m_renderInformation->totalPhotons += simulatePhotonCount;
            m_renderInformation->gamma = settings.gammaCorrection;
            m_renderInformation->numBounces = settings.numBounces;
            /** Any shared GPU/CPU debug information must be stored before here**/
            queue.memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation));

            auto cameraGPU = sycl::malloc_device<PinholeCamera>(1, queue);
            queue.memcpy(cameraGPU, &m_camera, sizeof(PinholeCamera));

            queue.wait();
            if (settings.kernelType == KERNEL_PATH_TRACER_2DGS && m_gpu.numGaussians > 0) {
                sycl::range<1> globalRange(simulatePhotonCount);
                queue.submit([&](sycl::handler& cgh) {
                    // Capture GPUData, etc. by value or reference as needed
                    LightTracerKernel kernel(m_gpu, simulatePhotonCount, m_cameraTransform, cameraGPU,
                                             settings.numBounces, rngGPU); // TODO set numBounces from renderInformation instead
                    cgh.parallel_for(globalRange, kernel);
                });
            }
            else if (settings.kernelType == KERNEL_PATH_TRACER_MESH) {
                sycl::range<1> globalRange(simulatePhotonCount);
                queue.submit([&](sycl::handler& cgh) {
                    // Capture GPUData, etc. by value or reference as needed
                    PathTracerMeshKernels kernel(m_gpu, simulatePhotonCount, m_cameraTransform, cameraGPU,
                                                 settings.numBounces, rngGPU);
                    cgh.parallel_for(globalRange, kernel);
                });
            }
            queue.wait_and_throw();

            sycl::free(cameraGPU, queue);
            sycl::free(rngGPU, queue);

            /** Any shared GPU/CPU debug information must be stored after here**/
            queue.memcpy(m_renderInformation.get(), m_gpu.renderInformation, sizeof(RenderInformation));
            queue.memcpy(m_imageMemory, m_gpu.imageMemory, m_width * m_height * sizeof(float)).wait();

            Log::Logger::getInstance()->trace(
                "simulated {:.3f} Billion photons. About {}k Photons hit the sensor",
                m_renderInformation->totalPhotons / 1e9,
                m_renderInformation->photonsAccumulated / 1000);
        }
        catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
    }

    void PhotonTracer::resetState() {
        m_selector.getQueue().fill(m_gpu.imageMemory, static_cast<float>(0), m_width * m_height).wait();
        m_renderInformation->frameID = 0;
        m_renderInformation->photonsAccumulated = 0;
        m_renderInformation->totalPhotons = 0;
        m_selector.getQueue().memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation)).wait();

    }

    void PhotonTracer::prepareImageAndInfoBuffers() {
        // Allocate device memory for RGBA image (4 floats per pixel)
        m_gpu.imageMemory = sycl::malloc_device<float>(m_width * m_height, m_selector.getQueue());
        if (!m_gpu.imageMemory) {
            throw std::runtime_error("Device memory allocation failed.");
        }
        m_gpu.contribution = sycl::malloc_device<float>(m_width * m_height, m_selector.getQueue());
        if (!m_gpu.contribution) {
            throw std::runtime_error("Device memory allocation failed.");
        }
        m_gpu.renderInformation = sycl::malloc_device<RenderInformation>(1, m_selector.getQueue());
        if (!m_gpu.renderInformation) {
            throw std::runtime_error("Device memory allocation failed.");
        }
        m_selector.getQueue().memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation));
        // Initialize device memory to 0
        m_selector.getQueue().fill(m_gpu.imageMemory, 0.0f, m_width * m_height).wait();
        // Initialize host memory to 0
        std::fill(m_imageMemory, m_imageMemory + (m_width * m_height * sizeof(float)), 0.0f);
        Log::Logger::getInstance()->info("Prepared Path Tracer SYCL image. Image dimensions are: {}x{}", m_width,
                                         m_height);
    }

    void PhotonTracer::upload(std::weak_ptr<Scene> scene) {
        prepareImageAndInfoBuffers();
        uploadVertexData(scene);
        uploadGaussianData(scene);
    }

    void PhotonTracer::freeResources() {
        m_selector.getQueue().wait();
        if (m_gpu.imageMemory) {
            sycl::free(m_gpu.imageMemory, m_selector.getQueue());
            m_gpu.imageMemory = nullptr;
        }
        if (m_gpu.contribution) {
            sycl::free(m_gpu.contribution, m_selector.getQueue());
            m_gpu.contribution = nullptr;
        }
        if (m_gpu.vertices) {
            sycl::free(m_gpu.vertices, m_selector.getQueue());
            m_gpu.vertices = nullptr;
        }
        if (m_gpu.indices) {
            sycl::free(m_gpu.indices, m_selector.getQueue());
            m_gpu.indices = nullptr;
        }
        if (m_gpu.indexOffsets) {
            sycl::free(m_gpu.indexOffsets, m_selector.getQueue());
            m_gpu.indexOffsets = nullptr;
        }
        if (m_gpu.vertexOffsets) {
            sycl::free(m_gpu.vertexOffsets, m_selector.getQueue());
            m_gpu.vertexOffsets = nullptr;
        }
        if (m_gpu.transforms) {
            sycl::free(m_gpu.transforms, m_selector.getQueue());
            m_gpu.transforms = nullptr;
        }
        if (m_gpu.materials) {
            sycl::free(m_gpu.materials, m_selector.getQueue());
            m_gpu.materials = nullptr;
        }
        if (m_gpu.tagComponents) {
            sycl::free(m_gpu.tagComponents, m_selector.getQueue());
            m_gpu.tagComponents = nullptr;
        }
        if (m_gpu.gaussianInputAssembly) {
            sycl::free(m_gpu.gaussianInputAssembly, m_selector.getQueue());
            m_gpu.gaussianInputAssembly = nullptr;
        }

        m_selector.getQueue().wait();
    }


    void PhotonTracer::uploadGaussiansFromTensors(GPUDataTensors& data) {
#ifdef DIFF_RENDERER_ENABLED
        freeResources();
        prepareImageAndInfoBuffers();
        auto& queue = m_selector.getQueue();
        // 2) Move Tensors to CPU (if they aren't already) so we can extract values
        //    (SYCL can't just copy directly from a PyTorch CUDA device pointer.)
        //    If data is already on CPU, this .cpu() will be basically a no-op.
        auto positionsCpu = data.positions.cpu();
        auto scalesCpu = data.scales.cpu();
        auto normalsCpu = data.normals.cpu();

        auto emissionsCpu = data.emissions.cpu();
        auto colorsCpu = data.colors.cpu();
        auto specularCpu = data.specular.cpu();
        auto diffuseCpu = data.diffuse.cpu();

        // 3) Some basic sanity checks on tensor shapes
        //    Here we assume:
        //      positions: [N, 3]
        //      scales:    [N, 1] or just [N]
        //      normals:   [N, 3]
        TORCH_CHECK(positionsCpu.dim() == 2 && positionsCpu.size(1) == 3,
                    "positions must have shape [N,3]");
        TORCH_CHECK(normalsCpu.dim() == 2 && normalsCpu.size(1) == 3,
                    "normals must have shape [N,3]");

        // For scales, accept either [N] or [N,1]
        TORCH_CHECK((scalesCpu.dim() == 2 && scalesCpu.size(1) == 2),
                    "scales must have shape [N,2]");

        // 4) Get the number of gaussians (N)
        const auto N = positionsCpu.size(0);

        // 5) Create a host vector of GaussianInputAssembly
        std::vector<GaussianInputAssembly> hostGaussians(N);

        // 6) Pointers to the underlying float data (on CPU).
        //    We'll read them row-by-row.
        //    Example: positionsCpu.data_ptr<float>() returns a pointer to the 2D array in row-major order.
        const float* posPtr = positionsCpu.data_ptr<float>();
        const float* scalesPtr = scalesCpu.data_ptr<float>();
        const float* normalsPtr = normalsCpu.data_ptr<float>();

        const float* emissionsPtr = emissionsCpu.data_ptr<float>();
        const float* colorsPtr = colorsCpu.data_ptr<float>();
        const float* specularPtr = specularCpu.data_ptr<float>();
        const float* diffusePtr = diffuseCpu.data_ptr<float>();

        // 7) Fill the hostGaussians array
        //    (positions = 3 floats, normals = 3 floats, scale = 1 float, plus defaults)
        for (int i = 0; i < N; ++i) {
            GaussianInputAssembly point{};

            // positions: [i,0..2]
            point.position.x = posPtr[i * 3 + 0];
            point.position.y = posPtr[i * 3 + 1];
            point.position.z = posPtr[i * 3 + 2];

            // scales: either shape [N,1] or [N].
            // If [N,1], then index is i*1, else it's i
            if (scalesCpu.dim() == 2) {
                point.scale.x = scalesPtr[i * 2 + 0];
                point.scale.y = scalesPtr[i * 2 + 1];
            }

            // normals: [i,0..2]
            point.normal.x = normalsPtr[i * 3 + 0];
            point.normal.y = normalsPtr[i * 3 + 1];
            point.normal.z = normalsPtr[i * 3 + 2];

            // Fill default appearance properties
            point.emission =      emissionsPtr[i]; // emission = 0
            point.color =         colorsPtr[i]; // color = 1
            point.diffuse =       specularPtr[i]; // diffuse = 0.5
            point.specular =      diffusePtr[i]; // specular = 0.5
            point.phongExponent = 32; // phongExponent = 32

            hostGaussians[i] = point;
        }

        // 8) Allocate device memory and copy
        m_gpu.gaussianInputAssembly = sycl::malloc_device<GaussianInputAssembly>(N, queue);
        queue.memcpy(m_gpu.gaussianInputAssembly, hostGaussians.data(), N * sizeof(GaussianInputAssembly));
        queue.wait();

        // 9) Set number of gaussians
        m_gpu.numGaussians = N;

        // Log
        Log::Logger::getInstance()->info("uploadFromTensors: Uploaded {} Gaussians", N);
#endif
    }

    void PhotonTracer::uploadGaussianData(std::weak_ptr<Scene>& scene) {
        auto scenePtr = scene.lock();
        auto& queue = m_selector.getQueue();
        std::vector<GaussianInputAssembly> gaussianInputAssembly;
        std::vector<TransformComponent> transformMatrices; // Transformation matrices for entities
        auto& registry = scenePtr->getRegistry();
        // Find all entities with GaussianComponent
        auto view = scenePtr->getRegistry().view<GaussianComponent2DGS>();
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

        m_gpu.gaussianInputAssembly = sycl::malloc_device<GaussianInputAssembly>(gaussianInputAssembly.size(), queue);
        queue.memcpy(m_gpu.gaussianInputAssembly, gaussianInputAssembly.data(),
                     gaussianInputAssembly.size() * sizeof(GaussianInputAssembly));
        m_gpu.numGaussians = gaussianInputAssembly.size(); // Number of entities for rendering

        Log::Logger::getInstance()->info("Uploaded  {} Gaussians to renderkernel", m_gpu.numGaussians);
        queue.wait();
    }

    void PhotonTracer::uploadVertexData(std::weak_ptr<Scene>& scene) {
        auto scenePtr = scene.lock();
        std::vector<InputAssembly> vertexData;
        std::vector<uint32_t> indices;
        std::vector<uint32_t> indexOffsets; // Offset for each entity's indices
        std::vector<uint32_t> vertexOffsets; // Offset for each entity's vertices
        std::vector<TransformComponent> transformMatrices; // Transformation matrices for entities
        std::vector<MaterialComponent> materials; // Transformation matrices for entities
        std::vector<TagComponent> tagComponents; // Transformation matrices for entities
        auto view = scenePtr->getRegistry().view<MeshComponent, TransformComponent>();
        uint32_t currentVertexOffset = 0;
        uint32_t currentIndexOffset = 0;

        for (auto e : view) {
            Entity entity(e, scenePtr.get());
            std::string tag = entity.getName();
            // Initialize a flag to determine if we should skip this entity
            bool skipEntity = false;
            // Start with the current entity
            Entity current = entity;
            // Traverse up the parent hierarchy
            while (current.getParent()) {
                // Move to the parent entity
                current = current.getParent();
                // Check if the parent has both GroupComponent and VisibilityComponent
                if (current.hasComponent<GroupComponent>() && current.hasComponent<VisibleComponent>()) {
                    // Retrieve the VisibilityComponent
                    auto& visibility = current.getComponent<VisibleComponent>();
                    // If visibility is set to false, mark to skip this entity
                    if (!visibility.visible) {
                        skipEntity = true;
                        break; // No need to check further ancestors
                    }
                }
            }
            if (entity.hasComponent<CameraComponent>())
                skipEntity = true;

            auto& meshComponent = entity.getComponent<MeshComponent>();
            if (meshComponent.meshDataType() != OBJ_FILE)
                skipEntity = true;

            // If an ancestor with visible == false was found, skip to the next entity
            if (skipEntity) {
                continue;
            }


            auto& transform = entity.getComponent<TransformComponent>();
            transformMatrices.emplace_back(transform);

            auto tagComponent = entity.getComponent<TagComponent>();
            tagComponents.emplace_back(tagComponent);

            if (entity.hasComponent<MaterialComponent>()) {
                auto& material = entity.getComponent<MaterialComponent>();
                float diff = material.diffuse;
                float specular = material.specular;
                materials.emplace_back(material);
            }


            std::shared_ptr<MeshData> meshData = m_meshManager.getMeshData(meshComponent);
            // Store vertex offset
            vertexOffsets.push_back(currentVertexOffset);
            // Add vertex data
            for (auto& vert : meshData->vertices) {
                InputAssembly input{};
                input.position = vert.pos;
                input.color = vert.color;
                input.normal = vert.normal;
                vertexData.emplace_back(input);
            }
            currentVertexOffset += meshData->vertices.size();
            // Store index offset
            indexOffsets.push_back(currentIndexOffset);
            // Add index data
            for (auto idx : meshData->indices) {
                indices.push_back(idx + vertexOffsets.back()); // Adjust indices by vertex offset
            }
            currentIndexOffset += meshData->indices.size();
        }
        // Upload vertex data to GPU
        auto& queue = m_selector.getQueue();
        m_gpu.vertices = sycl::malloc_device<InputAssembly>(vertexData.size(), queue);
        queue.memcpy(m_gpu.vertices, vertexData.data(), vertexData.size() * sizeof(InputAssembly));
        // Upload index data to GPU
        m_gpu.indices = sycl::malloc_device<uint32_t>(indices.size(), queue);
        queue.memcpy(m_gpu.indices, indices.data(), indices.size() * sizeof(uint32_t));
        // Upload transform matrices to GPU
        m_gpu.transforms = sycl::malloc_device<TransformComponent>(transformMatrices.size(), queue);
        queue.memcpy(m_gpu.transforms, transformMatrices.data(), transformMatrices.size() * sizeof(TransformComponent));
        m_gpu.materials = sycl::malloc_device<MaterialComponent>(materials.size(), queue);
        queue.memcpy(m_gpu.materials, materials.data(), materials.size() * sizeof(MaterialComponent));
        m_gpu.tagComponents = sycl::malloc_device<TagComponent>(tagComponents.size(), queue);
        queue.memcpy(m_gpu.tagComponents, tagComponents.data(), tagComponents.size() * sizeof(TagComponent));
        // Upload offsets (if necessary for rendering)
        m_gpu.vertexOffsets = sycl::malloc_device<uint32_t>(vertexOffsets.size(), queue);
        queue.memcpy(m_gpu.vertexOffsets, vertexOffsets.data(), vertexOffsets.size() * sizeof(uint32_t));
        m_gpu.indexOffsets = sycl::malloc_device<uint32_t>(indexOffsets.size(), queue);
        queue.memcpy(m_gpu.indexOffsets, indexOffsets.data(), indexOffsets.size() * sizeof(uint32_t));
        queue.wait();

        m_gpu.totalVertices = currentVertexOffset; // Number of entities for rendering
        m_gpu.totalIndices = currentIndexOffset; // Number of entities for rendering
        m_gpu.numEntities = static_cast<uint32_t>(transformMatrices.size()); // Number of entities for rendering
    }


    PhotonTracer::~PhotonTracer() {
        if (m_imageMemory) {
            delete[] m_imageMemory;
        }

        freeResources();
    }

    void PhotonTracer::saveAsPPM(const std::filesystem::path& filename) const {
        std::ofstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename.string());
        }

        // Write the PPM header
        file << "P6\n" << m_width << " " << m_height << "\n255\n";

        // Write pixel data in RGB format
        for (uint32_t y = 0; y < m_height; ++y) {
            for (uint32_t x = 0; x < m_width; ++x) {
                uint32_t pixelIndex = (y * m_width + x) * 4; // RGBA8: 4 bytes per pixel

                // Extract R, G, B components (ignore A)
                file.put(m_imageMemory[pixelIndex + 0]); // R
                file.put(m_imageMemory[pixelIndex + 1]); // G
                file.put(m_imageMemory[pixelIndex + 2]); // B
            }
        }

        file.close();
    }

    void PhotonTracer::saveAsPFM(const std::filesystem::path& filename) const {
        std::ofstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename.string());
        }

        // Write the PFM header
        // "PF" indicates a color image. Use "Pf" for grayscale.
        file << "PF\n" << m_width << " " << m_height << "\n-1.0\n";

        // PFM expects the data in binary format, row by row from top to bottom
        // Assuming your m_imageMemory is in RGBA format with floats

        // Allocate a temporary buffer for RGB data
        std::vector<float> rgbData(m_width * m_height * 3);

        for (uint32_t y = 0; y < m_height; ++y) {
            for (uint32_t x = 0; x < m_width; ++x) {
                uint32_t pixelIndex = (y * m_width + x); // RGBA: 4 floats per pixel
                uint32_t rgbIndex = (y * m_width + x) * 3;

                rgbData[rgbIndex + 0] = m_imageMemory[pixelIndex]; // R
                rgbData[rgbIndex + 1] = m_imageMemory[pixelIndex]; // G
                rgbData[rgbIndex + 2] = m_imageMemory[pixelIndex]; // B
            }
        }

        // Write the RGB float data
        file.write(reinterpret_cast<const char*>(rgbData.data()), rgbData.size() * sizeof(float));

        if (!file) {
            throw std::runtime_error("Failed to write PFM data to file: " + filename.string());
        }

        file.close();
    }

    // For editorCamera
    void PhotonTracer::setActiveCamera(const TransformComponent& transformComponent, float width, float height) {
        PinholeParameters pinholeParameters;
        SharedCameraSettings cameraSettings;
        pinholeParameters.width = width;
        pinholeParameters.height = height;
        pinholeParameters.cx = width / 2;
        pinholeParameters.cy = height / 2;
        pinholeParameters.fx = 600.0f;
        pinholeParameters.fy = 600.0f;
        PinholeCamera camera = PinholeCamera(cameraSettings, pinholeParameters);
        m_camera = std::move(camera);
        m_cameraTransform = transformComponent;
    }

    // For pinhole scene camera
    void PhotonTracer::setActiveCamera(const std::shared_ptr<PinholeCamera>& camera,
                                       const TransformComponent* cameraTransform) {
        m_camera = *camera;
        m_cameraTransform = *cameraTransform;
    }


    /* Draw rays
     {
        auto view = m_scene->getRegistry().view<CameraComponent, TransformComponent, MeshComponent>();
        for (auto e: view) {
            Entity entity(e, m_scene.get());
            auto &transform = entity.getComponent<TransformComponent>();
            auto camera = std::dynamic_pointer_cast<PinholeCamera>(entity.getComponent<CameraComponent>().camera);
            if (!camera || entity.getComponent<CameraComponent>().renderFromViewpoint())
                continue;
            float fx = camera->m_fx;
            float fy = camera->m_fy;
            float cx = camera->m_cx;
            float cy = camera->m_cy;
            float width = camera->m_width;
            float height = camera->m_height;


            // Helper lambda to create a ray entity
            auto updateRayEntity = [&](Entity cornerEntity, float x, float y) {
                MeshComponent *mesh;
                if (!cornerEntity.hasComponent<MeshComponent>())
                    mesh = &cornerEntity.addComponent<MeshComponent>(CYLINDER);
                else
                    mesh = &cornerEntity.getComponent<MeshComponent>();

                if (!cornerEntity.hasComponent<TemporaryComponent>())
                    cornerEntity.addComponent<TemporaryComponent>();


                cornerEntity.getComponent<TransformComponent>() = transform;
                auto cylinderParams = std::dynamic_pointer_cast<CylinderMeshParameters>(mesh->meshParameters);
                // The cylinder magnitude is how long the cylinder is.
                // Start the cylinder at the camera origin
                cylinderParams->origin = glm::vec3(0.0f, 0.0f, 0.0f);

                // Choose a plane at Z = -1 for visualization. Objects in front of the camera have negative Z.
                float Z_plane = -1.0f;

                auto mapPixelTo3D = [&](float u, float v) {
                    float X = -(u - cx) * Z_plane / fx;
                    float Y = -(v - cy) * Z_plane / fy; // Notice the minus sign before (v - cy)
                    float Z = Z_plane;
                    return glm::vec3(X, Y, Z);
                };
                glm::vec3 direction = mapPixelTo3D(x, y);


                cylinderParams->direction = glm::normalize(direction);
                cylinderParams->magnitude = glm::length(direction);
                cylinderParams->radius = 0.01f;
                mesh->updateMeshData = true;
            };

            auto groupEntity = m_scene->getOrCreateEntityByName("Rays");
            if (!groupEntity.hasComponent<GroupComponent>())
                groupEntity.addComponent<GroupComponent>();
            if (!groupEntity.hasComponent<TemporaryComponent>())
                groupEntity.addComponent<TemporaryComponent>();
            if (!groupEntity.hasComponent<VisibleComponent>())
                groupEntity.addComponent<VisibleComponent>(); // For visibility toggling

            auto topLeftEntity = m_scene->getOrCreateEntityByName("TopLeft");
            auto topRightEntity = m_scene->getOrCreateEntityByName("TopRight");

            auto bottomLeftEntity = m_scene->getOrCreateEntityByName("BottomLeft");
            auto bottomRightEntity = m_scene->getOrCreateEntityByName("BottomRight");


            updateRayEntity(topLeftEntity, 0.0f, 0.0f);
            updateRayEntity(topRightEntity, width, 0.0f);
            updateRayEntity(bottomLeftEntity, width, height);
            updateRayEntity(bottomRightEntity, 0.0f, height);

            topLeftEntity.setParent(groupEntity);
            topRightEntity.setParent(groupEntity);
            bottomLeftEntity.setParent(groupEntity);
            bottomRightEntity.setParent(groupEntity);

            //auto centerRayEntity = m_scene->getOrCreateEntityByName("CenterRay");
            //updateRayEntity(centerRayEntity, width / 2, height / 2);

            // Generate rays for every 10th pixel
            for (int x = 0; x < width; x += 100) {
                for (int y = 0; y < height; y += 100) {
                    // Create a unique name for the ray entity
                    std::string rayEntityName = "Ray_" + std::to_string(x) + "_" + std::to_string(y);

                    // Get or create the entity for this ray
                    auto rayEntity = m_scene->getOrCreateEntityByName(rayEntityName);
                    rayEntity.setParent(groupEntity);

                    // Update the ray entity's position or other attributes based on the pixel coordinates
                    updateRayEntity(rayEntity, static_cast<float>(x), static_cast<float>(y));
                }
            }
        }
    }
    */
}
