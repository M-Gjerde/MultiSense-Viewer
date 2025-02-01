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
#include "Viewer/Rendering/RenderResources/PathTracer/PathTracer2DGSKernelBackward.h"

namespace VkRender::PathTracer {
    PhotonTracer::PhotonTracer(Application* ctx, const PipelineSettings& pipelineSettings,
                               std::shared_ptr<Scene>& scene) :
        m_pipelineSettings(pipelineSettings),
        m_context(ctx) {
        // Load the scene into gpu memory
        // Create image memory
        // Allocate host memory for RGBA image (4 floats per pixel)
        m_imageMemory = new float[pipelineSettings.width * pipelineSettings.height];
        m_backwardInfo.gradients = new glm::vec3[pipelineSettings.photonCount];
        m_backwardInfo.sumGradients = new glm::vec3();
        m_renderInformation = std::make_unique<RenderInformation>();
        pipelineSettings.queue.wait();
        prepareImageAndInfoBuffers();
        uploadVertexData(scene);
        uploadGaussianData(scene);
        pipelineSettings.queue.wait();

        Log::Logger::getInstance()->info("PathTracer created, Propterties: PhotonCount: {}, Bounces: {}, Image Size: {}x{}", m_pipelineSettings.photonCount, m_pipelineSettings.numBounces, m_pipelineSettings.width, m_pipelineSettings.height);
    }

    RenderInformation PhotonTracer::getRenderInfo() {
        return *m_renderInformation;
    }

    void PhotonTracer::update(RenderSettings& renderSettings) {
        try {
            auto& queue = m_pipelineSettings.queue;
            // Update shared GPU/CPU render information
            m_renderInformation->frameID++;
            m_renderInformation->totalPhotons += m_pipelineSettings.photonCount;
            m_renderInformation->gamma = renderSettings.gammaCorrection;
            m_renderInformation->numBounces = m_pipelineSettings.numBounces;
            queue.memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation));
            queue.memcpy(m_gpu.pinholeCamera, &renderSettings.camera, sizeof(PinholeCamera));
            queue.memcpy(m_gpu.cameraTransform, &renderSettings.cameraTransform, sizeof(TransformComponent));

            // Kernel Launch
            sycl::range<1> globalRange(m_pipelineSettings.photonCount);

            switch (renderSettings.kernelType) {
            case KERNEL_PATH_TRACER_2DGS:
                if (m_gpu.numGaussians > 0) {
                    queue.submit([&](sycl::handler& cgh) {
                        LightTracerKernel kernel(m_gpu, m_gpuDataOutput, m_pcg32);
                        cgh.parallel_for(globalRange, kernel);
                    });
                }
                break;
            case KERNEL_PATH_TRACER_MESH:
                queue.submit([&](sycl::handler& cgh) {
                    PathTracerMeshKernels kernel(m_gpu, renderSettings.cameraTransform, m_gpu.pinholeCamera, m_pcg32);
                    cgh.parallel_for(globalRange, kernel);
                });
                break;
            default:
                // Handle unsupported kernel type if necessary
                    break;
            }
            queue.wait_and_throw();

            // Retrieve updated information from GPU
            queue.memcpy(m_renderInformation.get(), m_gpu.renderInformation, sizeof(RenderInformation));
            queue.memcpy(m_imageMemory, m_gpu.imageMemory,
                         m_pipelineSettings.width * m_pipelineSettings.height * sizeof(float));
            queue.wait();

            double totalM = static_cast<double>(m_renderInformation->totalPhotons) / 1e6;
            double sensorK = static_cast<double>(m_renderInformation->photonsAccumulated) / 1000.0;
            Log::Logger::getInstance()->info(
                "Simulated {}M photons. About {}k photons hit the sensor",
                totalM, sensorK);

        }
        catch (const sycl::exception& e) {
            Log::Logger::getInstance()->warning("Caught exception: {}", e.what());
            std::cerr << "Exception: " << e.what() << std::endl;
        }
    }

    PhotonTracer::BackwardInfo PhotonTracer::backward(RenderSettings& renderSettings) {

        try {
            auto& queue = m_pipelineSettings.queue;
            uint64_t simulatePhotonCount = m_pipelineSettings.photonCount;
            uint32_t imageSize = m_pipelineSettings.width * m_pipelineSettings.height;

            queue.memcpy(m_gpu.gradientImage, m_backwardInfo.gradientImage,  sizeof(float) * imageSize);

            m_renderInformation->frameID++;
            m_renderInformation->totalPhotons += m_pipelineSettings.photonCount;
            m_renderInformation->gamma = renderSettings.gammaCorrection;
            m_renderInformation->numBounces = m_pipelineSettings.numBounces;
            queue.memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation));
            queue.memcpy(m_gpu.pinholeCamera, &renderSettings.camera, sizeof(PinholeCamera));
            queue.memcpy(m_gpu.cameraTransform, &renderSettings.cameraTransform, sizeof(TransformComponent));

            sycl::range<1> globalRange(simulatePhotonCount);
            queue.submit([&](sycl::handler& cgh) {
                // Capture GPUData, etc. by value or reference as needed
                LightTracerKernelBackward kernel(m_gpu, m_gpuDataOutput, m_pcg32);
                cgh.parallel_for(globalRange, kernel);
            });

            queue.wait_and_throw();
            queue.memcpy(m_backwardInfo.gradients, m_gpu.gradients, simulatePhotonCount * sizeof(glm::vec3));
            queue.memcpy(m_backwardInfo.sumGradients, m_gpu.sumGradients,  sizeof(glm::vec3));
            queue.wait();
        }
        catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
        return m_backwardInfo;

    }

    void PhotonTracer::resetImage() {
        auto& queue = m_pipelineSettings.queue;
        queue.fill(m_gpu.imageMemory, static_cast<float>(0),
                   m_pipelineSettings.width * m_pipelineSettings.height).wait();
        m_renderInformation->frameID = 0;
        m_renderInformation->photonsAccumulated = 0;
        m_renderInformation->totalPhotons = 0;
        queue.memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation)).wait();
    }

    void PhotonTracer::prepareImageAndInfoBuffers() {
        uint32_t imageSize = m_pipelineSettings.width * m_pipelineSettings.height;
        auto& queue = m_pipelineSettings.queue;
        // Allocate device memory for RGBA image (4 floats per pixel)
        m_gpu.imageMemory = sycl::malloc_device<float>(imageSize, queue);
        if (!m_gpu.imageMemory) {
            throw std::runtime_error("Device memory allocation failed.");
        }

        m_gpu.renderInformation = sycl::malloc_device<RenderInformation>(1, queue);
        if (!m_gpu.renderInformation) {
            throw std::runtime_error("Device memory allocation failed.");
        }
        queue.memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation));
        // Initialize device memory to 0
        queue.fill(m_gpu.imageMemory, 0.0f, imageSize).wait();
        // Initialize host memory to 0
        std::fill(m_imageMemory, m_imageMemory + imageSize, 0.0f);

        // Initialize RNGs
        // Generate a random seed using std::random_device.
        size_t simulatePhotonCount = m_pipelineSettings.photonCount;
        // Seed a fast PRNG using std::random_device only once
        std::random_device rd;
        std::mt19937_64 engine(rd());
        std::uniform_int_distribution<uint64_t> dist;

        std::vector<PCG32> rng(simulatePhotonCount);
        for (uint64_t i = 0; i < simulatePhotonCount; ++i) {
            uint64_t randomNumber = dist(engine);
            rng[i].init(randomNumber, i);
        }
        // Allocate and copy RNG to GPU
        m_pcg32 = sycl::malloc_device<PCG32>(simulatePhotonCount, queue);
        queue.memcpy(m_pcg32, rng.data(), sizeof(PCG32) * simulatePhotonCount);
        m_gpu.pinholeCamera = sycl::malloc_device<PinholeCamera>(1, queue);
        m_gpu.cameraTransform = sycl::malloc_device<TransformComponent>(1, queue);

        m_gpuDataOutput = sycl::malloc_device<GPUDataOutput>(simulatePhotonCount, queue);
        GPUDataOutput output{};
        queue.fill(m_gpuDataOutput, output, simulatePhotonCount).wait();


        queue.wait();
    }


    void PhotonTracer::freeResources() {
        auto& queue = m_pipelineSettings.queue;
        queue.wait();
        if (m_gpu.imageMemory) {
            sycl::free(m_gpu.imageMemory, queue);
            m_gpu.imageMemory = nullptr;
        }
        if (m_gpu.vertices) {
            sycl::free(m_gpu.vertices, queue);
            m_gpu.vertices = nullptr;
        }
        if (m_gpu.indices) {
            sycl::free(m_gpu.indices, queue);
            m_gpu.indices = nullptr;
        }
        if (m_gpu.indexOffsets) {
            sycl::free(m_gpu.indexOffsets, queue);
            m_gpu.indexOffsets = nullptr;
        }
        if (m_gpu.vertexOffsets) {
            sycl::free(m_gpu.vertexOffsets, queue);
            m_gpu.vertexOffsets = nullptr;
        }
        if (m_gpu.transforms) {
            sycl::free(m_gpu.transforms, queue);
            m_gpu.transforms = nullptr;
        }
        if (m_gpu.materials) {
            sycl::free(m_gpu.materials, queue);
            m_gpu.materials = nullptr;
        }
        if (m_gpu.tagComponents) {
            sycl::free(m_gpu.tagComponents, queue);
            m_gpu.tagComponents = nullptr;
        }
        if (m_gpu.gaussianInputAssembly) {
            sycl::free(m_gpu.gaussianInputAssembly, queue);
            m_gpu.gaussianInputAssembly = nullptr;
        }
        if (m_gpu.gradients) {
            sycl::free(m_gpu.gradients, queue);
            m_gpu.gradients = nullptr;
        }
        if (m_gpu.sumGradients) {
            sycl::free(m_gpu.sumGradients, queue);
            m_gpu.sumGradients = nullptr;
        }
        if (m_gpu.gradientImage) {
            sycl::free(m_gpu.gradientImage, queue);
            m_gpu.gradientImage = nullptr;
        }

        if (m_gpu.pinholeCamera) {
            sycl::free(m_gpu.pinholeCamera, queue);
            m_gpu.pinholeCamera = nullptr;
        }
        if (m_gpu.cameraTransform) {
            sycl::free(m_gpu.cameraTransform, queue);
            m_gpu.cameraTransform = nullptr;
        }
        if (m_gpu.renderInformation) {
            sycl::free(m_gpu.renderInformation, queue);
            m_gpu.renderInformation = nullptr;
        }

        if (m_pcg32) {
            sycl::free(m_pcg32, queue);
            m_pcg32 = nullptr;
        }
        if (m_gpuDataOutput) {
            sycl::free(m_gpuDataOutput, queue);
            m_gpuDataOutput = nullptr;
        }
        queue.wait();
        Log::Logger::getInstance()->info("Freeing Path tracer GPU/SYCL resources");
    }


    void PhotonTracer::uploadGaussiansFromTensors(GPUDataTensors& data) {

#ifdef DIFF_RENDERER_ENABLED
        freeResources();
        prepareImageAndInfoBuffers();

        auto& queue = m_pipelineSettings.queue;
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
            point.emission = emissionsPtr[i]; // emission = 0
            point.color = colorsPtr[i]; // color = 1
            point.diffuse = specularPtr[i]; // diffuse = 0.5
            point.specular = diffusePtr[i]; // specular = 0.5
            point.phongExponent = 32; // phongExponent = 32

            hostGaussians[i] = point;
        }

        // 8) Allocate device memory and copy
        m_gpu.gaussianInputAssembly = sycl::malloc_device<GaussianInputAssembly>(N, queue);
        queue.memcpy(m_gpu.gaussianInputAssembly, hostGaussians.data(), N * sizeof(GaussianInputAssembly));

        uint32_t numPhotons = 500000;
        m_gpu.gradients = sycl::malloc_device<glm::vec3>(numPhotons, queue);
        queue.fill(m_gpu.gradients, glm::vec3(0.0f), numPhotons);

        m_gpu.sumGradients = sycl::malloc_device<glm::vec3>(1, queue);
        queue.fill(m_gpu.sumGradients, glm::vec3(0.0f), 1);

        uint32_t imageSize = m_pipelineSettings.width * m_pipelineSettings.height;
        m_gpu.gradientImage = sycl::malloc_device<float>(imageSize, queue);
        queue.fill(m_gpu.gradientImage, 0.0f, imageSize);

        // 9) Set number of gaussians
        m_gpu.numGaussians = N;
        queue.wait();

        // Log
        Log::Logger::getInstance()->info("uploadFromTensors: Uploaded {} Gaussians", N);
#endif

    }

    void PhotonTracer::uploadGaussianData(std::shared_ptr<Scene>& scene) {
        auto& queue = m_pipelineSettings.queue;
        std::vector<GaussianInputAssembly> gaussianInputAssembly;
        std::vector<TransformComponent> transformMatrices; // Transformation matrices for entities
        auto& registry = scene->getRegistry();
        // Find all entities with GaussianComponent
        auto view = scene->getRegistry().view<GaussianComponent2DGS>();
        for (auto e : view) {
            auto& component = Entity(e, scene.get()).getComponent<GaussianComponent2DGS>();
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
            auto& transform = Entity(e, scene.get()).getComponent<TransformComponent>();
            transformMatrices.emplace_back(transform);
        }

        m_gpu.gaussianInputAssembly = sycl::malloc_device<GaussianInputAssembly>(gaussianInputAssembly.size(), queue);
        queue.memcpy(m_gpu.gaussianInputAssembly, gaussianInputAssembly.data(),
                     gaussianInputAssembly.size() * sizeof(GaussianInputAssembly));
        m_gpu.numGaussians = gaussianInputAssembly.size(); // Number of entities for rendering

        Log::Logger::getInstance()->info("Uploaded  {} Gaussians to renderkernel", m_gpu.numGaussians);
        queue.wait();
    }

    void PhotonTracer::uploadVertexData(std::shared_ptr<Scene>& scene) {
        std::vector<InputAssembly> vertexData;
        std::vector<uint32_t> indices;
        std::vector<uint32_t> indexOffsets; // Offset for each entity's indices
        std::vector<uint32_t> vertexOffsets; // Offset for each entity's vertices
        std::vector<TransformComponent> transformMatrices; // Transformation matrices for entities
        std::vector<MaterialComponent> materials; // Transformation matrices for entities
        std::vector<TagComponent> tagComponents; // Transformation matrices for entities
        auto view = scene->getRegistry().view<MeshComponent, TransformComponent>();
        uint32_t currentVertexOffset = 0;
        uint32_t currentIndexOffset = 0;

        for (auto e : view) {
            Entity entity(e, scene.get());
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

            MeshManager meshManager;
            std::shared_ptr<MeshData> meshData = meshManager.getMeshData(meshComponent);
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
        // Upload vertex data to GPU // Split each assignment for debug purposes
        auto& queue = m_pipelineSettings.queue;
        m_gpu.vertices = sycl::malloc_device<InputAssembly>(vertexData.size(), queue);
        queue.memcpy(m_gpu.vertices, vertexData.data(), vertexData.size() * sizeof(InputAssembly));
        queue.wait();

        // Upload index data to GPU
        m_gpu.indices = sycl::malloc_device<uint32_t>(indices.size(), queue);
        queue.memcpy(m_gpu.indices, indices.data(), indices.size() * sizeof(uint32_t));
        queue.wait();

        // Upload transform matrices to GPU
        m_gpu.transforms = sycl::malloc_device<TransformComponent>(transformMatrices.size(), queue);
        queue.memcpy(m_gpu.transforms, transformMatrices.data(), transformMatrices.size() * sizeof(TransformComponent));
        queue.wait();

        // Upload material data to GPU
        m_gpu.materials = sycl::malloc_device<MaterialComponent>(materials.size(), queue);
        queue.memcpy(m_gpu.materials, materials.data(), materials.size() * sizeof(MaterialComponent));
        queue.wait();

        // Upload tag components to GPU
        m_gpu.tagComponents = sycl::malloc_device<TagComponent>(tagComponents.size(), queue);
        queue.memcpy(m_gpu.tagComponents, tagComponents.data(), tagComponents.size() * sizeof(TagComponent));
        queue.wait();

        // Upload vertex offsets to GPU (if necessary for rendering)
        m_gpu.vertexOffsets = sycl::malloc_device<uint32_t>(vertexOffsets.size(), queue);
        queue.memcpy(m_gpu.vertexOffsets, vertexOffsets.data(), vertexOffsets.size() * sizeof(uint32_t));
        queue.wait();

        // Upload index offsets to GPU
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
        if (m_backwardInfo.gradients) {
            delete[] m_backwardInfo.gradients;
        }
        if (m_backwardInfo.sumGradients) {
            delete m_backwardInfo.sumGradients;
        }
        freeResources();
    }


    /* Draw rays
     {
        auto view = m_scene->getRegistry().view<CameraComponent, TransformComponent, MeshComponent>();
        for (auto e: view) {
            Entity entity(e, m_scene.get());
            auto &transform = entity.getComponent<TransformComponent>();
            auto camera = std::dynamic_pointer_cast<PinholeCamera>(entity.getComponent<CameraComponent>().camera);
            if (!camera || entity.getComponent<CameraComponent>().isActiveCamera())
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
