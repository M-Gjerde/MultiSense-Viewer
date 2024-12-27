//
// Created by magnus on 11/27/24.
//

#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/RenderResources/Raytracer/RayTracer.h"

#include "LightTracerKernels.h"
#include "Viewer/Rendering/RenderResources/Raytracer/RayTracerKernels.h"
#include "Viewer/Rendering/Components/GaussianComponent.h"
#include "Viewer/Tools/SyclDeviceSelector.h"

namespace VkRender::RT {
    RayTracer::RayTracer(Application* ctx, std::shared_ptr<Scene>& scene, uint32_t width, uint32_t height) : m_context(
        ctx) {
        m_scene = scene;
        m_width = width;
        m_height = height;
        m_camera = BaseCamera(width / height);
        // Load the scene into gpu memory
        // Create image memory
        m_imageMemory = new uint8_t[width * height * 4]; // Assuming RGBA8 image
        m_gpu.imageMemory = sycl::malloc_device<uint8_t>(m_width * m_height * 4, m_selector.getQueue());
        if (!m_gpu.imageMemory) {
            throw std::runtime_error("Device memory allocation failed.");
        }

        upload(scene);
    }

    void RayTracer::upload(std::shared_ptr<Scene> scene){
        if (m_gpu.gaussianInputAssembly){
            sycl::free(m_gpu.gaussianInputAssembly, m_selector.getQueue());
            m_gpu.gaussianInputAssembly = nullptr;
        }
        if (m_gpu.vertices){
            sycl::free(m_gpu.vertices, m_selector.getQueue());
            m_gpu.vertices = nullptr;
        }
        if (m_gpu.indices){
            sycl::free(m_gpu.indices, m_selector.getQueue());
            m_gpu.indices = nullptr;
        }
        uploadVertexData(scene);
        uploadGaussianData(scene);
    }

    void RayTracer::uploadGaussianData(std::shared_ptr<Scene>& scene) {
        std::vector<GaussianInputAssembly> gaussianData;

        auto view = scene->getRegistry().view<GaussianComponent, TransformComponent>();
        for (auto e : view) {
            Entity entity(e, scene.get());
            std::string tag = entity.getName();
            auto& gaussianComponent = entity.getComponent<GaussianComponent>();

            for (size_t i = 0; i < gaussianComponent.size(); ++i) {
                GaussianInputAssembly point{};
                point.position = gaussianComponent.means[i];
                point.normal = glm::vec3(0.0f, 0.0f, 1.0f);;
                point.scale = gaussianComponent.scales[i];
                point.intensity = gaussianComponent.opacities[i];
                gaussianData.push_back(point);
            }
        }
        m_gpu.numGaussians = gaussianData.size();

        auto& queue = m_selector.getQueue();
        m_gpu.gaussianInputAssembly = sycl::malloc_device<GaussianInputAssembly>(gaussianData.size(), queue);
        queue.memcpy(m_gpu.gaussianInputAssembly, gaussianData.data(), gaussianData.size() * sizeof(GaussianInputAssembly));
        queue.wait();
    }

    void RayTracer::uploadVertexData(std::shared_ptr<Scene>& scene) {
        std::vector<InputAssembly> vertexData;
        std::vector<uint32_t> indices;

        auto view = scene->getRegistry().view<MeshComponent, TransformComponent>();
        for (auto e : view) {
            Entity entity(e, scene.get());
            std::string tag = entity.getName();
            auto& meshComponent = entity.getComponent<MeshComponent>();
            if (meshComponent.meshDataType() != OBJ_FILE)
                continue;
            std::shared_ptr<MeshData> meshData = m_meshManager.getMeshData(meshComponent);
            for (auto& vert : meshData->vertices) {
                InputAssembly input{};
                input.position = vert.pos;
                input.color = vert.color;
                input.normal = vert.normal;
                vertexData.emplace_back(input);
            }
            indices = meshData->indices;
            m_gpu.numVertices = vertexData.size();
            m_gpu.numIndices = indices.size();
        }

        auto& queue = m_selector.getQueue();
        m_gpu.vertices = sycl::malloc_device<InputAssembly>(vertexData.size(), queue);
        queue.memcpy(m_gpu.vertices, vertexData.data(), vertexData.size() * sizeof(InputAssembly));
        m_gpu.indices = sycl::malloc_device<uint32_t>(indices.size(), queue);
        queue.memcpy(m_gpu.indices, indices.data(), indices.size() * sizeof(uint32_t));
        queue.wait();
    }

    void RayTracer::update(bool update) {
        {
            auto view = m_scene->getRegistry().view<CameraComponent, TransformComponent, MeshComponent>();
            for (auto e : view) {
                Entity entity(e, m_scene.get());
                auto& transform = entity.getComponent<TransformComponent>();
                auto camera = std::dynamic_pointer_cast<PinholeCamera>(entity.getComponent<CameraComponent>().camera);
                if (!camera)
                    continue;
                float fx = camera->m_fx;
                float fy = camera->m_fy;
                float cx = camera->m_cx;
                float cy = camera->m_cy;
                float width = camera->m_width;
                float height = camera->m_height;


                // Helper lambda to create a ray entity
                auto updateRayEntity = [&](Entity cornerEntity, float x, float y) {
                    MeshComponent* mesh;
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


        if (update) {
            auto& queue = m_selector.getQueue();

            auto cameraEntity = m_scene->getEntityByName("Camera");
            if (cameraEntity) {
                auto camera = std::dynamic_pointer_cast<PinholeCamera>(cameraEntity.getComponent<CameraComponent>().camera);
                auto& transform = cameraEntity.getComponent<TransformComponent>();


                uint32_t totalPhotons = 100000;
                sycl::range<1> globalRange(totalPhotons);

                queue.submit([&](sycl::handler& cgh) {
                    // Capture GPUData, etc. by value or reference as needed
                    RenderKernelLightTracing kernel(m_gpu, totalPhotons, m_width, m_height, m_width * m_height * 4, transform, *camera, 1);
                    cgh.parallel_for(globalRange, kernel);
                });

                queue.memcpy(m_imageMemory, m_gpu.imageMemory, m_width * m_height * 4);
                queue.wait();

                //saveAsPPM("sycl.ppm");;
            }
        }
    }

    RayTracer::~RayTracer() {
        if (m_imageMemory) {
            delete[] m_imageMemory;
        }

        if (m_gpu.imageMemory) {
            sycl::free(m_gpu.imageMemory, m_selector.getQueue());
        }
    }

    void RayTracer::saveAsPPM(const std::filesystem::path& filename) const {
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
}
