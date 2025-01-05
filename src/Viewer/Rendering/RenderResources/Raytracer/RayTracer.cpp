//
// Created by magnus on 11/27/24.
//

#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/RenderResources/Raytracer/RayTracer.h"

#include "LightTracerKernels.h"
#include "Viewer/Rendering/RenderResources/Raytracer/PathTracerMeshKernels.h"

#include "Viewer/Rendering/Components/GaussianComponent.h"
#include "Viewer/Tools/SyclDeviceSelector.h"
#include "RayTracerKernels.h"

namespace VkRender::RT {
    RayTracer::RayTracer(Application *ctx, std::shared_ptr<Scene> &scene, uint32_t width, uint32_t height) : m_context(
            ctx), m_selector(SyclDeviceSelector(SyclDeviceSelector::DeviceType::GPU)) {
        ;

        m_scene = scene;
        m_width = width;
        m_height = height;
        // Load the scene into gpu memory
        // Create image memory
        // Allocate host memory for RGBA image (4 floats per pixel)
        m_imageMemory = new float[width * height * 4];

        // Allocate device memory for RGBA image (4 floats per pixel)
        m_gpu.imageMemory = sycl::malloc_device<float>(width * height * 4, m_selector.getQueue());
        if (!m_gpu.imageMemory) {
            throw std::runtime_error("Device memory allocation failed.");
        }
        // Allocate device memory for RGBA image (4 floats per pixel)
        m_gpu.imageMemory = sycl::malloc_device<float>(width * height * 4, m_selector.getQueue());
        if (!m_gpu.imageMemory) {
            throw std::runtime_error("Device memory allocation failed.");
        }

        m_renderInformation = std::make_unique<RenderInformation>();
        m_gpu.renderInformation = sycl::malloc_device<RenderInformation>(1, m_selector.getQueue());
        if (!m_gpu.renderInformation) {
            throw std::runtime_error("Device memory allocation failed.");
        }
        m_selector.getQueue().memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation));

        // Initialize device memory to 0
        m_selector.getQueue().fill(m_gpu.imageMemory, 0.0f, width * height * 4).wait();

        // Initialize host memory to 0
        std::fill(m_imageMemory, m_imageMemory + (width * height * 4), 0.0f);
        Log::Logger::getInstance()->info("Creating Ray Tracer. Image dimensions are: {}x{}", width, height);
        upload(scene);
    }

    void RayTracer::upload(std::shared_ptr<Scene> scene) {
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
        uploadVertexData(scene);
        uploadGaussianData(scene);
    }

    void RayTracer::uploadGaussianData(std::shared_ptr<Scene> &scene) {
        auto &queue = m_selector.getQueue();
        std::vector<GaussianInputAssembly> gaussianInputAssembly;
        auto &registry = scene->getRegistry();
        // Find all entities with GaussianComponent
        auto view = scene->getRegistry().view<GaussianComponent2DGS>();
        for (auto e: view) {
            auto &component = Entity(e, scene.get()).getComponent<GaussianComponent2DGS>();
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
        }

        m_gpu.gaussianInputAssembly = sycl::malloc_device<GaussianInputAssembly>(gaussianInputAssembly.size(), queue);
        queue.memcpy(m_gpu.gaussianInputAssembly, gaussianInputAssembly.data(),
                     gaussianInputAssembly.size() * sizeof(uint32_t));
        queue.wait();

        m_gpu.numGaussians = gaussianInputAssembly.size(); // Number of entities for rendering
    }

    void RayTracer::uploadVertexData(std::shared_ptr<Scene> &scene) {
        std::vector<InputAssembly> vertexData;
        std::vector<uint32_t> indices;
        std::vector<uint32_t> indexOffsets;       // Offset for each entity's indices
        std::vector<uint32_t> vertexOffsets;     // Offset for each entity's vertices
        std::vector<TransformComponent> transformMatrices; // Transformation matrices for entities
        std::vector<MaterialComponent> materials; // Transformation matrices for entities
        std::vector<TagComponent> tagComponents; // Transformation matrices for entities
        auto view = scene->getRegistry().view<MeshComponent, TransformComponent>();
        uint32_t currentVertexOffset = 0;
        uint32_t currentIndexOffset = 0;

        for (auto e: view) {
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
                    auto &visibility = current.getComponent<VisibleComponent>();
                    // If visibility is set to false, mark to skip this entity
                    if (!visibility.visible) {
                        skipEntity = true;
                        break; // No need to check further ancestors
                    }
                }
            }
            if (entity.hasComponent<CameraComponent>())
                skipEntity = true;
            // If an ancestor with visible == false was found, skip to the next entity
            if (skipEntity) {
                continue;
            }
            auto &transform = entity.getComponent<TransformComponent>();
            transformMatrices.emplace_back(transform);

            auto tagComponent = entity.getComponent<TagComponent>();
            tagComponents.emplace_back(tagComponent);

            if (entity.hasComponent<MaterialComponent>()) {
                auto &material = entity.getComponent<MaterialComponent>();
                materials.emplace_back(material);
            }


            auto &meshComponent = entity.getComponent<MeshComponent>();
            if (meshComponent.meshDataType() != OBJ_FILE)
                continue;

            std::shared_ptr<MeshData> meshData = m_meshManager.getMeshData(meshComponent);
            // Store vertex offset
            vertexOffsets.push_back(currentVertexOffset);
            // Add vertex data
            for (auto &vert: meshData->vertices) {
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
            for (auto idx: meshData->indices) {
                indices.push_back(idx + vertexOffsets.back()); // Adjust indices by vertex offset
            }
            currentIndexOffset += meshData->indices.size();
        }
        // Upload vertex data to GPU
        auto &queue = m_selector.getQueue();
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


    void RayTracer::update(const EditorImageUI &editorImageUI) {
        /*
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

        m_renderInformation->frameID++;
        uint32_t totalPhotons = 100000000;

        auto &queue = m_selector.getQueue();

        if (editorImageUI.clearImageMemory) {
            queue.fill(m_gpu.imageMemory, static_cast<float>(0), m_width * m_height).wait();
            m_frameID = 0;
            m_renderInformation->photonsAccumulated = 0;
        }
        queue.memcpy(m_gpu.renderInformation, m_renderInformation.get(), sizeof(RenderInformation));

        auto cameraEntity = m_scene->getEntityByName("Camera");
        if (cameraEntity && !editorImageUI.clearImageMemory) {
            auto camera = cameraEntity.getComponent<CameraComponent>().getPinholeCamera();
            auto transform = cameraEntity.getComponent<TransformComponent>();

            PinholeCamera cam;
            auto cameraGPU = sycl::malloc_device<PinholeCamera>(1, queue);
            queue.memcpy(cameraGPU, camera.get(), sizeof(PinholeCamera));


            if (editorImageUI.kernel == "Path Tracer: 2DGS") {
                sycl::range<1> globalRange(totalPhotons);
                queue.submit([&](sycl::handler &cgh) {
                    // Capture GPUData, etc. by value or reference as needed
                    RenderKernelLightTracing kernel(m_gpu, totalPhotons, m_width, m_height, m_width * m_height * 4,
                                                    transform, cameraGPU, 1);
                    cgh.parallel_for(globalRange, kernel);
                });
            } else if (editorImageUI.kernel == "Path Tracer: Mesh") {
                sycl::range<1> globalRange(totalPhotons);
                queue.submit([&](sycl::handler &cgh) {
                    // Capture GPUData, etc. by value or reference as needed
                    PathTracerMeshKernels kernel(m_gpu, totalPhotons, transform, cameraGPU, 4, m_frameID);
                    cgh.parallel_for(globalRange, kernel);
                });
            } else if (editorImageUI.kernel == "Hit-Test") {
                uint32_t tileWidth = 16;
                uint32_t tileHeight = 16;
                sycl::range localWorkSize(tileHeight, tileWidth);
                sycl::range globalWorkSize(m_height, m_width);

                queue.submit([&](sycl::handler &h) {
                    // Create a kernel instance with the required parameters
                    Kernels::RenderKernel kernel(m_gpu, m_width, m_height, m_width * m_height * 4, transform,
                                                 cameraGPU);
                    h.parallel_for<class RenderKernel>(
                            sycl::nd_range<2>(globalWorkSize, localWorkSize), kernel);
                });
            }
            queue.wait();

            sycl::free(cameraGPU, queue);
            m_frameID++;
        }

        queue.memcpy(m_renderInformation.get(), m_gpu.renderInformation, sizeof(RenderInformation));
        queue.memcpy(m_imageMemory, m_gpu.imageMemory, m_width * m_height * sizeof(float)).wait();
        if (editorImageUI.saveImage)
            saveAsPFM("cornell.pfm");

        Log::Logger::getInstance()->infoWithFrequency("PhotonCount", 60, "simulated {} Billion photons. About {} Photons, {}% hit the sensor",
            totalPhotons/1e9 * m_renderInformation->frameID, m_renderInformation->photonsAccumulated / 1000, m_renderInformation->photonsAccumulated / totalPhotons * 100);
    }

    RayTracer::~RayTracer() {
        if (m_imageMemory) {
            delete[] m_imageMemory;
        }

        if (m_gpu.imageMemory) {
            sycl::free(m_gpu.imageMemory, m_selector.getQueue());
        }
    }

    void RayTracer::saveAsPPM(const std::filesystem::path &filename) const {
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

    void RayTracer::saveAsPFM(const std::filesystem::path &filename) const {
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
        file.write(reinterpret_cast<const char *>(rgbData.data()), rgbData.size() * sizeof(float));

        if (!file) {
            throw std::runtime_error("Failed to write PFM data to file: " + filename.string());
        }

        file.close();
    }
}
