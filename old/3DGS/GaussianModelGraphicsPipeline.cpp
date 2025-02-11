//
// Created by magnus on 8/15/24.
//
#include <glm/ext/quaternion_float.hpp>

#include "GaussianModelGraphicsPipeline.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/VkRender/RenderResources/2DGS/RasterizerUtils2DGS.h"
#include "Rasterizer.h"

namespace VkRender {


    GaussianModelGraphicsPipeline::GaussianModelGraphicsPipeline(VulkanDevice &vulkanDevice,
                                                                 RenderPassInfo &renderPassInfo,
                                                                 uint32_t width,
                                                                 uint32_t height) :
            m_vulkanDevice(vulkanDevice),
            m_renderPassInfo(std::move(renderPassInfo)),
            m_width(width), m_height(height) {

        try {
            // Create a queue using the CPU device selector
            auto gpuSelector = [](const sycl::device &dev) {
                if (dev.is_gpu()) {
                    return 1; // Positive value to prefer GPU devices
                } else {
                    return -1; // Negative value to reject non-GPU devices
                }
            };

            auto cpuSelector = [](const sycl::device &dev) {
                if (dev.is_cpu()) {
                    return 1; // Positive value to prefer GPU devices
                } else {
                    return -1; // Negative value to reject non-GPU devices
                }
            };    // Define a callable device selector using a lambda

            queue = sycl::queue(gpuSelector);
            // Use the queue for your computation
        } catch (const sycl::exception &e) {
            Log::Logger::getInstance()->warning("GPU device not found");
            Log::Logger::getInstance()->info("Falling back to default device selector");
            // Fallback to default device selector
            queue = sycl::queue(sycl::property::queue::in_order());
        }

        m_numSwapChainImages = renderPassInfo.swapchainImageCount;
        m_renderData.resize(m_numSwapChainImages);

        m_vertexShader = "SYCLRenderer.vert";
        m_fragmentShader = "SYCLRenderer.frag";


        m_textureVideo = std::make_shared<TextureVideo>(width, height, &m_vulkanDevice,
                                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                        VK_FORMAT_R8G8B8A8_UNORM);

        setupUniformBuffers();
        setupDescriptors();
        setupPipeline();

        Log::Logger::getInstance()->info("Selected Device {}",
                                         queue.get_device().get_info<sycl::info::device::name>().c_str());
    }


    void GaussianModelGraphicsPipeline::bind(GaussianModelComponent &modelComponent) {

        auto &gs = modelComponent.getGaussians();
        Log::Logger::getInstance()->info("Loaded {} Gaussians", gs.getSize());
        try {
            uint32_t numPoints = gs.getSize();
            m_numPoints = numPoints;
            m_shDim = gs.getShDim();

            positionBuffer = sycl::malloc_device<glm::vec3>(numPoints, queue);
            normalsBuffer = sycl::malloc_device<glm::vec3>(numPoints, queue);
            scalesBuffer = sycl::malloc_device<glm::vec3>(numPoints, queue);
            quaternionBuffer = sycl::malloc_device<glm::quat>(numPoints, queue);
            opacityBuffer = sycl::malloc_device<float>(numPoints, queue);
            sphericalHarmonicsBuffer = sycl::malloc_device<float>(gs.sphericalHarmonics.size(), queue);

            queue.memcpy(positionBuffer, gs.positions.data(), numPoints * sizeof(glm::vec3));
            queue.memcpy(normalsBuffer, gs.normals.data(), numPoints * sizeof(glm::vec3));
            queue.memcpy(scalesBuffer, gs.scales.data(), numPoints * sizeof(glm::vec3));
            queue.memcpy(quaternionBuffer, gs.quats.data(), numPoints * sizeof(glm::quat));
            queue.memcpy(opacityBuffer, gs.opacities.data(), numPoints * sizeof(float));
            queue.memcpy(sphericalHarmonicsBuffer, gs.sphericalHarmonics.data(),
                         gs.sphericalHarmonics.size() * sizeof(float));
            queue.wait_and_throw();

            numTilesTouchedBuffer = sycl::malloc_device<uint32_t>(numPoints, queue);
            pointOffsets = sycl::malloc_device<uint32_t>(numPoints, queue);
            pointsBuffer = sycl::malloc_device<Rasterizer::GaussianPoint>(numPoints, queue);

            uint32_t sortBufferSize = (1 << 25);
            keysBuffer = sycl::malloc_device<uint32_t>(sortBufferSize, queue);
            valuesBuffer = sycl::malloc_device<uint32_t>(sortBufferSize, queue);

            m_imageSize = m_width * m_height * 4;
            m_image = reinterpret_cast<uint8_t *>(std::malloc(m_width * m_height * 4));
            sorter = std::make_unique<Sorter>(queue, sortBufferSize);

            imageBuffer = sycl::malloc_device<uint8_t>(m_width * m_height * 4, queue);
            rangesBuffer = sycl::malloc_device<glm::ivec2>((m_width / 16) * (m_height / 16), queue);
            m_boundBuffers = true;


        } catch (sycl::exception &e) {
            std::cerr << "Caught a SYCL exception: " << e.what() << std::endl;
            return;
        } catch (std::exception &e) {
            std::cerr << "Caught a standard exception: " << e.what() << std::endl;
            return;
        } catch (...) {
            std::cerr << "Caught an unknown exception." << std::endl;
            return;
        }


        queue.wait_and_throw();

    }


    GaussianModelGraphicsPipeline::~GaussianModelGraphicsPipeline() {
        if (m_boundBuffers) {
            sycl::free(positionBuffer, queue);
            sycl::free(normalsBuffer, queue);
            sycl::free(scalesBuffer, queue);
            sycl::free(quaternionBuffer, queue);
            sycl::free(opacityBuffer, queue);
            sycl::free(sphericalHarmonicsBuffer, queue);
            sycl::free(numTilesTouchedBuffer, queue);
            sycl::free(pointsBuffer, queue);
            sycl::free(imageBuffer, queue);
            sycl::free(rangesBuffer, queue);

            sycl::free(keysBuffer, queue);
            sycl::free(valuesBuffer, queue);
            free(m_image);
        }

        cleanUp();
    }

    uint8_t *GaussianModelGraphicsPipeline::getImage() {
        return m_image;
    }

    uint32_t GaussianModelGraphicsPipeline::getImageSize() {
        return m_imageSize;
    }


    void GaussianModelGraphicsPipeline::generateImage(Camera &camera, int colorType) {
        auto params = Rasterizer::getHtanfovxyFocal(camera.m_Fov, camera.m_height, camera.m_width);
        glm::mat4 viewMatrix = camera.matrices.view;
        glm::mat4 projectionMatrix = camera.matrices.perspective;
        glm::vec3 camPos = camera.pose.pos;
        // Flip the second row of the projection matrix
        projectionMatrix[1] = -projectionMatrix[1];
        uint32_t BLOCK_X = 16, BLOCK_Y = 16;
        const uint32_t imageWidth = m_width;
        const uint32_t imageHeight = m_height;
        const uint32_t tileWidth = 16;
        const uint32_t tileHeight = 16;
        glm::vec3 tileGrid((imageWidth + BLOCK_X - 1) / BLOCK_X, (imageHeight + BLOCK_Y - 1) / BLOCK_Y, 1);
        uint32_t numTiles = tileGrid.x * tileGrid.y;

        // Start timing
        // Preprocess
        try {
            auto startAll = std::chrono::high_resolution_clock::now();

            auto startWaitForQueue = std::chrono::high_resolution_clock::now();
            queue.wait();
            std::chrono::duration<double, std::milli> waitForQueueDuration =
                    std::chrono::high_resolution_clock::now() - startWaitForQueue;

            queue.fill(imageBuffer, static_cast<uint8_t>(0x00), m_width * m_height * 4);

            size_t numPoints = m_numPoints;
            Rasterizer::PreprocessInfo scene{};
            scene.projectionMatrix = projectionMatrix;
            scene.viewMatrix = viewMatrix;
            scene.params = params;
            scene.camPos = camPos;
            scene.tileGrid = tileGrid;
            scene.height = m_height;
            scene.width = m_width;
            scene.shDim = m_shDim;
            scene.shDegree = 1.0;
            scene.colorMethod = colorType;

            auto startPreprocess = std::chrono::high_resolution_clock::now();
            queue.fill(numTilesTouchedBuffer, 0x00, numPoints).wait();

            queue.submit([&](sycl::handler &h) {

                h.parallel_for(sycl::range<1>(numPoints),
                               Rasterizer::Preprocess(positionBuffer, normalsBuffer, scalesBuffer,
                                                      quaternionBuffer, opacityBuffer,
                                                      sphericalHarmonicsBuffer,
                                                      numTilesTouchedBuffer,
                                                      pointsBuffer, &scene));
            }).wait();

            auto endPreprocess = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> preprocessDuration = endPreprocess - startPreprocess;
            auto startInclusiveSum = std::chrono::high_resolution_clock::now();
            //queue.fill(pointOffsets, 0x00, numPoints).wait();

            queue.submit([&](sycl::handler &h) {
                h.parallel_for(sycl::range<1>(1),
                               Rasterizer::InclusiveSum(numTilesTouchedBuffer, pointOffsets, numPoints));
            }).wait();


            uint32_t numRendered = 1;
            queue.memcpy(&numRendered, pointOffsets + (numPoints - 1), sizeof(uint32_t)).wait();
            //printf("3DGS Rendering: Total Gaussians: %u. %uM\n", numRendered, numRendered / 1e6);

            Log::Logger::getInstance()->info("3DGS Rendering: Total Gaussians: {}. {:.3f}M", numRendered,
                                             numRendered / 1e6);
            if (numRendered > 0) {


                std::chrono::duration<double, std::milli> inclusiveSumDuration =
                        std::chrono::high_resolution_clock::now() - startInclusiveSum;

                auto startDuplicateGaussians = std::chrono::high_resolution_clock::now();
                float *depthValues = sycl::malloc_device<float>(numPoints, queue);

                queue.submit([&](sycl::handler &h) {
                    h.parallel_for<class duplicates>(sycl::range<1>(numPoints),
                                                     Rasterizer::DuplicateGaussians(pointsBuffer, pointOffsets,
                                                                                    keysBuffer,
                                                                                    valuesBuffer,
                                                                                    depthValues,
                                                                                    numRendered, tileGrid));
                }).wait();

                std::chrono::duration<double, std::milli> duplicateGaussiansDuration =
                        std::chrono::high_resolution_clock::now() - startDuplicateGaussians;
                auto startSorting = std::chrono::high_resolution_clock::now();

                float maxDepthValue = -1;
                float minDepthValue = -1;
                if (colorType == 2) {
                    std::vector<float> depthValuesHost(numPoints);
                    queue.memcpy(depthValuesHost.data(), depthValues, sizeof(float) * numPoints).wait();
                    std::sort(depthValuesHost.begin(), depthValuesHost.end());
                    maxDepthValue = depthValuesHost.back();
                    // Find the minimum non-zero value
                    minDepthValue = 0.0f; // Initialize to 0 or another appropriate default value
                    for (const auto &value: depthValuesHost) {
                        if (value > 0.0f) {
                            minDepthValue = value;
                            break; // Stop as soon as we find the first non-zero value
                        }
                    }
                }


                sycl::free(depthValues, queue);

                /*
                uint32_t numKeys = 1 << 13;
                std::random_device rd;
                std::mt19937 gen(rd());
                gen.seed(42);
                std::uniform_int_distribution<uint32_t> dis(0, 1 << 5);

                std::vector<uint32_t> keys(numKeys);
                for (auto &key: keys) {
                    key = dis(gen);
                }
                queue.memcpy(keysBuffer, keys.data(), numKeys * sizeof(uint32_t));
                //queue.memcpy(keys.data(), keysBuffer, numRendered * sizeof(uint32_t)).wait();
                */

                // Load keys

                sorter->performOneSweep(keysBuffer, valuesBuffer, numRendered);
                //sorter->verifySort(keysBuffer, numRendered); //, true, keys);


                std::chrono::duration<double, std::milli> sortingDuration =
                        std::chrono::high_resolution_clock::now() - startSorting;

                auto startIdentifyTileRanges = std::chrono::high_resolution_clock::now();
                queue.memset(rangesBuffer, static_cast<int>(0x00), numTiles * (sizeof(int) * 2)).wait();

                queue.submit([&](sycl::handler &h) {
                    h.parallel_for<class IdentifyTileRanges>(numRendered,
                                                             Rasterizer::IdentifyTileRanges(rangesBuffer, keysBuffer,
                                                                                            numRendered));
                }).wait();
                std::chrono::duration<double, std::milli> identifyTileRangesDuration =
                        std::chrono::high_resolution_clock::now() - startIdentifyTileRanges;

                auto startRenderGaussians = std::chrono::high_resolution_clock::now();
                // Compute the global work size ensuring it is a multiple of the local work size
                sycl::range<2> localWorkSize(tileHeight, tileWidth);
                size_t globalWidth = ((imageWidth + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
                size_t globalHeight = ((imageHeight + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
                sycl::range<2> globalWorkSize(globalHeight, globalWidth);
                uint32_t horizontal_blocks = (imageWidth + tileWidth - 1) / tileWidth;


                queue.submit([&](sycl::handler &h) {
                    auto range = sycl::nd_range<2>(globalWorkSize, localWorkSize);
                    h.parallel_for<class RenderGaussians>(range,
                                                          Rasterizer::RasterizeGaussians(rangesBuffer, keysBuffer,
                                                                                         valuesBuffer, pointsBuffer,
                                                                                         imageBuffer, numRendered,
                                                                                         imageWidth, imageHeight,
                                                                                         horizontal_blocks, numTiles,
                                                                                         maxDepthValue, minDepthValue));
                }).wait();


                /*
                queue.submit([&](sycl::handler &h) {
                    sycl::range<1> globalWorkSize(imageHeight * imageWidth * 4);
                    sycl::range<1> localWorkSize(32);

                    h.parallel_for<class RenderGaussians>(sycl::range<1>(m_width * m_height),
                                                          Rasterizer::RasterizeGaussiansPerPixel(rangesBuffer, keysBuffer,
                                                                                         valuesBuffer, pointsBuffer,
                                                                                         imageBuffer, numRendered,
                                                                                         imageWidth, imageHeight,
                                                                                         horizontal_blocks, numTiles));
                });
                */
                std::chrono::duration<double, std::milli> renderGaussiansDuration =
                        std::chrono::high_resolution_clock::now() - startRenderGaussians;
                sorter->resetMemory();

                auto startCopyImageToHost = std::chrono::high_resolution_clock::now();
                queue.memcpy(m_image, imageBuffer, m_width * m_height * 4);
                queue.wait();

                Rasterizer::saveAsPPM(m_image, m_width, m_height, "../output.ppm");
                std::chrono::duration<double, std::milli> copyImageDuration =
                        std::chrono::high_resolution_clock::now() - startCopyImageToHost;


                std::chrono::duration<double, std::milli> totalDuration =
                        std::chrono::high_resolution_clock::now() - startAll;

                bool exceedTimeLimit = (totalDuration.count() > 500);
                logTimes(waitForQueueDuration, preprocessDuration, inclusiveSumDuration, duplicateGaussiansDuration,
                         sortingDuration, identifyTileRangesDuration, renderGaussiansDuration, copyImageDuration,
                         totalDuration, exceedTimeLimit);

            } else {
                std::memset(m_image, static_cast<uint8_t>(0x00), m_width * m_height * 4);
            }

        } catch (sycl::exception &e) {
            std::cerr << "Caught a SYCL exception: " << e.what() << std::endl;
            return;
        } catch (std::exception &e) {
            std::cerr << "Caught a standard exception: " << e.what() << std::endl;
            return;
        } catch (...) {
            std::cerr << "Caught an unknown exception." << std::endl;
            return;
        }

        int rowSize = m_width * 4; // 4 bytes per pixel for RGBA8
        std::vector<uint8_t> tempRow(rowSize);

        for (int y = 0; y < m_height / 2; ++y) {
            uint8_t *row1 = m_image + y * rowSize;
            uint8_t *row2 = m_image + (m_height - y - 1) * rowSize;

            // Swap rows
            std::memcpy(tempRow.data(), row1, rowSize);
            std::memcpy(row1, row2, rowSize);
            std::memcpy(row2, tempRow.data(), rowSize);
        }

        m_textureVideo->updateTextureFromBuffer(m_image, m_width * m_height * 4);

    }

    void GaussianModelGraphicsPipeline::logTimes(std::chrono::duration<double, std::milli> t1,
                                                 std::chrono::duration<double, std::milli> t2,
                                                 std::chrono::duration<double, std::milli> t3,
                                                 std::chrono::duration<double, std::milli> t4,
                                                 std::chrono::duration<double, std::milli> t5,
                                                 std::chrono::duration<double, std::milli> t6,
                                                 std::chrono::duration<double, std::milli> t7,
                                                 std::chrono::duration<double, std::milli> t8,
                                                 std::chrono::duration<double, std::milli> t9,
                                                 bool error) {
        if (error) {
            Log::Logger::getInstance()->error("3DGS Rendering: Wait for ready queue: {}", t1.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Preprocess: {}", t2.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Inclusive Sum: {}", t3.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Duplicate Gaussians: {}",
                                              t4.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Sorting: {}", t5.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Identify Tile Ranges: {}",
                                              t6.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Render Gaussians: {}", t7.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Copy image to host: {}", t8.count());
            Log::Logger::getInstance()->error("3DGS Rendering: Total function duration: {}", t9.count());
        } else {
            Log::Logger::getInstance()->info("3DGS Rendering: Wait for ready queue: {}", t1.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Preprocess: {}", t2.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Inclusive Sum: {}", t3.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Duplicate Gaussians: {}",
                                             t4.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Sorting: {}", t5.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Identify Tile Ranges: {}",
                                             t6.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Render Gaussians: {}", t7.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Copy image to host: {}", t8.count());
            Log::Logger::getInstance()->info("3DGS Rendering: Total function duration: {}", t9.count());
        }


    }

    void GaussianModelGraphicsPipeline::cleanUp() {
        auto logicalDevice = m_vulkanDevice.m_LogicalDevice;
        VkFence fence;
        VkFenceCreateInfo fenceInfo = Populate::fenceCreateInfo(0);
        vkCreateFence(logicalDevice, &fenceInfo, nullptr, &fence);

        Indices &indices = this->indices;
        Vertices &vertices = this->vertices;
        VkDescriptorSetLayout layout = m_sharedRenderData.descriptorSetLayout;
        VkDescriptorPool pool = m_sharedRenderData.descriptorPool;

        VulkanResourceManager::getInstance().deferDeletion(
                [logicalDevice, indices, vertices, layout, pool]() {
                    vkDestroyDescriptorSetLayout(logicalDevice, layout, nullptr);
                    vkDestroyDescriptorPool(logicalDevice, pool, nullptr);

                    if (vertices.buffer != VK_NULL_HANDLE) {
                        vkDestroyBuffer(logicalDevice, vertices.buffer, nullptr);
                    }
                    if (vertices.memory != VK_NULL_HANDLE) {
                        vkFreeMemory(logicalDevice, vertices.memory, nullptr);
                    }
                    if (indices.buffer != VK_NULL_HANDLE) {
                        vkDestroyBuffer(logicalDevice, indices.buffer, nullptr);
                    }
                    if (indices.memory != VK_NULL_HANDLE) {
                        vkFreeMemory(logicalDevice, indices.memory, nullptr);
                    }
                },
                fence, "Cleaning Up GraphicsPipeline);


    }

    void GaussianModelGraphicsPipeline::setupUniformBuffers() {
        for (auto &data: m_renderData) {
            m_vulkanDevice.createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                        &data.fragShaderParamsBuffer, sizeof(VkRender::ShaderValuesParams));
            m_vulkanDevice.createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                        &data.mvpBuffer, sizeof(VkRender::UBOMatrix));

            data.mvpBuffer.map();
            data.fragShaderParamsBuffer.map();
        }
    }


    void GaussianModelGraphicsPipeline::setupDescriptors() {
        std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         m_numSwapChainImages * 2},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_numSwapChainImages * 2},
        };

        VkDescriptorPoolCreateInfo descriptorPoolCI{};
        descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        descriptorPoolCI.pPoolSizes = poolSizes.data();
        descriptorPoolCI.maxSets = m_numSwapChainImages * static_cast<uint32_t>(poolSizes.size());
        CHECK_RESULT(
                vkCreateDescriptorPool(m_vulkanDevice.m_LogicalDevice, &descriptorPoolCI, nullptr,
                                       &m_sharedRenderData.descriptorPool));


        // Scene (matrices and environment maps)
        {
            std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
                    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1,
                                                                      VK_SHADER_STAGE_VERTEX_BIT |
                                                                      VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                    {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            };
            VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
            descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            descriptorSetLayoutCI.pBindings = setLayoutBindings.data();
            descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
            CHECK_RESULT(
                    vkCreateDescriptorSetLayout(m_vulkanDevice.m_LogicalDevice, &descriptorSetLayoutCI,
                                                nullptr,
                                                &m_sharedRenderData.descriptorSetLayout));

            for (auto &resource: m_renderData) {
                VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
                descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocInfo.descriptorPool = m_sharedRenderData.descriptorPool;
                descriptorSetAllocInfo.pSetLayouts = &m_sharedRenderData.descriptorSetLayout;
                descriptorSetAllocInfo.descriptorSetCount = 1;
                VkResult res = vkAllocateDescriptorSets(m_vulkanDevice.m_LogicalDevice, &descriptorSetAllocInfo,
                                                        &resource.descriptorSet);
                if (res != VK_SUCCESS)
                    throw std::runtime_error("Failed to allocate descriptor sets");

                std::array<VkWriteDescriptorSet, 3> writeDescriptorSets{};

                writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writeDescriptorSets[0].descriptorCount = 1;
                writeDescriptorSets[0].dstSet = resource.descriptorSet;
                writeDescriptorSets[0].dstBinding = 0;
                writeDescriptorSets[0].pBufferInfo = &resource.mvpBuffer.m_descriptorBufferInfo;

                writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                writeDescriptorSets[1].descriptorCount = 1;
                writeDescriptorSets[1].dstSet = resource.descriptorSet;
                writeDescriptorSets[1].dstBinding = 1;
                writeDescriptorSets[1].pBufferInfo = &resource.fragShaderParamsBuffer.m_descriptorBufferInfo;

                writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writeDescriptorSets[2].descriptorCount = 1;
                writeDescriptorSets[2].dstSet = resource.descriptorSet;
                writeDescriptorSets[2].dstBinding = 2;
                writeDescriptorSets[2].pImageInfo = &m_textureVideo->m_descriptor;

                vkUpdateDescriptorSets(m_vulkanDevice.m_LogicalDevice,
                                       static_cast<uint32_t>(writeDescriptorSets.size()),
                                       writeDescriptorSets.data(), 0, nullptr);
            }
        }
    }


    void GaussianModelGraphicsPipeline::setupPipeline() {

        // Vertex bindings an attributes
        VkVertexInputBindingDescription vertexInputBinding = {0, sizeof(VkRender::Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3},
                {2, 0, VK_FORMAT_R32G32_SFLOAT,    sizeof(float) * 6},
        };
        VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
        vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputStateCI.vertexBindingDescriptionCount = 1;
        vertexInputStateCI.pVertexBindingDescriptions = &vertexInputBinding;
        vertexInputStateCI.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
        vertexInputStateCI.pVertexAttributeDescriptions = vertexInputAttributes.data();
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages(2);
        VkShaderModule vertModule{};
        VkShaderModule fragModule{};
        shaderStages[0] = Utils::loadShader(m_vulkanDevice.m_LogicalDevice, "spv/" + m_vertexShader,
                                            VK_SHADER_STAGE_VERTEX_BIT, &vertModule);
        shaderStages[1] = Utils::loadShader(m_vulkanDevice.m_LogicalDevice, "spv/" + m_fragmentShader,
                                            VK_SHADER_STAGE_FRAGMENT_BIT, &fragModule);

        VulkanGraphicsPipelineCreateInfo createInfo(m_renderPassInfo.renderPass, m_vulkanDevice);
        createInfo.rasterizationStateCreateInfo = Populate::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL,
                                                                                                 VK_CULL_MODE_NONE,
                                                                                                 VK_FRONT_FACE_COUNTER_CLOCKWISE);
        createInfo.msaaSamples = m_renderPassInfo.sampleCount;
        createInfo.shaders = shaderStages;
        createInfo.descriptorSetLayout = m_sharedRenderData.descriptorSetLayout;
        createInfo.vertexInputState = vertexInputStateCI;

        m_sharedRenderData.graphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(createInfo);

        for (auto shaderStage: shaderStages) {
            vkDestroyShaderModule(m_vulkanDevice.m_LogicalDevice, shaderStage.module, nullptr);
        }

    }

    void GaussianModelGraphicsPipeline::update(uint32_t currentFrame) {
        memcpy(m_renderData[currentFrame].fragShaderParamsBuffer.mapped,
               &m_fragParams, sizeof(VkRender::ShaderValuesParams));
        memcpy(m_renderData[currentFrame].mvpBuffer.mapped,
               &m_vertexParams, sizeof(VkRender::UBOMatrix));

    }

    void GaussianModelGraphicsPipeline::updateTransform(TransformComponent &transform) {
        m_vertexParams.model = transform.GetTransform();

    }

    void GaussianModelGraphicsPipeline::updateView(const Camera &camera) {
        m_vertexParams.view = camera.matrices.view;
        m_vertexParams.projection = camera.matrices.perspective;
        m_vertexParams.camPos = camera.pose.pos;
    }


    void GaussianModelGraphicsPipeline::draw(CommandBuffer &cmdBuffers) {
        const uint32_t &cbIndex = *cmdBuffers.frameIndex;
        vkCmdBindPipeline(cmdBuffers.buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                          m_sharedRenderData.graphicsPipeline->getPipeline());
        vkCmdBindDescriptorSets(cmdBuffers.buffers[cbIndex], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_sharedRenderData.graphicsPipeline->getPipelineLayout(), 0, static_cast<uint32_t>(1),
                                &m_renderData[cbIndex].descriptorSet, 0, nullptr);
        VkDeviceSize offsets[1] = {0};
        vkCmdBindVertexBuffers(cmdBuffers.buffers[cbIndex], 0, 1, &vertices.buffer, offsets);
        if (indices.buffer != VK_NULL_HANDLE) {
            vkCmdBindIndexBuffer(cmdBuffers.buffers[cbIndex], indices.buffer, 0,
                                 VK_INDEX_TYPE_UINT32);
        }
        if (indices.buffer != VK_NULL_HANDLE) {
            vkCmdDrawIndexed(cmdBuffers.buffers[cbIndex], indices.indexCount, 1,
                             0, 0, 0);
        } else {
            vkCmdDraw(cmdBuffers.buffers[cbIndex], vertices.vertexCount, 1, 0, 0);
        }
    }

    void GaussianModelGraphicsPipeline::setTexture(const VkDescriptorImageInfo *info) {
        VkWriteDescriptorSet writeDescriptorSets{};

        for (const auto &data: m_renderData) {
            writeDescriptorSets.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDescriptorSets.descriptorCount = 1;
            writeDescriptorSets.dstSet = data.descriptorSet;
            writeDescriptorSets.dstBinding = 2;
            writeDescriptorSets.pImageInfo = info;
            vkUpdateDescriptorSets(m_vulkanDevice.m_LogicalDevice, 1, &writeDescriptorSets, 0, nullptr);

        }
    }


    void GaussianModelGraphicsPipeline::bind(VkRender::MeshComponent *modelComponent) {
        // Bind vertex/index buffers from model
        indices.indexCount = modelComponent->m_indices.size();
        size_t vertexBufferSize = modelComponent->m_vertices.size() * sizeof(VkRender::Vertex);
        size_t indexBufferSize = modelComponent->m_indices.size() * sizeof(uint32_t);

        assert(vertexBufferSize > 0);

        struct StagingBuffer {
            VkBuffer buffer;
            VkDeviceMemory memory;
        } vertexStaging{}, indexStaging{};

        // Create staging buffers
        // Vertex data
        CHECK_RESULT(m_vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertexBufferSize,
                &vertexStaging.buffer,
                &vertexStaging.memory,
                modelComponent->m_vertices.data()));
        // Index data
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_vulkanDevice.createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    indexBufferSize,
                    &indexStaging.buffer,
                    &indexStaging.memory,
                    modelComponent->m_indices.data()));
        }

        // Create m_vulkanDevice local buffers
        // Vertex buffer
        CHECK_RESULT(m_vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                vertexBufferSize,
                &vertices.buffer,
                &vertices.memory));
        // Index buffer
        if (indexBufferSize > 0) {
            CHECK_RESULT(m_vulkanDevice.createBuffer(
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    indexBufferSize,
                    &indices.buffer,
                    &indices.memory));
        }

        // Copy from staging buffers
        VkCommandBuffer copyCmd = m_vulkanDevice.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

        VkBufferCopy copyRegion = {};

        copyRegion.size = vertexBufferSize;
        vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, vertices.buffer, 1, &copyRegion);

        if (indexBufferSize > 0) {
            copyRegion.size = indexBufferSize;
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, indices.buffer, 1, &copyRegion);
        }

        m_vulkanDevice.flushCommandBuffer(copyCmd, m_vulkanDevice.m_TransferQueue, true);

        vkDestroyBuffer(m_vulkanDevice.m_LogicalDevice, vertexStaging.buffer, nullptr);
        vkFreeMemory(m_vulkanDevice.m_LogicalDevice, vertexStaging.memory, nullptr);
        if (indexBufferSize > 0) {
            vkDestroyBuffer(m_vulkanDevice.m_LogicalDevice, indexStaging.buffer, nullptr);
            vkFreeMemory(m_vulkanDevice.m_LogicalDevice, indexStaging.memory, nullptr);
        }

    }
}