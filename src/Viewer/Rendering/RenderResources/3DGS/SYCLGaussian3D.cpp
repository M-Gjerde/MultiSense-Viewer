//
// Created by magnus on 10/21/24.
//

#include <sycl/sycl.hpp>
#include <vector>
#include <algorithm>
#include <execution>
#include <utility>

#include "Viewer/Rendering/RenderResources/3DGS/SYCLGaussian3D.h"
#include "Viewer/Rendering/RenderResources/3DGS/Rasterizer3DGS.h"
#include "Viewer/Scenes/Entity.h"

namespace VkRender {
    void SYCLGaussian3D::render(std::shared_ptr<Scene> &scene, std::shared_ptr<VulkanTexture2D> &outputTexture) {


        auto &registry = scene->getRegistry();
        // Find all entities with GaussianComponent
        registry.view<GaussianComponent>().each([&](auto entity, GaussianComponent &gaussianComp) {
            if (gaussianComp.addToRenderer) {
                std::vector<GaussianPoint> newPoints;

                for (size_t i = 0; i < gaussianComp.size(); ++i) {
                    GaussianPoint point{};
                    point.position = gaussianComp.means[i];
                    point.scale = gaussianComp.scales[i];
                    point.rotation = gaussianComp.rotations[i];
                    point.opacity = gaussianComp.opacities[i];
                    point.color = gaussianComp.colors[i];
                    if (!gaussianComp.shCoeffs.empty())
                        point.shCoeffs = gaussianComp.shCoeffs[i];

                    newPoints.push_back(point);
                }
                if (!newPoints.empty()) {
                    Log::Logger::getInstance()->info("Updating Gaussians in GPU memory");
                    updateGaussianPoints(newPoints);

                }

            }
        });


        if (m_numGaussians < 1) {
            return;
        }

        size_t imageSize = 0;
        auto view = scene->getRegistry().view<CameraComponent, TransformComponent, MeshComponent>();
        for (auto e: view) {
            Entity entity(e, scene.get());
            auto &transform = entity.getComponent<TransformComponent>();
            auto camera = std::dynamic_pointer_cast<PinholeCamera>(entity.getComponent<CameraComponent>().camera);
            if (!camera)
                continue;

            uint32_t BLOCK_X = 16, BLOCK_Y = 16;
            glm::vec3 tileGrid((outputTexture->width() + BLOCK_X - 1) / BLOCK_X,
                               (outputTexture->height() + BLOCK_Y - 1) / BLOCK_Y, 1);

            uint32_t numTiles = tileGrid.x * tileGrid.y;

            CameraParams params;
            params.focalX = camera->parameters().fx;
            params.focalY = camera->parameters().fy;

            m_preProcessData.camera = *camera;
            m_preProcessData.width = outputTexture->width();
            m_preProcessData.height = outputTexture->height();
            m_preProcessData.preProcessSettings.tileGrid = tileGrid;
            m_preProcessData.preProcessSettings.numTiles = numTiles;
            m_preProcessData.preProcessSettings.params = params;

            imageSize = outputTexture->width() * outputTexture->height() * 4;
        }

        auto *imageMemory = static_cast<uint8_t *>(malloc(imageSize));

        m_preProcessData.preProcessSettings.numPoints = m_numGaussians;
        m_queue.memcpy(m_preProcessDataPtr, &m_preProcessData, sizeof(PreProcessData)).wait();

        preProcessGaussians(imageMemory);
        //renderGaussiansWithProfiling(imageMemory, true);

        // Start rendering
        // Copy output to texture
        outputTexture->loadImage(imageMemory, imageSize); // TODO dont copy to staging buffers. Create a video texture type to be used for this

        // Free the allocated memory
        free(imageMemory);
    }

    void SYCLGaussian3D::updateGaussianPoints(const std::vector<GaussianPoint> &newPoints) {
        sycl::free(m_gaussianPointsPtr, m_queue);
        m_numGaussians = newPoints.size();
        m_gaussianPointsPtr = sycl::malloc_device<GaussianPoint>(m_numGaussians, m_queue);


        m_queue.memcpy(m_gaussianPointsPtr, newPoints.data(), newPoints.size() * sizeof(GaussianPoint)).wait();
    }

    void SYCLGaussian3D::copyAndSortKeysAndValues(uint32_t *keysBuffer, uint32_t *valuesBuffer, size_t numRendered) {
        // TODO this function creates a memory leak on host ram
        // Step 1: Allocate Unified Shared Memory (USM) for key-value pairs
        using KeyValue = std::pair<uint32_t, uint32_t>;

        // Allocate USM for the key-value pairs directly
        KeyValue *keyValuePairs = sycl::malloc_shared<KeyValue>(numRendered, m_queue);

        // Step 2: Initialize the key-value pairs from the provided buffers
        m_queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(numRendered), [=](sycl::id<1> idx) {
                keyValuePairs[idx[0]].first = keysBuffer[idx[0]];
                keyValuePairs[idx[0]].second = valuesBuffer[idx[0]];
            });
        }).wait();

        // Step 3: Sort the key-value pairs directly using the C++ standard algorithm with offloading
        std::sort(std::execution::par_unseq, keyValuePairs, keyValuePairs + numRendered,
                  [](const KeyValue &a, const KeyValue &b) {
                      return a.first < b.first;
                  });

        m_queue.wait();

        // Step 4: Write the sorted keys and values back to the provided buffers
        m_queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(numRendered), [=](sycl::id<1> idx) {
                keysBuffer[idx[0]] = keyValuePairs[idx[0]].first;
                valuesBuffer[idx[0]] = keyValuePairs[idx[0]].second;
            });
        }).wait();

        // Free the USM memory
        sycl::free(keyValuePairs, m_queue);
    }

    void SYCLGaussian3D::preProcessGaussians(uint8_t *imageMemory) {

        uint32_t imageWidth = m_preProcessData.width;
        uint32_t imageHeight = m_preProcessData.height;

        m_queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(m_numGaussians),
                           Rasterizer::Preprocess(m_gaussianPointsPtr, m_preProcessDataPtr));
        }).wait();

        auto *pointOffsets = sycl::malloc_device<uint32_t>(m_numGaussians, m_queue);
        m_queue.memset(pointOffsets, static_cast<uint32_t>(0x00), m_numGaussians * sizeof(uint32_t)).wait();

        uint32_t localSize = 16;  // Number of work-items in each work-group
        uint32_t globalSize = ((m_numGaussians + localSize - 1) / localSize) * localSize;
        uint32_t numWorkGroups = globalSize / localSize;
        sycl::nd_range<1> kernelRange({sycl::range<1>(globalSize), sycl::range<1>(localSize)});
        auto *groupTotals = sycl::malloc_device<uint32_t>(numWorkGroups, m_queue);

        // Kernel 2
        m_queue.submit([&](sycl::handler &h) {
            auto localMemory = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(localSize),
                                                                 h); // One entry per work-item
            auto caller = Rasterizer::InclusiveSum(m_gaussianPointsPtr, pointOffsets,
                                                   m_preProcessDataPtr->preProcessSettings.numPoints, localMemory,
                                                   groupTotals);
            h.parallel_for(kernelRange, caller);
        }).wait();

        m_queue.submit([&](sycl::handler &h) {
            h.single_task<class GroupTotalsScan>([=]() {
                // Perform exclusive scan on group totals
                for (size_t i = 1; i < numWorkGroups; ++i) {
                    groupTotals[i] += groupTotals[i - 1];
                }
            });
        }).wait();;
        // Step 4: Adjust local scans
        m_queue.submit([&](sycl::handler &h) {
            size_t numPoints = m_numGaussians;
            h.parallel_for<class AdjustLocalScans>(kernelRange, [=](sycl::nd_item<1> item) {
                size_t groupId = item.get_group_linear_id();
                uint32_t offset = (groupId > 0) ? groupTotals[groupId - 1] : 0;
                size_t globalIdx = item.get_global_id(0);
                // Adjust local scan result
                if (globalIdx < numPoints) {
                    pointOffsets[globalIdx] += offset;
                }
            });
        }).wait();;

        sycl::free(groupTotals, m_queue);

        uint32_t numRendered = 0;
        m_queue.memcpy(&numRendered, pointOffsets + (m_numGaussians - 1), sizeof(uint32_t)).wait();
        // Inclusive sum complete

        Log::Logger::getInstance()->traceWithFrequency("3dgsrendering", 60, "3DGS Rendering, Gaussians: {}",
                                                       numRendered);
        if (numRendered > 0) {

            auto *keysBuffer = sycl::malloc_device<uint32_t>(numRendered, m_queue);
            auto *valuesBuffer = sycl::malloc_device<uint32_t>(numRendered, m_queue);
            m_queue.wait();

            m_queue.submit([&](sycl::handler &h) {
                h.parallel_for<class duplicates>(sycl::range<1>(m_numGaussians),
                                                 Rasterizer::DuplicateGaussians(m_gaussianPointsPtr, pointOffsets,
                                                                                keysBuffer,
                                                                                valuesBuffer,
                                                                                numRendered,
                                                                                m_preProcessDataPtr->preProcessSettings.tileGrid));
            }).wait();


            copyAndSortKeysAndValues(keysBuffer, valuesBuffer, numRendered);

            auto *rangesBuffer = sycl::malloc_device<glm::ivec2>(m_preProcessData.preProcessSettings.numTiles, m_queue);
            m_queue.memset(rangesBuffer, static_cast<int>(0x00),
                           m_preProcessData.preProcessSettings.numTiles * sizeof(int) * 2).wait();


            m_queue.submit([&](sycl::handler &h) {
                h.parallel_for<class IdentifyTileRanges>(numRendered,
                                                         Rasterizer::IdentifyTileRanges(rangesBuffer, keysBuffer,
                                                                                        numRendered));
            }).wait();

            // Rasterize Gaussians
            const uint32_t tileWidth = 16;
            const uint32_t tileHeight = 16;

            sycl::range<2> localWorkSize(tileHeight, tileWidth);
            size_t globalWidth = ((imageWidth + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
            size_t globalHeight = ((imageHeight + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
            sycl::range<2> globalWorkSize(globalHeight, globalWidth);

            auto *imageBuffer = sycl::malloc_device<uint8_t>(imageWidth * imageHeight * 4, m_queue);
            m_queue.fill(imageBuffer, static_cast<uint8_t>(0x00), imageWidth * imageHeight * 4).wait();

            m_queue.submit([&](sycl::handler &h) {
                auto range = sycl::nd_range<2>(globalWorkSize, localWorkSize);
                h.parallel_for<class RenderGaussians>(range,
                                                      Rasterizer::RasterizeGaussians(rangesBuffer,
                                                                                     valuesBuffer, m_gaussianPointsPtr,
                                                                                     imageBuffer, numRendered,
                                                                                     imageWidth, imageHeight));
            }).wait();


            m_queue.memcpy(imageMemory, imageBuffer, imageWidth * imageHeight * 4);
            m_queue.wait();

            sycl::free(imageBuffer, m_queue);
            sycl::free(rangesBuffer, m_queue);
            sycl::free(keysBuffer, m_queue);
            sycl::free(valuesBuffer, m_queue);
        }
        sycl::free(pointOffsets, m_queue);

    }

    void SYCLGaussian3D::renderGaussiansWithProfiling(uint8_t *imageMemory, bool enable_profiling) {
        uint32_t imageWidth = m_preProcessData.width;
        uint32_t imageHeight = m_preProcessData.height;


        auto profileKernel = [&](sycl::event event, const std::string &kernel_name) {
            if (enable_profiling) {
                auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                auto duration_ns = end - start;
                std::cout << kernel_name << " kernel time: " << duration_ns * 1e-6 << " ms\n";
                m_durations.emplace_back(duration_ns * 1e-6);
            }
        };

        // Kernel 1
        auto event1 = m_queue.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(m_numGaussians),
                           Rasterizer::Preprocess(m_gaussianPointsPtr, m_preProcessDataPtr));
        });
        event1.wait();
        profileKernel(event1, "Preprocess");

        std::vector<GaussianPoint> hostPointsProcessed(m_numGaussians);
        m_queue.memcpy(hostPointsProcessed.data(), m_gaussianPointsPtr, m_numGaussians * sizeof(GaussianPoint)).wait();

        auto *pointOffsets = sycl::malloc_device<uint32_t>(m_numGaussians, m_queue);
        auto event2 = m_queue.memset(pointOffsets, static_cast<uint32_t>(0x00), m_numGaussians * sizeof(uint32_t));
        event2.wait();
        profileKernel(event2, "Memset Point Offsets");


        std::vector<GaussianPoint> hostPoints(m_numGaussians);
        std::cout << "TilesTouched: ";
        m_queue.memcpy(hostPoints.data(), m_gaussianPointsPtr, m_numGaussians * sizeof(GaussianPoint)).wait();
        for (int i = 0; i < m_numGaussians && i < 128; ++i) {
            std::cout << hostPoints[i].tilesTouched << " ";
            if ((i + 1) % 16 == 0)
                std::cout << "| ";
        }


        std::cout << std::endl;

        uint32_t localSize = 16;  // Number of work-items in each work-group
        uint32_t globalSize =
                ((m_preProcessDataPtr->preProcessSettings.numPoints + localSize - 1) / localSize) * localSize;
        uint32_t numWorkGroups = globalSize / localSize;
        sycl::nd_range<1> kernelRange({sycl::range<1>(globalSize), sycl::range<1>(localSize)});
        auto *groupTotals = sycl::malloc_device<uint32_t>(numWorkGroups, m_queue);
        // Kernel 2
        auto event3 = m_queue.submit([&](sycl::handler &h) {

            auto localMemory = sycl::local_accessor<uint32_t, 1>(sycl::range<1>(localSize),
                                                                 h); // One entry per work-item


            auto caller = Rasterizer::InclusiveSum(m_gaussianPointsPtr, pointOffsets,
                                                   m_preProcessDataPtr->preProcessSettings.numPoints, localMemory,
                                                   groupTotals);
            h.parallel_for(kernelRange, caller);
        });
        event3.wait();
        profileKernel(event3, "InclusiveSum");


        auto event4 = m_queue.submit([&](sycl::handler &h) {
            h.single_task<class GroupTotalsScan2>([=]() {
                // Perform exclusive scan on group totals
                for (size_t i = 1; i < numWorkGroups; ++i) {
                    groupTotals[i] += groupTotals[i - 1];
                }
            });
        });
        event4.wait();
        profileKernel(event4, "SumGroupTotals");


        std::vector<uint32_t> groupTotalsHost(numWorkGroups);
        std::cout << "groupTotalsHost: ";
        m_queue.memcpy(groupTotalsHost.data(), groupTotals, numWorkGroups * sizeof(uint32_t)).wait();
        for (int i = 0; i < 16; ++i) {
            std::cout << groupTotalsHost[i] << " ";
            if ((i + 1) % 16 == 0)
                std::cout << "| ";
        }
        std::cout << std::endl;


        // Step 4: Adjust local scans
        auto event5 = m_queue.submit([&](sycl::handler &h) {
            size_t numPoints = m_numGaussians;
            h.parallel_for<class AdjustLocalScans2>(kernelRange, [=](sycl::nd_item<1> item) {
                size_t groupId = item.get_group_linear_id();
                uint32_t offset = (groupId > 0) ? groupTotals[groupId - 1] : 0;
                size_t globalIdx = item.get_global_id(0);

                // Adjust local scan result
                if (globalIdx < numPoints) {
                    pointOffsets[globalIdx] += offset;
                }
            });
        });
        event5.wait();
        profileKernel(event5, "InclusiveSumGlobalOffsets");

        /*
        std::vector<uint32_t> inclusiveSum(m_numGaussians);
        std::cout << "inclusiveSum: ";
        m_queue.memcpy(inclusiveSum.data(), pointOffsets, m_numGaussians * sizeof(uint32_t)).wait();
        for (int i = 0; i < 128; ++i){
            std::cout << inclusiveSum[i] << " ";
            if ((i + 1) % 16 == 0)
                std::cout << "| ";
        }
        std::cout << std::endl;
        */

        uint32_t numRendered = 0;
        auto event6 = m_queue.memcpy(&numRendered, pointOffsets + (m_numGaussians - 1), sizeof(uint32_t));
        event6.wait();
        profileKernel(event6, "Memcpy numRendered");

        Log::Logger::getInstance()->traceWithFrequency("3dgsrendering", 60, "3DGS Rendering, Gaussians: {}",
                                                       numRendered);

        if (numRendered > 0) {
            auto *keysBuffer = sycl::malloc_device<uint32_t>(numRendered, m_queue);
            auto *valuesBuffer = sycl::malloc_device<uint32_t>(numRendered, m_queue);
            m_queue.wait();

            // Kernel 3
            auto event7 = m_queue.submit([&](sycl::handler &h) {
                h.parallel_for<>(sycl::range<1>(m_numGaussians),
                                 Rasterizer::DuplicateGaussians(m_gaussianPointsPtr, pointOffsets,
                                                                keysBuffer, valuesBuffer, numRendered,
                                                                m_preProcessDataPtr->preProcessSettings.tileGrid));
            });
            event7.wait();
            profileKernel(event7, "DuplicateGaussians");

            copyAndSortKeysAndValues(keysBuffer, valuesBuffer, numRendered);

            auto *rangesBuffer = sycl::malloc_device<glm::ivec2>(m_preProcessData.preProcessSettings.numTiles, m_queue);
            auto event8 = m_queue.memset(rangesBuffer, static_cast<int>(0x00),
                                         m_preProcessData.preProcessSettings.numTiles * sizeof(int) * 2);
            event8.wait();
            profileKernel(event8, "Memset Ranges Buffer");

            // Kernel 4
            auto event9 = m_queue.submit([&](sycl::handler &h) {
                h.parallel_for<>(numRendered, Rasterizer::IdentifyTileRanges(rangesBuffer, keysBuffer, numRendered));
            });
            event9.wait();
            profileKernel(event9, "IdentifyTileRanges");

            const uint32_t tileWidth = 16;
            const uint32_t tileHeight = 16;

            sycl::range<2> localWorkSize(tileHeight, tileWidth);
            size_t globalWidth = ((imageWidth + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
            size_t globalHeight = ((imageHeight + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];
            sycl::range<2> globalWorkSize(globalHeight, globalWidth);

            auto *imageBuffer = sycl::malloc_device<uint8_t>(imageWidth * imageHeight * 4, m_queue);
            auto event10 = m_queue.fill(imageBuffer, static_cast<uint8_t>(0x00), imageWidth * imageHeight * 4);
            event10.wait();
            profileKernel(event10, "Fill Image Buffer");

            // Kernel 5
            auto event11 = m_queue.submit([&](sycl::handler &h) {
                auto range = sycl::nd_range<2>(globalWorkSize, localWorkSize);
                h.parallel_for<>(range,
                                 Rasterizer::RasterizeGaussians(rangesBuffer, valuesBuffer,
                                                                m_gaussianPointsPtr, imageBuffer,
                                                                numRendered, imageWidth, imageHeight));
            });
            event11.wait();
            profileKernel(event11, "RenderGaussians");

            auto event12 = m_queue.memcpy(imageMemory, imageBuffer, imageWidth * imageHeight * 4);
            event12.wait();
            profileKernel(event12, "Memcpy Image Memory");

            sycl::free(imageBuffer, m_queue);
            sycl::free(rangesBuffer, m_queue);
            sycl::free(pointOffsets, m_queue);
            sycl::free(keysBuffer, m_queue);
            sycl::free(valuesBuffer, m_queue);
        }
        unsigned long sum = 0;
        for (unsigned long m_duration: m_durations) {
            sum += m_duration;
        }
        std::cout << "Total rendering time: " << sum << " ms" << std::endl;
        std::cout << "\n\n\n";
        m_durations.clear();
    }


    void SYCLGaussian3D::rasterizeGaussians(const GaussianComponent &gaussianComp) {
    }
}
