//
// Created by magnus on 1/15/25.
//

#include "Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRenderer.h"

#include "Viewer/Application/Application.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"
#include "Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRendererLayerUI.h"

#include <OpenImageDenoise/oidn.hpp>
#include <yaml-cpp/yaml.h>

namespace VkRender {
    EditorDifferentiableRenderer::EditorDifferentiableRenderer(EditorCreateInfo& createInfo, UUID uuid) : Editor(
        createInfo, uuid) {
        addUI("EditorDifferentiableRendererLayer");
        addUI("EditorUILayer");
        addUI("DebugWindow");
        addUIData<EditorDifferentiableRendererLayerUI>();

        m_descriptorRegistry.createManager(DescriptorManagerType::Viewport3DTexture, m_context->vkDevice());

        m_shaderSelectionBuffer.resize(m_context->swapChainBuffers().size());
        for (auto& frameIndex : m_shaderSelectionBuffer) {
            m_context->vkDevice().createBuffer(
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                frameIndex,
                sizeof(float), nullptr, "EditorDifferentiableRenderer:ShaderSelectionBuffer",
                m_context->getDebugUtilsObjectNameFunction());
        }
        m_colorTexture = EditorUtils::createEmptyTexture(m_createInfo.width, m_createInfo.height,
                                                         VK_FORMAT_R8G8B8A8_UNORM, m_context);

        m_activeSceneCamera = std::make_shared<ArcballCamera>();
        m_activeSceneCamera->setDefaultPosition({-90.0f, -60.0f}, 1.5f);

    }

    void EditorDifferentiableRenderer::onEditorResize() {
    }


    void EditorDifferentiableRenderer::onSceneLoad(std::shared_ptr<Scene> scene) {
        m_activeScene = scene;
        //initializeDifferentiableRenderer();
    }


    void EditorDifferentiableRenderer::onUpdate() {
        return;
        auto imageUI = std::dynamic_pointer_cast<EditorDifferentiableRendererLayerUI>(m_ui);
        if (imageUI->reloadRenderer) {
            initializeDifferentiableRenderer();
        }

        if (!m_photonRebuildModule)
            return;

        if (imageUI->uploadScene) {
            m_photonRebuildModule->uploadPathTracerFromTensor();
        }

        if (imageUI->switchKernelDevice) {
            //PathTracer::PhotonTracer::Settings settings;
            //settings.kernelDevice = imageUI->kernelDevice;
            //m_pathTracer->setExecutionDevice(settings);
            //m_pathTracer->upload(m_context->activeScene());
        }

        // ----------------------------------------------------------
        // 1. Possibly accumulate forward passes
        // ----------------------------------------------------------
        if (m_photonRebuildModule && (imageUI->step || imageUI->toggleStep)) {
            // Prepare path tracer forward settings

            m_photonRebuildModule->uploadPathTracerFromTensor(); // Upload path tracer with the new parameters

            // Forward pass (autograd-compatible)
            m_accumulatedTensor = m_photonRebuildModule->forward(m_renderSettings);
            m_numAccumulated++;

            // Optionally retrieve the float* for real-time display
            float* img = m_photonRebuildModule->getRenderedImage();
            uint32_t width = m_colorTexture->width();
            uint32_t height = m_colorTexture->height();
            if (!img) {
                std::cerr << "No rendered image returned; skipping display step.\n";
                return;
            }

            // Convert to RGBA for UI
            std::vector<uint8_t> convertedImage(width * height * 4); // RGBA
            for (uint32_t i = 0; i < width * height; ++i) {
                float r = img[i]; // If single channel, replicate to R/G/B
                convertedImage[i * 4 + 0] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
                convertedImage[i * 4 + 1] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
                convertedImage[i * 4 + 2] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
                convertedImage[i * 4 + 3] = 255;
            }
            m_colorTexture->loadImage(convertedImage.data(), convertedImage.size());
        }

        // ----------------------------------------------------------
        // 2. Backprop only once after enough forwards
        // ----------------------------------------------------------
        // e.g. if the user toggled a "finalize accumulation & train" button
        // or you simply want to backprop once after some number of passes
        if ((imageUI->backprop || imageUI->toggleStep) && m_accumulatedTensor.defined() && m_numAccumulated > 0) {
            // If you want the average instead of the sum, do so here:
            // Create or load your target tensor (matching shape)
            // For demonstration, just zero:
            int width = m_accumulatedTensor.size(1);
            int height = m_accumulatedTensor.size(0);
            // Load the target tensor
            torch::Tensor targetTensor = loadPFM("/home/magnus-desktop/datasets/PhotonRebuild/active/screenshot.pfm",
                                                 width, height);

            // Compute loss
            auto loss = torch::mean(torch::pow(targetTensor-m_accumulatedTensor, 2));

            // Backward
            loss.backward();

            // Example debug prints
            std::cout << "Loss: " << loss.item<float>() << std::endl;
            Log::Logger::getInstance()->info("Loss: {}", loss.item<float>());
            // Gradient checks: positions, scales, normals
            // (Make sure you've actually registered these as parameters in your module!)
            auto positions = m_photonRebuildModule->m_tensorData.positions;

            auto gradPositions = m_photonRebuildModule->m_tensorData.positions.grad();
            auto gradScales = m_photonRebuildModule->m_tensorData.scales.grad();
            auto gradNormals = m_photonRebuildModule->m_tensorData.normals.grad();

            if (positions.defined()) {
                // Check for NaNs or Infs
                if (positions.isnan().any().item<bool>()) {
                    std::cout << "positions contain NaNs!\n";
                }
                if (positions.isinf().any().item<bool>()) {
                    std::cout << "positions contain Infs!\n";
                }
                std::cout << "Positions: ("
                    << positions[0][0].item<float>() << ", "
                    << positions[0][1].item<float>() << ", "
                    << positions[0][2].item<float>() << ")"
                    << std::endl;

                Log::Logger::getInstance()->info("eo Gradient: ({},{},{})",
                    gradPositions[0][0].item<float>(),
                    gradPositions[0][1].item<float>(),
                    gradPositions[0][2].item<float>());

                Log::Logger::getInstance()->info("e0 Position: ({},{},{})",
                    positions[0][0].item<float>(),
                    positions[0][1].item<float>(),
                    positions[0][2].item<float>());
            }
            // Optimizer step
            m_optimizer->step();
            // Reset the accumulation if you only wanted to do a single backprop per accumulation
            m_accumulatedTensor = torch::Tensor();
            m_numAccumulated = 0;
            m_optimizer->zero_grad(); // Clear old gradients
        }
    }


    void EditorDifferentiableRenderer::onRender(CommandBuffer& commandBuffer) {
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>> renderGroups;
        collectRenderCommands(renderGroups, commandBuffer.frameIndex);

        // Render each group
        for (auto& [pipeline, commands] : renderGroups) {
            pipeline->bind(commandBuffer);
            for (auto& command : commands) {
                // Bind resources and draw
                bindResourcesAndDraw(commandBuffer, command);
            }
        }
    }

    void EditorDifferentiableRenderer::collectRenderCommands(
        std::unordered_map<std::shared_ptr<DefaultGraphicsPipeline>, std::vector<RenderCommand>>& renderGroups,
        uint32_t frameIndex) {
        if (!m_meshInstances) {
            m_meshInstances = EditorUtils::setupMesh(m_context);
            Log::Logger::getInstance()->info("Created MeshInstance for 3DViewport");
        }
        if (!m_meshInstances)
            return;
        PipelineKey key = {};
        key.setLayouts.resize(1);
        auto imageUI = std::dynamic_pointer_cast<EditorDifferentiableRendererLayerUI>(m_ui);

        // Prepare descriptor writes based on your texture or other resources
        std::array<VkWriteDescriptorSet, 2> writeDescriptors{};
        writeDescriptors[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptors[0].dstBinding = 0; // Binding index
        writeDescriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptors[0].descriptorCount = 1;
        writeDescriptors[0].pImageInfo = &m_colorTexture->getDescriptorInfo();
        writeDescriptors[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptors[1].dstBinding = 1; // Binding index
        writeDescriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptors[1].descriptorCount = 1;
        writeDescriptors[1].pBufferInfo = &m_shaderSelectionBuffer[frameIndex]->m_descriptorBufferInfo;
        std::vector descriptorWrites = {writeDescriptors[0], writeDescriptors[1]};
        VkDescriptorSet descriptorSet = m_descriptorRegistry.getManager(
            DescriptorManagerType::Viewport3DTexture).getOrCreateDescriptorSet(descriptorWrites);
        key.setLayouts[0] = m_descriptorRegistry.getManager(
            DescriptorManagerType::Viewport3DTexture).getDescriptorSetLayout();
        // Use default descriptor set layout
        key.vertexShaderName = "default2D.vert";
        key.fragmentShaderName = "EditorImageViewportTexture.frag";
        key.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        key.polygonMode = VK_POLYGON_MODE_FILL;
        std::vector<VkVertexInputBindingDescription> vertexInputBinding = {
            {0, sizeof(VkRender::ImageVertex), VK_VERTEX_INPUT_RATE_VERTEX}
        };
        std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
            {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
            {1, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 2},
        };
        key.vertexInputBindingDescriptions = vertexInputBinding;
        key.vertexInputAttributes = vertexInputAttributes;

        // Create or retrieve the pipeline
        RenderPassInfo renderPassInfo{};
        renderPassInfo.sampleCount = m_createInfo.pPassCreateInfo.msaaSamples;
        renderPassInfo.renderPass = m_renderPass->getRenderPass();
        renderPassInfo.debugName = "EditorDifferentiableRenderer::";
        auto pipeline = m_pipelineManager.getOrCreatePipeline(key, renderPassInfo, m_context);
        // Create the render command
        RenderCommand command;
        command.pipeline = pipeline;
        command.meshInstance = m_meshInstances.get();
        command.descriptorSets[DescriptorManagerType::Viewport3DTexture] = descriptorSet; // Assign the descriptor set
        // Add to render group
        renderGroups[pipeline].push_back(command);
    }

    void EditorDifferentiableRenderer::bindResourcesAndDraw(const CommandBuffer& commandBuffer,
                                                            RenderCommand& command) {
        VkCommandBuffer cmdBuffer = commandBuffer.getActiveBuffer();
        uint32_t frameIndex = commandBuffer.frameIndex;

        if (command.meshInstance->vertexBuffer) {
            VkBuffer vertexBuffers[] = {command.meshInstance->vertexBuffer->m_buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vertexBuffers, offsets);
            // Bind index buffer if the mesh has indices
            if (command.meshInstance->indexBuffer) {
                vkCmdBindIndexBuffer(cmdBuffer, command.meshInstance->indexBuffer->m_buffer, 0,
                                     VK_INDEX_TYPE_UINT32);
            }
        }

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          command.pipeline->pipeline()->getPipeline());


        for (auto& [index, descriptorSet] : command.descriptorSets) {
            vkCmdBindDescriptorSets(
                cmdBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                command.pipeline->pipeline()->getPipelineLayout(),
                0, // TODO can't reuse the approach in SceneRenderer since we have different manager types
                1,
                &descriptorSet,
                0,
                nullptr
            );
        }

        if (command.meshInstance->indexCount > 0) {
            vkCmdDrawIndexed(cmdBuffer, command.meshInstance->indexCount, 1, 0, 0, 0);
        }
    }


    void EditorDifferentiableRenderer::initializeDifferentiableRenderer() {
        auto view = m_activeScene->getRegistry().view<CameraComponent>();
        for (auto e : view) {
            Entity entity(e, m_activeScene.get());
            auto& cameraComponent = entity.getComponent<CameraComponent>();

            if (cameraComponent.renderFromViewpoint() && cameraComponent.cameraType ==
                CameraComponent::CameraType::PINHOLE) {
                // Use the first camera found that can render from viewpoint
                auto sceneCamera = cameraComponent.getPinholeCamera();

                float textureAspect = static_cast<float>(sceneCamera->parameters().width) / static_cast<float>(
                    sceneCamera->parameters().height);
                float editorAspect = static_cast<float>(m_createInfo.width) / static_cast<float>(m_createInfo.height);
                // Calculate scaling factors
                float scaleX = 1.0f, scaleY = 1.0f;
                if (editorAspect > textureAspect) {
                    scaleX = textureAspect / editorAspect;
                }
                else {
                    scaleY = editorAspect / textureAspect;
                }
                m_meshInstances.reset();
                m_meshInstances = nullptr;
                m_meshInstances = EditorUtils::setupMesh(m_context, scaleX, scaleY);
                float width = sceneCamera->parameters().width;
                float height = sceneCamera->parameters().height;

                /*
                m_pathTracer = std::make_unique<PathTracer::PhotonTracer>(m_context, m_activeScene, width, height);
                m_colorTexture = EditorUtils::createEmptyTexture(width, height, VK_FORMAT_R8G8B8A8_UNORM, m_context);

                m_pathTracer = std::make_unique<PathTracer::PhotonTracer>(m_context, m_activeScene, width, height);
                */
                //m_pathTracer->setActiveCamera(cameraComponent.getPinholeCamera(),&entity.getComponent<TransformComponent>());
                //m_pathTracer->upload(m_activeScene);
                m_photonRebuildModule = std::make_unique<PathTracer::PhotonRebuildModule>(
                    m_pathTracer.get(), m_context->activeScene());

                m_optimizer = std::make_unique<torch::optim::Adam>(
                    // We pass in the parameters of our module (or custom parameter list)
                    m_photonRebuildModule->parameters(),
                    // Then define the Adam options, e.g. learning rate = 1e-3
                    torch::optim::AdamOptions(0.1)
                );

                // Load the YAML file
                std::filesystem::path filePath = "/home/magnus-desktop/datasets/PhotonRebuild/active/render_info.yaml";
                if (std::filesystem::exists(filePath)) {
                    YAML::Node config = YAML::LoadFile(filePath);
                    // Retrieve values from YAML nodes
                    auto gamma = config["Gamma"].as<double>();
                    auto photonHitCount = config["PhotonHitCount"].as<uint64_t>();
                    auto photonsEmitted = config["PhotonsEmitted"].as<uint64_t>();
                    auto frameCount = config["FrameCount"].as<uint32_t>();
                    auto photonBounceCount = config["PhotonBounceCount"].as<uint32_t>();

                    // Print them out (or use them in your application)
                    std::cout << "Gamma: " << gamma << std::endl;
                    std::cout << "PhotonHitCount: " << photonHitCount << std::endl;
                    std::cout << "PhotonsEmitted: " << photonsEmitted << std::endl;
                    std::cout << "FrameCount: " << frameCount << std::endl;

                    m_renderSettings.gammaCorrection = gamma;
                }
                else {

                    Log::Logger::getInstance()->warning("Did not load params from dataset folder");
                }
                break;
            }
        }
    }

    torch::Tensor EditorDifferentiableRenderer::loadPFM(const std::string& filename, int expectedWidth,
                                                        int expectedHeight) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open PFM file: " + filename);
        }

        // Read header: "PF", width, height, scale
        std::string header;
        file >> header;
        if (header != "PF") {
            throw std::runtime_error("Unsupported PFM format (only RGB 'PF' supported). Got: " + header);
        }

        int width, height;
        float scale;
        file >> width >> height >> scale;
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // skip rest of line

        if (width != expectedWidth || height != expectedHeight) {
            throw std::runtime_error("PFM dimensions don't match the expected size.");
        }

        // Determine file endianness from sign of 'scale'
        bool fileIsLittleEndian = (scale < 0.f);
        float absScale = std::fabs(scale);

        // For x86, the machine is little-endian
        bool machineIsLittleEndian = true;
        bool needByteSwap = (fileIsLittleEndian != machineIsLittleEndian);

        // Allocate space (RGB => 3 channels)
        std::vector<float> data(width * height * 3);

        // Read raw bytes
        file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
        if (!file) {
            throw std::runtime_error("Failed to read PFM pixel data.");
        }

        // Byte-swap if needed
        if (needByteSwap) {
            for (auto& px : data) {
                uint8_t* b = reinterpret_cast<uint8_t*>(&px);
                std::swap(b[0], b[3]);
                std::swap(b[1], b[2]);
            }
        }

        // Scale pixels by absScale
        for (auto& px : data) {
            px *= absScale;
        }

        // Create Torch tensor of shape [height, width, 3]
        torch::Tensor tensor3D = torch::from_blob(data.data(), {height, width, 3}, torch::kFloat).clone();

        // If truly grayscale repeated in R/G/B, average them to get [height, width]
        torch::Tensor tensor2D = tensor3D.mean(2);


        if (tensor2D.isnan().any().item<bool>()) {
            throw std::runtime_error("PFM data contains NaN after load!");
        }
        if (tensor2D.isinf().any().item<bool>()) {
            throw std::runtime_error("PFM data contains Inf after load!");
        }

        return tensor2D;
    }
}
