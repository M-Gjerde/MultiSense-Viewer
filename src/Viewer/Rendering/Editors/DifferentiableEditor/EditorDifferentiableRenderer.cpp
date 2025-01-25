//
// Created by magnus on 1/15/25.
//

#include "Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRenderer.h"

#include "Viewer/Application/Application.h"
#include "Viewer/Rendering/Editors/CommonEditorFunctions.h"
#include "Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRendererLayerUI.h"
#include <OpenImageDenoise/oidn.hpp>


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
        initializeDifferentiableRenderer();

        m_optimizer = std::make_unique<torch::optim::Adam>(
            // We pass in the parameters of our module (or custom parameter list)
            m_photonRebuildModule->parameters(),
            // Then define the Adam options, e.g. learning rate = 1e-3
            torch::optim::AdamOptions(1e-3)
        );
    }


    void EditorDifferentiableRenderer::onUpdate() {
        auto imageUI = std::dynamic_pointer_cast<EditorDifferentiableRendererLayerUI>(m_ui);
        if (imageUI->reloadRenderer) {
            initializeDifferentiableRenderer();
        }

        if (imageUI->uploadScene) {
            m_pathTracer->upload(m_context->activeScene());
        }

        if (imageUI->switchKernelDevice) {
            PathTracer::PhotonTracer::Settings settings;
            settings.kernelDevice = imageUI->kernelDevice;
            m_pathTracer->setExecutionDevice(settings);
            m_pathTracer->upload(m_context->activeScene());
        }

        // ----------------------------------------------------------
        // Typical training (or single step) section
        // ----------------------------------------------------------
        if (m_photonRebuildModule && (imageUI->step || imageUI->toggleStep)) {
            // Example: Set up your path tracer’s forward settings
            PathTracer::PhotonTracer::Settings settings;
            settings.photonCount = imageUI->photonCount;
            settings.numBounces = imageUI->numBounces;
            settings.gammaCorrection = imageUI->gammaCorrection;
            settings.kernelType = PathTracer::KERNEL_PATH_TRACER_2DGS;

            // 1. ZERO OUT PREVIOUS GRADIENTS
            //    (Assuming m_photonRebuildModule is a torch::nn::Module
            //     with parameters and m_optimizer is e.g. torch::optim::Adam)
            m_optimizer->zero_grad();

            // 2. FORWARD PASS
            //    - If your forward method itself returns a torch::Tensor, you can skip
            //      the from_blob step below. If it doesn't, you need to create a tensor
            //      from the float* image data. Below is a typical pattern if you only
            //      have float* data from getRenderedImage().
            //    - Let's assume your forward method can also produce an Autograd-compatible tensor.
            //
            //    e.g.:
            //    torch::Tensor renderedTensor = m_photonRebuildModule->forward(settings, m_activeScene);
            //
            //    If it doesn't return a tensor directly, do:
            auto renderedTensor = m_photonRebuildModule->forward(settings, m_activeScene);

            float* img = m_photonRebuildModule->getRenderedImage();
            uint32_t width = m_colorTexture->width();
            uint32_t height = m_colorTexture->height();

            // Safety check
            if (!img) {
                std::cerr << "No rendered image returned; skipping training step.\n";
                return;
            }

            // If your renderer returns values in [0,1], you might not need further scaling.
            // If your renderer returns e.g. 0..255, scale it:
            //   renderedTensor /= 255.0f;

            // 3. CREATE OR GET TARGET TENSOR
            //    - In a real application, you presumably have a "reference" or "ground truth"
            //      you are trying to match. For demonstration, let's assume there's a
            //      pre-loaded target (the same shape) stored in m_targetImage.
            //    - Or you can construct a synthetic target.
            torch::Tensor targetTensor;
            {
                // ... Load or create your target here.
                // This must be the same shape as renderedTensor if you use MSE loss.
                // For demonstration, let's do a zero tensor as a placeholder:
                targetTensor = torch::zeros_like(renderedTensor);
            }

            // 4. COMPUTE THE LOSS
            //    - For instance, MSE:
            auto loss = torch::mse_loss(renderedTensor, targetTensor);

            // 5. BACKWARD PASS
            loss.backward();

            // 6. OPTIMIZER STEP (updates parameters of your differentiable module)
            m_optimizer->step();

            // Optionally, print or log the loss:
            std::cout << "Loss = " << loss.item<float>() << std::endl;

            // ----------------------------------------------------------
            // (Optional) Convert the rendered image for the UI
            // ----------------------------------------------------------
            std::vector<uint8_t> convertedImage(width * height * 4); // RGBA
            for (uint32_t i = 0; i < width * height; ++i) {
                float r = img[i]; // If single channel, store in R/G/B equally
                convertedImage[i * 4 + 0] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
                convertedImage[i * 4 + 1] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
                convertedImage[i * 4 + 2] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
                convertedImage[i * 4 + 3] = 255;
            }

            // Finally, upload the texture for display:
            m_colorTexture->loadImage(convertedImage.data(), convertedImage.size());
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
                m_pathTracer = std::make_unique<PathTracer::PhotonTracer>(m_context, m_activeScene, width, height);
                m_colorTexture = EditorUtils::createEmptyTexture(width, height, VK_FORMAT_R8G8B8A8_UNORM, m_context);

                m_pathTracer = std::make_unique<PathTracer::PhotonTracer>(m_context, m_activeScene, width, height);
                m_pathTracer->setActiveCamera(cameraComponent.getPinholeCamera(),
                                              &entity.getComponent<TransformComponent>());
                m_pathTracer->upload(m_activeScene);
                m_photonRebuildModule = std::make_unique<PathTracer::PhotonRebuildModule>(m_pathTracer.get());

                break;
            }
        }
    }
}
