//
// Created by magnus on 7/15/24.
//


#include "stb_image_write.h"
#include <tiffio.h>

#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Tools/Utils.h"

#include "Viewer/Application/Application.h"

#include "Viewer/Rendering/Core/VulkanRenderPass.h"
#include "Viewer/Rendering/Core/VulkanFramebuffer.h"

namespace VkRender {
    Editor::Editor(EditorCreateInfo &createInfo, UUID _uuid) : m_createInfo(createInfo),
                                                               m_context(createInfo.context),
                                                               m_sizeLimits(createInfo.pPassCreateInfo.width,
                                                                            createInfo.pPassCreateInfo.height),
                                                               m_uuid(_uuid) {
        m_ui = std::make_shared<EditorUI>();
        m_ui->borderSize = m_createInfo.borderSize;
        m_ui->height = m_createInfo.height;
        m_ui->width = m_createInfo.width;
        m_ui->x = m_createInfo.x;
        m_ui->y = m_createInfo.y;

        m_createInfo.pPassCreateInfo.debugInfo = editorTypeToString(m_createInfo.editorTypeDescription);
        m_renderPass = std::make_unique<VulkanRenderPass>(&m_createInfo.pPassCreateInfo);
        VulkanRenderPassCreateInfo offscreenRenderPassCreateInfo = m_createInfo.pPassCreateInfo;
        offscreenRenderPassCreateInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        offscreenRenderPassCreateInfo.type = DEPTH_RESOLVE_RENDER_PASS;
        m_offscreenRenderPass = std::make_unique<VulkanRenderPass>(&offscreenRenderPassCreateInfo);


        m_guiManager = std::make_unique<GuiManager>(m_context->vkDevice(),
                                                    m_renderPass->getRenderPass(), // TODO verify if this is ok?
                                                    m_ui.get(),
                                                    m_createInfo.pPassCreateInfo.msaaSamples,
                                                    m_createInfo.pPassCreateInfo.swapchainImageCount,
                                                    m_context,
                                                    ImGui::CreateContext(&m_createInfo.guiResources->fontAtlas),
                                                    m_createInfo.guiResources.get());

        Log::Logger::getInstance()->info("Creating new Editor. UUID: {}, size: {}x{}, at pos: ({},{})",
                                         m_uuid.operator std::string(), m_ui->width, m_ui->height, m_ui->x, m_ui->y);
        //m_guiManager->setSceneContext(scene);

        createOffscreenFramebuffer();
    }


    void Editor::resize(EditorCreateInfo &createInfo) {
        m_createInfo = createInfo;
        m_sizeLimits = EditorSizeLimits(createInfo.pPassCreateInfo.width, createInfo.pPassCreateInfo.height);
        m_ui->height = m_createInfo.height;
        m_ui->width = m_createInfo.width;
        m_ui->x = m_createInfo.x;
        m_ui->y = m_createInfo.y;

        m_renderPass = std::make_unique<VulkanRenderPass>(&m_createInfo.pPassCreateInfo);

        m_guiManager->resize(m_createInfo.width, m_createInfo.height, m_renderPass->getRenderPass(),
                             m_createInfo.pPassCreateInfo.msaaSamples,
                             m_createInfo.guiResources);

        Log::Logger::getInstance()->info("Resizing Editor. UUID: {} : {}, size: {}x{}, at pos: ({},{})",
                                         m_uuid.operator std::string(), m_createInfo.editorIndex, m_ui->width,
                                         m_ui->height, m_ui->x, m_ui->y);

        createOffscreenFramebuffer();

    }

    void Editor::loadScene(std::shared_ptr<Scene> scene) {
        onSceneLoad(scene);
    }

    void Editor::renderScene(CommandBuffer &drawCmdBuffers, const VkRenderPass &renderPass, uint32_t imageIndex,
                             VkFramebuffer *frameBuffers, const VkViewport &viewport, VkRect2D scissor, bool includeGUI,
                             uint32_t clearValueCount, VkClearValue *clearValues) {
        /// *** Color render pass *** ///
        VkRenderPassBeginInfo renderPassBeginInfo = Populate::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.offset.x = static_cast<int32_t>(viewport.x);
        renderPassBeginInfo.renderArea.offset.y = static_cast<int32_t>(viewport.y);
        renderPassBeginInfo.renderArea.extent.width = m_createInfo.width;
        renderPassBeginInfo.renderArea.extent.height = m_createInfo.height;
        renderPassBeginInfo.clearValueCount = clearValueCount;
        renderPassBeginInfo.pClearValues = clearValues;

        renderPassBeginInfo.framebuffer = frameBuffers[imageIndex];

        vkCmdBeginRenderPass(drawCmdBuffers.getActiveBuffer(), &renderPassBeginInfo,
                             VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(drawCmdBuffers.getActiveBuffer(), 0, 1, &viewport);
        vkCmdSetScissor(drawCmdBuffers.getActiveBuffer(), 0, 1, &scissor);

        onRender(drawCmdBuffers);

        if (includeGUI)
            m_guiManager->drawFrame(drawCmdBuffers.getActiveBuffer(), drawCmdBuffers.frameIndex, m_createInfo.width,
                                    m_createInfo.height, m_createInfo.x, m_createInfo.y);
        vkCmdEndRenderPass(drawCmdBuffers.getActiveBuffer());
    }

    void Editor::render(CommandBuffer &drawCmdBuffers) {
        if (!m_renderToOffscreen) {
            /// "Normal" Render pass
            VkViewport viewport{};
            viewport.x = static_cast<float>(m_createInfo.x);
            viewport.y = static_cast<float>(m_createInfo.y);
            viewport.width = static_cast<float>(m_createInfo.width);
            viewport.height = static_cast<float>(m_createInfo.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            VkRect2D scissor{};
            scissor.offset = {(m_createInfo.x), (m_createInfo.y)};
            scissor.extent = {static_cast<uint32_t>(m_createInfo.width), static_cast<uint32_t>(m_createInfo.height)};
            renderScene(drawCmdBuffers, m_renderPass->getRenderPass(), drawCmdBuffers.activeImageIndex,
                        m_createInfo.frameBuffers, viewport, scissor);
        } else {
            const uint32_t clearValueCount = 3;
            VkClearValue clearValues[clearValueCount];
            clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}}; // Clear color to black (or any other color)
            clearValues[1].depthStencil = {1.0f, 0}; // Clear depth to 1.0 and stencil to 0
            clearValues[2].color = {{0.1f, 0.1f, 0.1f, 1.0f}}; // Clear depth to 1.0 and stencil to 0
            /// "Normal" Render pass
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = static_cast<float>(m_createInfo.width);
            viewport.height = static_cast<float>(m_createInfo.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            VkRect2D scissor{};
            scissor.offset = {(0), (0)};
            scissor.extent = {static_cast<uint32_t>(m_createInfo.width), static_cast<uint32_t>(m_createInfo.height)};
            VkImageSubresourceRange subresourceRange = {};
            subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subresourceRange.levelCount = 1;
            subresourceRange.layerCount = 1;

            VkImageSubresourceRange depthSubresourceRange = {};
            depthSubresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            depthSubresourceRange.levelCount = 1;
            depthSubresourceRange.layerCount = 1;

            Utils::setImageLayout(drawCmdBuffers.getActiveBuffer(), m_offscreenFramebuffer.resolvedImage->image(),
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, subresourceRange,
                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

            Utils::setImageLayout(drawCmdBuffers.getActiveBuffer(), m_offscreenFramebuffer.resolvedDepthImage->image(),
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, depthSubresourceRange,
                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);


            renderScene(drawCmdBuffers, m_offscreenRenderPass->getRenderPass(), 0,
                        &m_offscreenFramebuffer.framebuffer->framebuffer(), viewport, scissor, false,
                        clearValueCount, clearValues);
            // Transition image to be in shader read mode

            if (m_saveNextFrame) {

                /// *** Copy Image Data to CPU Buffer *** ///
                Utils::setImageLayout(drawCmdBuffers.getActiveBuffer(), m_offscreenFramebuffer.resolvedDepthImage->image(),
                                      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, depthSubresourceRange,
                                      VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                CHECK_RESULT(m_context->vkDevice().createBuffer(
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    m_offscreenFramebuffer.resolvedDepthImage->getImageSize(),
                    &m_copyDataBuffer.buffer,
                    &m_copyDataBuffer.memory));

                // Copy the image to the buffer
                VkBufferImageCopy region = {};
                region.bufferOffset = 0;
                region.bufferRowLength = 0;
                region.bufferImageHeight = 0;
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                region.imageSubresource.mipLevel = 0;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = {0, 0, 0};
                region.imageExtent = {
                    static_cast<uint32_t>(m_createInfo.width), static_cast<uint32_t>(m_createInfo.height),
                    1
                };

                vkCmdCopyImageToBuffer(drawCmdBuffers.getActiveBuffer(), m_offscreenFramebuffer.resolvedDepthImage->image(),
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_copyDataBuffer.buffer, 1, &region);

                drawCmdBuffers.createEvent("CopyBufferEvent", m_context->vkDevice().m_LogicalDevice);

                Utils::setImageLayout(
                    drawCmdBuffers.getActiveBuffer(),
                    m_offscreenFramebuffer.resolvedDepthImage->image(),
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);

            }

            Utils::setImageLayout(drawCmdBuffers.getActiveBuffer(), m_offscreenFramebuffer.resolvedImage->image(),
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, subresourceRange,
                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

            Utils::setImageLayout(drawCmdBuffers.getActiveBuffer(), m_offscreenFramebuffer.resolvedDepthImage->image(),
                                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, depthSubresourceRange,
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);



            if (drawCmdBuffers.isEventSet("CopyBufferEvent", m_context->vkDevice().m_LogicalDevice)) {
                // Map memory and save image to a file
                /*
                void *data;
                vkMapMemory(m_context->vkDevice().m_LogicalDevice, m_copyDataBuffer.memory, 0,
                            m_offscreenFramebuffer.resolvedDepthImage->getImageSizeRBGA(), 0, &data);
                // Assuming the image is in RGBA format (4 channels) and that data is a pointer to the pixel data
                int width = m_createInfo.width;
                int height = m_createInfo.height;
                int channels = 3; // Assuming RGBA format
                uint8_t *pixelData = static_cast<uint8_t *>(data);
                uint8_t *rgbData = new uint8_t[width * height * channels];
                for (int i = 0; i < width * height; i++) {
                    rgbData[i * 3 + 0] = pixelData[i * 4 + 2]; // R
                    rgbData[i * 3 + 1] = pixelData[i * 4 + 1]; // G
                    rgbData[i * 3 + 2] = pixelData[i * 4 + 0]; // B
                    // Alpha is ignored, and we don't need to set it because we're only saving RGB
                }
                std::filesystem::path filePath = "output.png";
                // Save the image
                if (!std::filesystem::exists(filePath.parent_path())) {
                    try {
                        std::filesystem::create_directories(filePath.parent_path());
                    } catch (std::exception &e) {
                    }
                }
                stbi_write_png(filePath.string().c_str(), width, height, channels, rgbData, width * channels);
                */
                void *data;
                vkMapMemory(m_context->vkDevice().m_LogicalDevice, m_copyDataBuffer.memory, 0,
                            m_offscreenFramebuffer.resolvedDepthImage->getImageSize(), 0, &data);

                int width = m_createInfo.width;
                int height = m_createInfo.height;
                int channels = 1; // Grayscale, single-channel
                float *depthData = static_cast<float *>(data);

                // Optional: Normalize depth data to fit into an 8-bit format for visualization
                uint8_t *grayscaleData = new uint8_t[width * height];
                for (int i = 0; i < width * height; i++) {
                    auto depthValue = static_cast<float>(depthData[i]);
                    grayscaleData[i] = static_cast<uint8_t>(depthValue * 255.0f); // Scale to 0-255
                }

                std::filesystem::path filePath = "output_depth.png";
                if (!exists(filePath.parent_path())) {
                    try {
                        create_directories(filePath.parent_path());
                    } catch (std::exception &e) {
                        // Handle the error (e.g., log it)
                    }
                }
                // Save as a grayscale PNG
                stbi_write_png(filePath.string().c_str(), width, height, channels, grayscaleData, width * channels);
                Log::Logger::getInstance()->info("Writing png image to: {}", filePath.string().c_str());
                vkUnmapMemory(m_context->vkDevice().m_LogicalDevice, m_copyDataBuffer.memory);
                drawCmdBuffers.resetEvent("CopyBufferEvent", m_context->vkDevice().m_LogicalDevice);

                vkDestroyBuffer(m_context->vkDevice().m_LogicalDevice, m_copyDataBuffer.buffer, nullptr);
                vkFreeMemory(m_context->vkDevice().m_LogicalDevice, m_copyDataBuffer.memory, nullptr);

            }
        }

    }

    void Editor::update() {
        m_ui->x = m_createInfo.x;
        m_ui->y = m_createInfo.y;

        /*
        if (m_ui->reloadPipeline)
            onPipelineReload();

         */

        m_guiManager->update();
        onUpdate();
    }

    void Editor::updateBorderState(const glm::vec2 &mousePos) {
        // Check corners first to give them higher priority
        // Top-left corner
        if (mousePos.x >= m_ui->x && mousePos.x <= m_ui->x + (m_ui->borderSize) && mousePos.y >= m_ui->y &&
            mousePos.y <= m_ui->y + (m_ui->borderSize)) {
            m_ui->lastHoveredBorderType = EditorBorderState::TopLeft;
            return;
        }
        // Top-right corner
        if (mousePos.x >= m_ui->x + m_ui->width - (m_ui->borderSize) && mousePos.x <= m_ui->x + m_ui->width &&
            mousePos.y >= m_ui->y &&
            mousePos.y <= m_ui->y + (m_ui->borderSize)) {
            m_ui->lastHoveredBorderType = EditorBorderState::TopRight;
            return;
        }
        // Bottom-left corner
        if (mousePos.x >= m_ui->x && mousePos.x <= m_ui->x + (m_ui->borderSize) &&
            mousePos.y >= m_ui->y + m_ui->height - (m_ui->borderSize) &&
            mousePos.y <= m_ui->y + m_ui->height) {
            m_ui->lastHoveredBorderType = EditorBorderState::BottomLeft;
            return;
        }
        // Bottom-right corner
        if (mousePos.x >= m_ui->x + m_ui->width - (m_ui->borderSize) && mousePos.x <= m_ui->x + m_ui->width &&
            mousePos.y >= m_ui->y + m_ui->height - (m_ui->borderSize) &&
            mousePos.y <= m_ui->y + m_ui->height) {
            m_ui->lastHoveredBorderType = EditorBorderState::BottomRight;
            return;
        }

        // Check borders
        // Left border including borderSize pixels outside the window
        if (mousePos.x >= m_ui->x - m_ui->borderSize && mousePos.x <= m_ui->x + m_ui->borderSize &&
            mousePos.y >= m_ui->y && mousePos.y <= m_ui->y + m_ui->height) {
            m_ui->lastHoveredBorderType = EditorBorderState::Left;
            return;
        }
        // Right border including borderSize pixels outside the window
        if (mousePos.x >= m_ui->x + m_ui->width - m_ui->borderSize &&
            mousePos.x <= m_ui->x + m_ui->width + m_ui->borderSize &&
            mousePos.y >= m_ui->y && mousePos.y <= m_ui->y + m_ui->height) {
            m_ui->lastHoveredBorderType = EditorBorderState::Right;
            return;
        }
        // Top border including borderSize pixels outside the window
        if (mousePos.x >= m_ui->x && mousePos.x <= m_ui->x + m_ui->width &&
            mousePos.y >= m_ui->y - m_ui->borderSize && mousePos.y <= m_ui->y + m_ui->borderSize) {
            m_ui->lastHoveredBorderType = EditorBorderState::Top;
            return;
        }
        // Bottom border including borderSize pixels outside the window
        if (mousePos.x >= m_ui->x && mousePos.x <= m_ui->x + m_ui->width &&
            mousePos.y >= m_ui->y + m_ui->height - m_ui->borderSize &&
            mousePos.y <= m_ui->y + m_ui->height + m_ui->borderSize) {
            m_ui->lastHoveredBorderType = EditorBorderState::Bottom;
            return;
        }
        if (mousePos.x >= m_ui->x && mousePos.x <= m_ui->x + m_ui->width && mousePos.y >= m_ui->y &&
            mousePos.y <= m_ui->y + m_ui->height) {
            m_ui->lastHoveredBorderType = EditorBorderState::Inside;
            return;
        }
        // Outside the editor, not on any border
        m_ui->lastHoveredBorderType = EditorBorderState::None;
    }

    EditorBorderState Editor::checkLineBorderState(const glm::vec2 &mousePos, bool verticalResize) {
        // Check borders
        if (verticalResize) {
            // Top border including borderSize pixels outside the window
            if (mousePos.y >= m_ui->y - m_ui->borderSize && mousePos.y <= m_ui->y + m_ui->borderSize) {
                return EditorBorderState::Top;
            }
            // Bottom border including borderSize pixels outside the window
            if (mousePos.y >= m_ui->y + m_ui->height - m_ui->borderSize &&
                mousePos.y <= m_ui->y + m_ui->height + m_ui->borderSize) {
                return EditorBorderState::Bottom;
            }
        } else {
            // Left border including borderSize pixels outside the window
            if (mousePos.x >= m_ui->x - m_ui->borderSize && mousePos.x <= m_ui->x + m_ui->borderSize) {
                return EditorBorderState::Left;
            }
            // Right border including borderSize pixels outside the window
            if (mousePos.x >= m_ui->x + m_ui->width - m_ui->borderSize &&
                mousePos.x <= m_ui->x + m_ui->width + m_ui->borderSize) {
                return EditorBorderState::Right;
            }
        }

        // Inside the editor, not on any border
        return EditorBorderState::None;
    }

    bool Editor::validateEditorSize(EditorCreateInfo &createInfo) {
        // Ensure the x offset is within the allowed range
        if (createInfo.x < m_sizeLimits.MIN_OFFSET_X) {
            return false;
        }
        if (createInfo.x > m_sizeLimits.MAX_OFFSET_WIDTH) {
            return false;
        }
        // Ensure the y offset is within the allowed range
        if (createInfo.y < m_sizeLimits.MIN_OFFSET_Y) {
            return false;
        }
        if (createInfo.y > m_sizeLimits.MAX_OFFSET_HEIGHT) {
            return false;
        }
        // Ensure the width is within the allowed range considering the offset
        if (createInfo.width < m_sizeLimits.MIN_SIZE) {
            return false;
        }
        if (createInfo.width > m_sizeLimits.MAX_WIDTH - createInfo.x) {
            return false;
        }
        // Ensure the height is within the allowed range considering the offset
        if (createInfo.height < m_sizeLimits.MIN_SIZE) {
            return false;
        }
        if (createInfo.height > m_sizeLimits.MAX_HEIGHT - m_sizeLimits.MENU_BAR_HEIGHT) {
            return false;
        }
        return true;
    }

    void
    Editor::windowResizeEditorsHorizontal(int32_t dx, double widthScale, std::vector<std::unique_ptr<Editor> > &editors,
                                          uint32_t width) {
        std::vector<size_t> maxHorizontalEditors;
        std::map<size_t, std::vector<size_t> > indicesHorizontalEditors;
        for (size_t i = 0; auto &sortedEditor: editors) {
            auto &editor = sortedEditor;
            // Find the matching neighbors to the right (We sorted our editors list)
            auto &ci = editor->getCreateInfo();
            int32_t nextEditorX = ci.x + ci.width;
            for (size_t j = 0; auto &nextSortedEditor: editors) {
                auto &nextEditorPosX = nextSortedEditor->getCreateInfo().x;
                //Log::Logger::getInstance()->info("Comparing Editor {} to {}, pos-x {} to {}", ci.editorIndex, editors[nextSortedEditor->second].getCreateInfo().editorIndex, nextEditorX, nextEditorPosX);
                if (nextEditorX == nextEditorPosX) {
                    indicesHorizontalEditors[i].emplace_back(j);
                }
                j++;
            }
            i++;
        }
        // Now make sure we are filling the screen
        // Find the ones touching the application border to the right and add/remove width depending on how much we're missing
        for (size_t i = 0; auto &nextSortedEditor: editors) {
            auto &nextEditor = nextSortedEditor->getCreateInfo();
            if (nextEditor.x + nextEditor.width == width - dx) {
                maxHorizontalEditors.emplace_back(i);
            }
            i++;
        }

        for (auto &editorIdx: indicesHorizontalEditors) {
            size_t index = editorIdx.first;
            auto &ci = editors[index]->getCreateInfo();
            // ci and nextCI indicesHorizontalEditors should all match after resize
            auto newWidth = static_cast<int32_t>(ci.width * widthScale);
            if (newWidth < editors[index]->getSizeLimits().MIN_SIZE)
                newWidth = editors[index]->getSizeLimits().MIN_SIZE;
            Log::Logger::getInstance()->info("Editor {}, New Width: {}, Increase: {}", ci.editorIndex, newWidth,
                                             newWidth - ci.width);
            int32_t increase = newWidth - ci.width;
            ci.width = newWidth;
        }


        // Extract entries from the map to a vector
        std::vector<std::pair<size_t, std::vector<size_t> > > entries(indicesHorizontalEditors.begin(),
                                                                      indicesHorizontalEditors.end());

        // Comparator function to sort by ciX
        auto comparator = [&](const std::pair<size_t, std::vector<size_t> > &a,
                              const std::pair<size_t, std::vector<size_t> > &b) {
            // Assuming you want to sort based on the ciX value of the first editor in each vector
            size_t indexA = a.second.front(); // or however you decide which index to use
            size_t indexB = b.second.front();
            return editors[indexA]->getCreateInfo().x < editors[indexB]->getCreateInfo().x;
        };

        // Sort the vector using the comparator
        std::sort(entries.begin(), entries.end(), comparator);

        for (auto &editorIdx: entries) {
            auto &ci = editors[editorIdx.first]->getCreateInfo();

            int32_t nextX = ci.width + ci.x;
            for (auto &idx: editorIdx.second) {
                auto &nextCI = editors[idx]->getCreateInfo();
                nextCI.x = nextX;
                Log::Logger::getInstance()->info("Editor {}, next X: {}. From editor {}: width+x: {}",
                                                 nextCI.editorIndex, nextCI.x, ci.editorIndex,
                                                 ci.x + ci.width);
            }
        } // Perform the actual resize events


        // Map to store counts and indices of editors bordering the same editor
        std::map<size_t, std::pair<size_t, std::vector<size_t> > > identicalBorders;

        // Iterate over the map to count bordering editors and store indices
        for (const auto &editorIndices: entries) {
            const size_t thisEditor = editorIndices.first;
            const std::vector<size_t> &bordersToEditors = editorIndices.second;

            for (const size_t borderedEditor: bordersToEditors) {
                // Increment the count of the bordered editor
                identicalBorders[borderedEditor].first++;
                // Store the index of the editor sharing the border
                identicalBorders[borderedEditor].second.push_back(thisEditor);
            }
        }

        for (const auto &borderInfo: identicalBorders) {
            size_t editorIndex = borderInfo.first;
            size_t count = borderInfo.second.first;
            const std::vector<size_t> &sharingEditors = borderInfo.second.second;
            // Find the editor with the largest width
            size_t maxWidthIndex = *std::max_element(sharingEditors.begin(), sharingEditors.end(),
                                                     [&](size_t a, size_t b) {
                                                         return editors[a]->getCreateInfo().width <
                                                                editors[b]->getCreateInfo().width;
                                                     });
            int largestPos =
                    editors[maxWidthIndex]->getCreateInfo().width + editors[maxWidthIndex]->getCreateInfo().x;
            // Loop over the others and check if their pos does not match, their width is adjusted such that width + x matches largestPos:
            // Loop over the others and adjust their width if needed
            for (size_t index: sharingEditors) {
                auto &editorCreateInfo = editors[index]->getCreateInfo();
                int currentPos = editorCreateInfo.width + editorCreateInfo.x;
                if (currentPos != largestPos) {
                    // Adjust the width so that width + x matches largestPos
                    editorCreateInfo.width = largestPos - editorCreateInfo.x;
                }
            }
        }


        for (auto &idx: maxHorizontalEditors) {
            auto &ci = editors[idx]->getCreateInfo();
            int32_t posRightSide = ci.x + ci.width;
            int diff = width - posRightSide;
            if (diff)
                ci.width += diff;
        }
    }

    void
    Editor::windowResizeEditorsVertical(int32_t dy, double heightScale, std::vector<std::unique_ptr<Editor> > &editors,
                                        uint32_t height) {
        std::vector<size_t> maxHorizontalEditors;
        std::map<size_t, std::vector<size_t> > indicesVertical;
        for (size_t i = 0; auto &sortedEditor: editors) {
            auto &editor = sortedEditor;
            // Find the matching neighbors to the right (We sorted our editors list)
            auto &ci = editor->getCreateInfo();
            int32_t nextEditorY = ci.y + ci.height;
            for (size_t j = 0; auto &nextSortedEditor: editors) {
                auto &nextEditorPosY = nextSortedEditor->getCreateInfo().y;
                //Log::Logger::getInstance()->info("Comparing Editor {} to {}, pos-x {} to {}", ci.editorIndex, editors[nextSortedEditor->second].getCreateInfo().editorIndex, nextEditorY, nextEditorPosY);
                if (nextEditorY == nextEditorPosY) {
                    indicesVertical[i].emplace_back(j);
                }
                j++;
            }
            i++;
        }
        // Now make sure we are filling the screen
        // Find the ones touching the application border to the right and add/remove width depending on how much we're missing
        for (size_t i = 0; auto &nextSortedEditor: editors) {
            auto &nextEditor = nextSortedEditor->getCreateInfo();
            if (nextEditor.y + nextEditor.height == height - dy) {
                maxHorizontalEditors.emplace_back(i);
            }
            i++;
        }

        for (auto &editorIdx: indicesVertical) {
            size_t index = editorIdx.first;
            auto &ci = editors[index]->getCreateInfo();
            // ci and nextCI indicesVertical should all match after resize
            auto newHeight = static_cast<int32_t>(ci.height * heightScale);
            if (newHeight < editors[index]->getSizeLimits().MIN_SIZE)
                newHeight = editors[index]->getSizeLimits().MIN_SIZE;
            Log::Logger::getInstance()->info("Editor {}, New Height: {}, Increase: {}", ci.editorIndex, newHeight,
                                             newHeight - ci.height);
            ci.height = newHeight;
        }

        // Extract entries from the map to a vector
        std::vector<std::pair<size_t, std::vector<size_t> > > entries(indicesVertical.begin(),
                                                                      indicesVertical.end());

        // Comparator function to sort by ciX
        auto comparator = [&](const std::pair<size_t, std::vector<size_t> > &a,
                              const std::pair<size_t, std::vector<size_t> > &b) {
            // Assuming you want to sort based on the ciX value of the first editor in each vector
            size_t indexA = a.second.front(); // or however you decide which index to use
            size_t indexB = b.second.front();
            return editors[indexA]->getCreateInfo().y < editors[indexB]->getCreateInfo().y;
        };

        // Sort the vector using the comparator
        std::sort(entries.begin(), entries.end(), comparator);


        for (auto &editorIdx: entries) {
            auto &ci = editors[editorIdx.first]->getCreateInfo();

            int32_t nextY = ci.height + ci.y;
            for (auto &idx: editorIdx.second) {
                auto &nextCI = editors[idx]->getCreateInfo();
                nextCI.y = nextY;
                Log::Logger::getInstance()->info("Editor {}, next X: {}. From editor {}: height+y: {}",
                                                 nextCI.editorIndex, nextCI.y, ci.editorIndex,
                                                 ci.y + ci.height);
            }
        } // Perform the actual resize events


        // Map to store counts and indices of editors bordering the same editor
        std::map<size_t, std::pair<size_t, std::vector<size_t> > > identicalBorders;

        // Iterate over the map to count bordering editors and store indices
        for (const auto &editorIndices: entries) {
            const size_t thisEditor = editorIndices.first;
            const std::vector<size_t> &bordersToEditors = editorIndices.second;

            for (const size_t borderedEditor: bordersToEditors) {
                // Increment the count of the bordered editor
                identicalBorders[borderedEditor].first++;
                // Store the index of the editor sharing the border
                identicalBorders[borderedEditor].second.push_back(thisEditor);
            }
        }

        for (const auto &borderInfo: identicalBorders) {
            size_t editorIndex = borderInfo.first;
            size_t count = borderInfo.second.first;
            const std::vector<size_t> &sharingEditors = borderInfo.second.second;
            // Find the editor with the largest width
            size_t maxWidthIndex = *std::max_element(sharingEditors.begin(), sharingEditors.end(),
                                                     [&](size_t a, size_t b) {
                                                         return editors[a]->getCreateInfo().height <
                                                                editors[b]->getCreateInfo().height;
                                                     });
            int largestPos =
                    editors[maxWidthIndex]->getCreateInfo().height + editors[maxWidthIndex]->getCreateInfo().y;
            // Loop over the others and check if their pos does not match, their height is adjusted such that height + x matches largestPos:
            // Loop over the others and adjust their height if needed
            for (size_t index: sharingEditors) {
                auto &editorCreateInfo = editors[index]->getCreateInfo();
                int currentPos = editorCreateInfo.height + editorCreateInfo.y;
                if (currentPos != largestPos) {
                    // Adjust the height so that height + x matches largestPos
                    editorCreateInfo.height = largestPos - editorCreateInfo.y;
                }
            }
        }
        for (auto &idx: maxHorizontalEditors) {
            auto &ci = editors[idx]->getCreateInfo();
            int32_t posHeight = ci.y + ci.height;
            int diff = height - posHeight;
            if (diff)
                ci.height += diff;
        }
    }


    void Editor::handleHoverState(std::unique_ptr<Editor> &editor, const VkRender::MouseButtons &mouse) {
        editor->updateBorderState(mouse.pos);

        editor->ui()->dragDelta = glm::ivec2(0.0f);
        if (editor->ui()->resizeActive) {
            // Use global mouse value, since we move it outside of the editor to resize it
            editor->ui()->cursorDelta.x = static_cast<int32_t>(mouse.dx);
            editor->ui()->cursorDelta.y = static_cast<int32_t>(mouse.dy);
            editor->ui()->cursorPos.x = editor->ui()->cursorPos.x + editor->ui()->cursorDelta.x;
            editor->ui()->cursorPos.y = editor->ui()->cursorPos.y + editor->ui()->cursorDelta.y;
        } else if (editor->ui()->lastHoveredBorderType == None) {
            editor->ui()->cursorPos = glm::ivec2(0.0f);
            editor->ui()->cursorDelta = glm::ivec2(0.0f);
            editor->ui()->lastPressedPos = glm::ivec2(0.0f);
        } else {
            int32_t newCursorPosX = std::min(std::max(static_cast<int32_t>(mouse.x) - editor->ui()->x, 0),
                                             editor->ui()->width);
            int32_t newCursorPosY = std::min(std::max(static_cast<int32_t>(mouse.y) - editor->ui()->y, 0),
                                             editor->ui()->height);
            editor->ui()->cursorDelta.x = newCursorPosX - editor->ui()->cursorPos.x;
            editor->ui()->cursorDelta.y = newCursorPosY - editor->ui()->cursorPos.y;
            editor->ui()->cursorPos.x = newCursorPosX;
            editor->ui()->cursorPos.y = newCursorPosY;
        }
        editor->ui()->cornerBottomLeftHovered = editor->ui()->lastHoveredBorderType == EditorBorderState::BottomLeft;
        editor->ui()->resizeHovered = (EditorBorderState::Left == editor->ui()->lastHoveredBorderType ||
                                       EditorBorderState::Right == editor->ui()->lastHoveredBorderType ||
                                       EditorBorderState::Top == editor->ui()->lastHoveredBorderType ||
                                       EditorBorderState::Bottom == editor->ui()->lastHoveredBorderType);
        editor->ui()->hovered = editor->ui()->lastHoveredBorderType != EditorBorderState::None;

        if (editor->ui()->hovered) {
            Log::Logger::getInstance()->traceWithFrequency("hovertag", 300, "Hovering editor: {}. Type: {}",
                                                           editor->m_createInfo.editorIndex, editorTypeToString(
                                                               editor->getCreateInfo().editorTypeDescription));
        }
    }

    void Editor::handleClickState(std::unique_ptr<Editor> &editor, const VkRender::MouseButtons &mouse) {
        if (mouse.left && mouse.action == GLFW_PRESS) {
            handleLeftMouseClick(editor);
        }
        if (mouse.right && mouse.action == GLFW_PRESS) {
            handleRightMouseClick(editor);
        }
    }

    void Editor::handleLeftMouseClick(std::unique_ptr<Editor> &editor) {
        editor->ui()->lastPressedPos = editor->ui()->cursorPos;
        editor->ui()->lastClickedBorderType = editor->ui()->lastHoveredBorderType;
        editor->ui()->resizeActive = !editor->ui()->cornerBottomLeftHovered && editor->ui()->resizeHovered;
        editor->ui()->active = editor->ui()->lastHoveredBorderType != EditorBorderState::None;
        if (editor->ui()->cornerBottomLeftHovered) {
            editor->ui()->cornerBottomLeftClicked = true;
        }
    }

    void Editor::handleRightMouseClick(std::unique_ptr<Editor> &editor) {
        editor->ui()->lastRightClickedBorderType = editor->ui()->lastHoveredBorderType;
        if (editor->ui()->resizeHovered) {
            editor->ui()->rightClickBorder = true;
        }
    }


    void Editor::handleDragState(std::unique_ptr<Editor> &editor, const VkRender::MouseButtons &mouse) {
        if (!mouse.left) return;
        if (editor->ui()->lastClickedBorderType != EditorBorderState::None) {
            int32_t dragX = editor->ui()->cursorPos.x - editor->ui()->lastPressedPos.x;
            int32_t dragY = editor->ui()->cursorPos.y - editor->ui()->lastPressedPos.y;
            editor->ui()->dragDelta = glm::ivec2(dragX, dragY);
            //Log::Logger::getInstance()->info("Editor {}, DragDelta: {},{}", editor->ui()->index, editor->ui()->dragDelta.x, editor->ui()->dragDelta.y);
            editor->ui()->dragHorizontal = editor->ui()->dragDelta.x > 50;
            editor->ui()->dragVertical = editor->ui()->dragDelta.y < -50;
            editor->ui()->dragActive = dragX > 0 || dragY > 0;
        }
    }

    void
    Editor::handleIndirectClickState(std::vector<std::unique_ptr<Editor> > &editors, std::unique_ptr<Editor> &editor,
                                     const VkRender::MouseButtons &mouse) {
        if (mouse.left && mouse.action == GLFW_PRESS) {
            //&& (!anyCornerHovered && !anyCornerClicked)) {
            for (auto &otherEditor: editors) {
                if (editor != otherEditor && otherEditor->ui()->lastClickedBorderType == EditorBorderState::None &&
                    editor->ui()->lastClickedBorderType != EditorBorderState::None &&
                    (editor->ui()->resizeHovered || otherEditor->ui()->resizeHovered)) {
                    checkAndSetIndirectResize(editor, otherEditor, mouse);
                }
            }
        }
    }

    void Editor::checkAndSetIndirectResize(std::unique_ptr<Editor> &editor, std::unique_ptr<Editor> &otherEditor,
                                           const VkRender::MouseButtons &mouse) {
        auto otherBorder = otherEditor->checkLineBorderState(mouse.pos, true);
        if (otherBorder & EditorBorderState::HorizontalBorders) {
            otherEditor->ui()->resizeActive = true;
            otherEditor->ui()->active = true;
            otherEditor->ui()->indirectlyActivated = true;
            otherEditor->ui()->lastClickedBorderType = otherBorder;
            otherEditor->ui()->lastHoveredBorderType = otherBorder;
            editor->ui()->lastClickedBorderType = editor->checkLineBorderState(mouse.pos, true);

            Log::Logger::getInstance()->info(
                "Indirect access from Editor {} to Editor {}' border: {}. Our editor resize {} {}",
                editor->m_createInfo.editorIndex,
                otherEditor->m_createInfo.editorIndex,
                otherEditor->ui()->lastClickedBorderType, editor->ui()->resizeActive,
                editor->ui()->lastClickedBorderType);
        }
        otherBorder = otherEditor->checkLineBorderState(mouse.pos, false);
        if (otherBorder & EditorBorderState::VerticalBorders) {
            otherEditor->ui()->resizeActive = true;
            otherEditor->ui()->active = true;
            otherEditor->ui()->indirectlyActivated = true;
            otherEditor->ui()->lastClickedBorderType = otherBorder;
            otherEditor->ui()->lastHoveredBorderType = otherBorder;
            editor->ui()->lastClickedBorderType = editor->checkLineBorderState(mouse.pos, false);
            Log::Logger::getInstance()->info(
                "Indirect access from Editor {} to Editor {}' border: {}. Our editor resize {} {}",
                editor->m_createInfo.editorIndex,
                otherEditor->m_createInfo.editorIndex,
                otherEditor->ui()->lastClickedBorderType, editor->ui()->resizeActive,
                editor->ui()->lastClickedBorderType);
        }
    }


    void Editor::checkIfEditorsShouldMerge(std::vector<std::unique_ptr<Editor> > &editors) {

        for (size_t i = 0; i < editors.size(); ++i) {
            if (editors[i]->ui()->shouldMerge)
                continue;

            if (editors[i]->ui()->rightClickBorder &&
                editors[i]->ui()->lastRightClickedBorderType & EditorBorderState::VerticalBorders) {
                for (size_t j = i + 1; j < editors.size(); ++j) {
                    if (editors[j]->ui()->rightClickBorder &&
                        editors[j]->ui()->lastRightClickedBorderType & EditorBorderState::VerticalBorders) {
                        auto ci2 = editors[j]->ui();
                        auto ci1 = editors[i]->ui();

                        // otherEditor is on the rightmost side
                        bool matchTopCorner = ci1->x + ci1->width == ci2->x; // Top corner of editor
                        bool matchBottomCorner = ci1->height == ci2->height;

                        // otherEditor is on the leftmost side
                        bool matchTopCornerLeft = ci2->x + ci2->width == ci1->x;
                        bool matchBottomCornerLeft = ci1->height == ci2->height;


                        if ((matchTopCorner && matchBottomCorner) || (matchTopCornerLeft && matchBottomCornerLeft)) {
                            ci1->shouldMerge = true;
                            ci2->shouldMerge = true;
                        }
                    }
                }
            }

            if (editors[i]->ui()->rightClickBorder &&
                editors[i]->ui()->lastRightClickedBorderType & EditorBorderState::HorizontalBorders) {
                for (size_t j = i + 1; j < editors.size(); ++j) {
                    if (editors[j]->ui()->rightClickBorder &&
                        editors[j]->ui()->lastRightClickedBorderType & EditorBorderState::HorizontalBorders) {
                        auto ci2 = editors[j]->ui();
                        auto ci1 = editors[i]->ui();
                        // otherEditor is on the topmost side
                        bool matchLeftCorner = ci1->y + ci1->height == ci2->y; // Top corner of editor
                        bool matchRightCorner = ci1->width == ci2->width;

                        // otherEditor is on the bottom
                        bool matchLeftCornerBottom = ci2->y + ci2->height == ci1->y;
                        bool matchRightCornerBottom = ci1->width == ci2->width;

                        if ((matchLeftCorner && matchRightCorner) ||
                            (matchLeftCornerBottom && matchRightCornerBottom)) {
                            ci1->shouldMerge = true;
                            ci2->shouldMerge = true;
                        }
                    }
                }
            }
        }
    }

    bool Editor::isValidResize(EditorCreateInfo &newEditorCI, std::unique_ptr<Editor> &editor) {
        return editor->validateEditorSize(newEditorCI);
    }

    void Editor::createOffscreenFramebuffer() { {
            VkImageCreateInfo imageCI = Populate::imageCreateInfo();
            imageCI.imageType = VK_IMAGE_TYPE_2D;
            imageCI.format = m_createInfo.pPassCreateInfo.depthFormat;
            imageCI.extent = {static_cast<uint32_t>(m_createInfo.width), static_cast<uint32_t>(m_createInfo.height), 1};
            imageCI.mipLevels = 1;
            imageCI.arrayLayers = 1;
            imageCI.samples = m_createInfo.pPassCreateInfo.msaaSamples;
            imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
            imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageViewCI.format = m_createInfo.pPassCreateInfo.depthFormat;
            imageViewCI.subresourceRange.baseMipLevel = 0;
            imageViewCI.subresourceRange.levelCount = 1;
            imageViewCI.subresourceRange.baseArrayLayer = 0;
            imageViewCI.subresourceRange.layerCount = 1;
            imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (m_createInfo.pPassCreateInfo.depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
                imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
            VulkanImageCreateInfo createInfo(m_context->vkDevice(), m_context->allocator(), imageCI, imageViewCI);
            createInfo.debugInfo =
                    "OffScreenFrameBufferDepthImage: " + editorTypeToString(m_createInfo.editorTypeDescription);
            createInfo.setLayout = true;
            createInfo.srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            createInfo.dstLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            createInfo.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

            m_offscreenFramebuffer.depthStencil = std::make_shared<VulkanImage>(createInfo);

            imageCI.imageType = VK_IMAGE_TYPE_2D;
            imageCI.format = m_createInfo.pPassCreateInfo.depthFormat;
            imageCI.extent = {static_cast<uint32_t>(m_createInfo.width), static_cast<uint32_t>(m_createInfo.height), 1};
            imageCI.mipLevels = 1;
            imageCI.arrayLayers = 1;
            imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

            imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageViewCI.format = m_createInfo.pPassCreateInfo.depthFormat;
            imageViewCI.subresourceRange.baseMipLevel = 0;
            imageViewCI.subresourceRange.levelCount = 1;
            imageViewCI.subresourceRange.baseArrayLayer = 0;
            imageViewCI.subresourceRange.layerCount = 1;
            imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (m_createInfo.pPassCreateInfo.depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
                imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
            VulkanImageCreateInfo createInfoResolved(m_context->vkDevice(), m_context->allocator(), imageCI,
                                                     imageViewCI);
            createInfoResolved.debugInfo =
                    "OffScreenFrameBufferDepthImage: " + editorTypeToString(m_createInfo.editorTypeDescription);
            createInfoResolved.setLayout = true;
            createInfoResolved.srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            createInfoResolved.dstLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            createInfoResolved.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

            m_offscreenFramebuffer.resolvedDepthImage = std::make_shared<VulkanImage>(createInfoResolved);
        } {
            VkImageCreateInfo imageCI = Populate::imageCreateInfo();
            imageCI.imageType = VK_IMAGE_TYPE_2D;
            imageCI.format = m_createInfo.pPassCreateInfo.swapchainColorFormat;
            imageCI.extent = {static_cast<uint32_t>(m_createInfo.width), static_cast<uint32_t>(m_createInfo.height), 1};
            imageCI.mipLevels = 1;
            imageCI.arrayLayers = 1;
            imageCI.samples = m_createInfo.pPassCreateInfo.msaaSamples;
            imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
            imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
            imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageViewCI.format = m_createInfo.pPassCreateInfo.swapchainColorFormat;
            imageViewCI.subresourceRange.baseMipLevel = 0;
            imageViewCI.subresourceRange.levelCount = 1;
            imageViewCI.subresourceRange.baseArrayLayer = 0;
            imageViewCI.subresourceRange.layerCount = 1;
            imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            VulkanImageCreateInfo createInfo(m_context->vkDevice(), m_context->allocator(), imageCI, imageViewCI);
            createInfo.debugInfo =
                    "OffScreenFrameBufferColorImage: " + editorTypeToString(m_createInfo.editorTypeDescription);
            createInfo.setLayout = true;
            createInfo.srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            createInfo.dstLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            createInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            m_offscreenFramebuffer.colorImage = std::make_shared<VulkanImage>(createInfo);
        } {
            VkImageCreateInfo imageCI = Populate::imageCreateInfo();
            imageCI.imageType = VK_IMAGE_TYPE_2D;
            imageCI.format = m_createInfo.pPassCreateInfo.swapchainColorFormat;
            imageCI.extent = {static_cast<uint32_t>(m_createInfo.width), static_cast<uint32_t>(m_createInfo.height), 1};
            imageCI.mipLevels = 1;
            imageCI.arrayLayers = 1;
            imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCI.usage =
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VkImageViewCreateInfo imageViewCI = Populate::imageViewCreateInfo();
            imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageViewCI.format = m_createInfo.pPassCreateInfo.swapchainColorFormat;
            imageViewCI.subresourceRange.baseMipLevel = 0;
            imageViewCI.subresourceRange.levelCount = 1;
            imageViewCI.subresourceRange.baseArrayLayer = 0;
            imageViewCI.subresourceRange.layerCount = 1;
            imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            VulkanImageCreateInfo createInfo(m_context->vkDevice(), m_context->allocator(), imageCI, imageViewCI);
            createInfo.debugInfo =
                    "OffScreenFrameBufferResolvedColorImage: " + editorTypeToString(m_createInfo.editorTypeDescription);
            createInfo.setLayout = true;
            createInfo.srcLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            createInfo.dstLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            createInfo.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            m_offscreenFramebuffer.resolvedImage = std::make_shared<VulkanImage>(createInfo);
        } {
            VulkanFramebufferCreateInfo fbCreateInfo(m_context->vkDevice());
            fbCreateInfo.width = m_createInfo.width;
            fbCreateInfo.height = m_createInfo.height;
            fbCreateInfo.renderPass = m_offscreenRenderPass->getRenderPass();
            std::vector<VkImageView> attachments(4);
            attachments[0] = m_offscreenFramebuffer.colorImage->view();
            attachments[1] = m_offscreenFramebuffer.depthStencil->view();
            attachments[2] = m_offscreenFramebuffer.resolvedImage->view();
            attachments[3] = m_offscreenFramebuffer.resolvedDepthImage->view();
            fbCreateInfo.frameBufferAttachments = attachments;
            m_offscreenFramebuffer.framebuffer = std::make_unique<VulkanFramebuffer>(fbCreateInfo);
        }
    }
}
