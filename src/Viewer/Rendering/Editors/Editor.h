//
// Created by magnus on 7/15/24.
//

#ifndef MULTISENSE_VIEWER_EDITOR_H
#define MULTISENSE_VIEWER_EDITOR_H

#include "Viewer/Rendering/Core/KeyInput.h"
#include "Viewer/Scenes/Entity.h"
#include "Viewer/Rendering/Components/MaterialComponent.h"
#include "Viewer/Rendering/Components/MeshComponent.h"
#include "Viewer/Rendering/Components/PointCloudComponent.h"
#include "Viewer/Rendering/Core/RenderDefinitions.h"
#include "Viewer/Rendering/Editors/EditorIncludes.h"
#include "Viewer/Rendering/Core/VulkanRenderPass.h"
#include "Viewer/Rendering/ImGui/GuiManager.h"
#include "Viewer/Rendering/Core/UUID.h"


namespace VkRender {
    class Application;


    struct EditorCreateInfo {
        VulkanRenderPassCreateInfo pPassCreateInfo{};
        VulkanDevice *vulkanDevice = nullptr;
        VmaAllocator *allocator = nullptr;
        Application* context;
        VkFramebuffer *frameBuffers{};

        int32_t x = 0;
        int32_t y = 0;
        int32_t borderSize = 5;
        int32_t width = 10;
        int32_t height = 10;

        EditorType editorTypeDescription = EditorType::TestWindow;
        bool resizeable = true;
        size_t editorIndex = 0;

        bool disableImGUI = false;
        ImGuiContext *uiContext = nullptr;


        std::shared_ptr<GuiAssets> guiResources;

        EditorCreateInfo(std::shared_ptr<GuiAssets> guiRes, Application* ctx,
                         VulkanDevice *dev, VmaAllocator *alloc, VkFramebuffer* fbs)
                : vulkanDevice(dev), allocator(alloc), guiResources(std::move(guiRes)), context(ctx), frameBuffers(fbs) {
        }


        static void copy(EditorCreateInfo *dst, EditorCreateInfo *src) {
            dst->pPassCreateInfo = src->pPassCreateInfo;
            dst->vulkanDevice = src->vulkanDevice;
            dst->allocator = src->allocator;
            dst->frameBuffers = src->frameBuffers;
            dst->width = src->width;
            dst->height = src->height;

            dst->x = src->x;
            dst->y = src->y;
            dst->borderSize = src->borderSize;
            dst->editorTypeDescription = src->editorTypeDescription;
            dst->resizeable = src->resizeable;
            dst->editorIndex = src->editorIndex;
            dst->uiContext = src->uiContext;
        };
    };

    class Editor {
    public:

        Editor() = delete;

        explicit Editor(EditorCreateInfo &createInfo, UUID uuid = UUID());

        // Implement move constructor
        Editor(Editor &&other) noexcept: m_context(other.m_context),
                                         m_createInfo(other.m_createInfo),
                                         m_sizeLimits(other.m_createInfo.pPassCreateInfo.width,
                                                      other.m_createInfo.pPassCreateInfo.height) {
            swap(*this, other);
        }

        // and move assignment operator
        Editor &operator=(Editor &&other) noexcept {
            if (this != &other) { // Check for self-assignment
                swap(*this, other);
            }
            return *this;
        }

        // No copying allowed
        Editor(const Editor &) = delete;

        Editor &operator=(const Editor &) = delete;

        // Implement a swap function
        friend void swap(Editor &first, Editor &second) noexcept {
            std::swap(first.m_guiManager, second.m_guiManager);
            std::swap(first.m_ui, second.m_ui);
            std::swap(first.m_renderPass, second.m_renderPass);
            std::swap(first.m_createInfo, second.m_createInfo);
            std::swap(first.m_context, second.m_context);
            std::swap(first.m_uuid, second.m_uuid);
        }

        // Comparison operator
        bool operator==(const Editor &other) const {
            return other.getUUID() == getUUID();
        }

        virtual ~Editor() = default;

        void addUI(const std::string &layerName) {
            m_guiManager->pushLayer(layerName, this);
        }
        /**@brief Extends the UI data struct with custom type*/
        template <typename T>
        void addUIData() {
            static_assert(std::is_base_of<EditorUI, T>::value, "T must derive from EditorUI");
            m_ui = std::make_shared<T>(*m_ui); // Replace m_ui with a new instance of type T
        }
        const EditorSizeLimits &getSizeLimits() const { return m_sizeLimits; }

        EditorCreateInfo &getCreateInfo() { return m_createInfo; }

        const EditorCreateInfo &getCreateInfo() const { return m_createInfo; }

        ImGuiContext *guiContext() const { return m_guiManager->m_imguiContext; }

        UUID getUUID() const { return m_uuid; }

        std::shared_ptr<EditorUI> ui() { return m_ui; }
        const GuiResourcesData& guiResources(){
            return m_guiManager->guiResources();
        }

        OffscreenFramebuffer& getOffscreenFramebuffer(){return m_offscreenFramebuffer;}

        void renderScene(CommandBuffer & command_buffer, const VkRenderPass& renderPass, uint32_t frameBufferCount, VkFramebuffer* frameBuffers, const VkViewport & viewport, VkRect2D scissor, bool includeGUI = true, uint32_t clear_value_count = 0, VkClearValue * clear_values = nullptr);

        void render(CommandBuffer &drawCmdBuffers);

        virtual void onRender(CommandBuffer &drawCmdBuffers) {
        }

        virtual void onRenderDepthOnly(CommandBuffer &drawCmdBuffers) {
        }

        virtual void onMouseMove(const VkRender::MouseButtons &mouse) {}

        virtual void onMouseScroll(float change) {}

        virtual void onKeyCallback(const Input &input) {}

        virtual void onUpdate() {
        }

        virtual void onPipelineReload() {
        }

        virtual void onSceneLoad(std::shared_ptr<Scene> scene) {
        }

        virtual void onFileDrop(const std::filesystem::path &path) {}

        virtual void onEditorResize() {}

        virtual void onComponentAdded(Entity entity, MeshComponent& meshComponent){}
        virtual void onComponentRemoved(Entity entity, MeshComponent& meshComponent){}
        virtual void onComponentUpdated(Entity entity, MeshComponent& meshComponent){}

        virtual void onComponentAdded(Entity entity, MaterialComponent& meshComponent){}
        virtual void onComponentRemoved(Entity entity, MaterialComponent& meshComponent){}
        virtual void onComponentUpdated(Entity entity, MaterialComponent& meshComponent){}

        virtual void onComponentAdded(Entity entity, PointCloudComponent& meshComponent){}
        virtual void onComponentRemoved(Entity entity, PointCloudComponent& meshComponent){}
        virtual void onComponentUpdated(Entity entity, PointCloudComponent& meshComponent){}

        void loadScene(std::shared_ptr<Scene> ptr);

        void update();

        void updateBorderState(const glm::vec2 &mousePos);

        EditorBorderState checkLineBorderState(const glm::vec2 &mousePos, bool verticalResize);

        bool validateEditorSize(EditorCreateInfo &createInfo);

        void resize(EditorCreateInfo &createInfo);

        static void
        windowResizeEditorsHorizontal(int32_t dx, double widthScale, std::vector<std::unique_ptr<Editor>> &editors,
                                      uint32_t width);

        static void
        windowResizeEditorsVertical(int32_t dy, double heightScale, std::vector<std::unique_ptr<Editor>> &editors,
                                    uint32_t height);

        static void
        handleIndirectClickState(std::vector<std::unique_ptr<Editor>> &editors, std::unique_ptr<Editor> &editor,
                                 const MouseButtons &mouse);

        static bool isValidResize(EditorCreateInfo &newEditorCI, std::unique_ptr<Editor> &editor);

        static void checkIfEditorsShouldMerge(std::vector<std::unique_ptr<Editor>> &editors);

        static void checkAndSetIndirectResize(std::unique_ptr<Editor> &editor, std::unique_ptr<Editor> &otherEditor,
                                              const MouseButtons &mouse);

        static void handleRightMouseClick(std::unique_ptr<Editor> &editor);

        static void handleLeftMouseClick(std::unique_ptr<Editor> &editor);

        static void handleClickState(std::unique_ptr<Editor> &editor, const MouseButtons &mouse);

        static void handleHoverState(std::unique_ptr<Editor> &editor, const MouseButtons &mouse);

        static void handleDragState(std::unique_ptr<Editor> &editor, const MouseButtons &mouse);
        bool m_saveNextFrame = false;

    private:
        EditorSizeLimits m_sizeLimits;
        std::unique_ptr<GuiManager> m_guiManager;
        void createOffscreenFramebuffer();

    protected:
        Application* m_context;
        std::unique_ptr<VulkanRenderPass> m_renderPass;
        std::unique_ptr<VulkanRenderPass> m_offscreenRenderPass;
        EditorCreateInfo m_createInfo;

        void renderDepthPass(CommandBuffer &drawCmdBuffers);

        OffscreenFramebuffer m_offscreenFramebuffer;

        UUID m_uuid;
        std::shared_ptr<EditorUI> m_ui;

        bool m_renderDepthOnly = false;
        bool m_renderToOffscreen = false;
        bool m_disableImGUI = false;

        struct StagingBuffer {
            VkBuffer buffer;
            VkDeviceMemory memory;
        };
        StagingBuffer m_copyDataBuffer{};
    };
}

#endif //MULTISE{}NSE_VIEWER_EDITOR_H
