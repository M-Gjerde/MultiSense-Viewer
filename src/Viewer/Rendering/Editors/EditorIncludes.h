//
// Created by magnus on 7/29/24.
//

#ifndef MULTISENSE_VIEWER_EDITORINCLUDES_H
#define MULTISENSE_VIEWER_EDITORINCLUDES_H

#include <fmt/core.h>
#include <fmt/ostream.h>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>

#include "Viewer/Application/pch.h"
#include "Viewer/Rendering/Editors/EditorDefinitions.h"

#include "Viewer/Rendering/Core/VulkanFramebuffer.h"
#include "Viewer/Rendering/Core/VulkanImage.h"

namespace VkRender {
    struct OffscreenFramebuffer {
        std::shared_ptr<VulkanImage> colorImage;
        std::shared_ptr<VulkanImage> resolvedImage;
        std::shared_ptr<VulkanImage> depthStencil;
        std::shared_ptr<VulkanImage> resolvedDepthImage;
        std::unique_ptr<VulkanFramebuffer> framebuffer;
    };


    typedef enum EditorBorderState {
        None = 0,        // Cursor is not on any border
        Left = 1 << 0,   // Cursor is on the left border (1)
        Right = 1 << 1,   // Cursor is on the right border (2)
        Top = 1 << 2,   // Cursor is on the top border (4)
        Bottom = 1 << 3,   // Cursor is on the bottom border (8)
        TopLeft = 1 << 4,   // Cursor is on the top-left corner (16)
        TopRight = 1 << 5,   // Cursor is on the top-right corner (32)
        BottomLeft = 1 << 6,   // Cursor is on the bottom-left corner (64)
        BottomRight = 1 << 7,   // Cursor is on the bottom-right corner (128)
        Inside = 1 << 8,   // Cursor is inside (256)

        AnyBorders = Left | Right | Top | Bottom,
        VerticalBorders = Left | Right,
        HorizontalBorders = Top | Bottom,
        AnyCorner = TopLeft | TopRight | BottomRight | BottomLeft,
    } EditorBorderState;
}

// Overload the << operator for EditorBorderState
static std::ostream &operator<<(std::ostream &os, const VkRender::EditorBorderState &state) {
    switch (state) {
        case VkRender::None:
            os << "None";
            break;
        case VkRender::Left:
            os << "Left";
            break;
        case VkRender::Right:
            os << "Right";
            break;
        case VkRender::Top:
            os << "Top";
            break;
        case VkRender::Bottom:
            os << "Bottom";
            break;
        case VkRender::TopLeft:
            os << "TopLeft";
            break;
        case VkRender::TopRight:
            os << "TopRight";
            break;
        case VkRender::BottomLeft:
            os << "BottomLeft";
            break;
        case VkRender::BottomRight:
            os << "BottomRight";
            break;
        case VkRender::Inside:
            os << "Inside";
            break;
        default:
            os << "Unknown";
            break;
    }
    return os;
}

// Define a custom formatter for EditorBorderState for fmt within the fmt namespace
namespace fmt {
    template<>
    struct formatter<VkRender::EditorBorderState> : ostream_formatter {
    };
}

namespace VkRender {

    struct EditorSizeLimits {
        int MENU_BAR_HEIGHT = 25;
        int MIN_SIZE = 20;
        int MIN_OFFSET_X = 0;
        int MIN_OFFSET_Y = MENU_BAR_HEIGHT;
        int MAX_OFFSET_WIDTH;
        int MAX_OFFSET_HEIGHT;
        int MAX_WIDTH;
        int MAX_HEIGHT;

        EditorSizeLimits(uint32_t appWidth, uint32_t appHeight) {
            MAX_OFFSET_WIDTH = appWidth - MIN_SIZE;
            MAX_OFFSET_HEIGHT = appHeight - MIN_SIZE;

            MAX_WIDTH = appWidth;
            MAX_HEIGHT = appHeight;
        }
    };

    /** @brief Layout variables for UI Layer and is shared across layers */
    struct EditorUILayoutInfo{
        float uiHeight = 25.0f;
        float uiWidth = 50.0f;
        float borderSize = 3.0f;
        float uiYOffset = uiHeight + borderSize;
        float uiXOffset = uiWidth + borderSize;


    };


    struct EditorUI {

        int32_t x = 0;
        int32_t y = 0;
        int32_t borderSize = 0;
        int32_t width = 0;
        int32_t height = 0;
        bool resizeActive = false;
        bool resizeHovered = false;
        bool cornerBottomLeftHovered = false; // Hover state
        bool cornerBottomLeftClicked = false; // Single push down event
        bool rightClickBorder = false; // Single push down event
        bool dragHorizontal = false;  // Holding after click event
        bool dragVertical = false;  // Holding after click event
        bool dragActive = false;  // Holding after click event
        bool splitting = false;
        bool shouldMerge = false;
        EditorBorderState lastClickedBorderType = EditorBorderState::None;
        EditorBorderState lastRightClickedBorderType = EditorBorderState::None;
        EditorBorderState lastHoveredBorderType = EditorBorderState::None;
        glm::ivec2 lastPressedPos{};
        glm::ivec2 dragDelta{};
        glm::ivec2 cursorPos{0};
        glm::ivec2 cursorDelta{0};
        glm::ivec2 resizeIntervalHoriz{};
        glm::ivec2 resizeIntervalVertical{};

        ImVec4 backgroundColor{};
        ImVec4 backgroundColorHovered{};
        ImVec4 backgroundColorActive{};

        EditorUILayoutInfo layoutConstants;
        bool showDebugWindow = false;

        bool active = false;
        bool hovered = false;
        bool indirectlyActivated = false;

        // Selection menu
        std::string type;
        EditorType selectedType;
        bool changed = false;

        // assign some random background colors
        EditorUI() {
            std::vector<ImVec4> c{{1.0f, 0.0f, 0.0f, 0.4f},
                                  {0.0f, 1.0f, 0.0f, 0.4f}, // 1: Green
                                  {0.0f, 0.0f, 1.0f, 0.4f}, // 2: Blue
                                  {1.0f, 1.0f, 0.0f, 0.4f}, // 3: Yellow
                                  {1.0f, 0.0f, 1.0f, 0.4f}, // 4: Magenta
                                  {0.0f, 1.0f, 1.0f, 0.4f}, // 5: Cyan
                                  {0.5f, 0.5f, 0.5f, 0.4f}}; // 6: Gray
            std::random_device rd;  // Seed for the random number engine
            std::mt19937 gen(rd()); // Mersenne Twister random number engine
            std::uniform_int_distribution<> dis(0, c.size() - 1); // Uniform distribution
            size_t index = dis(gen); // Generate random index

            backgroundColor = {c[index].x, c[index].y, c[index].z, c[index].w};
            backgroundColorHovered = backgroundColor;
            backgroundColorHovered.w = 0.5f;
            backgroundColorActive = backgroundColor;
            backgroundColorActive.w = 0.7f;
        }
        virtual ~EditorUI() = default;  // Ensure polymorphic behavior

    };

}
#endif //MULTISENSE_VIEWER_EDITORINCLUDES_H
