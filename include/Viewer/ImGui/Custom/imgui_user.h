/**
 * @file: MultiSense-Viewer/include/Viewer/ImGui/Custom/imgui_user.h
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-8-2, mgjerde@carnegierobotics.com, Created file.
 **/
#ifndef MULTISENSE_VIEWER_IMGUI_USER_H
#define MULTISENSE_VIEWER_IMGUI_USER_H
#define IMGUI_DEFINE_MATH_OPERATORS

DISABLE_WARNING_PUSH
DISABLE_WARNING_ALL
#include <imgui_internal.h>
DISABLE_WARNING_POP
#include <string>

/**
 * @brief Custom IMGUI modules made for this project
 */
namespace ImGui {
    // Helper to display a little (?) mark which shows a tooltip when hovered.
// In your own code you may want to display an actual icon if you are using a merged icon fonts (see docs/FONTS.md)
    static void HelpMarker(const char *desc, ImVec4 textColor = ImVec4(0.0f, 0.0f, 0.0f, 0.0f)) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.0f, 5.0f));

        if (textColor.w != 0.0f)
            ImGui::PushStyleColor(ImGuiCol_Text, textColor);
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
        if (textColor.w != 0.0f)
            ImGui::PopStyleColor();
        ImGui::PopStyleVar();

    }


    struct Funcs {
        static int MyResizeCallback(ImGuiInputTextCallbackData *data) {
            if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
                auto *my_str = (std::string *) data->UserData;
                IM_ASSERT(my_str->data() == data->Buf);
                my_str->resize(data->BufSize); // NB: On resizing calls, generally data->BufSize == data->BufTextLen + 1
                data->Buf = my_str->data();
            }
            return 0;
        }

        static bool
        MyInputText(const char *label, std::string *my_str, ImGuiInputTextFlags flags = 0) {
            IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
            return ImGui::InputText(label, my_str->data(), (size_t) my_str->size(),
                                    flags | ImGuiInputTextFlags_CallbackResize,
                                    Funcs::MyResizeCallback, (void *) my_str);
        }
    };

    struct InputTextCallback_UserData
    {
        std::string*            Str;
        ImGuiInputTextCallback  ChainCallback;
        void*                   ChainCallbackUserData;
    };

    static int InputTextCallback(ImGuiInputTextCallbackData* data)
    {
        InputTextCallback_UserData* user_data = (InputTextCallback_UserData*)data->UserData;
        if (data->EventFlag == ImGuiInputTextFlags_CallbackResize)
        {
            // Resize string callback
            // If for some reason we refuse the new length (BufTextLen) and/or capacity (BufSize) we need to set them back to what we want.
            std::string* str = user_data->Str;
            IM_ASSERT(data->Buf == str->c_str());
            str->resize(data->BufTextLen);
            data->Buf = (char*)str->c_str();
        }
        else if (user_data->ChainCallback)
        {
            // Forward to user callback, if any
            data->UserData = user_data->ChainCallbackUserData;
            return user_data->ChainCallback(data);
        }
        return 0;
    }

    static bool CustomInputTextWithHint(const char* label, const char* hint, std::string* str, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL)
    {
        IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
        flags |= ImGuiInputTextFlags_CallbackResize;

        InputTextCallback_UserData cb_user_data{};
        cb_user_data.Str = str;
        cb_user_data.ChainCallback = callback;
        cb_user_data.ChainCallbackUserData = user_data;
        return InputTextWithHint(label, hint, (char*)str->c_str(), str->capacity() + 1, flags, InputTextCallback, &cb_user_data);
    }
    // Tip: use ImGui::PushID()/PopID() to push indices or pointers in the ID stack.
// Then you can keep 'str_id' empty or the same for all your buttons (instead of creating a string based on a non-string id)
    inline bool
    HoveredInvisibleButton(const char *str_id, bool *hovered, bool *held, const ImVec2 &size_arg,
                           ImGuiButtonFlags flags) {
        ImGuiContext &g = *GImGui;
        ImGuiWindow *window = GetCurrentWindow();
        if (window->SkipItems)
            return false;

        // Cannot use zero-size for InvisibleButton(). Unlike Button() there is not way to fallback using the label size.
        IM_ASSERT(size_arg.x != 0.0f && size_arg.y != 0.0f);

        const ImGuiID id = window->GetID(str_id);
        ImVec2 size = CalcItemSize(size_arg, 0.0f, 0.0f);
        ImVec2 bbMax = window->DC.CursorPos;
        bbMax.x += size.x;
        bbMax.y += size.y;
        const ImRect bb(window->DC.CursorPos, bbMax);
        ItemSize(size);
        if (!ItemAdd(bb, id))
            return false;

        bool pressed = ButtonBehavior(bb, id, hovered, held, flags);

        IMGUI_TEST_ENGINE_ITEM_INFO(id, str_id, g.LastItemData.StatusFlags);
        return pressed;
    }

    inline bool
    ImageButtonText(const char *str_id, int *idx, int defaultValue, const ImVec2 btnSize, ImTextureID user_texture_id,
                    const ImVec2 &size,
                    const ImVec2 &uv0,
                    const ImVec2 &uv1, const ImVec4 &tint_col) {
        ImGuiContext &g = *GImGui;
        ImGuiWindow *window = g.CurrentWindow;
        if (window->SkipItems)
            return false;

        ImVec2 posMinScreen = ImGui::GetCursorScreenPos();
        ImVec2 posMin = ImGui::GetCursorPos();

        ImGui::SetCursorScreenPos(posMinScreen);
        ImGui::PushID(1);
        bool hovered = false;
        bool held;
        bool clicked = false;
        if (HoveredInvisibleButton(str_id, &hovered, &held, btnSize, 0)) {
            if (*idx != defaultValue)
                *idx = defaultValue;
            else if (*idx == defaultValue)
                *idx = 0; // reset variable
            clicked = true;
        }
        ImGui::PopID();
        ImGui::SameLine();
        // Screen pos for adding to windowDrawList rects
        ImVec2 posMax = posMinScreen;
        posMax.x += btnSize.x;
        posMax.y += btnSize.y;
        ImVec4 color = (*idx == defaultValue) ? ImColor(0.666f, 0.674f, 0.658f, 1.0f) :
                       hovered ? ImColor(0.462f, 0.474f, 0.494f, 1.0f) :
                       ImColor(0.411f, 0.419f, 0.407f, 1.0f);
        ImGui::GetWindowDrawList()->AddRectFilled(posMinScreen, posMax,
                                                  ImColor(color), 10.0f, 0);


        // Window relative pos for text and img element
        posMax = posMin;
        posMax.x += btnSize.x;
        posMax.y += btnSize.y;
        ImVec2 txtSize = ImGui::CalcTextSize(str_id);
        ImGui::SetCursorPos(ImVec2(posMin.x + 10.0f, posMin.y + ((posMax.y - posMin.y) / 2) - (txtSize.y / 2)));
        ImGui::Text("%s", str_id);
        ImGui::SameLine();
        ImGui::PushID(0);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + ((btnSize.y - size.y) / 2));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((btnSize.x - size.x - txtSize.x - 30.0f)));
        ImageButtonEx(window->GetID(str_id), user_texture_id, size, uv0, uv1, color, tint_col);
        ImGui::PopID();
        return clicked;
    };

    inline bool ButtonWithGif(const char *str_id, const ImVec2 btnSize, ImTextureID user_texture_id, const ImVec2 &size,
                              const ImVec2 &uv0, const ImVec2 &uv1, const ImVec4 &tint_col, ImVec4 btnColor) {
        ImGuiContext &g = *GImGui;
        ImGuiWindow *window = g.CurrentWindow;
        if (window->SkipItems)
            return false;

        ImVec2 posMinScreen = ImGui::GetCursorScreenPos();
        ImVec2 posMin = ImGui::GetCursorPos();

        ImGui::SetCursorScreenPos(posMinScreen);
        ImGui::PushID(1);
        bool hovered = false;
        bool held;
        bool clicked = false;
        if (HoveredInvisibleButton(str_id, &hovered, &held, btnSize, 0)) {
            clicked = true;
        }
        ImGui::PopID();
        ImGui::SameLine();
        // Screen pos for adding to windowDrawList rects
        ImVec2 posMax = posMinScreen;
        posMax.x += btnSize.x;
        posMax.y += btnSize.y;

        ImVec4 color = btnColor;
        ImGui::GetWindowDrawList()->AddRectFilled(posMinScreen, posMax,
                                                  ImColor(color), 10.0f, 0);

        // Window relative pos for text and img element
        posMax = posMin;
        posMax.x += btnSize.x;
        posMax.y += btnSize.y;
        ImVec2 txtSize = ImGui::CalcTextSize(str_id);
        ImGui::SetCursorPos(ImVec2(posMin.x + 40.0f, posMin.y + ((posMax.y - posMin.y) / 2) - (txtSize.y / 2)));
        ImGui::Text("%s", str_id);
        ImGui::SameLine();
        ImGui::PushID(0);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + ((btnSize.y - size.y) / 2));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ((btnSize.x - size.x - txtSize.x - 50.0f)));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
        clicked |= ImageButtonEx(window->GetID(str_id), user_texture_id, size, uv0, uv1, color, tint_col);
        ImGui::PopStyleVar();
        ImGui::PopID();
        return clicked;
    };


    static inline bool isHoverable(ImGuiWindow *window, ImGuiHoveredFlags flags) {
        // An active popup disable hovering on other windows (apart from its own children)
        // FIXME-OPT: This could be cached/stored within the window.
        ImGuiContext &g = *GImGui;
        if (g.NavWindow)
            if (ImGuiWindow *focused_root_window = g.NavWindow->RootWindow)
                if (focused_root_window->WasActive && focused_root_window != window->RootWindow) {
                    // For the purpose of those flags we differentiate "standard popup" from "modal popup"
                    // NB: The order of those two tests is important because Modal windows are also Popups.
                    if (focused_root_window->Flags & ImGuiWindowFlags_Modal)
                        return false;
                    if ((focused_root_window->Flags & ImGuiWindowFlags_Popup) &&
                        !(flags & ImGuiHoveredFlags_AllowWhenBlockedByPopup))
                        return false;
                }
        return true;
    }

    inline bool IsWindowHoveredByName(std::string name, ImGuiHoveredFlags flags) {
        IM_ASSERT((flags & (ImGuiHoveredFlags_AllowWhenOverlapped | ImGuiHoveredFlags_AllowWhenDisabled)) ==
                  0);   // Flags not supported by this function
        ImGuiContext &g = *GImGui;
        ImGuiWindow *ref_window = g.HoveredWindow;
        if (ref_window == NULL)
            return false;

        if (g.HoveredWindow->RootWindow->Name != name)
            return false;

        if (!isHoverable(ref_window, flags))
            return false;
        if (!(flags & ImGuiHoveredFlags_AllowWhenBlockedByActiveItem))
            if (g.ActiveId != 0 && !g.ActiveIdAllowOverlap && g.ActiveId != ref_window->MoveId)
                return false;

        return true;
    }

}
#endif //MULTISENSE_VIEWER_IMGUI_USER_H
