//
// Created by mgjer on 20/10/2023.
//

#ifndef MULTISENSE_VIEWER_LAYERUTILS_H
#define MULTISENSE_VIEWER_LAYERUTILS_H

#include "Viewer/Tools/Macros.h"

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shobjidl.h>
#include <locale>
#include <codecvt>
#else
DISABLE_WARNING_PUSH
DISABLE_WARNING_OLD_STYLE_CAST
#include <gtk/gtk.h>
DISABLE_WARNING_POP
#endif

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>

#include "Viewer/Rendering/ImGui/Widgets.h"

namespace VkRender::LayerUtils {
    typedef enum FileTypeLoadFlow {
        OBJ_FILE,
        PLY_3DGS,
        PLY_MESH,
        TEXTURE_FILE,
        VERTEX_SHADER_FILE,
        FRAGMENT_SHADER_FILE,
        SAVE_SCENE,
        SAVE_SCENE_AS,
        SAVE_PROJECT_AS,
        SELECT_FOLDER,
    } FileTypeLoadFlow;

    struct LoadFileInfo {
        std::filesystem::path path;
        FileTypeLoadFlow filetype = OBJ_FILE;
    };

    // Struct to hold parameters and result
    // Define the type of dialog action
    enum class DialogAction {
        OPEN_FILE,
        SELECT_FOLDER,
        SAVE_FILE
    };

    // Struct to hold parameters and result
    struct FileDialogData {
        std::string dialogName;
        std::vector<std::string> filetypes;
        std::string setCurrentFolder;
        LayerUtils::FileTypeLoadFlow flow;
        std::promise<LoadFileInfo> promise; // Store promise to communicate result
        DialogAction action; // Specify whether selecting file or folder
    };


#ifdef WIN32

    static LoadFileInfo saveFile(const std::string &dialogName, const std::filesystem::path &setCurrentFolder,
                                 LayerUtils::FileTypeLoadFlow flow) {
        PWSTR path = nullptr;
        std::string filePath;
        HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        if (SUCCEEDED(hr)) {
            IFileSaveDialog *pfd;
            hr = CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_ALL, IID_IFileSaveDialog,
                                  reinterpret_cast<void **>(&pfd));
            if (SUCCEEDED(hr)) {
                // Set the file types to display only .multisense files. Notice the double null termination.
                COMDLG_FILTERSPEC rgSpec[] = {
                    {L"MultiSense Files", L"*.multisense"},
                    {L"All Files", L"*.*"}
                };
                pfd->SetFileTypes(ARRAYSIZE(rgSpec), rgSpec);

                // Set the default folder
                if (!setCurrentFolder.empty()) {
                    IShellItem *psiFolder;
                    std::string tempFolderStr = setCurrentFolder.string();
                    std::wstring folderWStr(tempFolderStr.begin(), tempFolderStr.end());
                    hr = SHCreateItemFromParsingName(folderWStr.c_str(), nullptr, IID_PPV_ARGS(&psiFolder));
                    if (SUCCEEDED(hr)) {
                        pfd->SetFolder(psiFolder);
                        psiFolder->Release();
                    }
                }

                // Set the default extension to .multisense
                pfd->SetDefaultExtension(L"multisense");

                // Show the dialog
                hr = pfd->Show(nullptr);
                if (SUCCEEDED(hr)) {
                    IShellItem *psi;
                    hr = pfd->GetResult(&psi);
                    if (SUCCEEDED(hr)) {
                        hr = psi->GetDisplayName(SIGDN_FILESYSPATH, &path);
                        psi->Release();
                        if (SUCCEEDED(hr)) {
                            // Convert the selected file path to a narrow string
                            int count = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
                            filePath.resize(count - 1);
                            WideCharToMultiByte(CP_UTF8, 0, path, -1, &filePath[0], count, nullptr, nullptr);

                            // Ensure the .multisense extension is added if the user didn't include it
                            if (filePath.find(".multisense") == std::string::npos) {
                                filePath += ".multisense";
                            }
                        }
                    }
                }
                pfd->Release();
            }
            CoUninitialize();
        }
        if (path) {
            CoTaskMemFree(path);
        }
        return {filePath, flow};
    }


    static LoadFileInfo selectFile(const std::string &dialogName, const std::vector<std::string> &filetypes,
                                   const std::filesystem::path &setCurrentFolder, LayerUtils::FileTypeLoadFlow flow) {
        PWSTR path = nullptr;
        std::string filePath;
        HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        if (SUCCEEDED(hr)) {
            IFileOpenDialog *pfd;
            hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL, IID_IFileOpenDialog,
                                  reinterpret_cast<void **>(&pfd));
            if (SUCCEEDED(hr)) {
                // Set the file types to display only. Notice the double null termination.
                for (auto &filetype: filetypes) {
                    std::string fileWithDot = "*" + filetype;
                    std::wstring wFileType = std::wstring(filetype.begin(), filetype.end());
                    std::wstring fileTypeFilter = std::wstring(fileWithDot.begin(), fileWithDot.end());
                    COMDLG_FILTERSPEC rgSpec[] = {
                        {wFileType.c_str(), fileTypeFilter.c_str()},
                        {L"All Files", L"*.*"},
                    };
                    pfd->SetFileTypes(ARRAYSIZE(rgSpec), rgSpec);
                }
                // Set the default folder
                if (!setCurrentFolder.empty()) {
                    IShellItem *psiFolder;
                    std::string tempFolderStr = setCurrentFolder.string();
                    std::wstring folderWStr(tempFolderStr.begin(), tempFolderStr.end());
                    hr = SHCreateItemFromParsingName(folderWStr.c_str(), nullptr, IID_PPV_ARGS(&psiFolder));
                    if (SUCCEEDED(hr)) {
                        pfd->SetFolder(psiFolder);
                        psiFolder->Release();
                    }
                }
                // Show the dialog
                hr = pfd->Show(nullptr);
                if (SUCCEEDED(hr)) {
                    IShellItem *psi;
                    hr = pfd->GetResult(&psi);
                    if (SUCCEEDED(hr)) {
                        hr = psi->GetDisplayName(SIGDN_FILESYSPATH, &path);
                        psi->Release();
                        if (SUCCEEDED(hr)) {
                            // Convert the selected file path to a narrow string
                            int count = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
                            filePath.resize(count - 1);
                            WideCharToMultiByte(CP_UTF8, 0, path, -1, &filePath[0], count, nullptr, nullptr);
                        }
                    }
                }
                pfd->Release();
            }
            CoUninitialize();
        }
        if (path) {
            CoTaskMemFree(path);
        }
        return {filePath, flow};
    }

    static LoadFileInfo selectFolder(const std::string &dialogName, const std::string &setCurrentFolder,
                                     LayerUtils::FileTypeLoadFlow flow) {
        PWSTR path = nullptr;
        std::string folderPath;
        HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        if (SUCCEEDED(hr)) {
            IFileOpenDialog *pfd;
            hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_ALL, IID_IFileOpenDialog,
                                  reinterpret_cast<void **>(&pfd));
            if (SUCCEEDED(hr)) {
                // Set the options to pick folders instead of files
                DWORD dwFlags;
                hr = pfd->GetOptions(&dwFlags);
                if (SUCCEEDED(hr)) {
                    hr = pfd->SetOptions(dwFlags | FOS_PICKFOLDERS);
                }

                // Set the default folder
                if (!setCurrentFolder.empty()) {
                    IShellItem *psiFolder;
                    std::wstring folderWStr(setCurrentFolder.begin(), setCurrentFolder.end());
                    hr = SHCreateItemFromParsingName(folderWStr.c_str(), nullptr, IID_PPV_ARGS(&psiFolder));
                    if (SUCCEEDED(hr)) {
                        pfd->SetFolder(psiFolder);
                        psiFolder->Release();
                    }
                }

                // Show the dialog
                hr = pfd->Show(nullptr);
                if (SUCCEEDED(hr)) {
                    IShellItem *psi;
                    hr = pfd->GetResult(&psi);
                    if (SUCCEEDED(hr)) {
                        hr = psi->GetDisplayName(SIGDN_FILESYSPATH, &path);
                        psi->Release();
                        if (SUCCEEDED(hr)) {
                            // Convert the selected folder path to a narrow string
                            int count = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
                            folderPath.resize(count - 1);
                            WideCharToMultiByte(CP_UTF8, 0, path, -1, &folderPath[0], count, nullptr, nullptr);
                        }
                    }
                }
                pfd->Release();
            }
            CoUninitialize();
        }
        if (path) {
            CoTaskMemFree(path);
        }
        return {folderPath, flow};
    }

#else
    DISABLE_WARNING_PUSH
    DISABLE_WARNING_ALL


    // Function to display the file dialog in the main thread
    static gboolean show_dialog(gpointer user_data) {
        // Cast the user_data back to FileDialogData pointer
        auto* data = static_cast<FileDialogData*>(user_data);

        // Set the action based on the type of dialog
        GtkFileChooserAction gtkAction = (data->action == DialogAction::OPEN_FILE)
                                         ? GTK_FILE_CHOOSER_ACTION_OPEN
                                         : GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER;

        // File or folder dialog creation
        GtkWidget *dialog = gtk_file_chooser_dialog_new(
                data->dialogName.c_str(),
                nullptr,
                gtkAction,
                ("_Cancel"), GTK_RESPONSE_CANCEL,
                ("_Select"), GTK_RESPONSE_ACCEPT,
                nullptr
        );

        if (data->action == DialogAction::OPEN_FILE) {
            GtkFileFilter *filter = gtk_file_filter_new();
            gtk_file_filter_set_name(filter, "Select files");

            // Add file types dynamically for file selection
            for (const auto& filetype : data->filetypes) {
                gtk_file_filter_add_pattern(filter, ("*" + filetype).c_str());
            }

            gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
        }

        gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dialog), data->setCurrentFolder.c_str());

        LoadFileInfo fileInfo;

        // Run the dialog and handle the response
        if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
            char *selected_file = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
            fileInfo.path = selected_file;
            fileInfo.filetype = data->flow;
            g_free(selected_file);
        }

        gtk_widget_destroy(dialog);

        // Set the promise value (fileInfo) when done
        data->promise.set_value(fileInfo);

        delete data;  // Clean up the dynamically allocated memory

        return FALSE;  // Remove idle function after execution
    }

    // Function to display the file save dialog in the main thread
    static gboolean show_save_dialog(gpointer user_data) {
        // Cast the user_data back to FileDialogData pointer
        auto* data = static_cast<FileDialogData*>(user_data);

        // Set the action for saving a file
        GtkWidget *dialog = gtk_file_chooser_dialog_new(
                data->dialogName.c_str(),
                nullptr,
                GTK_FILE_CHOOSER_ACTION_SAVE,   // Save file action
                ("_Cancel"), GTK_RESPONSE_CANCEL,
                ("_Save"), GTK_RESPONSE_ACCEPT,
                nullptr
        );

        gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dialog), data->setCurrentFolder.c_str());
        gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(dialog), TRUE);  // Ask for overwrite confirmation

        // Filter for the .multisense file extension
        data->filetypes;
        GtkFileFilter *filter = gtk_file_filter_new();
        gtk_file_filter_set_name(filter, "Supported Files");
        // Add patterns for each filetype in data->filetypes
        for (const auto& filetype : data->filetypes) {
            std::string pattern = "*." + filetype;
            gtk_file_filter_add_pattern(filter, pattern.c_str());
        }
        gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);

        LoadFileInfo fileInfo;

        // Run the dialog and handle the response
        if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
            char *selected_file = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
            fileInfo.path = selected_file;

            // Check if the selected file has a valid extension; add the first one if missing
            bool hasValidExtension = false;
            for (const auto& filetype : data->filetypes) {
                if (fileInfo.path.string().ends_with("." + filetype)) {
                    hasValidExtension = true;
                    break;
                }
            }
            if (!hasValidExtension && !data->filetypes.empty()) {
                fileInfo.path += "." + data->filetypes.front();
            }

            fileInfo.filetype = data->flow;
            g_free(selected_file);
        }

        gtk_widget_destroy(dialog);

        // Set the promise value (fileInfo) when done
        data->promise.set_value(fileInfo);

        delete data;  // Clean up the dynamically allocated memory

        return FALSE;  // Remove idle function after execution
    }

// Function to save a file, ensuring it runs on the main thread
    static LoadFileInfo saveFile(const std::string& dialogName,  const std::vector<std::string>& type, const std::string& setCurrentFolder, LayerUtils::FileTypeLoadFlow flow) {
        // Create a new FileDialogData struct dynamically
        auto* dialogData = new FileDialogData{
                dialogName, type, setCurrentFolder, flow, std::promise<LoadFileInfo>(), DialogAction::SAVE_FILE
        };

        std::future<LoadFileInfo> future = dialogData->promise.get_future();

        // Schedule the save dialog to run in the main thread using g_idle_add
        g_idle_add(show_save_dialog, dialogData);

        // Wait for the future to be fulfilled (blocking until dialog is handled)
        return future.get();
    }

// Function to select a file, ensuring it runs on the main thread
    static LoadFileInfo selectFile(const std::string& dialogName, const std::vector<std::string>& filetypes, const std::string& setCurrentFolder, LayerUtils::FileTypeLoadFlow flow) {
        // Create a new FileDialogData struct dynamically
        // Memory is cleaned up in the dialog
        auto* dialogData = new FileDialogData{
                dialogName, filetypes, setCurrentFolder, flow, std::promise<LoadFileInfo>(), DialogAction::OPEN_FILE
        };

        std::future<LoadFileInfo> future = dialogData->promise.get_future();

        // Schedule the dialog to run in the main thread using g_idle_add
        g_idle_add(show_dialog, dialogData);

        // Wait for the future to be fulfilled (blocking until dialog is handled)
        return future.get();
    }

    static  LoadFileInfo selectFolder(const std::string& dialogName, const std::filesystem::path& openLocation, LayerUtils::FileTypeLoadFlow flow) {
        std::vector<std::string> filetypes;
        // Memory is cleaned up in the dialog
        auto* dialogData = new FileDialogData{
                dialogName, filetypes, openLocation, flow, std::promise<LoadFileInfo>(), DialogAction::SELECT_FOLDER
        };
        std::future<LoadFileInfo> future = dialogData->promise.get_future();
        // Schedule the dialog to run in the main thread using g_idle_add
        g_idle_add(show_dialog, dialogData);
        // Wait for the future to be fulfilled (blocking until dialog is handled)
        return future.get();
    }
    /*
    static LoadFileInfo selectFile(const std::string& dialogName, const std::vector<std::string>& filetypes, const std::string& setCurrentFolder, LayerUtils::FileTypeLoadFlow flow) {
        std::filesystem::path filename;

        int argc = 0;
        char **argv = nullptr;
        gtk_init(&argc, &argv);

        GtkWidget *dialog = gtk_file_chooser_dialog_new(
                dialogName.c_str(),
                nullptr,
                GTK_FILE_CHOOSER_ACTION_OPEN,
                ("_Cancel"), GTK_RESPONSE_CANCEL,
                ("_Open"), GTK_RESPONSE_ACCEPT,
                nullptr
        );

        GtkFileFilter *filter = gtk_file_filter_new();
        gtk_file_filter_set_name(filter, "Select files"); // Set filter name dynamically based on fileType

        for (const auto& filetype : filetypes)
            gtk_file_filter_add_pattern(filter, ("*" + filetype).c_str()); // Set filter pattern dynamically

        gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);
        gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dialog), setCurrentFolder.c_str());

        if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT) {
            char *selected_file = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
            filename = selected_file;
            g_free(selected_file);
        }

        gtk_widget_destroy(dialog);
        while (gtk_events_pending()) {
            gtk_main_iteration();
        }

        return {filename, flow};
    }
*/



    DISABLE_WARNING_POP
#endif

    /*
        struct WidgetPosition {
            float paddingX = -1.0f;
            ImVec4 textColor = VkRender::Colors::CRLTextWhite;
            bool sameLine = false;

            float maxElementWidth = 300.0f;
        };

        static void
        createWidgets(VkRender::GuiObjectHandles &handles, const ScriptWidgetPlacement &area,
                      WidgetPosition pos = WidgetPosition()) {
            for (auto &elem: Widgets::make()->elements[area]) {
                if (pos.paddingX != -1) {
                    ImGui::Dummy(ImVec2(pos.paddingX, 0.0f));
                    ImGui::SameLine();
                }

                switch (elem.type) {
                    case WIDGET_CHECKBOX:
                        ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);
                        if (ImGui::Checkbox(elem.label.c_str(), elem.checkbox) &&
                            ImGui::IsItemActivated()) {
                            handles.usageMonitor->userClickAction(elem.label, "WIDGET_CHECKBOX",
                                                                   ImGui::GetCurrentWindow()->Name);
                        }
                        ImGui::PopStyleColor();
                        break;

                    case WIDGET_FLOAT_SLIDER:
                        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                        ImGui::SetNextItemWidth(pos.maxElementWidth);
                        if (ImGui::SliderFloat(elem.label.c_str(), elem.value, elem.minValue, elem.maxValue) &&
                            ImGui::IsItemActivated()) {
                            handles.usageMonitor->userClickAction(elem.label, "WIDGET_FLOAT_SLIDER",
                                                                   ImGui::GetCurrentWindow()->Name);
                        }
                        ImGui::PopStyleColor();
                        break;
                    case WIDGET_INT_SLIDER:
                        ImGui::PushStyleColor(ImGuiCol_Text, VkRender::Colors::CRLTextWhite);
                        ImGui::SetNextItemWidth(pos.maxElementWidth);
                        if (elem.active)
                            *elem.active = false;
                        ImGui::SliderInt(elem.label.c_str(), elem.intValue, elem.intMin, elem.intMax);
                        if (ImGui::IsItemDeactivatedAfterEdit()) {
                            handles.usageMonitor->userClickAction(elem.label, "WIDGET_INT_SLIDER",
                                                                   ImGui::GetCurrentWindow()->Name);
                            if (elem.active)
                                *elem.active = true;
                        }
                        ImGui::PopStyleColor();
                        break;
                    case WIDGET_TEXT:
                        ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);
                        ImGui::Text("%s", elem.label.c_str());
                        ImGui::PopStyleColor();

                        break;
                    case WIDGET_INPUT_TEXT:
                        ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);
                        ImGui::SetNextItemWidth(pos.maxElementWidth);
                        ImGui::InputText(elem.label.c_str(), elem.buf, 1024, 0);
                        ImGui::PopStyleColor();
                        break;
                    case WIDGET_BUTTON:
                        ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);
                        *elem.button = ImGui::Button(elem.label.c_str());
                        ImGui::PopStyleColor();
                        break;
                    case WIDGET_GLM_VEC_3:
                        ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);
                        ImGui::SetNextItemWidth(35.0f);
                        ImGui::InputText(("x: " + elem.label).c_str(), elem.glm.xBuf, 16, ImGuiInputTextFlags_CharsDecimal);
                        ImGui::SetNextItemWidth(35.0f);
                        ImGui::InputText(("y: " + elem.label).c_str(), elem.glm.yBuf, 16, ImGuiInputTextFlags_CharsDecimal);
                        ImGui::SetNextItemWidth(35.0f);
                        ImGui::InputText(("z: " + elem.label).c_str(), elem.glm.zBuf, 16, ImGuiInputTextFlags_CharsDecimal);
                        try {
                            elem.glm.vec3->x = std::stof(elem.glm.xBuf);
                            elem.glm.vec3->y = std::stof(elem.glm.yBuf);
                            elem.glm.vec3->z = std::stof(elem.glm.zBuf);
                        }
                        catch (...) {
                        }

                        ImGui::PopStyleColor();
                        break;
                    case WIDGET_SELECT_DIR_DIALOG:
                        ImGui::PushStyleColor(ImGuiCol_Text, pos.textColor);

                        if (ImGui::Button(elem.label.c_str())) {
                            if (!elem.future->valid())
                                *elem.future = std::async(selectFolder, "");
                        }

                        ImGui::PopStyleColor();
                        break;
                    default:
                        break;
                }
                if (pos.sameLine)
                    ImGui::SameLine();
            }
            ImGui::Dummy(ImVec2());
        }
        */
}


#endif //MULTISENSE_VIEWER_LAYERUTILS_H
