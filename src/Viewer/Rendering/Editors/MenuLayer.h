//
// Created by magnus on 4/7/24.
//

#ifndef MULTISENSE_VIEWER_MENULAYER_H
#define MULTISENSE_VIEWER_MENULAYER_H

#include <Viewer/Application/ProjectSerializer.h>
#include <Viewer/Scenes/SceneSerializer.h>

#include "Viewer/Rendering/ImGui/Layer.h"
#include "CommonEditorFunctions.h"

/** Is attached to the renderer through the GuiManager and instantiated in the GuiManager Constructor through
 *         pushLayer<[LayerName]>();
 *
**/

namespace VkRender {


    class MenuLayer : public VkRender::Layer {

    public:
        /** Called once upon this object creation**/
        void onAttach() override {
#ifdef __linux__
            int argc = 0;
            char **argv = NULL;
            gtk_init(&argc, &argv);  // Initialize GTK
#endif
        }

        /** Called after frame has finished rendered **/
        void onFinishedRender() override {

        }


        /** Called once per frame **/
        void onUIRender() override {


            // Push style variables
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 5.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f, 10.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 3.0f));

            // Set colors
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.4f, 0.4f, 0.4f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));

            float menuBarHeight = 25.0f;
            ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, menuBarHeight));
            if (ImGui::BeginMainMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    // Projects Menu
                    if (ImGui::BeginMenu("Projects")) {
                        auto projectDir = Utils::getProjectsPath(); // This should return a vector of project file paths
                        std::vector<std::filesystem::path> projectFiles;
                        if (std::filesystem::exists(projectDir) && std::filesystem::is_directory(projectDir)) {
                            for (const auto& entry : std::filesystem::directory_iterator(projectDir)) {
                                if (entry.is_regular_file() && entry.path().extension() == ".project") { // Use your project file extension
                                    projectFiles.push_back(entry.path());
                                }
                            }
                        }

                        for (const auto& projectFile : projectFiles) {
                            bool isCurrentProject = m_context->isCurrentProject(projectFile.filename().replace_extension().string());

                            if (ImGui::MenuItem(projectFile.filename().replace_extension().c_str(), nullptr, isCurrentProject)) {
                                if (!isCurrentProject) {
                                    Project project;
                                    ProjectSerializer serializer(project);
                                    if (serializer.deserialize(projectFile)) {
                                        m_context->loadProject(project);
                                        ImGui::SetCurrentContext(m_context->getMainUIContext());
                                    }

                                }
                            }

                        }

                        if (ImGui::MenuItem("Save current layout as Project..", nullptr)) {
                            auto &userSetting = ApplicationConfig::getInstance().getUserSetting();
                            auto openLocation = std::filesystem::exists(userSetting.lastActiveScenePath.parent_path()) ? userSetting.lastActiveScenePath.parent_path() : Utils::getSystemHomePath();
                            std::vector<std::string> types{"project"};
                            EditorUtils::saveFileDialog("Save scene as", types, LayerUtils::SAVE_PROJECT_AS, &loadFileFuture, openLocation);
                        }
                        ImGui::EndMenu();  // End the Projects submenu
                    }

                    // Scenes Menu
                    if (ImGui::BeginMenu("Scenes")) {

                        auto &userSetting = ApplicationConfig::getInstance().getUserSetting();

                        if (ImGui::MenuItem("New Scene", nullptr)) {
                            m_context->newScene();
                            userSetting.lastActiveScenePath.clear();
                        }

                        if (std::filesystem::exists(userSetting.lastActiveScenePath)) {
                            if (ImGui::MenuItem("Save Scene", nullptr)) {
                                SceneSerializer serializer(m_context->activeScene());
                                serializer.serialize(userSetting.lastActiveScenePath);
                                userSetting.assetsPath = userSetting.lastActiveScenePath.parent_path();
                            }
                        }
                        if (ImGui::MenuItem("Save Scene As..", nullptr)) {
                            auto openLocation = std::filesystem::exists(userSetting.lastActiveScenePath.parent_path()) ? userSetting.lastActiveScenePath.parent_path() : Utils::getSystemHomePath();
                            std::vector<std::string> types{"multisense"};
                            EditorUtils::saveFileDialog("Save scene as", types, LayerUtils::SAVE_SCENE_AS, &loadFileFuture, openLocation);
                        }
                        if (ImGui::MenuItem("Load Scene file", nullptr)) {
                            auto openLocation = std::filesystem::exists(userSetting.lastActiveScenePath.parent_path()) ? userSetting.lastActiveScenePath.parent_path() : Utils::getSystemHomePath();

                            std::vector<std::string> types{"multisense"};
                            EditorUtils::openImportFileDialog("Load Scene", types, LayerUtils::LOAD_SCENE, &loadFileFuture, openLocation);

                        }

                        if (ImGui::MenuItem("Set Assets Path", nullptr)) {
                            auto openLocation = std::filesystem::exists(userSetting.lastActiveScenePath.parent_path()) ? userSetting.lastActiveScenePath.parent_path() : Utils::getSystemHomePath();
                            EditorUtils::openImportFolderDialog("Load Scene", openLocation, LayerUtils::SELECT_FOLDER, &loadFileFuture);
                        }

                        ImGui::EndMenu();  // End the Scenes submenu
                    }

                    if (ImGui::MenuItem("Quit")) {
                        // Handle quitting the application
                        m_context->closeApplication();
                    }
                    ImGui::EndMenu();
                }

                if (ImGui::BeginMenu("View")) {
                    ImGui::MenuItem("Console", nullptr, &m_editor->ui()->showDebugWindow);
                    ImGui::EndMenu();
                }


                ImGui::EndMainMenuBar();
            }

            checkFileImportCompletion();
// TODO Not the right place to call ubuntu file dialog stuff.
#ifdef __linux__
            while (g_main_context_iteration(nullptr, false));
#endif
            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor(5);


        }

        void
        handleSelectedFileOrFolder(const LayerUtils::LoadFileInfo &loadFileInfo) {
            if (!loadFileInfo.path.empty()) {
                switch (loadFileInfo.filetype) {
                    case LayerUtils::LOAD_SCENE: {
                        auto scene = m_context->newScene();
                        if (scene) {
                            SceneSerializer serializer(scene);
                            serializer.deserialize(loadFileInfo.path);
                            auto &userSetting = ApplicationConfig::getInstance().getUserSetting();
                            userSetting.lastActiveScenePath = loadFileInfo.path;
                            userSetting.assetsPath = loadFileInfo.path.parent_path();
                        }
                    }
                        break;
                    case LayerUtils::SAVE_SCENE_AS: {
                        auto scene = m_context->activeScene();
                        if (scene) {
                            auto &userSetting = ApplicationConfig::getInstance().getUserSetting();
                            auto path = loadFileInfo.path;
                            SceneSerializer serializer(scene);
                            serializer.serialize(path);
                            userSetting.lastActiveScenePath = path;
                            userSetting.assetsPath = path.parent_path();
                        }
                    }
                        break;
                    case LayerUtils::SAVE_PROJECT_AS: {
                            auto project = m_context->getCurrentProject();
                            project.projectName = loadFileInfo.path.filename().replace_extension();
                            ProjectSerializer serializer(project);
                            serializer.serialize(loadFileInfo.path);
                            serializer.serialize(Utils::getProjectsPath() / loadFileInfo.path.filename());
                            ApplicationConfig::getInstance().getUserSetting().projectName = project.projectName;
                    }
                        break;
                    case LayerUtils::SELECT_FOLDER: {
                            auto scene = m_context->activeScene();
                            if (scene) {
                                std::filesystem::path assetsBasePath = loadFileInfo.path;
                                SceneSerializer serializer(scene);
                                serializer.deserialize(assetsBasePath);
                                auto &userSetting = ApplicationConfig::getInstance().getUserSetting();
                                userSetting.lastActiveScenePath = loadFileInfo.path;
                                userSetting.assetsPath = loadFileInfo.path.parent_path();
                            }
                    }
                        break;
                    default:
                        break;

                }
            }
        }

        void checkFileImportCompletion() {
            if (loadFileFuture.valid() &&
                loadFileFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                LayerUtils::LoadFileInfo loadFileInfo = loadFileFuture.get(); // Get the result from the future
                handleSelectedFileOrFolder(loadFileInfo);
            }
        }

        /** Called once upon this object destruction **/
        void onDetach()
        override {

        }

    private:
        std::future<LayerUtils::LoadFileInfo> loadFileFuture;

    };

}
#endif //MULTISENSE_VIEWER_MENULAYER_H
