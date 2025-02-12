//
// Created by magnus on 2/3/25.
//

#define IMGUI_DEFINE_MATH_OPERATORS

#include "Viewer/Rendering/ImGui/AdditionalWindows/ToolWindow.h"

#include <Viewer/Application/Application.h>
#include <Viewer/Rendering/Editors/Editor.h>
#include <Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRenderer.h>
#include <Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRendererLayerUI.h>

namespace VkRender {
    /** Called once upon this object creation**/
    void ToolWindow::onAttach() {
        for (auto &editor: m_context->m_editors) {
            if (editor->getCreateInfo().editorTypeDescription == EditorType::DifferentiableRenderer) {
                Editor *renderer = editor.get();
                m_diffRenderer = reinterpret_cast<EditorDifferentiableRenderer *>(renderer);
                break;
            }
        }
        for (auto &editor: m_context->m_editors) {
            if (editor->getCreateInfo().editorTypeDescription == EditorType::PathTracer) {
                Editor *renderer = editor.get();
                m_editorPathTracer = reinterpret_cast<EditorPathTracer *>(renderer);


                break;
            }
        }


        auto pathTracerUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_editorPathTracer->ui());
        pathTracerUI->kernelDevice = "GPU";
        pathTracerUI->photonCount = 10000000;
        pathTracerUI->numBounces = 0;
        pathTracerUI->shaderSelection.gammaCorrection = 1.1f;
        pathTracerUI->switchKernelDevice = true;
        pathTracerUI->useSceneCamera = true;
    }

    /** Called after frame has finished rendered **/
    void ToolWindow::onFinishedRender() {
    }

    /** Called once per frame **/
    void ToolWindow::onUIRender() {
        if (!m_editor->ui()->showPlotsWindow)
            return;

        // Begin a new ImGui window called "Debug Window".
        ImGui::Begin("Tool Window");
        uint32_t numCameras = 30;
        auto scene = m_context->activeScene();

        // Create a button labeled "Generate Cameras".
        if (ImGui::Button("Generate Cameras")) {
            // When the button is clicked, retrieve the active scene and call generateCameras.
            generateCameras(scene.get(), numCameras, 10);
        }


        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();


        ImGui::Text("Dataset Generation");
        ImGui::Spacing();

        // You can add more controls here as needed.
        auto pathTracerUI = std::dynamic_pointer_cast<EditorPathTracerLayerUI>(m_editorPathTracer->ui());
        auto optimizationUI = std::dynamic_pointer_cast<EditorDifferentiableRendererLayerUI>(m_diffRenderer->ui());

        const char *selections[] = {"CPU", "GPU"}; // TODO This should come from selectSyclDevices
        ImGui::SetNextItemWidth(100.0f);
        if (ImGui::Combo("##Select Device Type", &pathTracerUI->selectedDeviceIndex, selections,
                         IM_ARRAYSIZE(selections))) {
            pathTracerUI->kernelDevice = selections[pathTracerUI->selectedDeviceIndex];
            pathTracerUI->switchKernelDevice = true;
        }

        if (ImGui::Checkbox("Render Dataset", &m_checkRenderDataset)) {
            pathTracerUI->switchKernelDevice = true;
            pathTracerUI->useSceneCamera = true;
        }

        ImGui::Spacing();

        // Group photon settings under a collapsing header for better organization.
        if (ImGui::CollapsingHeader("Path Tracer Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Slider for Photon Count.
            // Adjust the min/max values as appropriate for your scene.
            ImGui::SliderInt("Photon Count", &pathTracerUI->photonCount, 0, 10000000);

            // Slider for Number of Bounces.
            // Here we assume bounces range from 0 to 10.
            ImGui::SliderInt("Bounces", &pathTracerUI->numBounces, 0, 10);

            // Slider for Gamma Correction.
            // The format "%.2f" shows the value with two decimals.
            ImGui::SliderFloat("Gamma Correction", &pathTracerUI->shaderSelection.gammaCorrection, 0.1f, 3.0f, "%.2f");
        }

        if (m_checkRenderDataset) {
            auto cameraView = scene->getRegistry().view<CameraComponent>();
            std::vector<Entity> cameraEntities;
            for (auto e: cameraView) {
                auto entity = Entity(e, scene.get());
                cameraEntities.push_back(entity);
            }
            auto &camera = cameraEntities[m_cameraID % cameraEntities.size()].getComponent<CameraComponent>();


            camera.isActiveCamera() = true;

            pathTracerUI->toggleRendering = true;

            pathTracerUI->bypassSave = true;


            if (m_editorPathTracer->getRenderInformation()->frameID >= 1) {
                m_cameraID++;
                auto &nextCamera = cameraEntities[m_cameraID % cameraEntities.size()].getComponent<CameraComponent>();
                nextCamera.isActiveCamera() = true;
                camera.isActiveCamera() = false;
                pathTracerUI->clearImageMemory = true;
            }

            if (m_cameraID == cameraEntities.size()) {
                m_cameraID = 0;
                m_checkRenderDataset = false;
                pathTracerUI->bypassSave = false;
                pathTracerUI->toggleRendering = false;
                pathTracerUI->useSceneCamera = false;
            }
        }


        // Create a button labeled "Generate Cameras".
        if (ImGui::Button("Stop")) {
            // When the button is clicked, retrieve the active scene and call generateCameras.
            pathTracerUI->toggleRendering = false;
            m_cameraID = 0;
            pathTracerUI->bypassSave = false;
        }

        ImGui::Spacing();

        ImGui::Separator();
        ImGui::Text("Optimization");
        ImGui::Spacing();
        if (ImGui::Checkbox("Iterate", &m_iterate)) {
            optimizationUI->reloadRenderer = true;
            optimizationUI->toggleStep = true;
            m_cameraID = 0;
        }

        if (m_iterate) {
            auto cameraView = scene->getRegistry().view<CameraComponent>();
            std::vector<Entity> cameraEntities;
            for (auto e: cameraView) {
                auto entity = Entity(e, scene.get());
                cameraEntities.push_back(entity);
            }
            auto &camera = cameraEntities[m_cameraID % cameraEntities.size()].getComponent<CameraComponent>();
            camera.isActiveCamera() = true;


            if (m_diffRenderer->m_stepIteration > m_cameraID) {
                m_cameraID++;
                camera.isActiveCamera() = false;
                cameraEntities[m_cameraID % cameraEntities.size()].getComponent<CameraComponent>().isActiveCamera() =
                        true;
            }
        }

        // Create a button labeled "Generate Cameras".
        if (ImGui::Button("Stop##Iterate")) {
            // When the button is clicked, retrieve the active scene and call generateCameras.
            optimizationUI->toggleStep = false;
        }

        // End the ImGui window.
        ImGui::End();
    }


    // Function to create N cameras evenly spaced on an upper hemisphere of a given radius.
    void ToolWindow::generateCameras(Scene *scene, int N, float radius) {
        // All cameras will look at the origin.
        glm::vec3 target(0.0f, 0.0f, 0.0f);
        // World up direction.
        glm::vec3 up(0.0f, 0.0f, 1.0f);

        // --- Fibonacci Spiral parameters for the hemisphere ---
        // When generating points on a full sphere, one common approach is to set:
        //    offset = 2.0 / N and y = (i * offset - 1) + offset/2,
        // which yields y in [-1,1]. For a hemisphere (upper half) we want y in [0,1].
        // One simple modification is to use:
        //    offset = 1.0 / N and y = 1.0 - (i + 0.5) * offset.
        // This will yield points with y from nearly 1 (the pole) down to nearly 0.
        float offset = 1.0f / static_cast<float>(N);
        // Golden angle in radians.
        float goldenAngle = glm::pi<float>() * (3.0f - std::sqrt(5.0f));

        for (int i = 0; i < N; i++) {
            // Compute the y coordinate (vertical component) so that points are evenly distributed.
            float z = 1.0f - (i + 0.5f) * offset; // y in (0,1)
            // Compute the radius of the horizontal circle at this y.
            float r = std::sqrt(1.0f - z * z);
            // Compute the azimuthal angle using the golden angle.
            float theta = goldenAngle * i;
            float x = r * std::cos(theta);
            float y = r * std::sin(theta);
            // Position in Cartesian coordinates on the unit hemisphere; scale to the desired radius.
            glm::vec3 pos = radius * glm::vec3(x, y, z);

            // Create a unique name for the camera.
            std::string cameraName = "camera" + std::to_string(i);
            auto entity = scene->getOrCreateEntityByName(cameraName);
            entity.addComponent<TemporaryComponent>();
            // Add the transform component and set its position.
            auto &transform = entity.getComponent<TransformComponent>();
            transform.setPosition(pos);

            // Compute the direction from the camera's position to the target.
            glm::vec3 direction = glm::normalize(target - pos);
            // Compute the rotation quaternion so that the camera's forward direction points to the target.
            // (Assuming the camera's forward is -Z.)
            glm::quat orientation = glm::quatLookAt(direction, up);
            transform.setRotationQuaternion(orientation);

            // Add and configure the camera component.
            auto &camera = entity.addComponent<CameraComponent>();
            camera.cameraType = CameraComponent::PINHOLE;
            camera.pinholeParameters.fx = 600;
            camera.pinholeParameters.fy = 600;
            camera.pinholeParameters.cx = 300;
            camera.pinholeParameters.cy = 300;
            camera.pinholeParameters.width = 600;
            camera.pinholeParameters.height = 600;
            camera.pinholeParameters.focalLength = 10;
            camera.pinholeParameters.fNumber = 4;
            camera.updateParametersChanged();

            // Optionally add a material component.
            auto &material = entity.addComponent<MaterialComponent>();

            auto &mesh = entity.addComponent<MeshComponent>(CAMERA_GIZMO_PINHOLE);
            mesh.polygonMode() = VK_POLYGON_MODE_LINE;
        }
    }

    /** Called once upon this object destruction **/
    void ToolWindow::onDetach() {
    }
}
