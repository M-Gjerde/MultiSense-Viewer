//
// Created by magnus on 2/3/25.
//

#define IMGUI_DEFINE_MATH_OPERATORS

#include "Viewer/Rendering/ImGui/AdditionalWindows/Plot3DWindow.h"

#include <Viewer/Application/Application.h>
#include <Viewer/Rendering/Editors/Editor.h>
#include <Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRenderer.h>

namespace VkRender {
    /** Called once upon this object creation**/
    void Plot3DWindow::onAttach() {
        ImPlot3D::CreateContext();

        for (auto& editor : m_context->m_editors) {
            if (editor->getCreateInfo().editorTypeDescription == EditorType::DifferentiableRenderer) {
                Editor* renderer = editor.get();
                diffRenderer = reinterpret_cast<EditorDifferentiableRenderer*>(renderer);
                break;
            }
        }
    }

    /** Called after frame has finished rendered **/
    void Plot3DWindow::onFinishedRender() {
    }

    /** Called once per frame **/
    void Plot3DWindow::onUIRender() {
        if (!m_editor->ui()->showPlotsWindow)
            return;

        EditorDifferentiableRenderer::DebugData debugData = diffRenderer->m_lastIteration;
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoSavedSettings;
        bool open = true;
        ImGui::SetNextWindowSize(ImVec2(600.0f, 600.0f), ImGuiCond_Always);
        ImGui::Begin("Plots Window", &open, window_flags);

        if (ImPlot3D::BeginPlot("3D Axis Vectors")) {
            ImPlot3D::SetupAxes("X", "Y", "Z");
            ImPlot3D::SetupAxesLimits(-4, 2, -4, 3, -2, 4);

            // X-axis vector
            float x_coords[2] = {0.0f, 1.0f};
            float y_coords_x[2] = {0.0f, 0.0f};
            float z_coords_x[2] = {0.0f, 0.0f};
            ImPlot3D::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 2);
            ImPlot3D::PlotLine("X-Axis", x_coords, y_coords_x, z_coords_x, 2);

            // Y-axis vector
            float x_coords_y[2] = {0.0f, 0.0f};
            float y_coords[2] = {0.0f, 1.0f};
            float z_coords_y[2] = {0.0f, 0.0f};
            ImPlot3D::SetNextLineStyle(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), 2);
            ImPlot3D::PlotLine("Y-Axis", x_coords_y, y_coords, z_coords_y, 2);

            // Z-axis vector
            float x_coords_z[2] = {0.0f, 0.0f};
            float y_coords_z[2] = {0.0f, 0.0f};
            float z_coords[2] = {0.0f, 1.0f};
            ImPlot3D::SetNextLineStyle(ImVec4(0.0f, 0.0f, 1.0f, 1.0f), 2);
            ImPlot3D::PlotLine("Z-Axis", x_coords_z, y_coords_z, z_coords, 2);

            // 4th vector from (1, 2, 3) to (-3, -2, -1)
            float x_coords_4th[2] = {0.0f, debugData.positionGradient.x * 100};
            float y_coords_4th[2] = {0.0f, debugData.positionGradient.y * 100};
            float z_coords_4th[2] = {0.0f, debugData.positionGradient.z * 100};
            ImPlot3D::SetNextLineStyle(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), 2);
            ImPlot3D::PlotLine("4th Vector", x_coords_4th, y_coords_4th, z_coords_4th, 2, ImPlot3DLineFlags_Segments);

            // Start and end markers for the 4th vector
            ImPlot3D::SetNextMarkerStyle(ImPlot3DMarker_Circle, 6, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            ImPlot3D::PlotScatter("Start Point", &x_coords_4th[0], &y_coords_4th[0], &z_coords_4th[0], 1);

            ImPlot3D::SetNextMarkerStyle(ImPlot3DMarker_Square, 6, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
            ImPlot3D::PlotScatter("End Point", &x_coords_4th[1], &y_coords_4th[1], &z_coords_4th[1], 1);

            ImPlot3D::EndPlot();
        }

        ImGui::End();
    }

    /** Called once upon this object destruction **/
    void Plot3DWindow::onDetach() {
    }
}
