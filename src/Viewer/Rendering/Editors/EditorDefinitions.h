//
// Created by mgjer on 04/08/2024.
//

#ifndef MULTISENSE_VIEWER_EDITORDEFINITIONS_H
#define MULTISENSE_VIEWER_EDITORDEFINITIONS_H

#include "Viewer/Application/pch.h"

namespace VkRender {

    // Define the EditorType enum
    enum class EditorType {
        None,
        SceneRenderer,
        Viewport3D,
        PathTracer,
        ImageEditor,
        Properties,
        GaussianViewer,
        SceneHierarchy,
        TestWindow,
        };

    static std::vector<EditorType> getAllEditorTypes() {
        return {
                EditorType::SceneRenderer,
                EditorType::Viewport3D,
                EditorType::PathTracer,
                EditorType::ImageEditor,
                EditorType::Properties,
                EditorType::GaussianViewer,
                EditorType::SceneHierarchy,
                EditorType::TestWindow};
    };
    static std::vector<EditorType> getSelectableEditorTypes() {
        return {
                EditorType::Viewport3D,
                EditorType::PathTracer,
                EditorType::ImageEditor,
                EditorType::GaussianViewer,
                EditorType::Properties,
                EditorType::SceneHierarchy,
                EditorType::TestWindow
        };
    };
    // Function to convert enum to string
    static std::string editorTypeToString(EditorType type) {
        switch(type) {
            case EditorType::SceneRenderer: return "Scene Renderer";
            case EditorType::SceneHierarchy: return "Scene Hierarchy";
            case EditorType::TestWindow: return "Test Window";
            case EditorType::Viewport3D: return "3D Viewport";
            case EditorType::PathTracer: return "Path Tracer";
            case EditorType::Properties: return "Properties";
            case EditorType::GaussianViewer: return "Gaussian Viewer";
            case EditorType::ImageEditor: return "Image Editor";
            default: return "Unknown";
        }
    }

    // Function to convert string to enum
    static EditorType stringToEditorType(const std::string &str) {
        if (str == "Scene Renderer") return EditorType::SceneRenderer;
        if (str == "Scene Hierarchy") return EditorType::SceneHierarchy;
        if (str == "Test Window") return EditorType::TestWindow;
        if (str == "3D Viewport") return EditorType::Viewport3D;
        if (str == "Path Tracer") return EditorType::PathTracer;
        if (str == "Properties") return EditorType::Properties;
        if (str == "Gaussian Viewer") return EditorType::GaussianViewer;
        if (str == "Image Editor") return EditorType::ImageEditor;
        throw std::invalid_argument("Unknown editor type string");
    }
}
#endif //MULTISENSE_VIEWER_EDITORDEFINITIONS_H
