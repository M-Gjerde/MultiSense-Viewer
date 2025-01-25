//
// Created by mgjer on 04/08/2024.
//

#ifndef MULTISENSE_VIEWER_EDITORFACTORY_H
#define MULTISENSE_VIEWER_EDITORFACTORY_H

#include <utility>

#include "Viewer/Rendering/Editors/Editor.h"
#include "Viewer/Rendering/Editors/3DViewport/Editor3DViewport.h"
#include "Viewer/Rendering/Editors/SceneHierarchy/EditorSceneHierarchy.h"
#include "Viewer/Rendering/Editors/Test/EditorTest.h"
#include "Viewer/Rendering/Editors/Properties/EditorProperties.h"
#include "Viewer/Rendering/Editors/ImageEditor/EditorImage.h"
#include "SceneRenderer.h"
#ifdef SYCL_ENABLED
#include "Viewer/Rendering/Editors/GaussianViewer/EditorGaussianViewer.h"
#include "Viewer/Rendering/Editors/PathTracer/EditorPathTracer.h"
#ifdef DIFF_RENDERER_ENABLED
#include "Viewer/Rendering/Editors/DifferentiableEditor/EditorDifferentiableRenderer.h"
#endif
#endif

namespace VkRender {

    // Define the factory class
    class EditorFactory {
    public:
        EditorFactory(){
            registerEditor(EditorType::SceneRenderer, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<SceneRenderer>(ci, uuid);
            });
            registerEditor(EditorType::SceneHierarchy, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorSceneHierarchy>(ci, uuid);
            });
            registerEditor(EditorType::Viewport3D, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<VkRender::Editor3DViewport>(ci, uuid);
            });
            registerEditor(EditorType::ImageEditor, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorImage>(ci, uuid);
            });
            registerEditor(EditorType::TestWindow, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorTest>(ci, uuid);
            });
            registerEditor(EditorType::Properties, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorProperties>(ci, uuid);
            });
#ifdef SYCL_ENABLED
            registerEditor(EditorType::GaussianViewer, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorGaussianViewer>(ci, uuid);
            });
            registerEditor(EditorType::PathTracer, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorPathTracer>(ci, uuid);
            });

#ifdef DIFF_RENDERER_ENABLED
            registerEditor(EditorType::DifferentiableRenderer, [](EditorCreateInfo &ci, UUID uuid) {
                return std::make_unique<EditorDifferentiableRenderer>(ci, uuid);
            });
#endif

#endif
        }

        using CreatorFunc = std::function<std::unique_ptr<Editor>(EditorCreateInfo&, UUID uuid)>;

        void registerEditor(EditorType type, CreatorFunc func) {
            m_creators[type] = std::move(func);
        }

        std::unique_ptr<Editor> createEditor(EditorType type, EditorCreateInfo &createInfo, UUID uuid) {
            auto it = m_creators.find(type);
            if (it != m_creators.end()) {
                return it->second(createInfo, uuid);
            }
            Log::Logger::getInstance()->warning("Failed to find editorType: {} in factory, reverting to {}", editorTypeToString(type),
                                                editorTypeToString(EditorType::TestWindow));
            createInfo.editorTypeDescription = EditorType::TestWindow;
            return m_creators[EditorType::TestWindow](createInfo, uuid);
        }

    private:
        std::unordered_map<EditorType, CreatorFunc> m_creators;
    };

}

#endif //MULTISENSE_VIEWER_EDITORFACTORY_H
