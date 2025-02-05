//
// Created by magnus on 2/3/25.
//

#ifndef TOOLWINDOW_H
#define TOOLWINDOW_H


#include <implot3d.h>

#include "Viewer/Rendering/ImGui/Layer.h"


namespace VkRender {
    class EditorPathTracer;
    class EditorDifferentiableRenderer;


    class ToolWindow : public VkRender::Layer {
    public:
        EditorDifferentiableRenderer* m_diffRenderer{};
        EditorPathTracer* m_editorPathTracer{};

        /** Called once upon this object creation**/
        void onAttach() override;

        /** Called after frame has finished rendered **/
        void onFinishedRender() override;

        /** Called once per frame **/
        void onUIRender() override;

        /** Called once upon this object destruction **/
        void onDetach() override;

    private:

        void generateCameras(Scene* scene, int N, float radius);
        uint32_t m_cameraID = 0;
        bool m_checkRenderDataset = false;
    };
}
#endif //TOOLWINDOW_H
