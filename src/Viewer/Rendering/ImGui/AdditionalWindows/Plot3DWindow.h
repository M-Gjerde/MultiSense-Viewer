//
// Created by magnus on 2/3/25.
//

#ifndef PLOT3DWINDOW_H
#define PLOT3DWINDOW_H


#include <implot3d.h>

#include "Viewer/Rendering/ImGui/Layer.h"


namespace VkRender {
    class EditorDifferentiableRenderer;


    class Plot3DWindow : public VkRender::Layer {
    public:
        EditorDifferentiableRenderer* diffRenderer{};

        /** Called once upon this object creation**/
        void onAttach() override;

        /** Called after frame has finished rendered **/
        void onFinishedRender() override;

        /** Called once per frame **/
        void onUIRender() override;

        /** Called once upon this object destruction **/
        void onDetach() override;
    };
}
#endif //PLOT3DWINDOW_H
