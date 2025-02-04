//
// Created by magnus on 1/15/25.
//

#ifndef MULTISENSE_VIEWER_EDITORPATHTRACERLAYER_H
#define MULTISENSE_VIEWER_EDITORPATHTRACERLAYER_H



#include "Viewer/Rendering/ImGui/Layer.h"


namespace VkRender {
    class EditorPathTracerLayer : public Layer {
    public:
        void onAttach() override;
        void onDetach() override;
        void onUIRender() override;
        void onFinishedRender() override;
    };
}

#endif //MULTISENSE_VIEWER_EDITORPATHTRACERLAYER_H
