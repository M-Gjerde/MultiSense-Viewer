//
// Created by magnus on 6/26/24.
//

#ifndef MULTISENSE_RENDERER_BRIDGE_H
#define MULTISENSE_RENDERER_BRIDGE_H
#include <string>
#include <vector>

#include "Viewer/Tools/AdapterUtils.h"
#include "MultiSenseTaskManager.h"
#include "CommonHeader.h"
#include "MultiSenseDevice.h"

namespace VkRender::MultiSense {

    class MultiSenseRendererBridge {
    public:
        void setSelectedMultiSenseProfile(std::shared_ptr<MultiSenseDevice> ref);
        MultiSenseProfileInfo& getSelectedMultiSenseProfile();

        void addNewProfile(MultiSenseProfileInfo, bool connectAndQuery = true);
        void removeProfile(std::shared_ptr<MultiSenseDevice> ref);
        void connect(bool retrieveCameraInfo = true);
        void disconnect();

        std::vector<std::shared_ptr<MultiSenseDevice>> getAllMultiSenseProfiles();

        std::vector<std::string> availableSources();

        void update();
        void setup();
        bool anyMultiSenseDeviceOnline();

        void getImage(MultiSenseStreamData* data);

    private:
        std::vector<std::shared_ptr<MultiSenseDevice>> m_multiSenseDevices;
        std::shared_ptr<MultiSenseDevice> m_selectedProfileRef;
        //AdapterUtils m_adapterUtils;
    };
}


#endif //MULTISENSE_RENDERER_BRIDGE_H
