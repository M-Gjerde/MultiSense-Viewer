//
// Created by mgjer on 14/07/2022.
//

#ifndef AUTOCONNECT_AUTOCONNECTWINDOWS_H
#define AUTOCONNECT_AUTOCONNECTWINDOWS_H


#include "AutoConnect.h"

class AutoConnectWindows : AutoConnect {


public:

    /** @Brief Starts the search for camera given a list containing network adapters Search is done in another thread**/
    void start(std::vector<AdapterSupportResult> vector) override;
    /** @Brief Function to search for network adapters **/
    std::vector<AdapterSupportResult> findEthernetAdapters() override;

    /** @Brief cleans up thread**/
    void stop() override;

    /** @Brief Function called after a search of adapters and at least one adapter was found **/
    void onFoundAdapters(std::vector<AdapterSupportResult> vector) override;
    /** @Brief Function called when a new IP is found. Return false if you want to keep searching or true to stop further IP searching **/
    FoundCameraOnIp onFoundIp(std::string string, AdapterSupportResult adapter) override;
    /** @Brief Function called when a camera has been found by a successfully connection by LibMultiSense **/
    void onFoundCamera() override;
    /** @Brief boolean set to true in onFoundCamera() **/
    bool isConnected() { return success; }

    AutoConnect::Result getResult();
    crl::multisense::Channel* getCameraChannel();

private:
    static void run(void* instance, std::vector<AdapterSupportResult> adapters);


};


#endif //AUTOCONNECT_AUTOCONNECTWINDOWS_H
