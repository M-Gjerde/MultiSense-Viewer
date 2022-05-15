//
// Created by magnus on 3/4/22.
//

#include <mutex>
#include <unordered_set>
#include <iostream>
#include "MultiSense/src/crl_camera/CRLBaseCamera.h"

void CRLBaseCamera::streamCallback(const crl::multisense::image::Header &image) {
    auto &buf = buffers_[image.source];

    // TODO: make this a method of the BufferPair or something
    std::scoped_lock lock(buf.swap_lock);

    if (buf.inactiveCBBuf != nullptr)  // initial state
    {
        cameraInterface->releaseCallbackBuffer(buf.inactiveCBBuf);
    }
    imagePointers[image.source] = image;

    buf.inactiveCBBuf = cameraInterface->reserveCallbackBuffer();
    buf.inactive = image;
}

void CRLBaseCamera::imageCallback(const crl::multisense::image::Header &header, void *userDataP) {
    auto cam = reinterpret_cast<CRLBaseCamera *>(userDataP);
    cam->streamCallback(header);
}


bool CRLBaseCamera::connect(const std::string& ip) {

    if (cameraInterface == nullptr) {
        cameraInterface = crl::multisense::Channel::Create(ip);
        if (cameraInterface != nullptr) {
            getCameraMetaData();
            addCallbacks();
            return true;
        }
    }
    // TODO Add Virtual Camera connection

    return false;
}

void CRLBaseCamera::getCameraMetaData() {
    cameraInterface->setMtu(
            7200);  //FROM CRL viewer --> (try to increase this to whatever we can get away with, or use in run-level config-file)
    cameraInterface->getImageConfig(cameraInfo.imgConf);
    cameraInterface->getNetworkConfig(cameraInfo.netConfig);
    cameraInterface->getVersionInfo(cameraInfo.versionInfo);
    cameraInterface->getDeviceInfo(cameraInfo.devInfo);
    cameraInterface->getDeviceModes(cameraInfo.supportedDeviceModes);
    cameraInterface->getImageCalibration(cameraInfo.camCal);
    cameraInterface->getEnabledStreams(cameraInfo.supportedSources);

}

void CRLBaseCamera::addCallbacks() {


    for (auto e: cameraInfo.supportedDeviceModes)
        cameraInfo.supportedSources |= e.supportedDataSources;

    // reserve double_buffers for each stream
    uint_fast8_t num_sources = 0;
    crl::multisense::DataSource d = cameraInfo.supportedSources;
    while (d) {
        num_sources += (d & 1);
        d >>= 1;
    }

    // --- initializing our callback buffers ---
    std::size_t bufSize = 1024 * 1024 * 10;  // 10mb for every image, like in LibMultiSense
    for (int i = 0;
         i < (num_sources * 2 + 1); ++i) // double-buffering for each stream, plus one for handling if those are full
    {
        cameraInfo.rawImages.push_back(new uint8_t[bufSize]);
    }

    // use these buffers instead of the default
    cameraInterface->setLargeBuffers(cameraInfo.rawImages, bufSize);

    // finally, add our callback
    if (cameraInterface->addIsolatedCallback(imageCallback, cameraInfo.supportedSources, this) !=
        crl::multisense::Status_Ok) {
        std::cerr << "Adding callback failed!\n";
    }

}



std::unordered_set<crl::multisense::DataSource> CRLBaseCamera::supportedSources() {
    // this method effectively restrics the supported sources for the classice libmultisense api
    std::unordered_set<crl::multisense::DataSource> ret;
    if (cameraInfo.supportedSources & crl::multisense::Source_Raw_Left) ret.insert(crl::multisense::Source_Raw_Left);
    if (cameraInfo.supportedSources & crl::multisense::Source_Raw_Right) ret.insert(crl::multisense::Source_Raw_Right);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Left) ret.insert(crl::multisense::Source_Luma_Left);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Right)
        ret.insert(crl::multisense::Source_Luma_Right);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Rectified_Left)
        ret.insert(crl::multisense::Source_Luma_Rectified_Left);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Rectified_Right)
        ret.insert(crl::multisense::Source_Luma_Rectified_Right);
    if (cameraInfo.supportedSources & crl::multisense::Source_Chroma_Aux)
        ret.insert(crl::multisense::Source_Chroma_Aux);
    if (cameraInfo.supportedSources & crl::multisense::Source_Chroma_Left)
        ret.insert(crl::multisense::Source_Chroma_Left);
    if (cameraInfo.supportedSources & crl::multisense::Source_Chroma_Right)
        ret.insert(crl::multisense::Source_Chroma_Right);
    if (cameraInfo.supportedSources & crl::multisense::Source_Disparity_Left)
        ret.insert(crl::multisense::Source_Disparity_Left);
    if (cameraInfo.supportedSources & crl::multisense::Source_Disparity_Right)
        ret.insert(crl::multisense::Source_Disparity_Right);
    if (cameraInfo.supportedSources & crl::multisense::Source_Disparity_Cost)
        ret.insert(crl::multisense::Source_Disparity_Cost);
    if (cameraInfo.supportedSources & crl::multisense::Source_Raw_Aux) ret.insert(crl::multisense::Source_Raw_Aux);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Aux) ret.insert(crl::multisense::Source_Luma_Aux);
    if (cameraInfo.supportedSources & crl::multisense::Source_Luma_Rectified_Aux)
        ret.insert(crl::multisense::Source_Luma_Rectified_Aux);
    if (cameraInfo.supportedSources & crl::multisense::Source_Chroma_Rectified_Aux)
        ret.insert(crl::multisense::Source_Chroma_Rectified_Aux);
    if (cameraInfo.supportedSources & crl::multisense::Source_Disparity_Aux)
        ret.insert(crl::multisense::Source_Disparity_Aux);
    return ret;
}


// Pick an image size
void CRLBaseCamera::selectResolution(uint32_t RequestedWidth,
                                     uint32_t RequestedHeight) {
    // Configure the sensor.
    crl::multisense::image::Config cfg;
    bool status = cameraInterface->getImageConfig(cfg);
    if (crl::multisense::Status_Ok != status) {
        printf("Failed to query image config: %d\n", status);
    }

    cfg.setResolution(RequestedWidth, RequestedHeight);
    status = cameraInterface->setImageConfig(cfg);
    if (crl::multisense::Status_Ok != status) {
        printf("Failed to configure sensor resolution and framerate\n");
    }
}

// Pick a disparity level
void CRLBaseCamera::selectDisparities(uint32_t disparities) {
    // Configure the sensor.
    crl::multisense::image::Config cfg;
    bool status = cameraInterface->getImageConfig(cfg);
    if (crl::multisense::Status_Ok != status) {
        printf("Failed to query image config: %d\n", status);
    }

    cfg.setDisparities(disparities);
    status = cameraInterface->setImageConfig(cfg);
    if (crl::multisense::Status_Ok != status) {
        printf("Failed to configure disparities\n");
    }
}

// Set camera frames per second
void CRLBaseCamera::selectFramerate(float FPS) {
    crl::multisense::image::Config cfg;
    // This routine also scales the camera values.
    crl::multisense::Status status;


    status = cameraInterface->getImageConfig(cfg);
    if (crl::multisense::Status_Ok != status) {
        printf("Failed to query image config: %d\n", status);
    }

    cfg.setFps(FPS);  // Can be 1.0 -> 30.0
    status = cameraInterface->setImageConfig(cfg);
    if (crl::multisense::Status_Ok != status) {
        printf("Failed to configure sensor framerate: %d\n", status);
    }
}