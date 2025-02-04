/**
 * @file: MultiSense-Viewer/src/CRLCamera/CRLPhysicalCamera.cpp
 *
 * Copyright 2022
 * Carnegie Robotics, LLC
 * 4501 Hatfield Street, Pittsburgh, PA 15201
 * http://www.carnegierobotics.com
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Robotics, LLC nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CARNEGIE ROBOTICS, LLC BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Significant history (date, user, action):
 *   2022-1-3, mgjerde@carnegierobotics.com, Created file.
 **/
#ifdef WIN32
#define _USE_MATH_DEFINES

#include <cmath>

#endif

#include <vulkan/vulkan_core.h>

#include <cmath>

#include "Viewer/CRLCamera/CRLPhysicalCamera.h"
#include "Viewer/Core/RenderDefinitions.h"
#include "Viewer/CRLCamera/CalibrationYaml.h"
#include "Viewer/Tools/Utils.h"

namespace VkRender::MultiSense {

    std::vector<crl::multisense::RemoteHeadChannel>
    CRLPhysicalCamera::connect(const VkRender::Device *dev, const std::string &ifName) {
        std::vector<crl::multisense::RemoteHeadChannel> indices;
        // If RemoteHead then attempt to connect 4 LibMultiSense channels
        // else create only one and place it at 0th index.
        if (dev->interruptConnection) {
            indices.clear();
            return indices;
        }

        Log::Logger::getInstance()->trace("Attempting to connect to ip: {}. Adapter is: {}", dev->IP, ifName);

        channelMap[static_cast<crl::multisense::RemoteHeadChannel>(0)] = std::make_unique<ChannelWrapper>(dev->IP);

        if (channelMap[static_cast<crl::multisense::RemoteHeadChannel>(0)].get()->ptr() != nullptr) {
            indices.emplace_back(static_cast<crl::multisense::RemoteHeadChannel>(0));
            crl::multisense::system::DeviceInfo devInfo;
            channelMap[static_cast<crl::multisense::RemoteHeadChannel>(0)].get()->ptr()->getDeviceInfo(devInfo);
            Log::Logger::getInstance()->trace("We got a connection! Device info: {}, {}", devInfo.name,
                                              devInfo.buildDate);
            setMtu(1400, static_cast<crl::multisense::RemoteHeadChannel>(0));

            if (!updateCameraInfo(const_cast<Device *>(dev), static_cast<crl::multisense::RemoteHeadChannel>(0))) {
                Log::Logger::getInstance()->error(
                        "Failed to fetch camera info. Viewer will not present the full set of options");
            } // TODO if this one fails we have to recheck our sources we present to the screen.
            addCallbacks(static_cast<crl::multisense::RemoteHeadChannel>(0));
        }

        if (dev->interruptConnection) {
            indices.clear();
            return indices;
        }

        return indices;
    }

    bool CRLPhysicalCamera::start(const std::string &dataSourceStr, crl::multisense::RemoteHeadChannel channelID) {
        crl::multisense::DataSource source = Utils::stringToDataSource(dataSourceStr);
        if (source == false)
            return false;
        // Start stream
        crl::multisense::Status status = channelMap[channelID]->ptr()->startStreams(source);
        if (status == crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Started stream: {} on channel {}",
                                             Utils::dataSourceToString(source).c_str(), channelID);
            return true;
        } else
            Log::Logger::getInstance()->info("Failed to start stream: {}  status code {}",
                                             Utils::dataSourceToString(source).c_str(), status);
        return false;
    }

    bool CRLPhysicalCamera::stop(const std::string &dataSourceStr, crl::multisense::RemoteHeadChannel channelID) {
        if (channelMap[channelID] == nullptr)
            return false;
        crl::multisense::DataSource src = Utils::stringToDataSource(dataSourceStr);
        if (!src) {
            Log::Logger::getInstance()->info("Failed to recognize '{}' source", dataSourceStr.c_str());
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->stopStreams(src);
        if (status == crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Stopped camera stream {} on channel {}", dataSourceStr.c_str(),
                                             channelID);
            return true;
        } else {
            Log::Logger::getInstance()->info("Failed to stop stream {}", dataSourceStr.c_str());
            return false;
        }
    }

    void CRLPhysicalCamera::remoteHeadCallback(const crl::multisense::image::Header &header, void *userDataP) {
        auto p = reinterpret_cast<ChannelWrapper *>(userDataP);
        //Log::Logger::getInstance()->info("Received m_Image {}x{} on channel {} with source {}", header.m_Width, header.m_Height, p->imageBuffer->id, header.source);
        p->imageBuffer->updateImageBuffer(std::make_shared<ImageBufferWrapper>(p->ptr(), header));
    }

    void CRLPhysicalCamera::imuCallback(const crl::multisense::imu::Header &header,
                                        void *userDataP) {
        auto p = reinterpret_cast<ChannelWrapper *>(userDataP);
        p->imuBuffer->updateIMUBuffer(std::make_shared<IMUBufferWrapper>(p->ptr(), header));

    }

    void CRLPhysicalCamera::addCallbacks(crl::multisense::RemoteHeadChannel channelID) {
        for (const auto &e: infoMap[channelID].supportedDeviceModes)
            infoMap[channelID].supportedSources |= e.supportedDataSources;
        // reserve double_buffers for each stream
        if (channelMap[channelID]->ptr()->addIsolatedCallback(imuCallback, channelMap[channelID].get()) ==
            crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Added imu callback for channel {}", channelID);
        } else {
            Log::Logger::getInstance()->info("Failed to add callback for channel {}", channelID);
        }

        if (channelMap[channelID]->ptr()->addIsolatedCallback(remoteHeadCallback,
                                                              infoMap[channelID].supportedSources,
                                                              channelMap[channelID].get()) ==
            crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Added image callback for channel {}", channelID);
        } else
            Log::Logger::getInstance()->info("Failed to add callback for channel {}", channelID);
    }

    CRLPhysicalCamera::CameraInfo CRLPhysicalCamera::getCameraInfo(crl::multisense::RemoteHeadChannel idx) const {
        auto it = infoMap.find(idx);
        if (it != infoMap.end())
            return it->second;
        else {
            Log::Logger::getInstance()->traceWithFrequency("getCameraInfoTag", 1000,
                                                           "Camera info for channel {} does not exist", idx);
            return {};
        }
    }

    bool CRLPhysicalCamera::getCameraStream(const std::string &stringSrc, VkRender::TextureData *tex,
                                            crl::multisense::RemoteHeadChannel channelID) const {
        auto it = channelMap.find(channelID);

        if (it == channelMap.end()) {
            return false;
        }

        auto src = Utils::stringToDataSource(stringSrc);
        crl::multisense::DataSource colorSource;
        crl::multisense::DataSource lumaSource;
        std::shared_ptr<ImageBufferWrapper> header;
        std::shared_ptr<ImageBufferWrapper> headerTwo;
        if (stringSrc == "Color Aux") {
            colorSource = crl::multisense::Source_Chroma_Aux;
            lumaSource = crl::multisense::Source_Luma_Aux;
            header = it->second->imageBuffer->getImageBuffer(channelID, lumaSource);
            headerTwo = it->second->imageBuffer->getImageBuffer(channelID, colorSource);
            if (headerTwo == nullptr) {
                return false;
            }

        } else if (stringSrc == "Color Rectified Aux") {
            colorSource = crl::multisense::Source_Chroma_Rectified_Aux;
            lumaSource = crl::multisense::Source_Luma_Rectified_Aux;
            header = it->second->imageBuffer->getImageBuffer(channelID, lumaSource);
            headerTwo = it->second->imageBuffer->getImageBuffer(channelID, colorSource);
            if (headerTwo == nullptr) {
                return false;
            }
        } else {
            header = it->second->imageBuffer->getImageBuffer(channelID, src);
        }

        if (header == nullptr) {
            return false;
        }

        tex->m_TimeSeconds = header->data().timeSeconds;
        tex->m_TimeMicroSeconds = header->data().timeMicroSeconds;
        switch (tex->m_Type) {
            case CRL_COLOR_IMAGE_YUV420: {
                if (headerTwo->data().source != src ||
                    tex->m_Width != header->data().width ||
                    tex->m_Height < header->data().height) {
                    //Log::Logger::getInstance()->error(
                    //        "In getCameraStream: Color Source and dimensions did not match expected values");
                    return false;
                }
                tex->m_Id = static_cast<uint32_t>(header->data().frameId);
                tex->m_Id2 = static_cast<uint32_t>(headerTwo->data().frameId);

                // These should be the same height. This can happen with the Color_Aux source when switching between max and h res.
                if (header->data().height == headerTwo->data().height) {
                    Log::Logger::getInstance()->error(
                            "In getCameraStream: Color Source and dimensions does not have matching height");
                    return false;
                }

                std::memcpy(tex->data, header->data().imageDataP, header->data().imageLength);
                // if data3 is not null we are using a manual ycbcr sampler. Therefore do a strided copy
                if (tex->data3 != nullptr) {
                    size_t channelSize = headerTwo->data().imageLength / 2;
                    const auto *p = reinterpret_cast<const uint16_t *>(headerTwo->data().imageDataP);
                    for (size_t i = 0; i < channelSize; i++) {
                        tex->data2[i] = (static_cast<uint16_t>( p[i] >> 0) & 0xFF);
                        tex->data3[i] = (static_cast<uint16_t>( p[i] >> 8) & 0xFF);
                    }
                } else
                    std::memcpy(tex->data2, headerTwo->data().imageDataP, headerTwo->data().imageLength);

                // Copy extra zeros to the bottom row if heights does not match
                if (tex->m_Height != header->data().height) {
                    uint32_t diff = tex->m_Height - header->data().height;
                    std::memset(tex->data + header->data().imageLength, 0x00, diff * tex->m_Width);

                    if (tex->data3 != nullptr) {
                        size_t diff2 = static_cast<size_t>((tex->m_Height) / 2) - headerTwo->data().height;
                        //std::memset(tex->data2 + (headerTwo->data().imageLength / 2), 0x00, diff2 * tex->m_Width);
                        //std::memset(tex->data3 + (headerTwo->data().imageLength / 2), 0x00, diff2 * tex->m_Width);

                    } else {
                        uint32_t diff2 = tex->m_Height / 2 - headerTwo->data().height;
                        std::memset(tex->data2 + headerTwo->data().imageLength, 0x00, diff2 * tex->m_Width);
                    }
                }
            }
                return true;

            case CRL_DISPARITY_IMAGE:
                DISABLE_WARNING_PUSH
                DISABLE_WARNING_IMPLICIT_FALLTHROUGH
                if (header->data().bitsPerPixel != 16) {
                    Log::Logger::getInstance()->warning("In getCameraStream: Unsupported disparity pixel depth");
                    return false;
                }DISABLE_WARNING_POP
            case CRL_GRAYSCALE_IMAGE:
            case CRL_POINT_CLOUD:
                if (header->data().source != src || tex->m_Width != header->data()
                        .width || tex->m_Height < header->data().height) {
                    Log::Logger::getInstance()->trace(
                            "In getCameraStream: Monochrome source and dimensions did not match expected values");
                    return false;
                }
                tex->m_Id = static_cast<uint32_t>(header->data().frameId);
                std::memcpy(tex->data, header->data().imageDataP, header->data().imageLength);
                // Copy extra zeros (black pixels) to the bottom row if heights does not match
                if (tex->m_Height != header->data().height) {
                    uint32_t diff = tex->m_Height - header->data().height;
                    std::memset(tex->data + header->data().imageLength, 0x00, diff * tex->m_Width);
                }

                return true;
            default:
                Log::Logger::getInstance()->warning("This texture type is not supported {}",
                                                    static_cast<int> (tex->m_Type));
                break;
        }
        return false;
    }


    bool CRLPhysicalCamera::updateCameraInfo(VkRender::Device *dev, crl::multisense::RemoteHeadChannel channelID) {
        // The calling function will repeat this loop until everything succeeds.
        bool allSucceeded = true;
        Log::Logger::getInstance()->trace("Updating Camera info for channel {}", channelID);
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        auto iter = channelMap.find(channelID);
        if (iter == channelMap.end() || iter->second == nullptr)
            return false;
        auto channelPtr = iter->second->ptr();
        if (channelPtr == nullptr)
            return false;

        auto &info = infoMap[channelID];

        allSucceeded &= updateAndLog([&](auto &data) { return channelPtr->getImageConfig(data); }, info.imgConf,
                                     "imgConf");
        if (dev->interruptConnection) return false;
        allSucceeded &= updateAndLog([&](auto &data) { return channelPtr->getDeviceModes(data); },
                                     info.supportedDeviceModes, "supportedDeviceModes");
        if (dev->interruptConnection) return false;
        allSucceeded &= updateAndLog([&](auto &data) { return channelPtr->getEnabledStreams(data); },
                                     info.supportedSources, "supportedSources");
        if (dev->interruptConnection) return false;
        allSucceeded &= updateAndLog([&](auto &data) { return channelPtr->getNetworkConfig(data); }, info.netConfig,
                                     "netConfig");
        if (dev->interruptConnection) return false;
        allSucceeded &= updateAndLog([&](auto &data) { return channelPtr->getVersionInfo(data); }, info.versionInfo,
                                     "versionInfo");
        if (dev->interruptConnection) return false;
        allSucceeded &= updateAndLog([&](auto &data) { return channelPtr->getDeviceInfo(data); }, info.devInfo,
                                     "devInfo");
        if (dev->interruptConnection) return false;
        allSucceeded &= updateAndLog([&](auto &data) { return channelPtr->getMtu(data); }, info.sensorMTU, "sensorMTU");

        if (info.devInfo.lightingType != 0) {
            if (dev->interruptConnection) return false;
            allSucceeded &= updateAndLog([&](auto &data) { return channelPtr->getLightingConfig(data); }, info.lightConf,
                                         "lightConf");
        }
        if (dev->interruptConnection) return false;
        allSucceeded &= updateAndLog([&](auto &data) { return channelPtr->getImageCalibration(data); },
                                     info.calibration, "calibration");
        if (dev->interruptConnection) return false;

        // TODO I want to update most info on startup but this function varies. On KS21 this will always fail.
        //  This also assumes that getDeviceInfo above also succeeded
        if (infoMap[channelID].devInfo.hardwareRevision != crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_KS21)
        {
            allSucceeded &= updateAndLog([&](auto &data) { return channelPtr->getAuxImageConfig(data); },
                                         info.auxImgConf, "auxImageConfig");
        }

        updateQMatrix(channelID);

        if (!allSucceeded) {
            Log::Logger::getInstance()->error(
                    "Querying the camera for config did not succeed, viewer may not work correctly");
        } else {
            Log::Logger::getInstance()->info("Querying the camera for config did succeeded");
        }

        // Also check for IMU information. This is not present on most devices.
        if (dev->interruptConnection) return false;
            getIMUConfig(channelID);

        return allSucceeded;
    }


    void CRLPhysicalCamera::getIMUConfig(crl::multisense::RemoteHeadChannel channelID){
        crl::multisense::Status                   status;

        std::vector<crl::multisense::imu::Info>   sensorInfos;
        std::vector<crl::multisense::imu::Config> sensorConfigs;
        uint32_t                 sensorSamplesPerMessage    = 0;
        uint32_t                 sensorMaxSamplesPerMessage = 0;

        auto* channel = channelMap.find(channelID)->second->ptr();

        status = channel->getImuInfo(sensorMaxSamplesPerMessage,
                                      sensorInfos);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Failed to query imu info: {}", crl::multisense::Channel::statusString(status));
            return;
        }
        status = channel->getImuConfig(sensorSamplesPerMessage, sensorConfigs);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Failed to query imu config: {}", crl::multisense::Channel::statusString(status));
            return;
        }

        for (const auto& info : sensorInfos){
            Log::Logger::getInstance()->info("Got IMU info: {}, device name: {}, units: {}", info.name, info.device, info.units);
        }

        infoMap[channelID].imuSensorInfos = sensorInfos;
        infoMap[channelID].imuSensorConfigs = sensorConfigs;

        size_t maxVal = 0;
        for(const auto& conf : sensorInfos){
            if (conf.rates.size() > maxVal){
                maxVal = conf.rates.size();
            }
        }
        infoMap[channelID].imuMaxTableIndex = static_cast<uint32_t>(maxVal);
        infoMap[channelID].hasIMUSensor = true;
    }


    bool CRLPhysicalCamera::setGamma(float gamma, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error("Attempted to set gamma on a channel that was not connected, Channel {}",
                                              channelID);
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to query gamma configuration");
            return false;
        }
        infoMap[channelID].imgConf.setGamma(gamma);
        status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to set gamma configuration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Set Gamma on channel {}", channelID);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->info("Failed to verify Gamma");
            return false;
        }
        return true;
    }

    bool CRLPhysicalCamera::setFps(float fps, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error("Attempted to set fps on a channel that was not connected, Channel {}",
                                              channelID);
            return false;
        }

        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to query m_Image configuration");
            return false;
        }
        infoMap[channelID].imgConf.setFps(fps);

        status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to set m_Image configuration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Set framerate on channel {}", channelID);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));


        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->info("Failed to verify fps");
            return false;
        }

        return true;
    }

    bool CRLPhysicalCamera::setGain(float gain, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);

        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error("Attempted to set gain on a channel that was not connected, Channel {}",
                                              channelID);
            return false;
        }

        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to query image configuration");
            return false;
        }

        infoMap[channelID].imgConf.setGain(gain);
        status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);

        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to set image configuration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Set gain on channel {}", channelID);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));


        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->info("Failed to update Gain");
            return false;
        }
        return true;
    }


    bool
    CRLPhysicalCamera::setResolution(CRLCameraResolution resolution, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID] == nullptr) {
            Log::Logger::getInstance()->error(
                    "Attempted to set resolution on a channel that was not connected, Channel {}", channelID);
            return false;
        }

        if (resolution == CRL_RESOLUTION_NONE) {
            Log::Logger::getInstance()->info("Resolution is not specified {}", static_cast<int> (resolution));
            return false;
        }
        uint32_t width = 0, height = 0, depth = 0;
        Utils::cameraResolutionToValue(resolution, &width, &height, &depth);
        if (width == 0 || height == 0 || depth == 0) {
            Log::Logger::getInstance()->error("Resolution mode not supported");
            return false;
        }
        int ret = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (ret != crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->error("failed to get m_Image config");
        }
        if (Utils::valueToCameraResolution(infoMap[channelID].imgConf.width(), infoMap[channelID].imgConf.height(),
                                           infoMap[channelID].imgConf.disparities()) ==
            resolution) {
            currentResolutionMap[channelID] = resolution;
            Log::Logger::getInstance()->info("Resolution already set to {}x{}x{}", width, height, depth);
            return false;
        }
        infoMap[channelID].imgConf.setResolution(width, height);
        infoMap[channelID].imgConf.setDisparities(depth);

        ret = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
        if (ret == crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Set resolution to {}x{}x{} on channel {}", width, height, depth,
                                             channelID);
            currentResolutionMap[channelID] = resolution;
            updateQMatrix(channelID);

        } else {
            Log::Logger::getInstance()->warning("Failed setting resolution to {}x{}x{}. Error: {}", width, height,
                                                depth, ret);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));


        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->info("Failed to query getImageConfig");
            return false;
        }

        return true;
    }

    bool CRLPhysicalCamera::setExposureParams(ExposureParams p, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error(
                    "Attempted to set exposure on a channel that was not connected, Channel {}", channelID);
            return false;
        }

        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Unable to query exposure configuration");
            return false;
        }
        if (p.autoExposure) {
            infoMap[channelID].imgConf.setAutoExposure(p.autoExposure);
            infoMap[channelID].imgConf.setAutoExposureMax(p.autoExposureMax);
            infoMap[channelID].imgConf.setAutoExposureDecay(p.autoExposureDecay);
            infoMap[channelID].imgConf.setAutoExposureTargetIntensity(p.autoExposureTargetIntensity);
            infoMap[channelID].imgConf.setAutoExposureThresh(p.autoExposureThresh);
            infoMap[channelID].imgConf.setAutoExposureRoi(p.autoExposureRoiX, p.autoExposureRoiY,
                                                          p.autoExposureRoiWidth,
                                                          p.autoExposureRoiHeight);

        } else {
            infoMap[channelID].imgConf.setAutoExposure(p.autoExposure);
            infoMap[channelID].imgConf.setExposure(p.exposure);
        }

        status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Unable to set exposure configuration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Set exposure on channel {}", channelID);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->warning("Failed to verify Exposure params");
            return false;
        }
        return true;
    }


    bool CRLPhysicalCamera::setPostFilterStrength(float filter, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error(
                    "Attempted to set post filter strength on a channel that was not connected, Channel {}", channelID);
            return false;
        }

        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Unable to query m_Image configuration");
            return false;
        }
        infoMap[channelID].imgConf.setStereoPostFilterStrength(filter);
        status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Unable to set m_Image configuration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Successfully set stereo post filter strength to {} on channel {}", filter,
                                             channelID);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        // Since we can only have one stereo setting among several remote head update all the infoMaps.
        for (const auto &channel: channelMap) {
            // don't bother for the VPB
            if (channel.first == crl::multisense::Remote_Head_VPB || channel.second->ptr() == nullptr)
                continue;
            if (crl::multisense::Status_Ok != channel.second->ptr()->getImageConfig(infoMap[channel.first].imgConf)) {
                Log::Logger::getInstance()->warning("Failed to verify fps");
                return false;
            }
        }
        return true;
    }

    bool CRLPhysicalCamera::setHDR(bool hdr, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error("Attempted to set hdr on a channel that was not connected, Channel {}",
                                              channelID);
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Unable to query hdr m_Image configuration");
            return false;
        }

        infoMap[channelID].imgConf.setHdr(hdr);
        status = channelMap[channelID]->ptr()->setImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Unable to set hdr configuration");
            return false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf)) {
            Log::Logger::getInstance()->warning("Failed to verifiy HDR");
            return false;
        }
        return true;
    }

    bool CRLPhysicalCamera::setLighting(LightingParams param, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error(
                    "Attempted to set light configuration on a channel that was not connected, Channel {}", channelID);
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->getLightingConfig(
                infoMap[channelID].lightConf);

        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Unable to query m_Image configuration");
            return false;
        }

        if (param.selection != -1) {
            infoMap[channelID].lightConf.setDutyCycle(param.selection, param.dutyCycle);
        } else
            infoMap[channelID].lightConf.setDutyCycle(param.dutyCycle);

        infoMap[channelID].lightConf.setNumberOfPulses(
                static_cast<uint32_t> (param.numLightPulses *
                                       1000));  // convert from ms to us and from float to uin32_t
        infoMap[channelID].lightConf.setStartupTime(
                static_cast<uint32_t> ((param.startupTime * 1000))); // convert from ms to us and from float to uin32_t
        infoMap[channelID].lightConf.setFlash(param.flashing);
        status = channelMap[channelID]->ptr()->setLightingConfig(infoMap[channelID].lightConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Unable to set m_Image configuration");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getLightingConfig(infoMap[channelID].lightConf)) {
            Log::Logger::getInstance()->warning("Failed to update '{}' ", "Lightning");
            return false;
        }

        return true;
    }

    bool CRLPhysicalCamera::setAuxImageConfig(AUXConfig auxConfig, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error(
                    "Attempted to set exposure on a channel that was not connected, Channel {}", channelID);
            return false;
        }

        crl::multisense::Status status = channelMap[channelID]->ptr()->getAuxImageConfig(infoMap[channelID].auxImgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->error("Unable to query aux image configuration");
            return false;
        }
        auto &p = infoMap[channelID].auxImgConf;

        p.setAutoExposure(auxConfig.ep.autoExposure);
        p.setAutoExposureMax(auxConfig.ep.autoExposureMax);
        p.setAutoExposureDecay(auxConfig.ep.autoExposureDecay);
        p.setAutoExposureTargetIntensity(auxConfig.ep.autoExposureTargetIntensity);
        p.setAutoExposureThresh(auxConfig.ep.autoExposureThresh);
        p.setAutoExposureRoi(auxConfig.ep.autoExposureRoiX, auxConfig.ep.autoExposureRoiY,
                             auxConfig.ep.autoExposureRoiWidth,
                             auxConfig.ep.autoExposureRoiHeight);
        p.setExposure(auxConfig.ep.exposure);


        p.setWhiteBalance(auxConfig.whiteBalanceRed, auxConfig.whiteBalanceBlue);
        p.setAutoWhiteBalance(auxConfig.whiteBalanceAuto);
        p.setAutoWhiteBalanceDecay(auxConfig.whiteBalanceDecay);
        p.setAutoWhiteBalanceThresh(auxConfig.whiteBalanceThreshold);
        p.setGain(auxConfig.gain);
        p.setGamma(auxConfig.gamma);
        p.setSharpeningLimit(static_cast<uint8_t>(auxConfig.sharpeningLimit));
        p.setSharpeningPercentage(auxConfig.sharpeningPercentage);
        p.enableSharpening(auxConfig.sharpening);

        status = channelMap[channelID]->ptr()->setAuxImageConfig(infoMap[channelID].auxImgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->error("Unable to set aux image configuration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Set aux image conf on channel {}", channelID);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getAuxImageConfig(infoMap[channelID].auxImgConf)) {
            Log::Logger::getInstance()->error("Failed to verify aux img conf");
            return false;
        }
        return true;
    }

    bool CRLPhysicalCamera::setMtu(uint32_t mtu, crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error("Attempted to set mtu on a channel that was not connected, Channel {}",
                                              channelID);
            return false;
        }
        int status = channelMap[channelID]->ptr()->setMtu(static_cast<int32_t> (mtu));

        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        int32_t getMtu = -1;
        channelMap[channelID]->ptr()->getMtu(getMtu);
        updateAndLog([&](auto &data) { return channelMap[channelID]->ptr()->getMtu(data); }, infoMap[channelID].sensorMTU, "sensorMTU");

        Log::Logger::getInstance()->info("Current mtu on channel {}", getMtu);
        if (status != crl::multisense::Status_Ok) {
            Log::Logger::getInstance()->info("Failed to set MTU {}", mtu);
            return false;
        } else {
            Log::Logger::getInstance()->info("Set MTU to {}", mtu);
            return true;
        }
    }

    bool CRLPhysicalCamera::getStatus(crl::multisense::RemoteHeadChannel channelID,
                                      crl::multisense::system::StatusMessage *msg) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr)
            return false;

        crl::multisense::Status status = channelMap[channelID]->ptr()->getDeviceStatus(*msg);
        return status == crl::multisense::Status_Ok;
    }

    bool CRLPhysicalCamera::setSensorCalibration(const std::string &intrinsicsFile, const std::string &extrinsicsFile,
                                                 crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error(
                    "Attempted to set sensor calibration on a channel that was not connected, Channel {}", channelID);
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageCalibration(
                infoMap[channelID].calibration);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->info("Unable to query calibration");
            return false;
        }
        bool hasAuxCamera = infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_C6S2_S27 ||
                            infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S30 ||
                            infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_MONOCAM||
                            infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_KS21i;
        std::ifstream inFile, exFile;
        std::map<std::string, std::vector<float> > data;
        inFile.open(intrinsicsFile.c_str());
        if (!inFile) {
            Log::Logger::getInstance()->error("Failed to open intrinsics file: {}.", intrinsicsFile);
            return false;
        }
        parseYaml(inFile, data);
        inFile.close();
        if (data["M1"].size() != 3 * 3 ||
            (data["D1"].size() != 5 && data["D1"].size() != 8) ||
            data["M2"].size() != 3 * 3 ||
            (data["D2"].size() != 5 && data["D2"].size() != 8) ||
            (hasAuxCamera && data["M3"].size() != 3 * 3) ||
            (hasAuxCamera && data["D3"].size() != 5 && data["D3"].size() != 8)) {
            Log::Logger::getInstance()->error("Intrinsics file error: {}. File not complete", intrinsicsFile);
            return false;
        }
        exFile.open(extrinsicsFile.c_str());
        if (!exFile) {
            Log::Logger::getInstance()->error("Failed to open extrinsics file: {}.", extrinsicsFile);
            return false;
        }
        parseYaml(exFile, data);
        exFile.close();
        if (data["R1"].size() != 3 * 3 ||
            data["P1"].size() != 3 * 4 ||
            data["R2"].size() != 3 * 3 ||
            data["P2"].size() != 3 * 4 ||
            (hasAuxCamera && (data["R3"].size() != 3 * 3 || data["P3"].size() != 3 * 4))) {
            Log::Logger::getInstance()->error("Extrinsics file error: {}. File not complete", intrinsicsFile);
            return false;
        }

        crl::multisense::image::Calibration calibration{};

        memcpy(&calibration.left.M[0][0], &data["M1"].front(), data["M1"].size() * sizeof(float));
        memset(&calibration.left.D[0], 0, sizeof(calibration.left.D));
        memcpy(&calibration.left.D[0], &data["D1"].front(), data["D1"].size() * sizeof(float));
        memcpy(&calibration.left.R[0][0], &data["R1"].front(), data["R1"].size() * sizeof(float));
        memcpy(&calibration.left.P[0][0], &data["P1"].front(), data["P1"].size() * sizeof(float));

        memcpy(&calibration.right.M[0][0], &data["M2"].front(), data["M2"].size() * sizeof(float));
        memset(&calibration.right.D[0], 0, sizeof(calibration.right.D));
        memcpy(&calibration.right.D[0], &data["D2"].front(), data["D2"].size() * sizeof(float));
        memcpy(&calibration.right.R[0][0], &data["R2"].front(), data["R2"].size() * sizeof(float));
        memcpy(&calibration.right.P[0][0], &data["P2"].front(), data["P2"].size() * sizeof(float));

        if (hasAuxCamera) {
            memcpy(&calibration.aux.M[0][0], &data["M3"].front(), data["M3"].size() * sizeof(float));
            memset(&calibration.aux.D[0], 0, sizeof(calibration.aux.D));
            memcpy(&calibration.aux.D[0], &data["D3"].front(), data["D3"].size() * sizeof(float));
            memcpy(&calibration.aux.R[0][0], &data["R3"].front(), data["R3"].size() * sizeof(float));
            memcpy(&calibration.aux.P[0][0], &data["P3"].front(), data["P3"].size() * sizeof(float));
        }


        status = channelMap[channelID]->ptr()->setImageCalibration(calibration);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->error("Unable to set calibration");
            return false;
        } else {
            Log::Logger::getInstance()->info("Successfully set new calibration");

        }

        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageCalibration(infoMap[channelID].calibration)) {
            Log::Logger::getInstance()->info("Failed to query image calibration");
            return false;
        }
        return true;
    }


    std::ostream &
    CRLPhysicalCamera::writeImageIntrinics(std::ostream &stream, crl::multisense::image::Calibration const &calibration,
                                           bool hasAuxCamera) {
        stream << "%YAML:1.0\n";
        writeMatrix(stream, "M1", 3, 3, &calibration.left.M[0][0]);
        writeMatrix(stream, "D1", 1, 8, &calibration.left.D[0]);
        writeMatrix(stream, "M2", 3, 3, &calibration.right.M[0][0]);
        writeMatrix(stream, "D2", 1, 8, &calibration.right.D[0]);

        if (hasAuxCamera) {
            writeMatrix(stream, "M3", 3, 3, &calibration.aux.M[0][0]);
            writeMatrix(stream, "D3", 1, 8, &calibration.aux.D[0]);
        }
        return stream;
    }


    std::ostream &
    CRLPhysicalCamera::writeImageExtrinics(std::ostream &stream, crl::multisense::image::Calibration const &calibration,
                                           bool hasAuxCamera) {
        stream << "%YAML:1.0\n";
        writeMatrix(stream, "R1", 3, 3, &calibration.left.R[0][0]);
        writeMatrix(stream, "P1", 3, 4, &calibration.left.P[0][0]);
        writeMatrix(stream, "R2", 3, 3, &calibration.right.R[0][0]);
        writeMatrix(stream, "P2", 3, 4, &calibration.right.P[0][0]);

        if (hasAuxCamera) {
            writeMatrix(stream, "R3", 3, 3, &calibration.aux.R[0][0]);
            writeMatrix(stream, "P3", 3, 4, &calibration.aux.P[0][0]);
        }
        return stream;
    }


    bool CRLPhysicalCamera::saveSensorCalibration(const std::string &savePath,
                                                  crl::multisense::RemoteHeadChannel channelID) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);

        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error(
                    "Attempted to save calibration from a channel that was not connected, Channel {}", channelID);
            return false;
        }

        if (crl::multisense::Status_Ok !=
            channelMap[channelID]->ptr()->getImageCalibration(infoMap[channelID].calibration)) {
            Log::Logger::getInstance()->info("Failed to query image calibration");
            return false;
        }

        bool hasAuxCamera = infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_C6S2_S27 ||
                            infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S30 ||
                            infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_MONOCAM ||
                            infoMap[channelID].devInfo.hardwareRevision ==
                            crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_KS21i;
        std::ofstream inFile, exFile;
        std::string intrinsicsFile = savePath + "/" + infoMap[channelID].devInfo.serialNumber + "_intrinsics.yml";
        std::string extrinsicsFile = savePath + "/" + infoMap[channelID].devInfo.serialNumber + "_extrinsics.yml";

        inFile.open(intrinsicsFile.c_str(), std::ios_base::out | std::ios_base::trunc);

        if (!inFile) {
            Log::Logger::getInstance()->error("Failed to open Intrinsics file for writing!");
            return false;
        }

        exFile.open(extrinsicsFile.c_str(), std::ios_base::out | std::ios_base::trunc);

        if (!exFile) {
            Log::Logger::getInstance()->error("Failed to open Extrinsics file for writing!");
            return false;
        }

        Log::Logger::getInstance()->trace("Writing intrinsics to file");
        writeImageIntrinics(inFile, infoMap[channelID].calibration, hasAuxCamera);
        Log::Logger::getInstance()->trace("Writing extrinsics to file");
        writeImageExtrinics(exFile, infoMap[channelID].calibration, hasAuxCamera);
        inFile.flush();
        exFile.flush();

        Log::Logger::getInstance()->info("Wrote camera calibration to file. HasAux: {}", hasAuxCamera);
        return true;
    }

    bool CRLPhysicalCamera::getIMUData(crl::multisense::RemoteHeadChannel channelID,
                                       std::vector<CRLPhysicalCamera::ImuData> *gyro,
                                       std::vector<CRLPhysicalCamera::ImuData> *accel) const {
        auto it = channelMap.find(channelID);
        if (it == channelMap.end()) {
            return false;
        }
        if (gyro == nullptr || accel == nullptr) {
            Log::Logger::getInstance()->trace("gyro or accel vectors cannot be nullptrs");
            return false;
        }
        auto header = it->second->imuBuffer->getIMUBuffer(channelID);
        if (header == nullptr) {
            return false;
        }

        double time = 0, prevTime = 0;
        for (auto iterator = header->data().samples.begin();
             iterator != header->data().samples.end(); ++iterator) {
            const crl::multisense::imu::Sample &s = *iterator;

            switch (s.type) {
                case crl::multisense::imu::Sample::Type_Accelerometer:
                    accel->push_back({s.x, s.y, s.z, s.time(), 0});
                    break;
                case crl::multisense::imu::Sample::Type_Gyroscope: {
                    time = s.time();
                    double dt = time - prevTime;

                    if (prevTime == 0)
                        dt = 0;

                    gyro->push_back({s.x, s.y, s.z, s.time(), dt});
                    prevTime = time;
                }
                    break;
                case crl::multisense::imu::Sample::Type_Magnetometer:
                    break;
            }

        }
        return true;
    }

    bool
    CRLPhysicalCamera::calculateIMURotation(VkRender::IMUData *data,
                                            crl::multisense::RemoteHeadChannel channelID, double alpha) const {

        std::vector<ImuData> gyro;
        std::vector<ImuData> accel;

        if (!getIMUData(channelID, &gyro, &accel))
            return false;

        double rollAcc = 0, pitchAcc = 0;
        for (size_t i = 0; i < accel.size(); ++i) {
            auto &a = accel[i];
            auto &g = gyro[i];
            if (a.time != g.time)
                continue;
            rollAcc = static_cast<double>(std::atan2(a.y, a.z));
            pitchAcc = static_cast<double>(std::atan2(-a.x, std::sqrt(a.y * a.y + a.z * a.z)));

            data->pitch = alpha * (data->pitch + (g.dTime * (static_cast<double>(g.y) * M_PI / 180))) +
                          (1 - alpha) * pitchAcc;
            data->roll =
                    alpha * (data->roll + (g.dTime * (static_cast<double>(g.x) * M_PI / 180))) + (1 - alpha) * rollAcc;
        }
        return true;
    }

    void CRLPhysicalCamera::updateQMatrix(crl::multisense::RemoteHeadChannel channelID) {
        /// Also update correct Q matrix with the new width
        // From LibMultisenseUtility

        channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        crl::multisense::image::Config c = infoMap[channelID].imgConf;

        float scale = ((static_cast<float>(infoMap[channelID].devInfo.imagerWidth) /
                        static_cast<float>(infoMap[channelID].imgConf.width())));
        float dcx = (infoMap[channelID].calibration.right.P[0][2] - infoMap[channelID].calibration.left.P[0][2]) *
                    (1.0f / scale);
        const float &fx = c.fx();
        const float &fy = c.fy();
        const float &cx = c.cx();
        const float &cy = c.cy();
        const float &tx = c.tx();
        // glm::mat4 indexing
        // [column][row]
        // Inserted values col by col
        glm::mat4 Q(0.0f);

        Q[0][0] = fy * tx;
        Q[1][1] = fx * tx;
        Q[2][3] = -fy;
        Q[3][0] = -fy * cx * tx;
        Q[3][1] = -fx * cy * tx;
        Q[3][2] = fx * fy * tx;
        Q[3][3] = fy * dcx;

        // keep as is
        infoMap[channelID].QMat = Q;
        infoMap[channelID].focalLength = fx;
        infoMap[channelID].pointCloudScale = scale;
        /// Finished updating Q matrix

        auto aux = infoMap[channelID].calibration.aux;
        glm::mat4 K(1.0f);
        K[0][0] = fx;
        K[1][1] = fy;
        K[2][0] = cx;
        K[2][1] = cy;

        auto r00 = aux.R[0][0];
        auto r01 = aux.R[1][0];
        auto r02 = aux.R[2][0];
        auto r10 = aux.R[0][1];
        auto r11 = aux.R[1][1];
        auto r12 = aux.R[2][1];
        auto r20 = aux.R[0][2];
        auto r21 = aux.R[1][2];
        auto r22 = aux.R[2][2];

        glm::mat4 T(1.0f);
        T[0][0] = r00;
        T[1][0] = r01;
        T[2][0] = r02;
        T[0][1] = r10;
        T[1][1] = r11;
        T[2][1] = r12;
        T[0][2] = r20;
        T[1][2] = r21;
        T[2][2] = r22;
        T[3][0] = infoMap[channelID].calibration.aux.P[0][3] / infoMap[channelID].calibration.aux.P[0][0]; // tx in mm
        T[3][1] = infoMap[channelID].calibration.aux.P[1][3] / infoMap[channelID].calibration.aux.P[0][0]; // ty in mm
        infoMap[channelID].KColorMatExtrinsic = T;
        infoMap[channelID].KColorMat = K;
    }

    bool CRLPhysicalCamera::getExposure(short channelID, bool hasAuxCamera) {
        std::scoped_lock<std::mutex> lock(setCameraDataMutex);
        if (channelMap[channelID]->ptr() == nullptr) {
            Log::Logger::getInstance()->error(
                    "Attempted to get exposure on a channel that was not connected, Channel {}", channelID);
            return false;
        }
        crl::multisense::Status status = channelMap[channelID]->ptr()->getImageConfig(infoMap[channelID].imgConf);
        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->error("Unable to query exposure configuration");
            return false;
        }

        if (hasAuxCamera) {
            status = channelMap[channelID]->ptr()->getAuxImageConfig(infoMap[channelID].auxImgConf);
            if (crl::multisense::Status_Ok != status) {
                std::string errString = crl::multisense::Channel::statusString(status);
                Log::Logger::getInstance()->error("Unable to query getAuxImageConfig in getExposure. Error: {}",
                                                  errString);

                if (status == crl::multisense::Status_Unsupported) {
                    Log::Logger::getInstance()->error(
                            "Viewer thinks this camera has aux camera but LibMultiSense says it is unsupported");
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    void CRLPhysicalCamera::setIMUConfig(uint32_t tableIndex, crl::multisense::RemoteHeadChannel channelID) {
        if (!infoMap[channelID].hasIMUSensor){
            Log::Logger::getInstance()->trace("No IMU Sensor present on device");
            return;
        }
        if (tableIndex > infoMap[channelID].imuMaxTableIndex){
            Log::Logger::getInstance()->warning("Table index is higher than allowed {}. max allowed: {}", tableIndex, infoMap[channelID].imuMaxTableIndex);
            return;
        }

        auto conf = infoMap[channelID].imuSensorConfigs;
        for( auto& config : conf){
            config.rateTableIndex = tableIndex;
        }

        // Set config
        auto channelP = channelMap[channelID]->ptr();

        crl::multisense::Status status = channelP->setImuConfig(false, 0, conf);

        if (crl::multisense::Status_Ok != status) {
            Log::Logger::getInstance()->warning("Failed to set imu config: {}", crl::multisense::Channel::statusString(status));
            return;
        } else {
            Log::Logger::getInstance()->info("Set imu config table index {}", tableIndex);

        }
    }

}
