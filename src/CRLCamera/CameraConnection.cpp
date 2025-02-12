/**
 * @file: MultiSense-Viewer/src/CRLCamera/CameraConnection.cpp
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
 *   2022-3-21, mgjerde@carnegierobotics.com, Created file.
 **/

#ifdef WIN32

#else

#include <net/if.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <unistd.h>

#endif

#include "Viewer/CRLCamera/CameraConnection.h"
#include "Viewer/Tools/Utils.h"

#define MAX_TASK_STACK_SIZE 4
#define MAX_NUM_REMOTEHEADS 4

namespace VkRender::MultiSense {
    void CameraConnection::updateActiveDevice(VkRender::Device *dev) {
        activeDeviceCameraStreams(dev);

        auto *p = &dev->parameters;
        // Populate dev-parameters with the currently set settings on the camera

        if (dev->parameters.updateGuiParams) {
            Log::Logger::getInstance()->info("Updating GUI with MultiSense parameters");
            const auto &conf = camPtr.getCameraInfo(dev->configRemoteHead).imgConf;
            p->stereo.ep.exposure = conf.exposure();
            p->stereo.ep.autoExposure = conf.autoExposure();
            p->stereo.ep.autoExposureThresh = conf.autoExposureThresh();
            p->stereo.ep.autoExposureDecay = conf.autoExposureDecay();
            p->stereo.ep.autoExposureMax = conf.autoExposureMax();
            p->stereo.ep.autoExposureTargetIntensity = conf.autoExposureTargetIntensity();
            p->stereo.ep.autoExposureRoiHeight = conf.autoExposureRoiHeight();
            p->stereo.ep.autoExposureRoiWidth = conf.autoExposureRoiWidth();
            p->stereo.ep.autoExposureRoiX = conf.autoExposureRoiX();
            p->stereo.ep.autoExposureRoiY = conf.autoExposureRoiY();
            p->stereo.gain = conf.gain();
            p->stereo.fps = conf.fps();
            p->stereo.gamma = conf.gamma();
            // Aux config
            const auto &auxConf = camPtr.getCameraInfo(dev->configRemoteHead).auxImgConf;
            p->aux.whiteBalanceAuto = auxConf.autoWhiteBalance();
            p->aux.whiteBalanceDecay = auxConf.autoWhiteBalanceDecay();
            p->aux.whiteBalanceThreshold = auxConf.autoWhiteBalanceThresh();
            p->aux.whiteBalanceBlue = auxConf.whiteBalanceBlue();
            p->aux.whiteBalanceRed = auxConf.whiteBalanceRed();
            p->aux.ep.exposure = auxConf.exposure();
            p->aux.ep.autoExposure = auxConf.autoExposure();
            p->aux.ep.autoExposureThresh = auxConf.autoExposureThresh();
            p->aux.ep.autoExposureDecay = auxConf.autoExposureDecay();
            p->aux.ep.autoExposureMax = auxConf.autoExposureMax();
            p->aux.ep.autoExposureTargetIntensity = auxConf.autoExposureTargetIntensity();
            p->aux.ep.autoExposureRoiHeight = auxConf.autoExposureRoiHeight();
            p->aux.ep.autoExposureRoiWidth = auxConf.autoExposureRoiWidth();
            p->aux.ep.autoExposureRoiX = auxConf.autoExposureRoiX();
            p->aux.ep.autoExposureRoiY = auxConf.autoExposureRoiY();
            p->aux.gain = auxConf.gain();
            p->aux.gamma = auxConf.gamma();
            p->aux.sharpening = auxConf.enableSharpening();
            p->aux.sharpeningPercentage = auxConf.sharpeningPercentage();
            p->aux.sharpeningLimit = auxConf.sharpeningLimit();

            const auto &lightConf = camPtr.getCameraInfo(dev->configRemoteHead).lightConf;
            p->light.numLightPulses = static_cast<float>(lightConf.getNumberOfPulses()) / 1000.0f;
            p->light.dutyCycle = lightConf.getDutyCycle(0);
            p->light.flashing = lightConf.getFlash();
            p->light.startupTime = static_cast<float>(lightConf.getStartupTime()) / 1000.0f;
            p->stereo.stereoPostFilterStrength = conf.stereoPostFilterStrength();
            p->stereo.mtu = camPtr.getCameraInfo(dev->configRemoteHead).sensorMTU;

            dev->parameters.updateGuiParams = false;
        }

        if (dev->parameters.stereo.update && pool->getTaskListSize() < MAX_TASK_STACK_SIZE) {
            Log::Logger::getInstance()->trace("Pushing {} to threadpool", "setAdditionalParametersTask");
            pool->Push(CameraConnection::setAdditionalParametersTask, this, p->stereo.fps, p->stereo.gain,
                       p->stereo.gamma,
                       p->stereo.stereoPostFilterStrength, p->stereo.hdrEnabled, p->stereo.mtu, dev,
                       dev->configRemoteHead);
        }

        if (dev->parameters.stereo.ep.update && pool->getTaskListSize() < MAX_TASK_STACK_SIZE) {
            Log::Logger::getInstance()->trace("Pushing {} to threadpool", "setExposureTask");
            pool->Push(CameraConnection::setExposureTask, this, &p->stereo.ep, dev, dev->configRemoteHead);
        }


        if (dev->parameters.aux.update && pool->getTaskListSize() < MAX_TASK_STACK_SIZE) {
            Log::Logger::getInstance()->trace("Pushing {} to threadpool", "setAuxConfigTask");
            pool->Push(CameraConnection::setAuxConfigTask, this, &p->aux, dev, dev->configRemoteHead);
        }


        if (dev->parameters.light.update && pool->getTaskListSize() < MAX_TASK_STACK_SIZE) {
            Log::Logger::getInstance()->trace("Pushing {} to threadpool", "setLightingTask");
            pool->Push(CameraConnection::setLightingTask, this, &p->light, dev, dev->configRemoteHead);
        }

        if (dev->parameters.calib.update && pool->getTaskListSize() < MAX_TASK_STACK_SIZE) {
            Log::Logger::getInstance()->trace("Pushing {} to threadpool", "setCalibrationTask");
            pool->Push(CameraConnection::setCalibrationTask, this, dev->parameters.calib.intrinsicsFilePath,
                       dev->parameters.calib.extrinsicsFilePath, dev->configRemoteHead,
                       &dev->parameters.calib.updateFailed);
            dev->parameters.calib.update = false;
        }


        if (dev->parameters.calib.save && pool->getTaskListSize() < MAX_TASK_STACK_SIZE) {
            Log::Logger::getInstance()->trace("Pushing {} to threadpool", "getCalibrationTask");
            pool->Push(CameraConnection::getCalibrationTask, this, dev->parameters.calib.saveCalibrationPath,
                       dev->configRemoteHead, &dev->parameters.calib.saveFailed);
            dev->parameters.calib.save = false;
        }
        // Set the correct resolution. Will only update if changed.

        for (auto &ch: dev->channelInfo) {
            if (ch.state == CRL_STATE_ACTIVE && ch.updateResolutionMode &&
                pool->getTaskListSize() < MAX_TASK_STACK_SIZE) {
                uint32_t width = 0, height = 0, depth = 0;
                Utils::cameraResolutionToValue(ch.selectedResolutionMode, &width, &height, &depth);
                Log::Logger::getInstance()->info("Requesting resolution {}x{}x{} on channel {}", width, height, depth,
                                                 ch.index);
                Log::Logger::getInstance()->trace("Pushing {} to threadpool", "setResolutionTask");
                pool->Push(CameraConnection::setResolutionTask, this, ch.selectedResolutionMode, dev, ch.index);
                ch.updateResolutionMode = false;
            }
        }

        // Stop if the stream is no longer requested
        for (auto &ch: dev->channelInfo) {
            if (ch.state != CRL_STATE_ACTIVE && pool->getTaskListSize() < MAX_TASK_STACK_SIZE)
                continue;
            for (const auto &enabled: ch.enabledStreams) {
                if (!Utils::isInVector(ch.requestedStreams, enabled) && !enabled.empty()) {
                    Log::Logger::getInstance()->trace("Pushing {} to threadpool", "stopStreamTask");
                    pool->Push(CameraConnection::stopStreamTask, this, enabled, ch.index);
                    Utils::removeFromVector(&ch.enabledStreams, enabled);
                }
            }
        }


        // Start requested streams
        for (auto &ch: dev->channelInfo) {
            if (ch.state != CRL_STATE_ACTIVE && pool->getTaskListSize() < MAX_TASK_STACK_SIZE)
                continue;
            for (const auto &requested: ch.requestedStreams) {
                if (!Utils::isInVector(ch.enabledStreams, requested) && requested != "Idle") {
                    Log::Logger::getInstance()->trace("Pushing {} to threadpool", "startStreamTask");
                    pool->Push(CameraConnection::startStreamTask, this, requested, ch.index);
                    ch.enabledStreams.emplace_back(requested);
                }
            }
        }

        // Query for status
        queryDevice(CameraConnection::getStatusTask, dev, &queryStatusTimer, INTERVAL_1_SECOND);

        // Query for camera configs
        if (!dev->updateDeviceConfigsSucceeded)
            queryDevice(CameraConnection::getCameraConfigsTask, dev, &queryDeviceConfigTimer, INTERVAL_5_SECONDS);


        // Read exposure setting around once a second if auto exposure is enabled Also put it into GUI structure
        auto time = std::chrono::steady_clock::now();
        auto elapsedTime =
                std::chrono::duration_cast<std::chrono::duration<float> >(time - queryExposureTimer);
        if (dev->parameters.stereo.ep.autoExposure && elapsedTime.count() > INTERVAL_1_SECOND &&
            pool->getTaskListSize() < MAX_TASK_STACK_SIZE) {
            queryExposureTimer = std::chrono::steady_clock::now();
            // We also want to make sure that we can query all the device info. Otherwise pinging the camera for updates especially auxImageConfig may not results in any usefull information
            Log::Logger::getInstance()->trace("Pushing {} to threadpool", "getExposureTask");
            pool->Push(CameraConnection::getExposureTask, this, dev, dev->configRemoteHead);
        }

        Log::Logger::getLogMetrics()->device.dev = dev;
    }


    // Existing queryDevice function
    void CameraConnection::queryDevice(
        std::function<void(void *, crl::multisense::RemoteHeadChannel, VkRender::Device *)> taskFunction,
        VkRender::Device *dev,
        std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<float> > *queryTimer,
        float updateFreqSec) {
        for (auto &ch: dev->channelInfo) {
            if (ch.state != CRL_STATE_ACTIVE)
                continue;

            auto time = std::chrono::steady_clock::now();
            auto timeSpan =
                    std::chrono::duration_cast<std::chrono::duration<float> >(time - *queryTimer);

            if (pool->getTaskListSize() < MAX_TASK_STACK_SIZE && timeSpan.count() > updateFreqSec) {
                *queryTimer = std::chrono::steady_clock::now();
                Log::Logger::getInstance()->trace("Pushing {} to threadpool",
                                                  (updateFreqSec == INTERVAL_1_SECOND
                                                       ? "getStatusTask"
                                                       : "getCameraConfigsTask"));
                pool->Push(taskFunction, this,
                           dev->isRemoteHead
                               ? crl::multisense::Remote_Head_VPB
                               : static_cast<crl::multisense::RemoteHeadChannel>(0), dev);
            }
        }
    }

    void CameraConnection::activeDeviceCameraStreams(VkRender::Device *dev) {
        // If we are in 3D mode automatically enable disparity and color streams.
        if (dev->selectedPreviewTab == CRL_TAB_3D_POINT_CLOUD) {
            if (dev->enableIMU) {
                if (!Utils::isInVector(dev->channelInfo[0].requestedStreams, "IMU"))
                    dev->channelInfo[0].requestedStreams.emplace_back("IMU");
            } else {
                Utils::removeFromVector(&dev->channelInfo[0].requestedStreams, "IMU");
            }

            auto &chInfo = dev->channelInfo.front();
            if (!Utils::isInVector(chInfo.requestedStreams, "Disparity Left")) {
                chInfo.requestedStreams.emplace_back("Disparity Left");
                Log::Logger::getInstance()->info(("Adding Disparity Left source to user requested sources"));
            }


            if (dev->useAuxForPointCloudColor) {
                auto streams = {"Color Rectified Aux", "Luma Rectified Aux"};
                for (const auto &s: streams) {
                    if (!Utils::isInVector(chInfo.requestedStreams, s)) {
                        chInfo.requestedStreams.emplace_back(s);
                        Log::Logger::getInstance()->info(("Adding {} source to user requested sources"), s);
                    }
                }
                // Stop Luma rectified left if not used in the 2D view
                bool inUse = false;
                for (const auto &win: dev->win) {
                    if (win.second.selectedSource == "Luma Rectified Left")
                        inUse = true;
                }
                if (!inUse) {
                    Utils::removeFromVector(&chInfo.requestedStreams, "Luma Rectified Left");
                }
            } else {
                if (!Utils::isInVector(chInfo.requestedStreams, "Luma Rectified Left")) {
                    chInfo.requestedStreams.emplace_back("Luma Rectified Left");
                    Log::Logger::getInstance()->info(("Adding {} source to user requested sources"),
                                                     "Luma Rectified Left");
                }
                bool auxLumaActive = false;
                bool auxColorActive = false;
                std::string colorRectified = "Color Rectified Aux";
                std::string auxLumaRectified = "Luma Rectified Aux";
                // Check if we have any color or luma
                for (const auto &win: dev->win) {
                    if (colorRectified == win.second.selectedSource) {
                        auxColorActive = true;
                    }
                    if (auxLumaRectified == win.second.selectedSource) {
                        auxLumaActive = true;
                    }
                }
                auto &cInfo = dev->channelInfo.front();

                if (!auxColorActive && !auxLumaActive) {
                    Utils::removeFromVector(&cInfo.requestedStreams, colorRectified);
                    Utils::removeFromVector(&cInfo.requestedStreams, auxLumaRectified);
                } else if (auxLumaActive && !auxColorActive) {
                    Utils::removeFromVector(&cInfo.requestedStreams, colorRectified);
                }
            }
        } else {
            // Remember to stop streams if we exit 3D view and our color streams are not in use
            bool auxLumaActive = false;
            bool auxColorActive = false;
            std::string colorRectified = "Color Rectified Aux";
            std::string auxLumaRectified = "Luma Rectified Aux";
            // Check if we have any color or luma
            for (const auto &win: dev->win) {
                if (colorRectified == win.second.selectedSource) {
                    auxColorActive = true;
                }
                if (auxLumaRectified == win.second.selectedSource) {
                    auxLumaActive = true;
                }
            }
            auto &chInfo = dev->channelInfo.front();

            if (!auxColorActive && !auxLumaActive) {
                Utils::removeFromVector(&chInfo.requestedStreams, colorRectified);
                Utils::removeFromVector(&chInfo.requestedStreams, auxLumaRectified);
            } else if (auxLumaActive && !auxColorActive) {
                Utils::removeFromVector(&chInfo.requestedStreams, colorRectified);
            }

            Utils::removeFromVector(&dev->channelInfo[0].requestedStreams, "IMU");
        }
    }

    float CameraConnection::findPercentile(uint16_t *image, size_t len, float percentile) {
        std::sort(image, image + len);
        size_t index = static_cast<size_t>(percentile * (len - 1) / 100);
        return image[index];
    }

    std::pair<float, float> CameraConnection::findUpperAndLowerDisparityBounds(void *ctx, VkRender::Device *dev) {
        auto *app = reinterpret_cast<CameraConnection *>(ctx);
        float minVal = 0;
        float maxVal = 255;
        if (!dev->channelInfo.empty()) {
            auto tex = VkRender::TextureData(CRL_DISPARITY_IMAGE, dev->channelInfo.front().selectedResolutionMode,
                                             true);
            // If we get an image attempt to update the GPU buffer
            if (app->camPtr.getCameraStream("Disparity Left", &tex, 0)) {
                // Find percentile should not run on main render thread as it slows down the application considerably.
                // Maybe just update the percentiles around once a second or so in a threaded operation
                minVal =
                        app->findPercentile(reinterpret_cast<uint16_t *>(tex.data),
                                            static_cast<size_t>(tex.m_Len / 2.0f), 10.0f) /
                        (16.0f * 255.0f);
                maxVal =
                        app->findPercentile(reinterpret_cast<uint16_t *>(tex.data),
                                            static_cast<size_t>(tex.m_Len / 2.0f), 90.0f) /
                        (16.0f * 255.0f);
            }
        }
        return {minVal, maxVal};
    }

    void CameraConnection::update(VkRender::Device &dev) {
        size_t numBins = 256;
        std::vector<size_t> histogram(numBins, 0);
        bool shouldNormalize = false;
        for (auto &window: dev.win) {
            if (window.second.effects.normalize) {
                shouldNormalize = true;
                break;
            }
        }

        // Normalize disparity values if it is running
        if (shouldNormalize && dev.selectedPreviewTab == CRL_TAB_2D_PREVIEW &&
            Utils::isStreamRunning(dev, "Disparity Left")) {
            auto time = std::chrono::steady_clock::now();
            auto timeSpan =
                    std::chrono::duration_cast<std::chrono::duration<float> >(time - calcDisparityNormValuesTimer);

            // Check if previous future is finished
            if (disparityNormFuture.valid() &&
                disparityNormFuture.wait_for(std::chrono::duration<float>(0)) == std::future_status::ready) {
                auto result = disparityNormFuture.get();
                Log::Logger::getInstance()->traceWithFrequency("normalize future result", 50,
                                                               "Calculated new norm {} to {}", result.first,
                                                               result.second);
                for (auto &window: dev.win) {
                    window.second.effects.data.minDisparityValue = result.first;
                    window.second.effects.data.maxDisparityValue = result.second;
                }
            }
            // Only create new future if updateIntervalSeconds second has passed or we're currently not running our previous future
            float updateIntervalSeconds = 0.1f;
            if (timeSpan.count() > updateIntervalSeconds &&
                (!disparityNormFuture.valid() ||
                 disparityNormFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) &&
                pool->getTaskListSize() < MAX_TASK_STACK_SIZE) {
                Log::Logger::getInstance()->traceWithFrequency("new normalize future", 50,
                                                               "Pushing new disparity normalizer calculation to pool");
                disparityNormFuture = pool->Push(CameraConnection::findUpperAndLowerDisparityBounds, this, &dev);
                calcDisparityNormValuesTimer = std::chrono::steady_clock::now();
            }
        }
    }

    void
    CameraConnection::onUIUpdate(std::vector<VkRender::Device> &devices, bool shouldConfigNetwork) {
        if (devices.empty())
            return;

        for (auto &dev: devices) {
            if (dev.state == CRL_STATE_INTERRUPT_CONNECTION) {
                Log::Logger::getInstance()->info("Profile {} set to CRL_STATE_INTERRUPT_CONNECTION", dev.name);
                dev.record.frame = false;
                dev.interruptConnection = true;
                Log::Logger::getInstance()->trace("Pushing {} to threadpool", "pushStopTask");
                pool->signalStop();
                saveProfileAndDisconnect(&dev);
                return;
            }
            // If we have requested (previous frame) to reset a connection or want to remove it from saved profiles
            if (dev.state == CRL_STATE_RESET || dev.state == CRL_STATE_DISCONNECT_AND_FORGET ||
                dev.state == CRL_STATE_LOST_CONNECTION) {
                Log::Logger::getInstance()->info(
                    "Profile {} set to CRL_STATE_RESET | CRL_STATE_LOST_CONNECTION | CRL_STATE_DISCONNECT_AND_FORGET",
                    dev.name);
                dev.selectedPreviewTab = CRL_TAB_2D_PREVIEW; // Note: Weird place to reset a UI element
                dev.record.frame = false;
                saveProfileAndDisconnect(&dev);
                pool->signalStop();
                return;
            }
            // Connect if we click a m_Device in the sidebar or if it is just added by add m_Device btn
            if ((dev.clicked && dev.state != CRL_STATE_ACTIVE) || dev.state == CRL_STATE_JUST_ADDED ||
                dev.state == CRL_STATE_JUST_ADDED_WINDOWS) {
                // reset other active m_Device if present. So loop over all devices again.
                Log::Logger::getInstance()->info("Profile {} set to CRL_STATE_ACTIVE | CRL_STATE_JUST_ADDED",
                                                 dev.name);

                VkRender::Device *otherDev = nullptr;
                for (auto &d: devices) {
                    if (d.state == CRL_STATE_ACTIVE && d.name != dev.name) {
                        d.state = CRL_STATE_RESET;
                        Log::Logger::getInstance()->info("Set dev state to RESET {}", d.name);
                        otherDev = &d;
                    }
                }
                if (otherDev) {
                    for (auto ch: otherDev->channelConnections)
                        camPtr.stop("All", ch); // Blocking operation

                    saveProfileAndDisconnect(otherDev);
                }


                dev.state = CRL_STATE_CONNECTING;
                Log::Logger::getInstance()->info("Set dev {}'s state to CRL_STATE_CONNECTING ", dev.name);

                // Re-create thread pool for a new connection in case we have old tasks from another connection in queue
                auto startTime = std::chrono::steady_clock::now();
                pool = std::make_unique<VkRender::ThreadPool>(3);
                std::chrono::duration<float> timeSpan =
                        std::chrono::duration_cast<std::chrono::duration<float> >(
                            std::chrono::steady_clock::now() - startTime);
                Log::Logger::getInstance()->trace("Creating cameraconnection threadpool took {}ms",
                                                  timeSpan.count() * 1000);
                // Perform connection by pushing a connect task.
                Log::Logger::getInstance()->trace("Pushing {} to threadpool", "connectCRLCameraTask");
                if (dev.simulatedDevice) {
                    pool->Push(CameraConnection::connectFakeCameraTask, this, &dev);
                } else {
                    bool delayConnection = (dev.state == CRL_STATE_JUST_ADDED_WINDOWS);
                    pool->Push(CameraConnection::connectCRLCameraTask, this, &dev, dev.isRemoteHead,
                               shouldConfigNetwork, delayConnection);
                }
                break;
            }

            // Rest of this function is only needed for active devices
            if (dev.state != CRL_STATE_ACTIVE) {
                continue;
            }

            if (!dev.simulatedDevice)
                updateActiveDevice(&dev);

            // Disable if we click a m_Device already connected
            if (dev.clicked && dev.state == CRL_STATE_ACTIVE) {
                // Disable all streams and delete camPtr on next update
                Log::Logger::getInstance()->info("Set profile {}'s state to CRL_STATE_RESET ", dev.name);
                dev.state = CRL_STATE_RESET;
                for (auto ch: dev.channelConnections) {
                    Log::Logger::getInstance()->trace("Pushing {} to threadpool", "stopStreamTask");
                    pool->Push(stopStreamTask, this, "All", ch);
                }
            }

            // Disable if we lost connection
            std::scoped_lock lock(statusCountMutex);
            if (m_FailedGetStatusCount >= MAX_FAILED_STATUS_ATTEMPTS &&
                (!Log::Logger::getLogMetrics()->device.ignoreMissingStatusUpdate)) {
                // Disable all streams and delete camPtr on next update
                Log::Logger::getInstance()->info("Call to reset state requested for profile {}. Lost connection..",
                                                 dev.name);
                dev.state = CRL_STATE_LOST_CONNECTION;
                Log::Logger::getInstance()->info("Set dev {}'s state to CRL_STATE_LOST_CONNECTION ", dev.name);
            }
        }
    }

    void CameraConnection::updateUIDataBlock(VkRender::Device &dev, CRLPhysicalCamera &camPtr) {
        dev.channelInfo.resize(1); // max number of remote heads
        dev.win.clear();
        for (crl::multisense::RemoteHeadChannel ch: dev.channelConnections) {
            VkRender::ChannelInfo chInfo;
            chInfo.availableSources.clear();
            chInfo.modes.clear();
            chInfo.availableSources.emplace_back("Idle");
            chInfo.index = ch;
            chInfo.state = CRL_STATE_ACTIVE;
            filterAvailableSources(&chInfo.availableSources, maskArrayAll, ch, camPtr);

            const auto &supportedModes = camPtr.getCameraInfo(ch).supportedDeviceModes;
            initCameraModes(&chInfo.modes, supportedModes);
            const auto &imgConf = camPtr.getCameraInfo(ch).imgConf;
            chInfo.selectedResolutionMode = Utils::valueToCameraResolution(imgConf.width(), imgConf.height(),
                                                                           imgConf.disparities());


            for (int i = 0; i < CRL_PREVIEW_TOTAL_MODES; ++i) {
                dev.win[static_cast<StreamWindowIndex>(i)].availableRemoteHeads.push_back(std::to_string(ch + 1));
                if (!chInfo.availableSources.empty())
                    dev.win[static_cast<StreamWindowIndex>(i)].selectedRemoteHeadIndex = ch;
            }

            // stop streams if there were any enabled, just so we can start with a clean slate
            //stopStreamTask(this, "All", ch);

            dev.channelInfo.at(ch) = chInfo;
        }


        // Update Debug Window
        auto &info = Log::Logger::getLogMetrics()->device.info;
        const auto &cInfo = camPtr.getCameraInfo(0).versionInfo;
        const auto &dInfo = camPtr.getCameraInfo(0).devInfo;

        info.firmwareBuildDate = cInfo.sensorFirmwareBuildDate;
        info.firmwareVersion = cInfo.sensorFirmwareVersion;
        info.serialNumber = dInfo.serialNumber;
        info.apiBuildDate = cInfo.apiBuildDate;
        info.apiVersion = cInfo.apiVersion;
        info.hardwareMagic = cInfo.sensorHardwareMagic;
        info.hardwareVersion = cInfo.sensorHardwareVersion;
        info.sensorFpgaDna = cInfo.sensorFpgaDna;
    }

    void CameraConnection::getProfileFromIni(VkRender::Device &dev) const {
        for (auto ch: dev.channelConnections) {
            CSimpleIniA ini;
            ini.SetUnicode();
            auto filePath = (Utils::getSystemCachePath() / "crl.ini");

            SI_Error rc = ini.LoadFile(filePath.c_str());
            if (rc < 0) {
            } // Handle error
            else {
                // A serial number is the section identifier of a profile. I can have three following states
                // - Not Exist
                // - Exist
                std::string cameraSerialNumber = camPtr.getCameraInfo(ch).devInfo.serialNumber;
                if (ini.SectionExists(cameraSerialNumber.c_str())) {
                    std::string profileName = ini.GetValue(cameraSerialNumber.c_str(), "ProfileName", "default");
                    std::string mode = ini.GetValue(cameraSerialNumber.c_str(),
                                                    ("Mode" + std::to_string(ch)).c_str(), "1");
                    std::string layout = ini.GetValue(cameraSerialNumber.c_str(), "Layout", "1");
                    Log::Logger::getInstance()->info("Using Layout {} and camera resolution {}", layout, mode);
                    dev.layout = static_cast<PreviewLayout>(std::stoi(layout));
                    dev.channelInfo.at(ch).selectedResolutionMode = static_cast<CRLCameraResolution>(std::stoi(mode));
                    dev.channelInfo.at(ch).selectedModeIndex = std::stoi(mode);
                    // Create previews
                    for (int i = 0; i <= CRL_PREVIEW_FOUR; ++i) {
                        std::string key = "Preview" + std::to_string(i + 1);
                        std::string source = std::string(ini.GetValue(cameraSerialNumber.c_str(), key.c_str(), ""));
                        if (!source.empty()) {
                            dev.win[static_cast<StreamWindowIndex>(i)].selectedSource = source.substr(0,
                                source.find_last_of(
                                    ':'));
                            dev.win[static_cast<StreamWindowIndex>(i)].selectedRemoteHeadIndex = static_cast<
                                crl::multisense::RemoteHeadChannel>(0);
                            Log::Logger::getInstance()->info(
                                ".ini file: found source '{}' for preview {} at head {}, Adding to requested source",
                                source.substr(0, source.find_last_of(':')),
                                i + 1, ch);
                            std::string requestSource = dev.win[static_cast<StreamWindowIndex>(i)].selectedSource;
                            if (requestSource == "Color Rectified Aux") {
                                dev.channelInfo.at(ch).requestedStreams.emplace_back("Luma Rectified Aux");
                            } else if (requestSource == "Color Aux") {
                                dev.channelInfo.at(ch).requestedStreams.emplace_back("Luma Aux");
                            }
                            dev.channelInfo.at(ch).requestedStreams.emplace_back(requestSource);
                        }
                    }
                }
            }
        }
    }

    void CameraConnection::initCameraModes(std::vector<std::string> *modes,
                                           const std::vector<crl::multisense::system::DeviceMode> &deviceModes) {
        for (const auto &mode: deviceModes) {
            std::string modeName = std::to_string(mode.width) + " x " + std::to_string(mode.height) + " x " +
                                   std::to_string(mode.disparities) + "x";
            modes->emplace_back(modeName);
            Log::Logger::getInstance()->info("Found supported mode: {}", modeName);
        }
        if (modes->empty())
            Log::Logger::getInstance()->warning("No supported modes were found. This may be an error");
    }

    void
    CameraConnection::filterAvailableSources(std::vector<std::string> *sources,
                                             const std::vector<uint32_t> &maskVec,
                                             crl::multisense::RemoteHeadChannel idx,
                                             CRLPhysicalCamera &camPtr) {
        crl::multisense::DataSource bits = camPtr.getCameraInfo(idx).supportedSources;
        for (auto mask: maskVec) {
            bool enabled = (bits & mask);
            if (enabled) {
                sources->emplace_back(Utils::dataSourceToString(mask));
                Log::Logger::getInstance()->info("Found supported source: {}", Utils::dataSourceToString(mask));
            }
        }
        if (sources->empty())
            Log::Logger::getInstance()->warning("No supported sources were found. This may be an error");
    }


    bool CameraConnection::setNetworkAdapterParameters(VkRender::Device &dev, bool shouldConfigNetwork) {
        std::string hostAddress = dev.IP;
        try {
            std::string last_element(hostAddress.substr(hostAddress.rfind('.')));
            hostAddress.replace(hostAddress.rfind('.'), last_element.length(), ".2");
        } catch (std::out_of_range &exception) {
            Log::Logger::getInstance()->error(
                "Trying to configure adapter '{}' with source IP: '{}', but address does not seem like an ipv4 address",
                dev.interfaceName, dev.IP);
            Log::Logger::getInstance()->error("Exception message: '{}'", exception.what());
            return false;
        }
        if (shouldConfigNetwork) {
            Log::Logger::getInstance()->info("User wants to configure network automatically");
#ifdef WIN32

            /*
            WinRegEditor regEditor(dev.interfaceName, dev.interfaceDescription, dev.interfaceIndex);
            if (regEditor.ready) {
                regEditor.readAndBackupRegisty();
                if (regEditor.backup.JumboPacket != "9014")
                {
                    Log::Logger::getInstance()->error("Setting jumbo packet to 9014");
                    regEditor.setJumboPacket("9014");
                    regEditor.restartNetAdapters();
                    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
                }
                else {
                    Log::Logger::getInstance()->error("Jumbo packet already set to 9014");

                }
            }
            else {
                Log::Logger::getInstance()->error("Failed to read current jumbo frame setting");
            }

            if (regEditor.setStaticIp(dev.interfaceIndex, hostAddress, "255.255.255.0")) {
                dev.systemNetworkChanged = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            // TODO
            // 5 Seconds to wait for adapter to change. This will vary from machine to machine and should be re-done
            // If possible then wait for a windows event that triggers when the adapter is ready

             */
#else
            int ioctl_result = -1;
            /** SET NETWORK PARAMETERS FOR THE ADAPTER **/
            if ((m_FD = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
                Log::Logger::getInstance()->error("Error in creating socket to configure network adapter: '{}'",
                                                  strerror(errno));
                return false;
            }
            // Specify interface m_Name
            const char *interface = dev.interfaceName.c_str();
            if (setsockopt(m_FD, SOL_SOCKET, SO_BINDTODEVICE, interface, 15) < 0) {
                Log::Logger::getInstance()->error("Could not bind socket to adapter {}, '{}'", dev.interfaceName,
                                                  strerror(errno));
            };
            struct ifreq ifr{};
            /// note: no pointer here
            struct sockaddr_in inet_addr{}, subnet_mask{};
            /* get interface m_Name */
            /* Prepare the struct ifreq */
            bzero(ifr.ifr_name, IFNAMSIZ);
            strncpy(ifr.ifr_name, interface, IFNAMSIZ);
            ifr.ifr_name[IF_NAMESIZE - 1] = '\0';
            /*** Call ioctl to get configure network interface ***/

            /// note: prepare the two struct sockaddr_in
            inet_addr.sin_family = AF_INET;
            inet_pton(AF_INET, hostAddress.c_str(), &(inet_addr.sin_addr));

            subnet_mask.sin_family = AF_INET;
            inet_pton(AF_INET, "255.255.255.0", &(subnet_mask.sin_addr));

            /// put addr in ifr structure
            memcpy(&(ifr.ifr_addr), &inet_addr, sizeof(struct sockaddr));
            ioctl_result = ioctl(m_FD, SIOCSIFADDR, &ifr);  // Set IP address
            if (ioctl_result < 0) {
                Log::Logger::getInstance()->error("Could not set ip address on {}, reason: {}", dev.interfaceName,
                                                  strerror(errno));

            }

            /// put mask in ifr structure
            memcpy(&(ifr.ifr_addr), &subnet_mask, sizeof(struct sockaddr));
            ioctl_result = ioctl(m_FD, SIOCSIFNETMASK, &ifr);   // Set subnet mask
            if (ioctl_result < 0) {
                Log::Logger::getInstance()->error("Could not set subnet mask address on {}, reason: {}",
                                                  dev.interfaceName,
                                                  strerror(errno));
            }

            strncpy(ifr.ifr_name, interface, sizeof(ifr.ifr_name));//interface m_Name where you want to set the MTU
            ifr.ifr_name[IF_NAMESIZE - 1] = '\0';
            ifr.ifr_mtu = 7200; //your MTU  here
            if (ioctl(m_FD, SIOCSIFMTU, reinterpret_cast<caddr_t>( &ifr)) < 0) {
                Log::Logger::getInstance()->error("Failed to set mtu size {} on adapter {}", 7200,
                                                  dev.interfaceName.c_str());
            } else {
                Log::Logger::getInstance()->error("Set Mtu size to {} on adapter {}", 7200,
                                                  dev.interfaceName.c_str());
            }

#endif
        }
        return true;
    }


    void CameraConnection::addIniEntry(CSimpleIniA *ini, std::string section, std::string key, std::string value) {
        int ret = ini->SetValue(section.c_str(), key.c_str(), value.c_str());

        if (ret < 0)
            Log::Logger::getInstance()->error("Section: {} Updated {} to {}", section, key, value);
        else
            Log::Logger::getInstance()->info("Section: {} Updated {} to {}", section, key, value);
    }

    void
    CameraConnection::deleteIniEntry(CSimpleIniA *ini, std::string section, std::string key, std::string value) {
        int ret = ini->DeleteValue(section.c_str(), key.c_str(), value.c_str());

        if (ret < 0)
            Log::Logger::getInstance()->error("Section: {} deleted {}", section, key, value);
        else
            Log::Logger::getInstance()->info("Section: {} deleted {}", section, key, value);
    }

    CameraConnection::~CameraConnection() {
        auto startTime = std::chrono::steady_clock::now();

        // Make sure delete the camPtr for physical
        // cameras so we run destructor on the physical camera class which
        // stops all streams on the camera
        if (pool) {
            pool->Stop();
        }
#ifndef WIN32
        if (m_FD != -1)
            close(m_FD);
#endif // !WIN32

        auto timeSpan = std::chrono::duration_cast<std::chrono::duration<float> >(
            std::chrono::steady_clock::now() - startTime);
        Log::Logger::getInstance()->trace("Cameraconnection destructor stopping threadpool took {}s", timeSpan.count());
    }


    void CameraConnection::connectCRLCameraTask(void *context, VkRender::Device *dev, bool isRemoteHead,
                                                bool shouldConfigNetwork, bool delayConnection) {
        auto *app = reinterpret_cast<CameraConnection *>(context);

        if (delayConnection) {
            auto startTime = std::chrono::steady_clock::now();
            auto time = std::chrono::steady_clock::now();
            std::chrono::duration<float> timeSpan =
                    std::chrono::duration_cast<std::chrono::duration<float> >(time - startTime);

            Log::Logger::getInstance()->info("Delay connection with 10 seconds on Windows to propagate IP changes");

            while (timeSpan.count() < 10.0f && !dev->interruptConnection) {
                time = std::chrono::steady_clock::now();
                timeSpan = std::chrono::duration_cast<std::chrono::duration<float> >(time - startTime);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }

        app->setNetworkAdapterParameters(*dev, shouldConfigNetwork);
        // Connect to camera
        Log::Logger::getInstance()->info("Creating connection to camera. Ip: {}, ifName {}", dev->IP,
                                         dev->interfaceDescription);
        // If we successfully connect
        dev->channelConnections = app->camPtr.connect(dev, dev->interfaceName);
        if (!dev->channelConnections.empty() && dev->state == CRL_STATE_CONNECTING) {
            // Check if we actually connected to a RemoteHead or not
            auto hwRev = app->camPtr.getCameraInfo(dev->channelConnections.front()).devInfo.hardwareRevision;
            for (std::vector v{
                     crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_REMOTE_HEAD_VPB,
                     crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_REMOTE_HEAD_STEREO,
                     crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_REMOTE_HEAD_MONOCAM
                 };
                 auto &e: v) {
                if (hwRev == e && !isRemoteHead) {
                    dev->state = CRL_STATE_UNAVAILABLE;
                    Log::Logger::getInstance()->error(
                        "User connected to a remote head but didn't check the remote head box");
                    return;
                }
            }

            app->updateUIDataBlock(*dev, app->camPtr);

            /*
            Log::Logger::getInstance()->info("Attempting to restore last session from .ini runtime config file");
            if (!isRemoteHead)
                app->getProfileFromIni(*dev);
            */
            // Set the resolution read from config file
            const auto &info = app->camPtr.getCameraInfo(dev->channelConnections.front()).devInfo;
            dev->cameraName = info.name;
            dev->serialName = info.serialNumber;
            dev->hasColorCamera =
                    info.hardwareRevision ==
                    crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_C6S2_S27 ||
                    info.hardwareRevision == crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_S30 ||
                    info.hardwareRevision == crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_MONOCAM ||
                    info.hardwareRevision == crl::multisense::system::DeviceInfo::HARDWARE_REV_MULTISENSE_KS21i;

            Log::Logger::getInstance()->info("Has color camera? {}", dev->hasColorCamera);

            app->m_FailedGetStatusCount = 0;
            app->queryStatusTimer = std::chrono::steady_clock::now();
            app->queryDeviceConfigTimer = std::chrono::steady_clock::now();
            app->queryExposureTimer = std::chrono::steady_clock::now();
            app->calcDisparityNormValuesTimer = std::chrono::steady_clock::now();
            dev->state = CRL_STATE_ACTIVE;

            dev->hasColorCamera ? dev->useAuxForPointCloudColor = 1 : dev->useAuxForPointCloudColor = 0;
            dev->hasImuSensor = app->camPtr.getCameraInfo(0).hasIMUSensor;
            dev->enableIMU = dev->hasImuSensor;

            if (dev->hasImuSensor) {
                Log::Logger::getInstance()->info("Found IMU Sensor");
                dev->rateTableIndex = app->camPtr.getCameraInfo(0).imuSensorConfigs.front().rateTableIndex;
            } else {
                Log::Logger::getInstance()->info("No IMU Sensor");
            }

            const auto &cInfo = app->camPtr.getCameraInfo(0).versionInfo;
            Log::Logger::getInstance()->info(
                "Connected to {}, FW version: {}, FW Build date: {}, LibMultiSense API: {} LibMultiSense API Date: {}",
                dev->cameraName, cInfo.sensorFirmwareVersion, cInfo.sensorFirmwareBuildDate, cInfo.apiVersion,
                cInfo.apiBuildDate);

            Log::Logger::getInstance()->info("Set dev {}'s state to CRL_STATE_ACTIVE ", dev->name);
        } else {
            dev->state = CRL_STATE_UNAVAILABLE;
            Log::Logger::getInstance()->info("Set dev {}'s state to CRL_STATE_UNAVAILABLE ", dev->name);
            Log::Logger::getInstance()->info("Failed to Connect to Ip: {}, ifName {}", dev->IP,
                                             dev->interfaceDescription);
        }
    }

    void CameraConnection::connectFakeCameraTask(void *context, VkRender::Device *dev) {
        auto *app = reinterpret_cast<CameraConnection *>(context);

        Log::Logger::getInstance()->info("Creating Fake connection to camera. Ip: {}, ifName {}, devInfo.name: {}",
                                         dev->IP,
                                         dev->interfaceDescription, app->camPtr.getCameraInfo(0).devInfo.name);
        // If we successfully connect
        //dev->channelConnections = app->camPtr.connect(dev, dev->interfaceName);
        Utils::initializeUIDataBlockWithTestData(*dev);
        dev->state = CRL_STATE_ACTIVE;
        Log::Logger::getInstance()->info("Set dev {}'s state to CRL_STATE_ACTIVE ", dev->name);
    }


    void CameraConnection::saveProfileAndDisconnect(VkRender::Device *dev) {
        Log::Logger::getInstance()->info("Disconnecting profile {} using camera {}", dev->name.c_str(),
                                         dev->cameraName.c_str());
        // Save settings to file. Attempt to create a new file if it doesn't exist
        /*
       CSimpleIniA ini;
       ini.SetUnicode();

       auto filePath = (Utils::getSystemCachePath() / "crl.ini");
       SI_Error rc = ini.LoadFile(filePath.c_str());
       if (rc < 0) {
           // File doesn't exist error, then create one
           if (rc == SI_FILE && errno == ENOENT) {
               std::ofstream output(filePath.c_str());
               output.close();
               rc = ini.LoadFile(filePath.c_str());
           } else
               Log::Logger::getInstance()->error("Failed to create profile configuration file\n");
       }
       std::string CRLSerialNumber = dev->serialName;
       // If sidebar is empty or we dont recognize any serial numbers in the crl.ini file then clear it.
       // new m_Entry given we have a valid ini file m_Entry
       if (rc >= 0 && !CRLSerialNumber.empty()) {
           // Profile Data
           addIniEntry(&ini, CRLSerialNumber, "ProfileName", dev->name);
           addIniEntry(&ini, CRLSerialNumber, "AdapterName", dev->interfaceName);
           addIniEntry(&ini, CRLSerialNumber, "CameraName", dev->cameraName);
           addIniEntry(&ini, CRLSerialNumber, "IP", dev->IP);
           addIniEntry(&ini, CRLSerialNumber, "AdapterIndex", std::to_string(dev->interfaceIndex));
           addIniEntry(&ini, CRLSerialNumber, "State", std::to_string(dev->state));
           // Preview Data per channel
           for (const auto &ch: dev->channelConnections) {
               std::string mode = "Mode" + std::to_string(ch);
               if (dev->channelInfo.empty() || dev->channelInfo[ch].modes.empty())
                   continue;

               auto resMode = dev->channelInfo[ch].modes[dev->channelInfo[ch].selectedModeIndex];
               if (Utils::stringToCameraResolution(resMode) == CRL_RESOLUTION_NONE)
                   resMode = "0";

               addIniEntry(&ini, CRLSerialNumber, mode,
                           std::to_string(static_cast<int>(Utils::stringToCameraResolution(
                                   resMode))
                           ));
               addIniEntry(&ini, CRLSerialNumber, "Layout", std::to_string(static_cast<int>(dev->layout)));
               for (int i = 0; i <= CRL_PREVIEW_FOUR; ++i) {
                   std::string source = dev->win[static_cast<StreamWindowIndex>(i)].selectedSource;
                   std::string remoteHead = std::to_string(
                       dev->win[static_cast<StreamWindowIndex>(i)].selectedRemoteHeadIndex);
                   std::string key = "Preview" + std::to_string(i + 1);
                   std::string value = (source.append(":" + remoteHead));
                   addIniEntry(&ini, CRLSerialNumber, key, value);
               }
           }
       }
       // delete m_Entry if we gave the disconnect and reset flag Otherwise just normal disconnect
       // save the m_DataPtr back to the file
*/
        if (dev->state == CRL_STATE_DISCONNECT_AND_FORGET || dev->state == CRL_STATE_INTERRUPT_CONNECTION) {
            //ini.Delete(CRLSerialNumber.c_str(), nullptr);
            dev->state = CRL_STATE_REMOVE_FROM_LIST;
            Log::Logger::getInstance()->info("Set dev {}'s state to CRL_STATE_REMOVE_FROM_LIST ", dev->name);
            //Log::Logger::getInstance()->info("Deleted saved profile for serial: {}", CRLSerialNumber);
        } else {
            dev->state = CRL_STATE_DISCONNECTED;
            Log::Logger::getInstance()->info("Set dev {}'s state to CRL_STATE_DISCONNECTED ", dev->name);
        }
        //rc = ini.SaveFile(filePath.c_str());
        //if (rc < 0) {
        //    Log::Logger::getInstance()->info("Failed to save crl.ini file. Err: {}", rc);
        //}

        dev->channelInfo.clear();
    }

    void CameraConnection::setExposureTask(void *context, ExposureParams *arg1, VkRender::Device *dev,
                                           crl::multisense::RemoteHeadChannel remoteHeadIndex) {
        auto *app = reinterpret_cast<CameraConnection *>(context);

        std::scoped_lock lock(app->writeParametersMtx);
        if (app->camPtr.setExposureParams(*arg1, remoteHeadIndex))
            app->updateFromCameraParameters(dev, remoteHeadIndex);
    }

    void CameraConnection::setAuxConfigTask(void *context, AUXConfig *arg1, VkRender::Device *dev,
                                            crl::multisense::RemoteHeadChannel remoteHeadIndex) {
        auto *app = reinterpret_cast<CameraConnection *>(context);

        std::scoped_lock lock(app->writeParametersMtx);
        if (app->camPtr.setAuxImageConfig(*arg1, remoteHeadIndex))
            app->updateFromCameraParameters(dev, remoteHeadIndex);
    }

    void CameraConnection::setLightingTask(void *context, LightingParams *arg1, VkRender::Device *dev,
                                           crl::multisense::RemoteHeadChannel remoteHeadIndex) {
        auto *app = reinterpret_cast<CameraConnection *>(context);
        std::scoped_lock lock(app->writeParametersMtx);
        if (app->camPtr.setLighting(*arg1, remoteHeadIndex))
            app->updateFromCameraParameters(dev, remoteHeadIndex);
    }

    void
    CameraConnection::setAdditionalParametersTask(void *context, float fps, float gain, float gamma, float spfs,
                                                  bool hdr, int mtu, VkRender::Device *dev,
                                                  crl::multisense::RemoteHeadChannel index) {
        auto *app = reinterpret_cast<CameraConnection *>(context);
        std::scoped_lock lock(app->writeParametersMtx);
        if (!app->camPtr.setGamma(gamma, index))                {app->updateFromCameraParameters(dev, index); return;}
        if (!app->camPtr.setGain(gain, index))                  {app->updateFromCameraParameters(dev, index); return;}
        if (!app->camPtr.setFps(fps, index))                    {app->updateFromCameraParameters(dev, index); return;}
        if (!app->camPtr.setPostFilterStrength(spfs, index))    {app->updateFromCameraParameters(dev, index); return;}
        if (!app->camPtr.setHDR(hdr, index))                    {app->updateFromCameraParameters(dev, index); return;}
        if (!app->camPtr.setMtu(mtu, index))                    {app->updateFromCameraParameters(dev, index); return;}


        app->updateFromCameraParameters(dev, index);
    }

    void CameraConnection::setResolutionTask(void *context, CRLCameraResolution arg1, VkRender::Device *dev,
                                             crl::multisense::RemoteHeadChannel idx) {
        auto *app = reinterpret_cast<CameraConnection *>(context);
        std::scoped_lock lock(app->writeParametersMtx);
        if (app->camPtr.setResolution(arg1, idx)) {
            app->updateFromCameraParameters(dev, idx);
        }
    }

    void CameraConnection::startStreamTask(void *context, std::string src,
                                           crl::multisense::RemoteHeadChannel remoteHeadIndex) {
        auto *app = reinterpret_cast<CameraConnection *>(context);
        app->camPtr.start(src, remoteHeadIndex);
    }

    void CameraConnection::stopStreamTask(void *context, std::string src,
                                          crl::multisense::RemoteHeadChannel remoteHeadIndex) {
        auto *app = reinterpret_cast<CameraConnection *>(context);
        if (app->camPtr.stop(src, remoteHeadIndex));
        else
            Log::Logger::getInstance()->info("Failed to disable stream {}", src);
    }

    void CameraConnection::updateFromCameraParameters(VkRender::Device *dev,
                                                      crl::multisense::RemoteHeadChannel remoteHeadIndex) const {
        // Query the camera for new values and update the GUI. It is a way to see if the actual value was set.
        const auto &conf = camPtr.getCameraInfo(remoteHeadIndex).imgConf;
        auto *p = &dev->parameters;
        p->stereo.ep.exposure = conf.exposure();
        p->stereo.ep.autoExposure = conf.autoExposure();
        p->stereo.ep.autoExposureThresh = conf.autoExposureThresh();
        p->stereo.ep.autoExposureDecay = conf.autoExposureDecay();
        p->stereo.ep.autoExposureMax = conf.autoExposureMax();
        p->stereo.ep.autoExposureTargetIntensity = conf.autoExposureTargetIntensity();
        p->stereo.ep.autoExposureRoiHeight = conf.autoExposureRoiHeight();
        p->stereo.ep.autoExposureRoiWidth = conf.autoExposureRoiWidth();
        p->stereo.ep.autoExposureRoiX = conf.autoExposureRoiX();
        p->stereo.ep.autoExposureRoiY = conf.autoExposureRoiY();
        p->stereo.gain = conf.gain();
        p->stereo.fps = conf.fps();
        p->stereo.gamma = conf.gamma();
        p->stereo.stereoPostFilterStrength = conf.stereoPostFilterStrength();

        const auto &auxConf = camPtr.getCameraInfo(dev->configRemoteHead).auxImgConf;
        p->aux.whiteBalanceAuto = auxConf.autoWhiteBalance();
        p->aux.whiteBalanceDecay = auxConf.autoWhiteBalanceDecay();
        p->aux.whiteBalanceThreshold = auxConf.autoWhiteBalanceThresh();
        p->aux.whiteBalanceBlue = auxConf.whiteBalanceBlue();
        p->aux.whiteBalanceRed = auxConf.whiteBalanceRed();
        p->aux.sharpening = auxConf.enableSharpening();
        p->aux.sharpeningPercentage = auxConf.sharpeningPercentage();
        p->aux.sharpeningLimit = auxConf.sharpeningLimit();
        p->aux.ep.exposure = auxConf.exposure();
        p->aux.ep.autoExposure = auxConf.autoExposure();
        p->aux.ep.autoExposureThresh = auxConf.autoExposureThresh();
        p->aux.ep.autoExposureDecay = auxConf.autoExposureDecay();
        p->aux.ep.autoExposureMax = auxConf.autoExposureMax();
        p->aux.ep.autoExposureTargetIntensity = auxConf.autoExposureTargetIntensity();
        p->aux.ep.autoExposureRoiHeight = auxConf.autoExposureRoiHeight();
        p->aux.ep.autoExposureRoiWidth = auxConf.autoExposureRoiWidth();
        p->aux.ep.autoExposureRoiX = auxConf.autoExposureRoiX();
        p->aux.ep.autoExposureRoiY = auxConf.autoExposureRoiY();
        p->aux.gain = auxConf.gain();
        p->aux.gamma = auxConf.gamma();
        p->stereo.mtu = camPtr.getCameraInfo(dev->configRemoteHead).sensorMTU;

        dev->parameters.updateGuiParams = false;
    }

    void CameraConnection::getStatusTask(void *context, crl::multisense::RemoteHeadChannel remoteHeadIndex,
                                         VkRender::Device *dev) {
        auto *app = reinterpret_cast<CameraConnection *>(context);
        std::scoped_lock lock(app->writeParametersMtx);
        crl::multisense::system::StatusMessage msg;
        if (app->camPtr.getStatus(remoteHeadIndex, &msg)) {
            Log::Logger::getLogMetrics()->device.upTime = msg.uptime;
            std::scoped_lock lock2(app->statusCountMutex);
            app->m_FailedGetStatusCount = 0;
        } else {
            std::scoped_lock lock2(app->statusCountMutex);
            Log::Logger::getInstance()->info("Failed to get channel {} status. Attempt: {}. Name: {}", remoteHeadIndex,
                                             app->m_FailedGetStatusCount, dev->name);
            app->m_FailedGetStatusCount++;
        }
        // Increment a counter
    }

    void CameraConnection::getCameraConfigsTask(void *context, crl::multisense::RemoteHeadChannel remoteHeadIndex,
                                                VkRender::Device *dev) {
        auto *app = reinterpret_cast<CameraConnection *>(context);
        std::scoped_lock lock(app->writeParametersMtx);
        dev->updateDeviceConfigsSucceeded = app->camPtr.updateCameraInfo(dev, remoteHeadIndex);
    }

    void CameraConnection::getExposureTask(void *context, VkRender::Device *dev,
                                           crl::multisense::RemoteHeadChannel index) {
        auto *app = reinterpret_cast<CameraConnection *>(context);
        std::scoped_lock lock(app->writeParametersMtx);
        if (app->camPtr.getExposure(index, dev->hasColorCamera)) {
            // Update GUI value
            const auto &conf = app->camPtr.getCameraInfo(index).imgConf;
            auto *p = &dev->parameters;
            p->stereo.ep.currentExposure = conf.exposure();

            if (dev->hasColorCamera) {
                const auto &auxConf = app->camPtr.getCameraInfo(index).auxImgConf;
                p->aux.ep.currentExposure = auxConf.exposure();
            }
        }
    }

    void CameraConnection::getCalibrationTask(void *context, const std::string &saveLocation,
                                              crl::multisense::RemoteHeadChannel index, bool *success) {
        auto *app = reinterpret_cast<CameraConnection *>(context);
        std::scoped_lock lock(app->writeParametersMtx);
        *success = app->camPtr.saveSensorCalibration(saveLocation, index);
    }

    void
    CameraConnection::setCalibrationTask(void *context, const std::string &intrinsicFilePath,
                                         const std::string &extrinsicFilePath,
                                         crl::multisense::RemoteHeadChannel index, bool *success) {
        auto *app = reinterpret_cast<CameraConnection *>(context);
        std::scoped_lock lock(app->writeParametersMtx);
        *success = app->camPtr.setSensorCalibration(intrinsicFilePath, extrinsicFilePath, index);
    }
}
