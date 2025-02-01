//
// Created by magnus on 4/9/23.
//

#include <iostream>
#include <future>

#include "UsageMonitor.h"
#include "Viewer/Tools/Logger.h"
#include "Viewer/Tools/Utils.h"
#include "ApplicationConfig.h"
#include "Viewer/Rendering/Core/ServerConnection.h"

UsageMonitor::UsageMonitor() {
    // Initialize usage file
    usageFilePath = Utils::getSystemCachePath() / "usage.json";
    logFilePath = Utils::getSystemCachePath() / "logger.log";

    if (!std::filesystem::exists(usageFilePath)) {
        initializeJSONFile();
    }
    VkRender::ApplicationConfig &config = VkRender::ApplicationConfig::getInstance();

    // Connect to CRL server
    server = std::make_unique<VkRender::ServerConnection>(config.getAnonymousIdentifier(), config.getServerInfo());

    /*
    getAppVersionRemoteFuture = std::async(std::launch::async, [&]() {
        return server->getLatestApplicationVersionRemote();
    });
    */

}


void UsageMonitor::addEvent() {
    std::ifstream input_file(usageFilePath);
    // Create a string stream and copy the file's contents into it
    std::stringstream buffer;
    buffer << input_file.rdbuf();
}

nlohmann::json UsageMonitor::openUsageFile() {
    std::ifstream input_file(usageFilePath);
    // Create a string stream and copy the file's contents into it
    std::stringstream buffer;
    buffer << input_file.rdbuf();

    // Parse the JSON from the string stream
    nlohmann::json jsonObj = parseJSON(buffer);
    return jsonObj;
}

void UsageMonitor::saveJsonToUsageFile(nlohmann::json jsonObj) {
    // Save the modified JSON to the file
    std::ofstream output_file(usageFilePath);
    output_file << jsonObj.dump(4);

}

nlohmann::json UsageMonitor::parseJSON(const std::stringstream &buffer) {
    try {
        return nlohmann::json::parse(buffer.str());
    } catch (nlohmann::json::exception &exception) {
        Log::Logger::getInstance()->error("JSON Parse error: {}", exception.what());
        initializeJSONFile();
        Log::Logger::getInstance()->info("Reverting all usage diagnostics data and resetting json file",
                                         exception.what());

        return nullptr;
    }
}

void UsageMonitor::initializeJSONFile() {
    try {
        if (std::filesystem::exists(usageFilePath)) {
            std::filesystem::remove(usageFilePath);
            Log::Logger::getInstance()->info("File {} deleted successfully", usageFilePath.string());
        }
    } catch (const std::filesystem::filesystem_error& e) {
        Log::Logger::getInstance()->error("Error deleting file {}: {}", usageFilePath.string(), e.what());
        // Handle the error, e.g., retrying the deletion, informing the user, etc.
    } catch (const std::exception& e) {
        // Catch any other standard exceptions
        Log::Logger::getInstance()->error("An error occurred: {}", e.what());
        // Handle the error
    } catch (...) {
        // Catch any other exceptions not caught by the previous catch blocks
        Log::Logger::getInstance()->error("An unknown error occurred while deleting file {}", usageFilePath.string());
        // Handle the error
    }

    VkRender::ApplicationConfig &config = VkRender::ApplicationConfig::getInstance();

    std::string logVersion = "1.0.0";
    std::ofstream output_file(usageFilePath);
    std::string jsonBoilerPlate = "{\n"
                                  "    \"log_version\": \"" + logVersion + "\",\n"
                                                                           "    \"os\": {\n"
                                                                           "         \"name\": \"" + config.getOS() +
                                  "\",\n"
                                  "         \"version\": \"" + config.getOsVersion() + "\",\n"
                                                                                       "         \"architecture\": \"" +
                                  config.getArchitecture() + "\"\n"
                                                             "    },\n"
                                                             "    \"app\": {\n"
                                                             "        \"name\": \"MultiSense Viewer\",\n"
                                                             "        \"version\": \"" + config.getAppVersion() + "\"\n"
                                                                                                                  "    },\n"
                                                                                                                  "    \"stats\": {\n"
                                                                                                                  "    }\n"
                                                                                                                  "}";

    output_file << jsonBoilerPlate;
    output_file.close();
}

void UsageMonitor::sendUsageLog() {
    server->sendUsageStatistics(usageFilePath, logFilePath);
}

bool UsageMonitor::getLatestAppVersionRemote(std::string *version) {
    bool success = false;
    if (getAppVersionRemoteFuture.valid() &&
        getAppVersionRemoteFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        *version = VkRender::ApplicationConfig::getInstance().getAppVersionRemote();
        success = getAppVersionRemoteFuture.get();
    }

    return success;
}


std::string UsageMonitor::getCurrentTimeString() {
#ifdef WIN32
    time_t currentTime;
    time(&currentTime);  // Get the current time
    std::tm tm;
    localtime_s(&tm, &currentTime);  // Convert to local time
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    auto timestamp = oss.str();
#else
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    auto timestamp = oss.str();
#endif
    return timestamp;
}


void UsageMonitor::userClickAction(const std::string &label, const std::string &type, const std::string &window) {
    try {
        nlohmann::json obj;
        obj["element"] = type;
        obj["element_id"] = label;
        obj["parent_window"] = window;
        obj["timestamp"] = getCurrentTimeString();

        // I do not want a return value here. otherwise it does nto make sense to make it a async operation.
        writeToUsageFileFuture = std::async(std::launch::async, &UsageMonitor::writeToUsageFileAsync, this, obj);

        Log::Logger::getInstance()->info("User click action: {}, Window: {}, Time: {}", label, window,
                                         getCurrentTimeString());
    } catch (nlohmann::json::exception &e) {
        Log::Logger::getInstance()->warning("Failed to record userClickAction: {}", e.what());
    }
}

void UsageMonitor::writeToUsageFileAsync(const nlohmann::json &obj) {
    auto usageLog = openUsageFile();
    if (!usageLog["stats"][sessionIndex]["interactions"].is_array()) {
        usageLog["stats"][sessionIndex]["interactions"] = nlohmann::json::array();
    }

    usageLog["stats"][sessionIndex]["interactions"].push_back(obj);
    saveJsonToUsageFile(usageLog);
}

void UsageMonitor::userEndSession() {
    try {
        nlohmann::json generalData;
        nlohmann::json settingsChanged;
        auto time = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - m_StartTime).count();
        generalData["time_spent_seconds"] = std::to_string(time);
        generalData["settings_changed"] = settingsChanged;
        auto usageLog = openUsageFile();
        usageLog["stats"][sessionIndex]["general"] = generalData;
        saveJsonToUsageFile(usageLog);

    } catch (nlohmann::json::exception &e) {
        Log::Logger::getInstance()->warning("Failed to save usage log in userEndSession: {}", e.what());
    }


}

void UsageMonitor::userStartSession(
        std::chrono::system_clock::time_point startTime) {
    m_StartTime = startTime;
    nlohmann::json obj;
    auto gpuDevice = VkRender::ApplicationConfig::getInstance().getGpuDevice();


    obj["event"] = "start application";
    obj["start_time"] = getCurrentTimeString();
    obj["graphics_device"] = gpuDevice;

    auto usageLog = openUsageFile();
    // Check if usageLog["stats"] is an array; if not, initialize it as an empty array
    if (!usageLog["stats"].is_array()) {
        usageLog["stats"] = nlohmann::json::array();
    }

    sessionIndex = static_cast<uint32_t>(usageLog["stats"].size());
    usageLog["stats"].push_back(obj);
    saveJsonToUsageFile(usageLog);

}
