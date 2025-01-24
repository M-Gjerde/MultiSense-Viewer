/**
 * @file: MultiSense-Viewer/src/Tools/Logger.cpp
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
 *   2022-09-12, mgjerde@carnegierobotics.com, Created file.
 **/

// C++ Header File(s)
#include <iostream>
#include <ctime>
#include <vector>
#include <regex>
#include <filesystem>


#ifdef WIN32
#define semPost(x) SetEvent(x)
#define semWait(x, y) WaitForSingleObject(x, y)
#define CTIME(time, size, timer) ctime_s(time, size, timer)
#else
#include<bits/stdc++.h>
#define semPost(x) sem_post(x)
#define CTIME(time, timer) ctime(timer)
#endif

#include "Viewer/Tools/Logger.h"

namespace Log {

    Logger *Logger::m_instance = nullptr;
    VkRender::ThreadPool *Logger::m_threadPool = nullptr;
    std::queue<std::string> *Logger::m_consoleLogQueue = nullptr;

    Logger::Logger(const std::string &logFileName) {
        // Check file size
        bool resetLogFile = false;
        double fileSizeMB = 0.0;
        if (std::filesystem::exists(logFileName)) {
            std::uintmax_t fileSize = std::filesystem::file_size(logFileName);
            fileSizeMB = static_cast<double>(fileSize) / (1024.0 * 1024.0);  // Convert to MB
            // Delete file if size exceeds 10 MB
            resetLogFile = fileSizeMB > 5.0;
            if (resetLogFile) {
                std::filesystem::remove(logFileName);
            }
        }

        m_file.open(logFileName.c_str(), std::ios::out | std::ios::app);
        m_logLevel = LOG_LEVEL_TRACE;
        m_logType = FILE_LOG;

        info("<=============================== START OF PROGRAM ===============================>");

        if (resetLogFile)
            info("Log file was larger than 10MB. Deleted: {} and created new empty", logFileName);
        else {
            info("Log file size is currently: {}MB", fileSizeMB);
        }
    }


    Logger::~Logger() {
        m_file.close();
        delete m_instance;
        delete m_threadPool;
        delete m_consoleLogQueue;
    }

    Logger *Logger::getInstance(const std::string &fileName) noexcept {
        if (m_instance == nullptr) {
            m_consoleLogQueue = new std::queue<std::string>();
            m_threadPool = new VkRender::ThreadPool(1);
            m_instance = new Logger(fileName);
            m_instance->info("Initialized logger instance, fileName: {} with log level: {}", fileName, logLevelToString(m_instance->m_logLevel));
        }
        return m_instance;
    }

    std::queue<std::string>* Logger::getConsoleLogQueue() noexcept{
        return m_consoleLogQueue;
    }

    void Logger::logIntoFile(void *ctx, std::string &data) {
        auto *app = static_cast<Logger *> (ctx);
        app->m_mutex.lock();
        app->m_file << getCurrentTime() << "  " << data << std::endl;
        app->m_consoleLogQueue->push(data);
        app->m_mutex.unlock();
    }

    void Logger::logOnConsole(std::string &data) {
        std::cout << getCurrentTime() << "  " << data << std::endl;
    }

    std::string Logger::getCurrentTime() {
        // Get the current time with milliseconds precision
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        // Convert current time to string excluding the year
        std::tm timeInfo{};
#ifdef _WIN32
        localtime_s(&timeInfo, &now_time_t);
#else
        localtime_r(&now_time_t, &timeInfo);
#endif

        std::ostringstream oss;
        oss << std::put_time(&timeInfo, "%a %b %d %H:%M:%S") << '.' << std::setw(3) << std::setfill('0') << now_ms.count();

        return oss.str();
    }

// Interface for Error Log
    void Logger::errorInternal(const char *text) noexcept {
        std::string data;
        data.append("[ERROR]: ");
        data.append(text);

        // ERROR must be capture
        if (m_logType == FILE_LOG) {
            m_threadPool->Push(Logger::logIntoFile, this, data);
        } else if (m_logType == CONSOLE) {
            m_threadPool->Push(Logger::logOnConsole, data);
        }
    }
    // Interface for Error Log
    void Logger::fatalInternal(const char *text) noexcept {
        std::string data;
        data.append("[FATAL ERROR]: ");
        data.append(text);

        // ERROR must be capture
        if (m_logType == FILE_LOG) {
            m_threadPool->Push(Logger::logIntoFile, this, data);
        } else if (m_logType == CONSOLE) {
            m_threadPool->Push(Logger::logOnConsole, data);
        }
    }

    void Logger::warningInternal(const char *text) noexcept {
        std::string data;
        data.append("[WARNING]: ");
        data.append(text);

        // ERROR must be capture
        if (m_logType == FILE_LOG) {
            m_threadPool->Push(Logger::logIntoFile, this, data);
        } else if (m_logType == CONSOLE) {
            m_threadPool->Push(Logger::logOnConsole, data);
        }
    }

// Interface for Info Log
    void Logger::infoInternal(const char *text) noexcept {
        std::string data;
        data.append("[INFO]: ");
        data.append(text);

        if ((m_logType == FILE_LOG) && (m_logLevel >= LOG_LEVEL_INFO)) {
            m_threadPool->Push(Logger::logIntoFile, this, data);
        } else if ((m_logType == CONSOLE) && (m_logLevel >= LOG_LEVEL_INFO)) {
            m_threadPool->Push(Logger::logOnConsole, data);
        }

    }

    void Logger::traceInternal(const char *text) noexcept {
        std::string data;
        data.append("[TRACE]: ");
        data.append(text);

        if ((m_logType == FILE_LOG) && (m_logLevel >= LOG_LEVEL_TRACE)) {
            m_threadPool->Push(Logger::logIntoFile, this, data);
        } else if ((m_logType == CONSOLE) && (m_logLevel >= LOG_LEVEL_TRACE)) {
            m_threadPool->Push(Logger::logOnConsole, data);
        }

    }

// Interface for Always Log
    void Logger::always(std::string text) noexcept {
        std::string data;
        data.append("[ALWAYS]: ");
        data.append(text);

        // No check for ALWAYS logs
        if (m_logType == FILE_LOG) {
            m_threadPool->Push(Logger::logIntoFile, this, data);
        } else if (m_logType == CONSOLE) {
            m_threadPool->Push(Logger::logOnConsole, data);
        }
    }

    std::string Logger::filterFilePath(const std::string &input) {
        // Filter away the absolute file path given by std::source_location, both for anonymous and readable logs purpose.
        // Magic regex expression, courtesy of chatgpt4
#ifdef WIN32
        std::regex folderPathRegex(R"(^((?:[a-zA-Z]:[\\\/])?(?:[\w\s-]+[\\\/])+))");
#else
        std::regex folderPathRegex(R"(^((?:\/|\.\.\/|\.\/)?(?:[\w\s-]+\/)+))");
#endif
        std::string result = std::regex_replace(input, folderPathRegex, "");


        return result;

    }

    void Logger::setLogLevel(LogLevel logLevel) {
        m_logLevel = logLevel;
    }

};