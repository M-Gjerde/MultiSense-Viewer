/**
 * @file: MultiSense-Viewer/include/Viewer/Tools/Logger.h
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
#ifndef LOGGER_H_
#define LOGGER_H_

// C++ Header File(s)
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <fmt/core.h>
#include <mutex>
#include <map>

#ifdef _WIN32

#include <process.h>

#else
// POSIX Socket Header File(s)
#include <cerrno>
#include <pthread.h>
#endif

#include <queue>
#include <unordered_map>

#ifndef __has_include
#error "__has_include not supported"
#else
#if __has_include(<source_location>)

#include <source_location>

#define HAS_SOURCE_LOCATION
#elif __has_include(<experimental/source_location>)
#include <experimental/source_location>
#define EXPERIMENTAL_SOURCE_LOCATION
#else
#error "Does not have source location as part of std location or experimental"
#endif
#endif

#include "Viewer/Tools/ThreadPool.h"


namespace Log {
    // Direct Interface for logging into log file or console using MACRO(s)
#define LOG_ERROR(x)    Logger::getInstance()->error(x)
#define LOG_ALARM(x)       Logger::getInstance()->alarm(x)
#define LOG_ALWAYS(x)    Logger::getInstance()->always(x)
#define LOG_INFO(x)     Logger::getInstance()->info(x)
#define LOG_TRACE(x)    Logger::getInstance()->trace(x)
#define LOG_DEBUG(x)    Logger::getInstance()->debug(x)

    // enum for LOG_LEVEL
    typedef enum LOG_LEVEL {
        DISABLE_LOG = 1,
        LOG_LEVEL_INFO = 2,
        LOG_LEVEL_BUFFER = 3,
        LOG_LEVEL_TRACE = 4,
        LOG_LEVEL_DEBUG = 5,
        ENABLE_LOG = 6,
    } LogLevel;


    // enum for LOG_TYPE
    typedef enum LOG_TYPE {
        NO_LOG = 1,
        CONSOLE = 2,
        FILE_LOG = 3,
    } LogType;



    struct FormatString {
        fmt::string_view m_Str;
#ifdef HAS_SOURCE_LOCATION
        std::source_location m_Loc;

        FormatString(const char *str, const std::source_location &loc = std::source_location::current()) : m_Str(str),
                                                                                                           m_Loc(loc) {}

#endif
#ifdef EXPERIMENTAL_SOURCE_LOCATION
        std::experimental::source_location m_Loc;

        FormatString(const char *str, const std::experimental::source_location &loc = std::experimental::source_location::current()) : m_Str(str), m_Loc(loc) {}
#endif
    };


    class Logger {
    public:

        static Logger *getInstance(const std::string &logFileName = "") noexcept;

        void errorInternal(const char *text) noexcept;

        void fatalInternal(const char *text) noexcept;

        /**@brief Using templates to allow user to use formattet logging.
     * @refitem @FormatString Is used to obtain m_Name of calling func, file and line number as default parameter */
        template<typename... Args>
        void error(const FormatString &format, Args &&... args) {
            verror(format, fmt::make_format_args(args...));
        }

        void verror(const FormatString &format, fmt::format_args args) {
            errorInternal(prepareMessage(format, args, frameNumber).c_str());
        }

        template<typename... Args>
        void fatal(const FormatString &format, Args &&... args) {
            vfatal(format, fmt::make_format_args(args...));
        }

        void vfatal(const FormatString &format, fmt::format_args args) {
            fatalInternal(prepareMessage(format, args, frameNumber).c_str());
        }


        void warningInternal(const char *text) noexcept;

        /**@brief Using templates to allow user to use formattet logging.
* @refitem @FormatString Is used to obtain m_Name of calling func, file and line number as default parameter */
        template<typename... Args>
        void warning(const FormatString &format, Args &&... args) {
            vWarning(format, fmt::make_format_args(args...));
        }

        void vWarning(const FormatString &format, fmt::format_args args) {
            warningInternal(prepareMessage(format, args, frameNumber).c_str());
        }

        /**@brief Using templates to allow user to use formattet logging.
         * @refitem @FormatString Is used to obtain name of calling func, file and line number as default parameter */
        template<typename... Args>
        void info(const FormatString &format, Args &&... args) {
            vInfo(format, fmt::make_format_args(args...));
        }

        void vInfo(const FormatString &format, fmt::format_args args) {
            infoInternal(prepareMessage(format, args, frameNumber).c_str());
        }

        template<typename... Args>
        void trace(const FormatString &format, Args &&... args) {
            vTrace(format, fmt::make_format_args(args...));
        }

        /**
         * log trace messages with a specific frequency. frequency == 1 means every frame otherwise every nth frame.
         * @tparam Args
         * @param tag Tag used to count the frequency
         * @param frequency ever nth frame this line should be logged
         * @param format format string
         * @param args args for formatted string
         */
        template<typename... Args>
        void
        traceWithFrequency(const std::string &tag, uint32_t frequency, const FormatString &format, Args &&... args) {
            if (m_frequencies.find(tag) != m_frequencies.end() && m_counter.find(tag) != m_counter.end()) {
                m_counter[tag]++;
            } else {
                m_counter.insert_or_assign(tag, static_cast<uint32_t>(1));
                m_frequencies.insert_or_assign(tag, frequency);
            }
            // I would like to use % == 0, but at least on windows if I do
            // m_counter.insert_or_assign(tag, static_cast<uint32_t>(0)); the m_counter map get initialized with NULL.
            // which makes the next line crash when I reference value in m_counter[tag].
            // Does not happen if I use any other value such as 1. No functional difference but just weird
            // as I cannot use 0 as a value so I am forced to count from 1. Even chatgpt is confused
            if (m_counter[tag] % m_frequencies[tag] == 1) {
                vTrace(tag, format, fmt::make_format_args(args...));
            }
        }

        /**
         * log trace messages with a specific frequency. frequency == 1 means every frame otherwise every nth frame.
         * @tparam Args
         * @param tag Tag used to count the frequency (should be unique across all log levels)
         * @param frequency ever nth frame this line should be logged (Usually the application runs at 60 fps - so value of 60 --> ca. once every second)
         * @param format format string
         * @param args args for formatted string
         */
        template<typename... Args>
        void
        warningWithFrequency(const std::string &tag, uint32_t frequency, const FormatString &format, Args &&... args) {
            if (m_frequencies.find(tag) != m_frequencies.end() && m_counter.find(tag) != m_counter.end()) {
                m_counter[tag]++;
            } else {
                m_counter.insert_or_assign(tag, static_cast<uint32_t>(1));
                m_frequencies.insert_or_assign(tag, frequency);
            }
            // I would like to use % == 0, but at least on windows if I do
            // m_counter.insert_or_assign(tag, static_cast<uint32_t>(0)); the m_counter map get initialized with NULL.
            // which makes the next line crash when I reference value in m_counter[tag].
            // Does not happen if I use any other value such as 1. No functional difference but just weird
            // as I cannot use 0 as a value so I am forced to count from 1. Even chatgpt is confused
            if (m_counter[tag] % m_frequencies[tag] == 1) {
                vWarning(tag, format, fmt::make_format_args(args...));
            }
        }

        void vTrace(const FormatString &format, fmt::format_args args) {
            traceInternal(prepareMessage(format, args, frameNumber).c_str());
        }

        void vTrace(const std::string &tag, const FormatString &format, fmt::format_args args) {
            traceInternal(prepareMessage(format, args, frameNumber, tag).c_str());
        }

        void vWarning(const std::string &tag, const FormatString &format, fmt::format_args args) {
            warningInternal(prepareMessage(format, args, frameNumber, tag).c_str());
        }


        uint32_t frameNumber = 0;

        void operator=(const Logger &obj) = delete;

        void always(std::string text) noexcept;

        void setLogLevel(LogLevel logLevel);

    protected:
        Logger(const std::string &logFileName);

        ~Logger();

        static std::string getCurrentTime();

    private:

        static inline std::string getLogStringFromEnum(const Log::LOG_LEVEL &logEnum) {
            switch (logEnum) {
                case Log::LOG_LEVEL_INFO:
                    return "LOG_INFO";
                case Log::LOG_LEVEL_TRACE:
                    return "LOG_TRACE";
                case Log::LOG_LEVEL_DEBUG:
                    return "LOG_DEBUG";
                default:
                    return "LOG_INFO";
            }
        };

        void infoInternal(const char *text) noexcept;

        void traceInternal(const char *text) noexcept;

        static void logOnConsole(std::string &data);

        static void logIntoFile(void *ctx, std::string &data);

        static std::string filterFilePath(const std::string &input);

        static inline std::string
        prepareMessage(const FormatString &format, fmt::format_args args, uint32_t frameNumber,
                       const std::string &tag = "") {
            const auto &loc = format.m_Loc;
            std::string s;
            fmt::vformat_to(std::back_inserter(s), format.m_Str, args);
            std::string filePath;
            if (tag.empty())
                filePath = fmt::format("{}:{}: ", loc.file_name(), loc.line());
            else
                filePath = fmt::format("{}: {}-{}: ", loc.file_name(), tag, loc.line());
            std::string msg = filterFilePath(filePath);
            msg.append(s);
            msg = msg.insert(0, (std::to_string(frameNumber) + "  "));
            return msg;
        }

        static Logger *m_instance;
        static VkRender::ThreadPool *m_threadPool;
        std::ofstream m_file;
        std::map<std::string, uint32_t> m_frequencies;
        std::map<std::string, uint32_t> m_counter;

        std::mutex m_mutex{};

        LogLevel m_logLevel{};
        LogType m_logType{};

    };

} // End of namespace

#endif // End of _LOGGER_H_

