// Copyright 2026 Algorithmiq
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "monoprop/monopropExport.h"

#include <quill/LogMacros.h>
#include <quill/Logger.h>

// the global logger is defined in QuillWrapper.cpp
extern monoprop_EXPORT quill::Logger* monoprop_global_logger;

// custom log macros using monoprop_global_logger
// standard
#define monoprop_LOG_TRACE_L3(fmt, ...)  QUILL_LOG_TRACE_L3(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOG_TRACE_L2(fmt, ...)  QUILL_LOG_TRACE_L2(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOG_TRACE_L1(fmt, ...)  QUILL_LOG_TRACE_L1(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOG_DEBUG(fmt, ...)     QUILL_LOG_DEBUG(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOG_INFO(fmt, ...)      QUILL_LOG_INFO(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOG_NOTICE(fmt, ...)    QUILL_LOG_NOTICE(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOG_WARNING(fmt, ...)   QUILL_LOG_WARNING(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOG_ERROR(fmt, ...)     QUILL_LOG_ERROR(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOG_CRITICAL(fmt, ...)  QUILL_LOG_CRITICAL(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOG_BACKTRACE(fmt, ...) QUILL_LOG_BACKTRACE(monoprop_global_logger, fmt, ##__VA_ARGS__)

// value-based
#define monoprop_LOGV_TRACE_L3(fmt, ...)  QUILL_LOGV_TRACE_L3(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGV_TRACE_L2(fmt, ...)  QUILL_LOGV_TRACE_L2(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGV_TRACE_L1(fmt, ...)  QUILL_LOGV_TRACE_L1(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGV_DEBUG(fmt, ...)     QUILL_LOGV_DEBUG(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGV_INFO(fmt, ...)      QUILL_LOGV_INFO(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGV_NOTICE(fmt, ...)    QUILL_LOGV_NOTICE(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGV_WARNING(fmt, ...)   QUILL_LOGV_WARNING(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGV_ERROR(fmt, ...)     QUILL_LOGV_ERROR(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGV_CRITICAL(fmt, ...)  QUILL_LOGV_CRITICAL(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGV_BACKTRACE(fmt, ...) QUILL_LOGV_BACKTRACE(monoprop_global_logger, fmt, ##__VA_ARGS__)

// JSON
#define monoprop_LOGJ_TRACE_L3(fmt, ...)  QUILL_LOGJ_TRACE_L3(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGJ_TRACE_L2(fmt, ...)  QUILL_LOGJ_TRACE_L2(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGJ_TRACE_L1(fmt, ...)  QUILL_LOGJ_TRACE_L1(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGJ_DEBUG(fmt, ...)     QUILL_LOGJ_DEBUG(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGJ_INFO(fmt, ...)      QUILL_LOGJ_INFO(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGJ_NOTICE(fmt, ...)    QUILL_LOGJ_NOTICE(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGJ_WARNING(fmt, ...)   QUILL_LOGJ_WARNING(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGJ_ERROR(fmt, ...)     QUILL_LOGJ_ERROR(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGJ_CRITICAL(fmt, ...)  QUILL_LOGJ_CRITICAL(monoprop_global_logger, fmt, ##__VA_ARGS__)
#define monoprop_LOGJ_BACKTRACE(fmt, ...) QUILL_LOGJ_BACKTRACE(monoprop_global_logger, fmt, ##__VA_ARGS__)
