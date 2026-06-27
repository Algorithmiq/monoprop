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

#include <quill/core/LogLevel.h>
#include <quill/core/PatternFormatterOptions.h>

namespace monoprop::logging {
monoprop_EXPORT auto setup_quill() -> void;

monoprop_EXPORT auto log_level() -> quill::LogLevel;

monoprop_EXPORT auto log_pattern_formatter_options() -> quill::PatternFormatterOptions;
} // namespace monoprop::logging
