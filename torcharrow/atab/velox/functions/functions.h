/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "f4d/exec/tests/utils/FunctionUtils.h"
#include "f4d/functions/lib/Re2Functions.h"
#include "f4d/parse/Expressions.h"
#include "string_functions.h"

namespace facebook::torcharrow::functions {

inline void registerTorchArrowFunctions() {
  facebook::velox::registerFunction<
      facebook::torcharrow::functions::udf_torcharrow_isalpha,
      bool,
      facebook::velox::Varchar>();
  facebook::velox::registerFunction<
      facebook::torcharrow::functions::udf_torcharrow_isalnum,
      bool,
      facebook::velox::Varchar>();
  facebook::velox::registerFunction<
      facebook::torcharrow::functions::udf_torcharrow_isinteger,
      bool,
      facebook::velox::Varchar>();

  facebook::velox::exec::registerStatefulVectorFunction(
      "match_re",
      velox::functions::re2MatchSignatures(),
      velox::functions::makeRe2Match);
}

inline void initializeTorchArrowTypeResolver() {
  facebook::velox::exec::test::registerTypeResolver();
}
}
