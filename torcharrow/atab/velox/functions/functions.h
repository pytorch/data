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

#include "f4d/parse/Expressions.h"
#include "string_functions.h"

namespace facebook::torcharrow::functions {

inline std::shared_ptr<const f4d::Type> torchArrowTypeResolver(
    const std::vector<std::shared_ptr<const f4d::core::ITypedExpr>>& inputs,
    const std::shared_ptr<const f4d::core::CallExpr>& expr) {
  // Based on
  // https://github.com/facebookexternal/f4d/blob/0706ac98733c0c6349c02a4a4f65d09b0c8209ed/f4d/exec/tests/utils/FunctionUtils.cpp#L67-L72

  std::vector<TypePtr> inputTypes;
  inputTypes.reserve(inputs.size());
  for (auto& input : inputs) {
    inputTypes.emplace_back(input->type());
  }

  auto func =
      f4d::exec::getVectorFunction(expr->getFunctionName(), inputTypes, {});
  if (func) {
    return func->inferType(inputTypes);
  }
  return nullptr;
}

inline void registerTorchArrowFunctions() {
  registerFunction<
      facebook::torcharrow::functions::udf_torcharrow_isalpha,
      bool,
      Varchar>();
}

inline void initializeTorchArrowTypeResolver() {
  f4d::core::Expressions::setTypeResolverHook(&torchArrowTypeResolver);
}
}
