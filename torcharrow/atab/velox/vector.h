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

#include <f4d/type/Type.h>
#include <f4d/vector/BaseVector.h>
#include <f4d/vector/ComplexVector.h>

namespace facebook::torcharrow {

velox::VectorPtr vectorSlice(const velox::BaseVector& src, int start, int end);

template <velox::TypeKind kind>
velox::VectorPtr
simpleVectorSlice(const velox::BaseVector& src, int start, int end);

velox::VectorPtr
arrayVectorSlice(const velox::ArrayVector& src, int start, int end);

velox::VectorPtr reshape(
    velox::VectorPtr vec,
    std::function<velox::vector_size_t(velox::vector_size_t)> offsets,
    std::function<velox::vector_size_t(velox::vector_size_t)> lengths,
    velox::vector_size_t size);

} // namespace facebook::torcharrow
