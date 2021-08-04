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

f4d::VectorPtr vectorSlice(const f4d::BaseVector& src, int start, int end);

template <f4d::TypeKind kind>
f4d::VectorPtr
simpleVectorSlice(const f4d::BaseVector& src, int start, int end);

f4d::VectorPtr
arrayVectorSlice(const f4d::ArrayVector& src, int start, int end);

f4d::VectorPtr reshape(
    f4d::VectorPtr vec,
    std::function<f4d::vector_size_t(f4d::vector_size_t)> offsets,
    std::function<f4d::vector_size_t(f4d::vector_size_t)> lengths,
    f4d::vector_size_t size);

} // namespace facebook::torcharrow
