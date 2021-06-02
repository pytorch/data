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

namespace facebook {

using namespace f4d;

namespace torcharrow {

VectorPtr vectorSlice(const BaseVector& src, int start, int end);

template <TypeKind kind>
VectorPtr simpleVectorSlice(const BaseVector& src, int start, int end);

VectorPtr arrayVectorSlice(const ArrayVector& src, int start, int end);

VectorPtr reshape(
    VectorPtr vec,
    std::function<vector_size_t(vector_size_t)> offsets,
    std::function<vector_size_t(vector_size_t)> lengths,
    vector_size_t size);

} // namespace torcharrow
} // namespace facebook
