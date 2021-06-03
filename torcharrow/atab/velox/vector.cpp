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
#include "vector.h"
#include <f4d/common/memory/Memory.h>
#include <f4d/type/Type.h>
#include <f4d/vector/BaseVector.h>
#include <f4d/vector/ComplexVector.h>
#include <iostream>

using namespace facebook::f4d;
using namespace facebook::torcharrow;

template <TypeKind kind>
VectorPtr facebook::torcharrow::simpleVectorSlice(
    const BaseVector& src,
    int start,
    int end) {
  using T = typename TypeTraits<kind>::NativeType;
  auto newVector =
      BaseVector::create(CppToType<T>::create(), end - start, src.pool());
  newVector.get()->copy(&src, 0, start, end - start);
  return newVector;
}

VectorPtr facebook::torcharrow::arrayVectorSlice(
    const ArrayVector& src,
    int start,
    int end) {
  auto length = end - start;
  std::shared_ptr<const Type> elementType = src.type();
  auto result = BaseVector::create(ARRAY(elementType), length, src.pool());
  auto ptr = result.get()->as<ArrayVector>();
  if (length > 0) {
    ptr->setElements(vectorSlice(
        *src.elements(),
        src.offsetAt(start),
        src.offsetAt(end - 1) + src.sizeAt(end - 1)));
  }

  for (int i = 0; i < length; i++) {
    auto isNull = src.isNullAt(start + i);
    ptr->setNull(i, isNull);
    if (!isNull) {
      auto offset = src.offsetAt(start + i) - src.offsetAt(start);
      auto size = src.sizeAt(start + i);
      ptr->setOffsetAndSize(i, offset, size);
    }
  }

  return result;
}

VectorPtr
facebook::torcharrow::vectorSlice(const BaseVector& src, int start, int end) {
  auto type = src.type();
  auto kind = type.get()->kind();
  switch (kind) {
    case TypeKind::ARRAY: {
      return arrayVectorSlice(*src.as<ArrayVector>(), start, end);
    }
    case TypeKind::MAP: {
      throw "Not implemented yet.";
    }
    default:
      return F4D_DYNAMIC_SCALAR_TYPE_DISPATCH(
          simpleVectorSlice, kind, src, start, end);
  }
}

VectorPtr facebook::torcharrow::reshape(
    VectorPtr vec,
    std::function<vector_size_t(vector_size_t)> offsets,
    std::function<vector_size_t(vector_size_t)> lengths,
    vector_size_t size) {
  auto result =
      BaseVector::create(ARRAY(vec.get()->type()), size, vec.get()->pool());
  result.get()->as<ArrayVector>()->setElements(vec);
  for (int i = 0; i < size; i++) {
    result.get()->as<ArrayVector>()->setOffsetAndSize(
        i, offsets(i), lengths(i));
  }
  return result;
}
