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
#include "column.h"
#include <common/base/VectorOperations.h>
#include <common/memory/Memory.h>
#include <type/Type.h>
#include <vector/BaseVector.h>
#include <vector/ComplexVector.h>

namespace facebook {

using namespace f4d;

namespace torcharrow {

template <TypeKind kind>
std::unique_ptr<BaseColumn>
createSimpleColumn(VectorPtr vec, vector_size_t offset, vector_size_t length) {
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_unique<SimpleColumn<T>>(vec, offset, length);
}

std::unique_ptr<BaseColumn> createColumn(VectorPtr vec) {
  return createColumn(vec, 0, vec.get()->size());
}

std::unique_ptr<BaseColumn>
createColumn(VectorPtr vec, vector_size_t offset, vector_size_t length) {
  auto type = vec.get()->type();
  auto kind = type.get()->kind();
  switch (kind) {
    case TypeKind::ARRAY: {
      return std::make_unique<ArrayColumn>(vec, offset, length);
    }
    case TypeKind::MAP: {
      return std::make_unique<MapColumn>(vec, offset, length);
    }
    case TypeKind::ROW: {
      return std::make_unique<RowColumn>(vec, offset, length);
    }
    default:
      return F4D_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createSimpleColumn, kind, vec, offset, length);
  }
}

std::unique_ptr<BaseColumn> ArrayColumn::valueAt(vector_size_t i) {
  TypePtr elementType = type()->as<TypeKind::ARRAY>().elementType();
  auto dataPtr = _delegate.get()->as<ArrayVector>();
  auto elements = dataPtr->elements();
  auto start = dataPtr->offsetAt(_offset + i);
  auto end = dataPtr->offsetAt(_offset + i) + dataPtr->sizeAt(_offset + i);
  auto sliceResult = vectorSlice(*elements.get(), start, end);
  return createColumn(sliceResult);
}

std::unique_ptr<BaseColumn> MapColumn::valueAt(vector_size_t i) {
  TypePtr keyType = type()->as<TypeKind::MAP>().keyType();
  TypePtr valueType = type()->as<TypeKind::MAP>().valueType();
  auto dataPtr = _delegate.get()->as<MapVector>();
  auto keys = dataPtr->mapKeys();
  auto values = dataPtr->mapValues();
  auto start = dataPtr->offsetAt(_offset + i);
  auto end = dataPtr->offsetAt(_offset + i) + dataPtr->sizeAt(_offset + i);
  auto slicedKeys = vectorSlice(*keys.get(), start, end);
  auto slicedValues = vectorSlice(*values.get(), start, end);
  auto slicedResult = BaseVector::create(type(), 1, pool_);
  slicedResult.get()->as<MapVector>()->setKeysAndValues(
      slicedKeys, slicedValues);
  return createColumn(slicedResult);
}

} // namespace torcharrow

} // namespace facebook
