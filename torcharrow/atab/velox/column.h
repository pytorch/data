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

#include "f4d/type/Type.h"
#include "f4d/vector/BaseVector.h"
#include "f4d/vector/FlatVector.h"
#include "f4d/vector/ComplexVector.h"
#include "vector.h"

namespace facebook {

using namespace f4d;

namespace torcharrow {

class NotAppendableException : public std::exception {
 public:
  virtual const char* what() const throw() {
    return "Cannot append in a view";
  }
};

class BaseColumn {
  friend class ArrayColumn;
  friend class MapColumn;
  friend class RowColumn;

 protected:
  VectorPtr _delegate;
  vector_size_t _offset;
  vector_size_t _length;
  vector_size_t _nullCount;

  void bumpLength() {
    _length++;
    _delegate.get()->resize(_offset + _length);
  }

  bool isAppendable() {
    return _offset + _length == _delegate.get()->size();
  }

 private:
  f4d::memory::MemoryPool* pool_ =
      &memory::getProcessDefaultMemoryManager().getRoot();

 public:
  BaseColumn(
      const BaseColumn& other,
      vector_size_t offset,
      vector_size_t length)
      : _delegate(other._delegate), _offset(offset), _length(length) {
    _nullCount = 0;
    for (int i = 0; i < length; i++) {
      if (_delegate.get()->isNullAt(_offset + i)) {
        _nullCount++;
      }
    }
  }
  BaseColumn(TypePtr type) : _offset(0), _length(0), _nullCount(0) {
    _delegate = BaseVector::create(type, 0, pool_);
  }
  BaseColumn(VectorPtr delegate)
      : _delegate(delegate),
        _offset(0),
        _length(delegate.get()->size()),
        _nullCount(delegate.get()->getNullCount().value_or(0)) {}

  virtual ~BaseColumn() = default;

  TypePtr type() {
    return _delegate.get()->type();
  }

  bool isNullAt(vector_size_t idx) const {
    return _delegate.get()->isNullAt(_offset + idx);
  }

  vector_size_t getOffset() const {
    return _offset;
  }

  vector_size_t getLength() const {
    return _length;
  }

  vector_size_t getNullCount() const {
    return _nullCount;
  }
};

std::unique_ptr<BaseColumn> createColumn(VectorPtr vec);

std::unique_ptr<BaseColumn>
createColumn(VectorPtr vec, vector_size_t offset, vector_size_t length);

template <typename T>
class SimpleColumn : public BaseColumn {
 public:
  SimpleColumn() : BaseColumn(CppToType<T>::create()) {}
  SimpleColumn(VectorPtr delegate) : BaseColumn(delegate) {}
  SimpleColumn(
      const SimpleColumn& other,
      vector_size_t offset,
      vector_size_t length)
      : BaseColumn(other, offset, length) {}

  T valueAt(int i) {
    return _delegate.get()->template as<SimpleVector<T>>()->valueAt(_offset + i);
  }

  void append(const T& value) {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto index = _delegate.get()->size();
    auto flatVector = _delegate->asFlatVector<T>();
    flatVector->resize(index + 1);
    flatVector->set(index, value);
    bumpLength();
  }

  void appendNull() {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto index = _delegate.get()->size();
    _delegate->resize(index + 1);
    _delegate->setNull(index, true);
    _nullCount++;
    bumpLength();
  }

  std::unique_ptr<SimpleColumn<T>> slice(
      vector_size_t offset,
      vector_size_t length) {
    return std::make_unique<SimpleColumn<T>>(*this, offset, length);
  }
};

template <typename T>
class FlatColumn : public SimpleColumn<T> {};

class ArrayColumn : public BaseColumn {
 public:
  ArrayColumn(TypePtr type) : BaseColumn(type) {}
  ArrayColumn(VectorPtr delegate) : BaseColumn(delegate) {}
  ArrayColumn(
      const ArrayColumn& other,
      vector_size_t offset,
      vector_size_t length)
      : BaseColumn(other, offset, length) {}

  void appendElement(const BaseColumn& new_element) {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto dataPtr = _delegate.get()->as<ArrayVector>();
    auto elements = dataPtr->elements();
    auto new_index = dataPtr->size();
    dataPtr->resize(new_index + 1);
    auto new_offset = elements.get()->size();
    auto new_size = new_element.getLength();
    elements.get()->append(new_element._delegate.get());
    dataPtr->setOffsetAndSize(new_index, new_offset, new_size);
    bumpLength();
  }

  void appendNull() {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto dataPtr = _delegate.get()->as<ArrayVector>();
    auto elements = dataPtr->elements();
    auto new_index = dataPtr->size();
    dataPtr->resize(new_index + 1);
    auto new_offset = elements.get()->size();
    auto new_size = 0;
    dataPtr->setOffsetAndSize(new_index, new_offset, new_size);
    dataPtr->setNull(new_index, true);
    bumpLength();
  }

  std::unique_ptr<BaseColumn> valueAt(vector_size_t i);

  std::unique_ptr<ArrayColumn> slice(
      vector_size_t offset,
      vector_size_t length) {
    return std::make_unique<ArrayColumn>(*this, offset, length);
  }
};

class MapColumn : public BaseColumn {
 public:
  MapColumn(TypePtr type) : BaseColumn(type) {}
  MapColumn(VectorPtr delegate) : BaseColumn(delegate) {}
  MapColumn(const MapColumn& other, vector_size_t offset, vector_size_t length)
      : BaseColumn(other, offset, length) {}

  vector_size_t offsetAt(vector_size_t index) const {
    return _delegate.get()->as<MapVector>()->offsetAt(_offset + index);
  }

  vector_size_t sizeAt(vector_size_t index) const {
    return _delegate.get()->as<MapVector>()->sizeAt(_offset + index);
  }

  void appendElement(const BaseColumn& newKey, const BaseColumn& newValue) {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto dataPtr = _delegate.get()->as<MapVector>();

    auto keys = dataPtr->mapKeys();
    auto values = dataPtr->mapValues();
    auto new_index = dataPtr->size();
    dataPtr->resize(new_index + 1);
    auto new_offset = keys.get()->size();
    auto new_size = newKey.getLength();
    keys.get()->append(newKey._delegate.get());
    values.get()->append(newValue._delegate.get());
    dataPtr->setOffsetAndSize(new_index, new_offset, new_size);
    bumpLength();
  }

  void appendNull() {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto dataPtr = _delegate.get()->as<MapVector>();

    auto keys = dataPtr->mapKeys();
    auto values = dataPtr->mapValues();
    auto new_index = dataPtr->size();
    dataPtr->resize(new_index + 1);
    auto new_offset = keys.get()->size();
    auto new_size = 0;
    dataPtr->setOffsetAndSize(new_index, new_offset, new_size);
    dataPtr->setNull(new_index, true);
    bumpLength();
  }

  std::unique_ptr<BaseColumn> valueAt(vector_size_t i);

  std::unique_ptr<BaseColumn> mapKeys() {
    auto dataPtr = _delegate.get()->as<MapVector>();
    auto keys = dataPtr->mapKeys();
    auto reshapedKeys = reshape(
        keys,
        std::bind(&MapVector::offsetAt, *dataPtr, std::placeholders::_1),
        std::bind(&MapVector::sizeAt, *dataPtr, std::placeholders::_1),
        dataPtr->size());
    return createColumn(reshapedKeys, _offset, _length);
  }

  std::unique_ptr<BaseColumn> mapValues() {
    auto dataPtr = _delegate.get()->as<MapVector>();
    auto values = dataPtr->mapValues();
    auto reshapedValues = reshape(
        values,
        std::bind(&MapVector::offsetAt, *dataPtr, std::placeholders::_1),
        std::bind(&MapVector::sizeAt, *dataPtr, std::placeholders::_1),
        dataPtr->size());
    return createColumn(reshapedValues, _offset, _length);
  }

  std::unique_ptr<MapColumn> slice(vector_size_t offset, vector_size_t length) {
    return std::make_unique<MapColumn>(*this, offset, length);
  }
};

class RowColumn : public BaseColumn {
 public:
  RowColumn(TypePtr type) : BaseColumn(type) {}
  RowColumn(VectorPtr delegate) : BaseColumn(delegate) {}
  RowColumn(const RowColumn& other, vector_size_t offset, vector_size_t length)
      : BaseColumn(other, offset, length) {}

  std::unique_ptr<BaseColumn> childAt(ChannelIndex index) {
    auto dataPtr = _delegate.get()->as<RowVector>();
    return createColumn(dataPtr->childAt(index), _offset, _length);
  }

  void setChild(ChannelIndex index, const BaseColumn& new_child) {
    auto dataPtr = _delegate.get()->as<RowVector>();
    dataPtr->children()[index] = new_child._delegate;
  }

  size_t childrenSize() {
    auto dataPtr = _delegate.get()->as<RowVector>();
    return dataPtr->childrenSize();
  }

  std::unique_ptr<RowColumn> slice(vector_size_t offset, vector_size_t length) {
    return std::make_unique<RowColumn>(*this, offset, length);
  }

  void setLength(vector_size_t length) {
    _length = length;
    auto dataPtr = _delegate.get()->as<RowVector>();
    dataPtr->resize(_offset + _length);
  }

  void setNullAt(vector_size_t idx) {
    auto dataPtr = _delegate.get()->as<RowVector>();
    if (!isNullAt(idx)) {
      _nullCount++;
    }
    dataPtr->setNull(_offset + idx, true);
  }

  std::unique_ptr<BaseColumn> copy() {
    auto dataPtr = _delegate.get()->as<RowVector>();
    auto newVector = RowVector::createEmpty(dataPtr->type(), dataPtr->pool());
    newVector.get()->resize(dataPtr->size());
    newVector.get()->copy(dataPtr, 0, 0, dataPtr->size());
    auto newColumn = createColumn(newVector, _offset, _length);
    return newColumn;
  }
};

} // namespace torcharrow
} // namespace facebook
