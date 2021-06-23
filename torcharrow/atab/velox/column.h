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

#include <f4d/common/memory/Memory.h>
#include <f4d/core/QueryCtx.h>
#include <memory>
#include "f4d/type/Type.h"
#include "f4d/common/base/Exceptions.h"
#include "f4d/exec/Expr.h"
#include "f4d/type/Type.h"
#include "f4d/vector/BaseVector.h"
#include "f4d/vector/ComplexVector.h"
#include "f4d/vector/FlatVector.h"
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

struct TorchArrowGlobalStatic {
  static core::QueryCtx& queryContext();
  static core::ExecCtx& execContext();

  static f4d::memory::MemoryPool* rootMemoryPool() {
     static f4d::memory::MemoryPool* const pool =
        &memory::getProcessDefaultMemoryManager().getRoot();
    return pool;
  }
};

class BaseColumn;

struct OperatorHandle {
  RowTypePtr inputRowType_;
  std::shared_ptr<exec::ExprSet> exprSet_;

  OperatorHandle(
      RowTypePtr inputRowType,
      std::shared_ptr<exec::ExprSet> exprSet)
      : inputRowType_(inputRowType),
        exprSet_(exprSet){}

  static std::unique_ptr<OperatorHandle> fromExpression(RowTypePtr inputRowType, const std::string& expr);

  static RowVectorPtr wrapRowVector(
      const std::vector<VectorPtr>& children,
      std::shared_ptr<const RowType> rowType) {
    return std::make_shared<RowVector>(
        TorchArrowGlobalStatic::rootMemoryPool(),
        rowType,
        BufferPtr(nullptr),
        children[0]->size(),
        children,
        folly::none);
  }

  // Input type VectorPtr (instead of BaseColumn) since it might be a ConstantVector
  std::unique_ptr<BaseColumn> call(VectorPtr a, VectorPtr b);

  std::unique_ptr<BaseColumn> call(const std::vector<VectorPtr>& args);
};

class BaseColumn {
  friend class ArrayColumn;
  friend class MapColumn;
  friend class RowColumn;
  friend struct OperatorHandle;

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

  // TODO: move this method as static...
  RowVectorPtr wrapRowVector(
      const std::vector<VectorPtr>& children,
      std::shared_ptr<const RowType> rowType) {
    return std::make_shared<RowVector>(
        pool_,
        rowType,
        BufferPtr(nullptr),
        children[0]->size(),
        children,
        folly::none);
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

  TypePtr type() const {
    return _delegate->type();
  }

  bool isNullAt(vector_size_t idx) const {
    return _delegate->isNullAt(_offset + idx);
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

  VectorPtr getUnderlyingVeloxVector() const {
    return _delegate;
  }

  // TODO: add output type
  static std::shared_ptr<exec::ExprSet> genUnaryExprSet(
      // input row type is required even for unary op since the input vector
      // needs to be wrapped into a RowVector before evaluation.
      std::shared_ptr<const facebook::f4d::RowType> inputRowType,
      TypePtr outputType,
      const std::string& functionName);

  std::unique_ptr<BaseColumn> applyUnaryExprSet(
      // input row type is required even for unary op since the input vector
      // needs to be wrapped into a RowVector before evaluation.
      std::shared_ptr<const facebook::f4d::RowType> inputRowType,
      std::shared_ptr<exec::ExprSet> exprSet);

  static std::shared_ptr<exec::ExprSet> genBinaryExprSet(
      std::shared_ptr<const facebook::f4d::RowType> inputRowType,
      std::shared_ptr<const facebook::f4d::Type> commonType,
      const std::string& functionName);

  // From f4d/type/Variant.h
  // TODO: refactor into some type utility class
  template <TypeKind Kind>
  static const std::shared_ptr<const Type> kind2type() {
    return TypeFactory<Kind>::create();
  }

  static TypeKind promoteNumericTypeKind(TypeKind a, TypeKind b) {
    constexpr auto b1 = TypeKind::BOOLEAN;
    constexpr auto i1 = TypeKind::TINYINT;
    constexpr auto i2 = TypeKind::SMALLINT;
    constexpr auto i4 = TypeKind::INTEGER;
    constexpr auto i8 = TypeKind::BIGINT;
    constexpr auto f4 = TypeKind::REAL;
    constexpr auto f8 = TypeKind::DOUBLE;
    constexpr auto num_numeric_types = static_cast<int>(TypeKind::DOUBLE) + 1;

    VELOX_CHECK(
        static_cast<int>(a) < num_numeric_types &&
        static_cast<int>(b) < num_numeric_types);

    if (a == b) {
      return a;
    }

    // Sliced from
    // https://github.com/pytorch/pytorch/blob/1c502d1f8ec861c31a08d580ae7b73b7fbebebed/c10/core/ScalarType.h#L402-L421
    static constexpr TypeKind
        _promoteTypesLookup[num_numeric_types][num_numeric_types] = {
            /*        b1  i1  i2  i4  i8  f4  f8*/
            /* b1 */ {b1, i1, i2, i4, i8, f4, f8},
            /* i1 */ {i1, i1, i2, i4, i8, f4, f8},
            /* i2 */ {i2, i2, i2, i4, i8, f4, f8},
            /* i4 */ {i4, i4, i4, i4, i8, f4, f8},
            /* i8 */ {i8, i8, i8, i8, i8, f4, f8},
            /* f4 */ {f4, f4, f4, f4, f4, f4, f8},
            /* f8 */ {f8, f8, f8, f8, f8, f8, f8},
        };
    return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
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
    return _delegate.get()->template as<SimpleVector<T>>()->valueAt(
        _offset + i);
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

  //
  // unary numeric column ops
  //

  // TODO: return SimpleColumn<T> instead?
  std::unique_ptr<BaseColumn> invert() {
    const static auto inputRowType = ROW({"c0"}, {CppToType<T>::create()});
    const static auto exprSet =
        BaseColumn::genUnaryExprSet(inputRowType, CppToType<T>::create(), "not");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> neg() {
    const static auto inputRowType = ROW({"c0"}, {CppToType<T>::create()});
    const static auto exprSet =
        BaseColumn::genUnaryExprSet(inputRowType, CppToType<T>::create(), "negate");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> abs() {
    const static auto inputRowType = ROW({"c0"}, {CppToType<T>::create()});
    const static auto exprSet =
        BaseColumn::genUnaryExprSet(inputRowType, CppToType<T>::create(), "abs");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> ceil() {
    const static auto inputRowType = ROW({"c0"}, {CppToType<T>::create()});
    const static auto exprSet = BaseColumn::genUnaryExprSet(
        inputRowType, CppToType<T>::create(), "ceil");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> floor() {
    const static auto inputRowType = ROW({"c0"}, {CppToType<T>::create()});
    const static auto exprSet = BaseColumn::genUnaryExprSet(
        inputRowType, CppToType<T>::create(), "floor");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> round() {
    const static auto inputRowType = ROW({"c0"}, {CppToType<T>::create()});
    const static auto exprSet = BaseColumn::genUnaryExprSet(
        inputRowType, CppToType<T>::create(), "round");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  //
  // binary numeric column ops
  //

  // TODO: Model binary functions as UDF.
  std::unique_ptr<OperatorHandle> createBinaryOperatorHandle(
      TypePtr otherType,
      const std::string& functionName) {
    auto inputRowType = ROW({"c0", "c1"}, {this->type(), otherType});
    TypeKind commonTypeKind =
        promoteNumericTypeKind(this->type()->kind(), otherType->kind());
    auto commonType =
        F4D_DYNAMIC_SCALAR_TYPE_DISPATCH(kind2type, commonTypeKind);
    auto exprSet =
        BaseColumn::genBinaryExprSet(inputRowType, commonType, functionName);

    return std::make_unique<OperatorHandle>(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> add(const BaseColumn& other) {
    constexpr auto num_numeric_types = static_cast<int>(TypeKind::DOUBLE) + 1;
    static std::array<
        std::unique_ptr<OperatorHandle>,
        num_numeric_types> /* library-local */ ops;

    int dispatch_id = static_cast<int>(other.type()->kind());
    if (ops[dispatch_id] == nullptr) {
      ops[dispatch_id] = createBinaryOperatorHandle(other.type(), "plus");
    }

    auto result = ops[dispatch_id]->call(_delegate, other.getUnderlyingVeloxVector());

    return result;
  }

  template <typename ScalarCppType>
  std::unique_ptr<BaseColumn> addScalar(ScalarCppType val) {
    const static auto scalarType = CppToType<ScalarCppType>::create();
    // TODO: this is incorrect, since tensor-scalar type promotion rule is
    // different from tensor-tensor type promotion rule
    const static auto operatorHandle =
        createBinaryOperatorHandle(scalarType, "plus");

    auto constantVectorPtr = std::make_shared<ConstantVector<ScalarCppType>>(
        TorchArrowGlobalStatic::rootMemoryPool(),
        _delegate->size(),
        false /* isNull */,
        std::move(val),
        cdvi::EMPTY_METADATA,
        folly::none /* representedBytes */);

    auto result = operatorHandle->call(_delegate, constantVectorPtr);
    return result;
  }

  //
  // string column ops
  //
  std::unique_ptr<BaseColumn> lower() {
    static_assert(
        std::is_same<StringView, T>(),
        "lower should only be called over VARCHAR column");

    const static auto inputRowType = ROW({"c0"}, {CppToType<T>::create()});
    const static auto op =
        OperatorHandle::fromExpression(inputRowType, "lower(c0)");
    return op->call({_delegate});
  }

  std::unique_ptr<BaseColumn> upper() {
    static_assert(
        std::is_same<StringView, T>(),
        "upper should only be called over VARCHAR column");

    const static auto inputRowType = ROW({"c0"}, {CppToType<T>::create()});
    const static auto op =
        OperatorHandle::fromExpression(inputRowType, "upper(c0)");
    return op->call({_delegate});
  }

  std::unique_ptr<BaseColumn> isalpha() {
    static_assert(
        std::is_same<StringView, T>(),
        "isalpha should only be called over VARCHAR column");

    const static auto inputRowType = ROW({"c0"}, {CppToType<T>::create()});
    const static auto op =
        OperatorHandle::fromExpression(inputRowType, "torcharrow_isalpha(c0)");
    return op->call({_delegate});
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
