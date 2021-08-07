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
#include <pybind11/pybind11.h>
#include <memory>
#include <string>
#include <unordered_map>
#include "f4d/common/base/Exceptions.h"
#include "f4d/expression/Expr.h"
#include "f4d/type/Type.h"
#include "f4d/vector/BaseVector.h"
#include "f4d/vector/ComplexVector.h"
#include "f4d/vector/FlatVector.h"
#include "vector.h"

// TODO: Move uses of static variables into .cpp. Static variables are local to
// the compilation units so every file that includes this header will have its
// own instance of the static variables, which in most cases is not what we want

namespace facebook::torcharrow {

class NotAppendableException : public std::exception {
 public:
  virtual const char* what() const throw() {
    return "Cannot append in a view";
  }
};

struct TorchArrowGlobalStatic {
  static velox::core::QueryCtx& queryContext();
  static velox::core::ExecCtx& execContext();

  static velox::memory::MemoryPool* rootMemoryPool() {
    static velox::memory::MemoryPool* const pool =
        &velox::memory::getProcessDefaultMemoryManager().getRoot();
    return pool;
  }
};

struct GenericUDFDispatchKey {
  std::string udfName;
  // TODO: use row type instead of string
  std::string typeSignature;

  GenericUDFDispatchKey(std::string udfName, std::string typeSignature)
      : udfName(std::move(udfName)), typeSignature(std::move(typeSignature)) {}
};

inline bool operator==(
    const GenericUDFDispatchKey& lhs,
    const GenericUDFDispatchKey& rhs) {
  return lhs.udfName == rhs.udfName && lhs.typeSignature == rhs.typeSignature;
}

velox::variant pyToVariant(const pybind11::handle& obj);

class BaseColumn;

struct OperatorHandle {
  velox::RowTypePtr inputRowType_;
  std::shared_ptr<velox::exec::ExprSet> exprSet_;

  OperatorHandle(
      velox::RowTypePtr inputRowType,
      std::shared_ptr<velox::exec::ExprSet> exprSet)
      : inputRowType_(inputRowType), exprSet_(exprSet) {}

  static std::unique_ptr<OperatorHandle> fromGenericUDF(
      velox::RowTypePtr inputRowType,
      const std::string& udfName);

  static std::unique_ptr<OperatorHandle> fromExpression(
      velox::RowTypePtr inputRowType,
      const std::string& expr);

  static velox::RowVectorPtr wrapRowVector(
      const std::vector<velox::VectorPtr>& children,
      std::shared_ptr<const velox::RowType> rowType) {
    return std::make_shared<velox::RowVector>(
        TorchArrowGlobalStatic::rootMemoryPool(),
        rowType,
        velox::BufferPtr(nullptr),
        children[0]->size(),
        children,
        folly::none);
  }

  // Specialized invoke methods for common arities
  // Input type velox::VectorPtr (instead of BaseColumn) since it might be a
  // ConstantVector
  // TODO: Use Column once ConstantColumn is supported
  std::unique_ptr<BaseColumn> call(velox::VectorPtr a, velox::VectorPtr b);

  std::unique_ptr<BaseColumn> call(const std::vector<velox::VectorPtr>& args);
};

class PromoteNumericTypeKind {
 public:
  static velox::TypeKind promoteColumnColumn(velox::TypeKind a, velox::TypeKind b) {
    return promote(a, b, PromoteStrategy::ColumnColumn);
  }

  // Assume a being a column and b being a scalar
  static velox::TypeKind promoteColumnScalar(velox::TypeKind a, velox::TypeKind b) {
    return promote(a, b, PromoteStrategy::ColumnScalar);
  }

 private:
  enum class PromoteStrategy {
    ColumnColumn,
    ColumnScalar,
  };

  static velox::TypeKind
  promote(velox::TypeKind a, velox::TypeKind b, PromoteStrategy promoteStrategy) {
    constexpr auto b1 = velox::TypeKind::BOOLEAN;
    constexpr auto i1 = velox::TypeKind::TINYINT;
    constexpr auto i2 = velox::TypeKind::SMALLINT;
    constexpr auto i4 = velox::TypeKind::INTEGER;
    constexpr auto i8 = velox::TypeKind::BIGINT;
    constexpr auto f4 = velox::TypeKind::REAL;
    constexpr auto f8 = velox::TypeKind::DOUBLE;
    constexpr auto num_numeric_types =
        static_cast<int>(velox::TypeKind::DOUBLE) + 1;

    VELOX_CHECK(
        static_cast<int>(a) < num_numeric_types &&
        static_cast<int>(b) < num_numeric_types);

    if (a == b) {
      return a;
    }

    switch (promoteStrategy) {
      case PromoteStrategy::ColumnColumn: {
        // Sliced from
        // https://github.com/pytorch/pytorch/blob/1c502d1f8ec861c31a08d580ae7b73b7fbebebed/c10/core/ScalarType.h#L402-L421
        static constexpr velox::TypeKind
            promoteTypesLookup[num_numeric_types][num_numeric_types] = {
                /*        b1  i1  i2  i4  i8  f4  f8*/
                /* b1 */ {b1, i1, i2, i4, i8, f4, f8},
                /* i1 */ {i1, i1, i2, i4, i8, f4, f8},
                /* i2 */ {i2, i2, i2, i4, i8, f4, f8},
                /* i4 */ {i4, i4, i4, i4, i8, f4, f8},
                /* i8 */ {i8, i8, i8, i8, i8, f4, f8},
                /* f4 */ {f4, f4, f4, f4, f4, f4, f8},
                /* f8 */ {f8, f8, f8, f8, f8, f8, f8},
            };
        return promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
      } break;
      case PromoteStrategy::ColumnScalar: {
        // TODO: Decide on how we want to handle column-scalar type promotion.
        // Current strategy is to always respect the type of the column for
        // int-int cases.
        static constexpr velox::TypeKind
            promoteTypesLookup[num_numeric_types][num_numeric_types] = {
                /*        b1  i1  i2  i4  i8  f4  f8*/
                /* b1 */ {b1, b1, b1, b1, b1, f4, f8},
                /* i1 */ {i1, i1, i1, i1, i1, f4, f8},
                /* i2 */ {i2, i2, i2, i2, i2, f4, f8},
                /* i4 */ {i4, i4, i4, i4, i4, f4, f8},
                /* i8 */ {i8, i8, i8, i8, i8, f4, f8},
                /* f4 */ {f4, f4, f4, f4, f4, f4, f8},
                /* f8 */ {f8, f8, f8, f8, f8, f8, f8},
            };
        return promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
      } break;
      default: {
        throw std::logic_error(
            "Unsupported promote: " +
            std::to_string(static_cast<int64_t>(promoteStrategy)));
      } break;
    }
  }
};

class BaseColumn {
  friend class ArrayColumn;
  friend class MapColumn;
  friend class RowColumn;
  friend struct OperatorHandle;

 protected:
  velox::VectorPtr _delegate;
  velox::vector_size_t _offset;
  velox::vector_size_t _length;
  velox::vector_size_t _nullCount;

  void bumpLength() {
    _length++;
    _delegate.get()->resize(_offset + _length);
  }

  bool isAppendable() {
    return _offset + _length == _delegate.get()->size();
  }

  // TODO: move this method as static...
  velox::RowVectorPtr wrapRowVector(
      const std::vector<velox::VectorPtr>& children,
      std::shared_ptr<const velox::RowType> rowType) {
    return std::make_shared<velox::RowVector>(
        pool_,
        rowType,
        velox::BufferPtr(nullptr),
        children[0]->size(),
        children,
        folly::none);
  }

 private:
  velox::memory::MemoryPool* pool_ =
      &velox::memory::getProcessDefaultMemoryManager().getRoot();

 public:
  BaseColumn(
      const BaseColumn& other,
      velox::vector_size_t offset,
      velox::vector_size_t length)
      : _delegate(other._delegate), _offset(offset), _length(length) {
    _nullCount = 0;
    for (int i = 0; i < length; i++) {
      if (_delegate.get()->isNullAt(_offset + i)) {
        _nullCount++;
      }
    }
  }
  explicit BaseColumn(velox::TypePtr type)
      : _offset(0), _length(0), _nullCount(0) {
    _delegate = velox::BaseVector::create(type, 0, pool_);
  }
  explicit BaseColumn(velox::VectorPtr delegate)
      : _delegate(delegate),
        _offset(0),
        _length(delegate.get()->size()),
        _nullCount(delegate.get()->getNullCount().value_or(0)) {}

  virtual ~BaseColumn() = default;

  velox::TypePtr type() const {
    return _delegate->type();
  }

  bool isNullAt(velox::vector_size_t idx) const {
    return _delegate->isNullAt(_offset + idx);
  }

  velox::vector_size_t getOffset() const {
    return _offset;
  }

  velox::vector_size_t getLength() const {
    return _length;
  }

  velox::vector_size_t getNullCount() const {
    return _nullCount;
  }

  velox::VectorPtr getUnderlyingVeloxVector() const {
    return _delegate;
  }

  // TODO: add output type
  static std::shared_ptr<velox::exec::ExprSet> genUnaryExprSet(
      // input row type is required even for unary op since the input vector
      // needs to be wrapped into a velox::RowVector before evaluation.
      std::shared_ptr<const velox::RowType> inputRowType,
      velox::TypePtr outputType,
      const std::string& functionName);

  std::unique_ptr<BaseColumn> applyUnaryExprSet(
      // input row type is required even for unary op since the input vector
      // needs to be wrapped into a velox::RowVector before evaluation.
      std::shared_ptr<const velox::RowType> inputRowType,
      std::shared_ptr<velox::exec::ExprSet> exprSet);

  static std::shared_ptr<velox::exec::ExprSet> genBinaryExprSet(
      std::shared_ptr<const velox::RowType> inputRowType,
      std::shared_ptr<const velox::Type> commonType,
      const std::string& functionName);

  // From f4d/type/velox::variant.h
  // TODO: refactor into some type utility class
  template <velox::TypeKind Kind>
  static const std::shared_ptr<const velox::Type> kind2type() {
    return velox::TypeFactory<Kind>::create();
  }

 public:
  // generic UDF
  static std::unique_ptr<BaseColumn> genericUnaryUDF(
      const std::string& udfName,
      const BaseColumn& col1);

  static std::unique_ptr<BaseColumn> genericBinaryUDF(
      const std::string& udfName,
      const BaseColumn& col1,
      const BaseColumn& col2);

  // factory methods to create columns
  static std::unique_ptr<BaseColumn> createConstantColumn(
      velox::variant value,
      velox::vector_size_t size);
};

std::unique_ptr<BaseColumn> createColumn(velox::VectorPtr vec);

std::unique_ptr<BaseColumn> createColumn(
    velox::VectorPtr vec,
    velox::vector_size_t offset,
    velox::vector_size_t length);

template <typename T>
class SimpleColumn : public BaseColumn {
 public:
  SimpleColumn() : BaseColumn(velox::CppToType<T>::create()) {}
  explicit SimpleColumn(velox::VectorPtr delegate) : BaseColumn(delegate) {}
  SimpleColumn(
      const SimpleColumn& other,
      velox::vector_size_t offset,
      velox::vector_size_t length)
      : BaseColumn(other, offset, length) {}

  T valueAt(int i) {
    return _delegate.get()->template as<velox::SimpleVector<T>>()->valueAt(
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
      velox::vector_size_t offset,
      velox::vector_size_t length) {
    return std::make_unique<SimpleColumn<T>>(*this, offset, length);
  }

  //
  // unary numeric column ops
  //

  // TODO: return SimpleColumn<T> instead?
  std::unique_ptr<BaseColumn> invert() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto exprSet = BaseColumn::genUnaryExprSet(
        inputRowType, velox::CppToType<T>::create(), "not");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> neg() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto exprSet = BaseColumn::genUnaryExprSet(
        inputRowType, velox::CppToType<T>::create(), "negate");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> abs() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto exprSet = BaseColumn::genUnaryExprSet(
        inputRowType, velox::CppToType<T>::create(), "abs");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> ceil() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto exprSet = BaseColumn::genUnaryExprSet(
        inputRowType, velox::CppToType<T>::create(), "ceil");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> floor() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto exprSet = BaseColumn::genUnaryExprSet(
        inputRowType, velox::CppToType<T>::create(), "floor");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> round() {
    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto exprSet = BaseColumn::genUnaryExprSet(
        inputRowType, velox::CppToType<T>::create(), "round");
    return this->applyUnaryExprSet(inputRowType, exprSet);
  }

  //
  // binary numeric column ops
  //

 private:
  enum class OpCode : int16_t {
    Plus = 0,
  };

  static std::string opCodeToName(OpCode opCode) {
    switch (opCode) {
      case OpCode::Plus: {
        return "plus";
      } break;
      default: {
        throw std::logic_error(
            "Unsupported OpCode: " +
            std::to_string(static_cast<int16_t>(opCode)));
      } break;
    }
  }

  // TODO: Model binary functions as UDF.
  std::unique_ptr<OperatorHandle> createBinaryOperatorHandle(
      velox::TypePtr otherType,
      velox::TypeKind commonTypeKind,
      OpCode opCode) {
    auto inputRowType = velox::ROW({"c0", "c1"}, {this->type(), otherType});
    auto commonType =
        VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(kind2type, commonTypeKind);
    auto exprSet = BaseColumn::genBinaryExprSet(
        inputRowType, commonType, opCodeToName(opCode));

    return std::make_unique<OperatorHandle>(inputRowType, exprSet);
  }

  std::unique_ptr<BaseColumn> dispatchBinaryOperation(
      velox::VectorPtr other,
      velox::TypeKind commonTypeKind,
      OpCode opCode) {
    // FIXME This is fragile as it assumes velox::TypeKind numbers numeric types
    // starting from 0 and has DOUBLE being the last one
    constexpr size_t num_numeric_types =
        static_cast<size_t>(velox::TypeKind::DOUBLE) + 1;
    constexpr size_t num_ops = static_cast<size_t>(OpCode::Plus) + 1;
    // Indices are [otherTypeKind][commonTypeKind][opCode]
    static std::unique_ptr<OperatorHandle> ops[num_numeric_types]
                                              [num_numeric_types][num_ops];

    size_t id0 = static_cast<size_t>(other->typeKind());
    size_t id1 = static_cast<size_t>(commonTypeKind);
    size_t id2 = static_cast<size_t>(opCode);
    if (ops[id0][id1][id2] == nullptr) {
      ops[id0][id1][id2] =
          createBinaryOperatorHandle(other->type(), commonTypeKind, opCode);
    }

    auto result = ops[id0][id1][id2]->call(_delegate, other);
    return result;
  }

 public:
  std::unique_ptr<BaseColumn> addColumn(const BaseColumn& other) {
    velox::TypeKind commonTypeKind = PromoteNumericTypeKind::promoteColumnColumn(
        this->type()->kind(), other.type()->kind());

    return dispatchBinaryOperation(
        other.getUnderlyingVeloxVector(), commonTypeKind, OpCode::Plus);
  }

  std::unique_ptr<BaseColumn> addScalar(const pybind11::handle& obj) {
    velox::variant val = pyToVariant(obj);
    velox::VectorPtr other = velox::BaseVector::createConstant(
        val, _delegate->size(), TorchArrowGlobalStatic::rootMemoryPool());

    velox::TypeKind commonTypeKind = PromoteNumericTypeKind::promoteColumnScalar(
        this->type()->kind(), other->typeKind());

    return dispatchBinaryOperation(other, commonTypeKind, OpCode::Plus);
  }

  //
  // string column ops
  //
  std::unique_ptr<BaseColumn> lower() {
    static_assert(
        std::is_same<velox::StringView, T>(),
        "lower should only be called over VARCHAR column");

    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto op =
        OperatorHandle::fromGenericUDF(inputRowType, "lower");
    return op->call({_delegate});
  }

  std::unique_ptr<BaseColumn> upper() {
    static_assert(
        std::is_same<velox::StringView, T>(),
        "upper should only be called over VARCHAR column");

    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto op =
        OperatorHandle::fromGenericUDF(inputRowType, "upper");
    return op->call({_delegate});
  }

  std::unique_ptr<BaseColumn> isalpha() {
    static_assert(
        std::is_same<velox::StringView, T>(),
        "isalpha should only be called over VARCHAR column");

    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto op =
        OperatorHandle::fromGenericUDF(inputRowType, "torcharrow_isalpha");
    return op->call({_delegate});
  }

  std::unique_ptr<BaseColumn> isalnum() {
    static_assert(
        std::is_same<velox::StringView, T>(),
        "isalnum should only be called over VARCHAR column");

    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto op =
        OperatorHandle::fromExpression(inputRowType, "torcharrow_isalnum(c0)");
    return op->call({_delegate});
  }

  std::unique_ptr<BaseColumn> isinteger() {
    static_assert(
        std::is_same<velox::StringView, T>(),
        "isinteger should only be called over VARCHAR column");

    const static auto inputRowType =
        velox::ROW({"c0"}, {velox::CppToType<T>::create()});
    const static auto op = OperatorHandle::fromExpression(
        inputRowType, "torcharrow_isinteger(c0)");
    return op->call({_delegate});
  }
};

template <typename T>
class FlatColumn : public SimpleColumn<T> {};

template <typename T>
class ConstantColumn : public SimpleColumn<T> {
 public:
  ConstantColumn(velox::variant value, velox::vector_size_t size)
      : SimpleColumn<T>(velox::BaseVector::createConstant(
            value,
            size,
            TorchArrowGlobalStatic::rootMemoryPool())) {}
};

class ArrayColumn : public BaseColumn {
 public:
  explicit ArrayColumn(velox::TypePtr type) : BaseColumn(type) {}
  explicit ArrayColumn(velox::VectorPtr delegate) : BaseColumn(delegate) {}
  ArrayColumn(
      const ArrayColumn& other,
      velox::vector_size_t offset,
      velox::vector_size_t length)
      : BaseColumn(other, offset, length) {}

  void appendElement(const BaseColumn& new_element) {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto dataPtr = _delegate.get()->as<velox::ArrayVector>();
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
    auto dataPtr = _delegate.get()->as<velox::ArrayVector>();
    auto elements = dataPtr->elements();
    auto new_index = dataPtr->size();
    dataPtr->resize(new_index + 1);
    auto new_offset = elements.get()->size();
    auto new_size = 0;
    dataPtr->setOffsetAndSize(new_index, new_offset, new_size);
    dataPtr->setNull(new_index, true);
    bumpLength();
  }

  std::unique_ptr<BaseColumn> valueAt(velox::vector_size_t i);

  std::unique_ptr<ArrayColumn> slice(
      velox::vector_size_t offset,
      velox::vector_size_t length) {
    return std::make_unique<ArrayColumn>(*this, offset, length);
  }
};

class MapColumn : public BaseColumn {
 public:
  explicit MapColumn(velox::TypePtr type) : BaseColumn(type) {}
  explicit MapColumn(velox::VectorPtr delegate) : BaseColumn(delegate) {}
  MapColumn(
      const MapColumn& other,
      velox::vector_size_t offset,
      velox::vector_size_t length)
      : BaseColumn(other, offset, length) {}

  velox::vector_size_t offsetAt(velox::vector_size_t index) const {
    return _delegate.get()->as<velox::MapVector>()->offsetAt(_offset + index);
  }

  velox::vector_size_t sizeAt(velox::vector_size_t index) const {
    return _delegate.get()->as<velox::MapVector>()->sizeAt(_offset + index);
  }

  void appendElement(const BaseColumn& newKey, const BaseColumn& newValue) {
    if (!isAppendable()) {
      throw NotAppendableException();
    }
    auto dataPtr = _delegate.get()->as<velox::MapVector>();

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
    auto dataPtr = _delegate.get()->as<velox::MapVector>();

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

  std::unique_ptr<BaseColumn> valueAt(velox::vector_size_t i);

  std::unique_ptr<BaseColumn> mapKeys() {
    auto dataPtr = _delegate.get()->as<velox::MapVector>();
    auto keys = dataPtr->mapKeys();
    auto reshapedKeys = reshape(
        keys,
        std::bind(&velox::MapVector::offsetAt, *dataPtr, std::placeholders::_1),
        std::bind(&velox::MapVector::sizeAt, *dataPtr, std::placeholders::_1),
        dataPtr->size());
    return createColumn(reshapedKeys, _offset, _length);
  }

  std::unique_ptr<BaseColumn> mapValues() {
    auto dataPtr = _delegate.get()->as<velox::MapVector>();
    auto values = dataPtr->mapValues();
    auto reshapedValues = reshape(
        values,
        std::bind(&velox::MapVector::offsetAt, *dataPtr, std::placeholders::_1),
        std::bind(&velox::MapVector::sizeAt, *dataPtr, std::placeholders::_1),
        dataPtr->size());
    return createColumn(reshapedValues, _offset, _length);
  }

  std::unique_ptr<MapColumn> slice(
      velox::vector_size_t offset,
      velox::vector_size_t length) {
    return std::make_unique<MapColumn>(*this, offset, length);
  }
};

class RowColumn : public BaseColumn {
 public:
  explicit RowColumn(velox::TypePtr type) : BaseColumn(type) {}
  explicit RowColumn(velox::VectorPtr delegate) : BaseColumn(delegate) {}
  RowColumn(
      const RowColumn& other,
      velox::vector_size_t offset,
      velox::vector_size_t length)
      : BaseColumn(other, offset, length) {}

  std::unique_ptr<BaseColumn> childAt(velox::ChannelIndex index) {
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    return createColumn(dataPtr->childAt(index), _offset, _length);
  }

  void setChild(velox::ChannelIndex index, const BaseColumn& new_child) {
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    dataPtr->children()[index] = new_child._delegate;
  }

  size_t childrenSize() {
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    return dataPtr->childrenSize();
  }

  std::unique_ptr<RowColumn> slice(
      velox::vector_size_t offset,
      velox::vector_size_t length) {
    return std::make_unique<RowColumn>(*this, offset, length);
  }

  void setLength(velox::vector_size_t length) {
    _length = length;
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    dataPtr->resize(_offset + _length);
  }

  void setNullAt(velox::vector_size_t idx) {
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    if (!isNullAt(idx)) {
      _nullCount++;
    }
    dataPtr->setNull(_offset + idx, true);
  }

  std::unique_ptr<BaseColumn> copy() {
    auto dataPtr = _delegate.get()->as<velox::RowVector>();
    auto newVector =
        velox::RowVector::createEmpty(dataPtr->type(), dataPtr->pool());
    newVector.get()->resize(dataPtr->size());
    newVector.get()->copy(dataPtr, 0, 0, dataPtr->size());
    auto newColumn = createColumn(newVector, _offset, _length);
    return newColumn;
  }
};

} // namespace facebook::torcharrow

namespace std {
template <>
struct hash<::facebook::torcharrow::GenericUDFDispatchKey> {
  size_t operator()(
      const ::facebook::torcharrow::GenericUDFDispatchKey& x) const {
    return std::hash<std::string>()(x.udfName) ^
        (~std::hash<std::string>()(x.typeSignature));
  }
};
} // namespace std
