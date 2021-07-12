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
#include <f4d/common/memory/Memory.h>
#include <f4d/type/Type.h>
#include <f4d/vector/ComplexVector.h>
#include <chrono>
#include <memory>
#include <ratio>
#include "f4d/common/base/Exceptions.h"
#include "f4d/core/Expressions.h"
#include "f4d/core/ITypedExpr.h"
#include "f4d/exec/Expr.h"
#include "f4d/functions/common/CoreFunctions.h"
#include "f4d/parse/Expressions.h"
#include "f4d/parse/ExpressionsParser.h"
#include "f4d/vector/BaseVector.h"

namespace py = pybind11;

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

template <TypeKind kind>
std::unique_ptr<BaseColumn> doCreateConstantColumn(
    variant value,
    vector_size_t size) {
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_unique<ConstantColumn<T>>(value, size);
}

std::unique_ptr<BaseColumn> BaseColumn::createConstantColumn(
    variant value,
    vector_size_t size) {
  // Note here we are doing the same type dispatch twice:
  //   1. first happens when dispatching to doCreateSimpleColumn
  //   2. second happens in constructor of ConstantColumn<T> when calling
  //      BaseVector::createConstant
  //
  // The second dispatch is required because the method
  // BaseVector::createConstant dispatch to (newConstant) is a not yet a public
  // Velox API. Otherwise, we can create the ConstantVector and wrap it into the
  // ConstantColumn in one dispatch.
  //
  // We can avoid the second dispatch either by making `newConstant` a public
  // Velox API, or have template method to create ConstantVector (which
  // essentially fork `newConstant` method).
  //
  // However, at some point we also want to revisit whether
  // SimpleColumn/ConstantColumn needs to be templated and thus we could
  // remove the first dispatch.
  return F4D_DYNAMIC_SCALAR_TYPE_DISPATCH(
      doCreateConstantColumn, value.kind(), value, size);
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

std::shared_ptr<exec::ExprSet> BaseColumn::genUnaryExprSet(
    std::shared_ptr<const facebook::f4d::RowType> inputRowType,
    TypePtr outputType,
    const std::string& functionName) {
  // Construct Typed Expression
  using InputExprList = std::vector<std::shared_ptr<const core::ITypedExpr>>;
  InputExprList inputTypedExprs{
      std::make_shared<core::InputTypedExpr>(inputRowType)};

  InputExprList fieldAccessTypedExprs{
      std::make_shared<core::FieldAccessTypedExpr>(
          inputRowType->childAt(0),
          std::move(inputTypedExprs),
          inputRowType->nameOf(0))};

  InputExprList callTypedExprs{std::make_shared<core::CallTypedExpr>(
      outputType,
      std::move(fieldAccessTypedExprs),
      functionName)};

  // Container for expressions that get evaluated together. Common
  // subexpression elimination and other cross-expression
  // optimizations take place within this set of expressions.
  return std::make_shared<exec::ExprSet>(
      std::move(callTypedExprs), &TorchArrowGlobalStatic::execContext());
}

std::unique_ptr<BaseColumn> BaseColumn::applyUnaryExprSet(
    std::shared_ptr<const facebook::f4d::RowType> inputRowType,
    std::shared_ptr<exec::ExprSet> exprSet) {
  auto inputRows = wrapRowVector({_delegate}, inputRowType);
  exec::EvalCtx evalCtx(
      &TorchArrowGlobalStatic::execContext(), exprSet.get(), inputRows.get());
  SelectivityVector select(_delegate->size());
  std::vector<VectorPtr> outputRows(1);
  exprSet->eval(0, 1, true, select, &evalCtx, &outputRows);

  // TODO: This causes an extra type-based dispatch.
  // We can optimize it by specializing applyUnaryExprSet method for
  // SimpleColumn.
  return createColumn(outputRows[0]);
}

std::shared_ptr<exec::ExprSet> BaseColumn::genBinaryExprSet(
    std::shared_ptr<const facebook::f4d::RowType> inputRowType,
    std::shared_ptr<const facebook::f4d::Type> commonType,
    const std::string& functionName) {
  // Construct Typed Expression
  using InputExprList = std::vector<std::shared_ptr<const core::ITypedExpr>>;
  InputExprList inputTypedExprs{
      std::make_shared<core::InputTypedExpr>(inputRowType)};

  InputExprList castedFieldAccessTypedExprs;
  for (int i = 0; i < 2; i++) {
    auto fieldAccessTypedExpr = std::make_shared<core::FieldAccessTypedExpr>(
        inputRowType->childAt(i),
        InputExprList(inputTypedExprs),
        inputRowType->nameOf(i));

    if (*inputRowType->childAt(i) == *commonType) {
      // no need to cast
      castedFieldAccessTypedExprs.push_back(fieldAccessTypedExpr);
    }
    else {
      // type promotion
      InputExprList fieldAccessTypedExprs{fieldAccessTypedExpr};
      castedFieldAccessTypedExprs.push_back(
          std::make_shared<core::CastTypedExpr>(
              commonType,
              fieldAccessTypedExprs,
              false /* nullOnFailure */));
    }
  }

  InputExprList callTypedExprs{std::make_shared<core::CallTypedExpr>(
      commonType,
      std::move(castedFieldAccessTypedExprs),
      functionName)};

  // Container for expressions that get evaluated together. Common
  // subexpression elimination and other cross-expression
  // optimizations take place within this set of expressions.
  return std::make_shared<exec::ExprSet>(
      std::move(callTypedExprs), &TorchArrowGlobalStatic::execContext());
}

std::unique_ptr<BaseColumn> BaseColumn::genericUnaryUDF(
    const std::string& udfName,
    const BaseColumn& col1) {
  auto rowType = ROW({"c0"}, {col1.getUnderlyingVeloxVector()->type()});
  GenericUDFDispatchKey key(udfName, rowType->toString());

  static std::
      unordered_map<GenericUDFDispatchKey, std::unique_ptr<OperatorHandle>>
          dispatchTable;

  auto iter = dispatchTable.find(key);
  if (iter == dispatchTable.end()) {
    iter = dispatchTable
               .insert({key, OperatorHandle::fromGenericUDF(rowType, udfName)})
               .first;
  }
  return iter->second->call({col1.getUnderlyingVeloxVector()});
}

std::unique_ptr<BaseColumn> BaseColumn::genericBinaryUDF(
    const std::string& udfName,
    const BaseColumn& col1,
    const BaseColumn& col2) {
  auto rowType =
      ROW({"c0", "c1"},
          {col1.getUnderlyingVeloxVector()->type(),
           col2.getUnderlyingVeloxVector()->type()});
  GenericUDFDispatchKey key(udfName, rowType->toString());

  static std::
      unordered_map<GenericUDFDispatchKey, std::unique_ptr<OperatorHandle>>
          dispatchTable;

  auto iter = dispatchTable.find(key);
  if (iter == dispatchTable.end()) {
    iter = dispatchTable
               .insert({key, OperatorHandle::fromGenericUDF(rowType, udfName)})
               .first;
  }
  return iter->second->call({col1.getUnderlyingVeloxVector(), col2.getUnderlyingVeloxVector()});
}

std::unique_ptr<OperatorHandle> OperatorHandle::fromGenericUDF(
    RowTypePtr inputRowType,
    const std::string& udfName) {
    // Generate the expression
    std::stringstream ss;
    ss << udfName << "(";
    bool first = true;
    for (int i = 0; i < inputRowType->size(); i++) {
      if (!first) {
        ss << ",";
      }
      ss << inputRowType->nameOf(i);
      first=false;
    }
    ss << ")";

    return OperatorHandle::fromExpression(inputRowType, ss.str());
}

std::unique_ptr<OperatorHandle> OperatorHandle::fromExpression(
    RowTypePtr inputRowType,
    const std::string& expr) {
  auto untypedExpr = parse::parseExpr(expr);
  auto typedExpr = core::Expressions::inferTypes(
      untypedExpr, inputRowType, TorchArrowGlobalStatic::execContext().pool());

  using TypedExprList = std::vector<std::shared_ptr<const core::ITypedExpr>>;
  TypedExprList typedExprs{typedExpr};
  return std::make_unique<OperatorHandle>(
      inputRowType,
      std::make_shared<exec::ExprSet>(
          std::move(typedExprs), &TorchArrowGlobalStatic::execContext()));
}

  std::unique_ptr<BaseColumn> OperatorHandle::call(VectorPtr a, VectorPtr b) {
    auto inputRows = wrapRowVector({a, b}, inputRowType_);
    exec::EvalCtx evalCtx(
        &TorchArrowGlobalStatic::execContext(),
        exprSet_.get(),
        inputRows.get());
    SelectivityVector select(a->size());
    std::vector<VectorPtr> outputRows(1);
    exprSet_->eval(0, 1, true, select, &evalCtx, &outputRows);

    // TODO: This causes an extra type-based dispatch.
    // We can optimize it by template OperatorHandle by return type
    return createColumn(outputRows[0]);
  }

  std::unique_ptr<BaseColumn> OperatorHandle::call(
      const std::vector<VectorPtr>& args) {
    auto inputRows = wrapRowVector(args, inputRowType_);
    exec::EvalCtx evalCtx(
        &TorchArrowGlobalStatic::execContext(),
        exprSet_.get(),
        inputRows.get());
    SelectivityVector select(args[0]->size());
    std::vector<VectorPtr> outputRows(1);
    exprSet_->eval(0, 1, true, select, &evalCtx, &outputRows);

    // TODO: This causes an extra type-based dispatch.
    // We can optimize it by template OperatorHandle by return type
    return createColumn(outputRows[0]);
  }

  core::QueryCtx& TorchArrowGlobalStatic::queryContext() {
    static core::QueryCtx queryContext;
    return queryContext;
  }

  core::ExecCtx& TorchArrowGlobalStatic::execContext() {
    static core::ExecCtx execContext(
        memory::getDefaultScopedMemoryPool(),
        &TorchArrowGlobalStatic::queryContext());
    return execContext;
  }

  // This method only supports a limited set Python object to variant conversion
  // to minimize code duplication.
  // TODO: Open source some part of utility codes in Koski (PyVelox?)
  variant pyToVariant(const pybind11::handle& obj) {
    if (py::isinstance<py::bool_>(obj)) {
      return variant::create<TypeKind::BOOLEAN>(py::cast<bool>(obj));
    } else if (py::isinstance<py::int_>(obj)) {
      return variant::create<TypeKind::BIGINT>(py::cast<long>(obj));
    } else if (py::isinstance<py::float_>(obj)) {
      return variant::create<TypeKind::REAL>(py::cast<float>(obj));
    } else if (py::isinstance<py::str>(obj)) {
      return variant::create<TypeKind::VARCHAR>(py::cast<std::string>(obj));
    } else if (obj.is_none()) {
      return variant();
    }
    else {
      VELOX_CHECK(false);
    }
  }

} // namespace torcharrow
} // namespace facebook
