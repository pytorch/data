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

#include "f4d/functions/Udf.h"
#include "f4d/functions/lib/string/StringCore.h"
#include "f4d/functions/lib/string/StringImpl.h"

namespace facebook::torcharrow::functions {

namespace internal {
class Utf8CatUtils {
 public:
  static inline bool isAlpha(utf8proc_category_t category) {
    return category == UTF8PROC_CATEGORY_LU /**< Letter, uppercase */ ||
        category == UTF8PROC_CATEGORY_LL /**< Letter, lowercase */ ||
        category == UTF8PROC_CATEGORY_LT /**< Letter, titlecase */ ||
        category == UTF8PROC_CATEGORY_LM /**< Letter, modifier */ ||
        category == UTF8PROC_CATEGORY_LO /**< Letter, other */;
  }

  static inline bool isNumber(utf8proc_category_t category) {
    return category == UTF8PROC_CATEGORY_ND /**< Number, decimal digit */ ||
        category == UTF8PROC_CATEGORY_NL /**< Number, letter */ ||
        category == UTF8PROC_CATEGORY_NO /**< Number, other */;
  }
};

} // namespace internal

/**
 * torcharrow_isalpha(string) → bool
 * Return True if the string is an alphabetic string, False otherwise.
 *
 * A string is alphabetic if all characters in the string are alphabetic
 * and there is at least one character in the string.
 **/
VELOX_UDF_BEGIN(torcharrow_isalpha)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<facebook::f4d::Varchar>& input) {
  using namespace f4d::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  size_t index = 0;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);

    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));
    if (!Utf8CatUtils::isAlpha(category)) {
      result = false;
      return true;
    }

    index += codePointSize;
  }
  result = true;
  return true;
}
VELOX_UDF_END();

/**
 * torcharrow_isalnum(string) → bool
 * Return True if all characters in the string are alphanumeric (either
 *alphabets or numbers), False otherwise.
 **/
VELOX_UDF_BEGIN(torcharrow_isalnum)
FOLLY_ALWAYS_INLINE
bool call(bool& result, const arg_type<facebook::f4d::Varchar>& input) {
  using namespace f4d::functions;
  using namespace internal;

  size_t size = input.size();
  if (size == 0) {
    result = false;
    return true;
  }

  size_t index = 0;
  while (index < size) {
    int codePointSize;
    utf8proc_int32_t codePoint =
        utf8proc_codepoint(input.data() + index, codePointSize);

    utf8proc_category_t category =
        static_cast<utf8proc_category_t>(utf8proc_category(codePoint));
    if (!(Utf8CatUtils::isAlpha(category) ||
          Utf8CatUtils::isNumber(category))) {
      result = false;
      return true;
    }

    index += codePointSize;
  }
  result = true;
  return true;
}
VELOX_UDF_END();
} // namespace facebook::torcharrow::functions
