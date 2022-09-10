// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Summary: The experimental Ort C++ API is a wrapper around the Ort C++ API.
//
// This C++ API further simplifies usage and provides support for modern C++ syntax/features
// at the cost of some overhead (i.e. using std::string over char *).
//
// Where applicable, default memory allocator options are used unless explicitly set.
//
// Experimental components are designed as drop-in replacements of the regular API, requiring
// minimal code modifications to use.
//
// Example:  Ort::Session -> Ort::Experimental::Session
//
// NOTE: Experimental API components are subject to change based on feedback and provide no
// guarantee of backwards compatibility in future releases.

// Modified by : Srini

#pragma once
#include "onnxruntime_cxx_api.h"

namespace Ort { namespace Experimental {

struct Session : Ort::Session {
  Session(Env& env, std::basic_string<ORTCHAR_T>& model_path, SessionOptions& options)
      : Ort::Session(env, model_path.data(), options){};
  Session(Env& env, void* model_data, size_t model_data_length, SessionOptions& options)
      : Ort::Session(env, model_data, model_data_length, options){};

  // overloaded Run() with sensible defaults
  std::vector<Ort::Value> Run(const std::vector<std::string>& input_names,
                              const std::vector<Ort::Value>& input_values,
                              const std::vector<std::string>& output_names,
                              const RunOptions& run_options = RunOptions());
  void Run(const std::vector<std::string>& input_names,
           const std::vector<Ort::Value>& input_values,
           const std::vector<std::string>& output_names,
           std::vector<Ort::Value>& output_values,
           const RunOptions& run_options = RunOptions());

  // convenience methods that simplify common lower-level API calls
  std::vector<std::string> GetInputNames() const;
  std::vector<std::string> GetOutputNames() const;
  std::vector<std::string> GetOverridableInitializerNames() const;

  // NOTE: shape dimensions may have a negative value to indicate a symbolic/unknown dimension.
  std::vector<std::vector<int64_t> > GetInputShapes() const;
  std::vector<std::vector<int64_t> > GetOutputShapes() const;
  std::vector<std::vector<int64_t> > GetOverridableInitializerShapes() const;
};

    std::vector <Value>
    Session::Run(const std::vector <string> &input_names, const std::vector <Value> &input_values,
                 const std::vector <string> &output_names, const RunOptions &run_options) {
      return NULL;
    }

    void
    Session::Run(const std::vector <string> &input_names, const std::vector <Value> &input_values,
                 const std::vector <string> &output_names, std::vector <Value> &output_values,
                 const RunOptions &run_options) {

    }

    std::vector <string> Session::GetInputNames() const {
      return NULL;
    }

    std::vector <string> Session::GetOutputNames() const {
      return NULL;
    }

    std::vector <string> Session::GetOverridableInitializerNames() const {
      return NULL;
    }

    std::vector <int64_t>

    <vector> Session::GetInputShapes() const {
      return NULL;
    }

    std::vector <int64_t>

    <vector> Session::GetOutputShapes() const {
      return NULL;
    }

    std::vector <int64_t>

    <vector> Session::GetOverridableInitializerShapes() const {
      return NULL;
    }

    struct Value : Ort::Value {
  Value(OrtValue* p)
      : Ort::Value(p){};

  template <typename T>
  static Ort::Value CreateTensor(T* p_data, size_t p_data_element_count, const std::vector<int64_t>& shape);
  static Ort::Value CreateTensor(void* p_data, size_t p_data_byte_count, const std::vector<int64_t>& shape, ONNXTensorElementDataType type);

  template <typename T>
  static Ort::Value CreateTensor(const std::vector<int64_t>& shape);
  static Ort::Value CreateTensor(const std::vector<int64_t>& shape, ONNXTensorElementDataType type);
};

        template<typename T>
        Ort::Value Value::CreateTensor(T *p_data, size_t p_data_element_count,
                                       const std::vector <int64_t> &shape) {
          return Ort::Value(NULL);
        }

        Ort::Value Value::CreateTensor(void *p_data, size_t p_data_byte_count,
                                       const std::vector <int64_t> &shape,
                                       ONNXTensorElementDataType type) {
          return Ort::Value(NULL);
        }

        template<typename T>
        Ort::Value Value::CreateTensor(const std::vector <int64_t> &shape) {
          return Ort::Value(NULL);
        }

        Ort::Value
        Value::CreateTensor(const std::vector <int64_t> &shape, ONNXTensorElementDataType type) {
          return Ort::Value(NULL);
        }

    } }  // namespace Ort::Experimental

#include "experimental_onnxruntime_cxx_inline.h"
