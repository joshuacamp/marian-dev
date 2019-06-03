#pragma once

#include "data/types.h"
#include "tensors/tensor.h"

namespace marian {
namespace data {

class SynonymMatrix {
private:
  // Data structures used to store the sparse matrix
  // in CSR format.
  std::vector<float> data_;
  std::vector<int> indices_;
  std::vector<int> indptr_;
  
  // Number of columns, rows in the matrix (should equal
  // output vocabulary dimensionality).
  size_t size_;
public:
  SynonymMatrix(const std::string& path);

  void load(const std::string& path); 

  std::vector<float> operator[](const std::vector<Word>& ids);
};

} // namespace data
} // namespace marian
