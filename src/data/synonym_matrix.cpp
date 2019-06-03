#include "data/synonym_matrix.h"

#include "marian.h"

namespace marian {
namespace data {

SynonymMatrix::SynonymMatrix(const std::string& path) {
  load(path);
}

void SynonymMatrix::load(const std::string& path) {
  // Read npz file.
  auto items = io::loadItems(path);
  for(auto item : items) {
    auto totalSize = item.shape.elements();
    if(item.name == "data") {
      data_.resize(totalSize);
      std::copy((float*)item.data(), ((float*)item.data()) + totalSize, data_.begin());
    }
    if(item.name == "indptr") {
      indptr_.resize(totalSize);
      size_ = totalSize;
      std::copy((int*)item.data(), ((int*)item.data()) + totalSize, indptr_.begin());
    }
    if(item.name == "indices") {
      indices_.resize(totalSize);
      std::copy((int*)item.data(), ((int*)item.data()) + totalSize, indices_.begin());
    }
  }
}

std::vector<float> SynonymMatrix::operator[](const std::vector<Word>& ids) {
  // TODO: comment.
  std::vector<float> targetDists;
  int rows = ids.size();
  targetDists.resize(rows * size_);
  std::fill(targetDists.begin(), targetDists.end(), 0.f);
  for (int i = 0; i < rows; ++i) {
    int id = ids[i];
    int len = indptr_[id + 1] - indptr_[id];
    for (int j = 0; j < len; ++j) {
      targetDists[(size_*i) + indices_[indptr_[id] + j]] =
          data_[indptr_[id] + j];
    }
  }
  return std::move(targetDists);
}

}  // namespace data
}  // namespace marian
