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
      std::copy((int*)item.data(), ((int*)item.data()) + totalSize, indptr_.begin());
    }
    if(item.name == "indices") {
      indptr_.resize(totalSize);
      size_ = totalSize;
      std::copy((int*)item.data(), ((int*)item.data()) + totalSize, indices_.begin());
    }
  }
}

std::vector<float> SynonymMatrix::operator[](const Word& id) {
  std::vector<float> targetDistribution;
  targetDistribution.resize(size_);
  std::fill(targetDistribution.begin(), targetDistribution.end(), 0.f);
  // TODO: check out-of-bounds
  int len = indptr_[id + 1] - indptr_[id];
  for (int j = 0; j < len; ++j) {
    targetDistribution[indices_[indptr_[j]]] = data_[indices_[indptr_[j]]];
  }
  return std::move(targetDistribution);
}

}  // namespace data
}  // namespace marian
