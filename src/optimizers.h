#pragma once

#include <map>
#include <memory>
#include <boost/any.hpp>
#include "tensor_operators.h"

namespace marian {

// @TODO: modify computation graph to group all paramters in single matrix object.
// This will allow to perform a single large SGD update per batch. Currently there
// are as many updates as different parameters.

class OptimizerBase {
  public:
    virtual void update(ExpressionGraphPtr, data::BatchPtr = nullptr) = 0;
};

typedef std::shared_ptr<OptimizerBase> OptimizerBasePtr;

class Sgd : public OptimizerBase {
  public:
    Sgd(float eta=0.01) : eta_(eta) {}

    void update(ExpressionGraphPtr graph, data::BatchPtr batch = nullptr) {
      graph->backprop(batch);

      for(auto& param : graph->params())
        Element(_1 -= eta_ * _2,
                param->val(), param->grad());
    }

  private:
    float eta_;
};

// @TODO: Add serialization for historic gradients and parameters
class Adagrad : public OptimizerBase {
  public:
    Adagrad(float eta=0.01, float eps=1e-8)
    : eta_(eta), eps_(eps),
      alloc_(newTensorAllocator<DeviceGPU>())
    {}

    void update(ExpressionGraphPtr graph, data::BatchPtr batch = nullptr) {
      graph->backprop(batch);

      if(!gt_) {
        int totalSize = graph->params().totalSize();
        alloc_->reserveExact(totalSize);
        alloc_->allocate(gt_, {1, totalSize});
        gt_->set(0);
      }

      Tensor pv = graph->params().vals();
      Tensor pg = graph->params().grads();

      Element(_1 += (_2 * _2),
              gt_, pg);

      Element(_1 -= (eta_ / (Sqrt(_2) + eps_)) * _3,
              pv, gt_, pg);
    }

  private:
    float eta_;
    float eps_;
    TensorAllocator alloc_;
    Tensor gt_;
};


// @TODO: Add serialization for historic gradients and parameters
// https://arxiv.org/pdf/1412.6980v8.pdf
class Adam : public OptimizerBase {
  public:
    Adam(float eta=0.001, float beta1=0.9, float beta2=0.999, float eps=1e-8)
    : eta_(eta), beta1_(beta1), beta2_(beta2), eps_(eps), t_(0),
      mtAlloc_(newTensorAllocator<DeviceGPU>()),
      vtAlloc_(newTensorAllocator<DeviceGPU>())
    {}

    void update(ExpressionGraphPtr graph, data::BatchPtr batch = nullptr) {
      graph->backprop(batch);

      if(!mt_) {
        int totalSize = graph->params().totalSize();
        mtAlloc_->reserveExact(totalSize);
        mtAlloc_->allocate(mt_, {1, totalSize});
        mt_->set(0);

        vtAlloc_->reserveExact(totalSize);
        vtAlloc_->allocate(vt_, {1, totalSize});
        vt_->set(0);
      }

      t_++;
      float denom1 = 1 - pow(beta1_, t_);
      float denom2 = 1 - pow(beta2_, t_);

      Tensor pv = graph->params().vals();
      Tensor pg = graph->params().grads();

      //clip(pg);

      Element(_1 = (beta1_ * _1) + ((1 - beta1_) * _2),
              mt_, pg);
      Element(_1 = (beta2_ * _1) + ((1 - beta2_) * (_2 * _2)),
              vt_, pg);

      Element(_1 -= eta_ * (_2 / denom1) / (Sqrt(_3 / denom2) + eps_),
              pv, mt_, vt_);
    }

  private:
    float eta_;
    float beta1_;
    float beta2_;
    float eps_;
    size_t t_;

    TensorAllocator mtAlloc_;
    Tensor mt_;
    TensorAllocator vtAlloc_;
    Tensor vt_;
};

template <class Algorithm, typename ...Args>
OptimizerBasePtr Optimizer(Args&& ...args) {
  return OptimizerBasePtr(new Algorithm(args...));
}

}
