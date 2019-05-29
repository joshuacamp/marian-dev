#pragma once

#include "marian.h"
#include "states.h"

#include "data/shortlist.h"
#include "data/synonym_matrix.h"
#include "layers/generic.h"

namespace marian {

class DecoderBase {
protected:
  Ptr<Options> options_;
  std::string prefix_{"decoder"};
  bool inference_{false};
  size_t batchIndex_{1};

  Ptr<data::Shortlist> shortlist_;
  Ptr<data::SynonymMatrix> synonymMatrix_;

public:
  DecoderBase(Ptr<Options> options)
      : options_(options),
        prefix_(options->get<std::string>("prefix", "decoder")),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", 1)) {}

  virtual ~DecoderBase() {}

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph>,
                                       Ptr<data::CorpusBatch> batch,
                                       std::vector<Ptr<EncoderState>>&)
      = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph>, Ptr<DecoderState>) = 0;

  virtual void embeddingsFromBatch(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state,
                                   Ptr<data::CorpusBatch> batch) {

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];
    int dimEmb = opt<int>("dim-emb");

    auto yEmbFactory = embedding()  //
        ("dimVocab", dimVoc)        //
        ("dimEmb", dimEmb);

    if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
      yEmbFactory("prefix", "Wemb");
    else
      yEmbFactory("prefix", prefix_ + "_Wemb");

    if(options_->has("embedding-fix-trg"))
      yEmbFactory("fixed", opt<bool>("embedding-fix-trg"));

    if(options_->hasAndNotEmpty("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      yEmbFactory("embFile", embFiles[batchIndex_])  //
          ("normalization", opt<bool>("embedding-normalization"));
    }

    auto yEmb = yEmbFactory.construct(graph);

    auto subBatch = (*batch)[batchIndex_];

    Expr y, yMask; std::tie
    (y, yMask) = yEmb->apply(subBatch);

    Expr yData;
    if(shortlist_) {
      yData = graph->indices(shortlist_->mappedIndices());
    } else if(synonymMatrix_) {
      std::vector<Expr> targetDistributions;
      for(auto index : subBatch->data()) {
        auto targetDistribution = (*synonymMatrix_)[index];
        targetDistributions.push_back(
            graph->constant({dimVoc}, inits::from_vector(targetDistribution)));
      }
      yData = concatenate(targetDistributions, /*axis =*/ -1);
    } else {
      yData = graph->indices(subBatch->data());
    }

    auto yShifted = shift(y, {1, 0, 0});

    state->setTargetEmbeddings(yShifted);
    state->setTargetMask(yMask);
    state->setTargetIndices(yData);
  }

  virtual void embeddingsFromPrediction(Ptr<ExpressionGraph> graph,
                                        Ptr<DecoderState> state,
                                        const std::vector<IndexType>& embIdx,
                                        int dimBatch,
                                        int dimBeam) {
    int dimTrgEmb = opt<int>("dim-emb");
    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    Expr selectedEmbs;
    if(embIdx.empty()) {
      selectedEmbs = graph->constant({1, 1, dimBatch, dimTrgEmb}, inits::zeros);
    } else {
      // embeddings are loaded from model during translation, no fixing required
      auto yEmbFactory = embedding()  //
          ("dimVocab", dimTrgVoc)     //
          ("dimEmb", dimTrgEmb);

      if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
        yEmbFactory("prefix", "Wemb");
      else
        yEmbFactory("prefix", prefix_ + "_Wemb");

      auto yEmb = yEmbFactory.construct(graph);

      selectedEmbs = yEmb->apply(embIdx, {dimBeam, 1, dimBatch, dimTrgEmb});
    }
    state->setTargetEmbeddings(selectedEmbs);
  }

  virtual const std::vector<Expr> getAlignments(int /*i*/ = 0) { return {}; };

  virtual Ptr<data::Shortlist> getShortlist() { return shortlist_; }
  virtual void setShortlist(Ptr<data::Shortlist> shortlist) {
    shortlist_ = shortlist;
  }

  virtual Ptr<data::SynonymMatrix> getSynonymMatrix() { return synonymMatrix_; }
  virtual void setSynonymMatrix(Ptr<data::SynonymMatrix> synonymMatrix) {
    synonymMatrix_ = synonymMatrix;
  }

  template <typename T>
  T opt(const std::string& key) const {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string& key, const T& def) {
    return options_->get<T>(key, def);
  }

  virtual void clear() = 0;
};

}  // namespace marian
