// SNT5/feat.h

#ifndef KALDI_VAD_GMM_C_VAD_GMM_H_
#define KALDI_VAD_GMM_C_VAD_GMM_H_

#include <iomanip>
#include <string>
#include <map>

#include "ivector/voice-activity-detection.h"
#include "feat.h"
#include "segmenter.h"
#include "hmm/hmm-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-matrix.h"
#include "lat/kaldi-lattice.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

  struct  VadOptions {
    FeatOptions mfcc_opts;
    DeltaFeaturesOptions delta_opts;
    DeltaFeaturesOptions ns_delta_opts;
    SlidingWindowCmnOptions cmn_opts;
    SlidingWindowCmnOptions ns_cmn_opts;
    //    VadEnergyOptions vad_opts;
    int32 num_gselect, ns_num_gselect;
    BaseFloat frame_shift, frame_overlap, chunk_time; 

    // These are optional and related
    // to the GMM-based speech/nonspeech detector.

    FasterDecoderOptions decoder_opts;
    std::string nonspeech_ubm_filename, speech_ubm_filename, decoding_method, decoding_model, decoding_graph, use_gpu_log;
    kaldi::segmenter::SegmentationPostProcessingOptions seg_opts;
    int pad_length, post_pad_length;
    float speech_to_sil_ratio, acoustic_wt;
    bool allow_partial;    
   VadOptions() 
   : use_gpu_log("no"),
      allow_partial(true)
    {}

    void Register(OptionsItf *po) {
      ParseOptions mfccopts("mfcc", po);
      mfcc_opts.Register(&mfccopts);
      
      ParseOptions po_ns("ns", po);
      ns_delta_opts.Register(&po_ns);
      ns_cmn_opts.Register(&po_ns);
      po_ns.Register("nonspeech-ubm", &nonspeech_ubm_filename, "TODO");
      po_ns.Register("speech-ubm", &speech_ubm_filename, "TODO");
      po_ns.Register("num-gselect", &ns_num_gselect, "TODO, num-gselect usage");

     
      //    vad_opts.Register(po);
      delta_opts.Register(po);
      cmn_opts.Register(po);
      
      (*po).Register("frame-shift", &frame_shift, "audio frame shift");
      (*po).Register("frame-overlap", &frame_overlap, "audio frame overlap");
      (*po).Register("chunk-time", &chunk_time, "audio chunk input time");
      (*po).Register("num-gselect", &num_gselect, "TODO, num-gselect usage");

      (*po).Register("use-gpu-log", &use_gpu_log,
                     "yes|no|optional|wait, only has effect if compiled with CUDA");
      
      (*po).Register("decoding-method", &decoding_method,
                     "Decoding method in segmentation");
      (*po).Register("decoding-model", &decoding_model,
                     "Decoding model in segmentation");
      (*po).Register("decoding-graph", &decoding_graph,
                     "Decoding graph in segmentation");
      (*po).Register("allow-partial", &allow_partial, "allow not reaching final state in decoding");
      (*po).Register("acwt", &acoustic_wt, "acoustic scale");
      (*po).Register("speech-to-sil-ratio", &speech_to_sil_ratio, "sp_sil_ratio");
      (*po).Register("pad-length", &pad_length,"first stage pad length");
      (*po).Register("post-pad-length",&post_pad_length, "2nd stage pad length");

      seg_opts.Register(po);
      
      decoder_opts.Register(po, true);
    }
  };

  using fst::SymbolTable;
  using fst::VectorFst;
  using fst::StdArc;
    
  class Vad {
 public:
    int32 num_chunks;
    BaseFloat prev_start, prev_end, curr_start, curr_end;
    bool isrecognize_cont;
    explicit Vad(const VadOptions &vad_opts);
    void reinitiate();
    bool Compute_online(const VectorBase<BaseFloat> &waveform,
                        std::vector<std::vector<BaseFloat> > *seg_times, bool isrecording_continue) ;

 private:
    DeltaFeaturesOptions delta_opts;
    DeltaFeaturesOptions ns_delta_opts;
    SlidingWindowCmnOptions cmn_opts;
    SlidingWindowCmnOptions ns_cmn_opts;
    //VadEnergyOptions vad_opts;
    int32 num_gselect, ns_num_gselect;

    //    int32 num_chunks = 0;
    //    BaseFloat prev_start = 0, prev_end = 0, curr_start = 0, curr_end = 0;
    Feat feat_mfcc_;
    FullGmm nonspeech_ubm;
    FullGmm speech_ubm;
    DiagGmm diag_nonspeech_ubm;
    DiagGmm diag_speech_ubm;

    BaseFloat frame_shift_, frame_overlap_, chunk_time_;
    FasterDecoderOptions decoder_opts_;
    kaldi::segmenter::SegmentationPostProcessingOptions seg_opts_;
    std::string decoding_method_, decoding_model_, decoding_graph_, use_gpu_log_;
    float speech_to_sil_ratio_, acoustic_scale_;
    int pad_length_, post_pad_length_;
    Vector<BaseFloat> prior_vec_;
    kaldi::segmenter::SegmentationPostProcessor post_processor;
    VectorFst<StdArc> *decode_fst_;
    TransitionModel trans_model_; 
    bool allow_partial_;
    KALDI_DISALLOW_COPY_AND_ASSIGN(Vad);
  };
}  // namespace kaldi

#endif  // KALDI_SNT5_FEAT_H_
