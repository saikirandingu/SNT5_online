// SNT5/feat.h

#ifndef KALDI_SNT5_ONLINE_ONLINE_VAD_BUFFER_ALGO_H_
#define KALDI_SNT5_ONLINE_ONLINE_VAD_BUFFER_ALGO_H_

#include <iomanip>
#include <string>
#include <map>

#include "feat.h"
#include "segmenter.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

#define SNT_KALDI_SMP_FREQ 8000

namespace kaldi {

  struct  VadOptions {
    FeatOptions mfcc_opts;
    DeltaFeaturesOptions delta_opts;
    DeltaFeaturesOptions ns_delta_opts;
    SlidingWindowCmnOptions cmn_opts;
    SlidingWindowCmnOptions ns_cmn_opts;
    //    VadEnergyOptions vad_opts;
    int32 num_gselect, ns_num_gselect, speech_threshold, 
	speech_buffer, sil_threshold;
    BaseFloat frame_shift, frame_overlap, chunk_time; 

    // These are optional and related
    // to the GMM-based speech/nonspeech detector.

    std::string nonspeech_ubm_filename, speech_ubm_filename, use_gpu_log;
    kaldi::segmenter::SegmentationPostProcessingOptions seg_opts;
    int pad_length, post_pad_length;
       
   VadOptions() 
   : use_gpu_log("no")
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
      
      (*po).Register("speech-threshold", &speech_threshold, "minimum no. of frames in buffer");
      (*po).Register("sil-threshold", &sil_threshold, "minimum no. of consecutive sil frames");
      (*po).Register("speech-buffer", &speech_buffer, "buffer for speech frames");
      (*po).Register("frame-shift", &frame_shift, "audio frame shift");
      (*po).Register("frame-overlap", &frame_overlap, "audio frame overlap");
      (*po).Register("chunk-time", &chunk_time, "audio chunk input time");
      (*po).Register("num-gselect", &num_gselect, "TODO, num-gselect usage");

      (*po).Register("use-gpu-log", &use_gpu_log,
                     "yes|no|optional|wait, only has effect if compiled with CUDA");
      (*po).Register("pad-length", &pad_length,"first stage pad length");
      (*po).Register("post-pad-length",&post_pad_length, "2nd stage pad length");

      seg_opts.Register(po);
   }
  };

   
  class Vad {
 public:
    
    BaseFloat prev_start, prev_end, curr_start, curr_end;
    bool isrecognize_cont;
    explicit Vad(const VadOptions &vad_opts);
    void reinitiate();
    bool Compute_online(const VectorBase<BaseFloat> &waveform,
                        std::vector<std::vector<BaseFloat> > *seg_times) ;

 private:
    DeltaFeaturesOptions delta_opts;
    DeltaFeaturesOptions ns_delta_opts;
    SlidingWindowCmnOptions cmn_opts;
    SlidingWindowCmnOptions ns_cmn_opts;
    //VadEnergyOptions vad_opts;
    int32 num_gselect, ns_num_gselect, total_frames_,
    speech_buffer_, speech_threshold_, sil_threshold_,
    speech_flag_, sil_flag_, num_sil_;
  
    //    BaseFloat prev_start = 0, prev_end = 0, curr_start = 0, curr_end = 0;
    Feat feat_mfcc_;
    FullGmm nonspeech_ubm;
    FullGmm speech_ubm;
    DiagGmm diag_nonspeech_ubm;
    DiagGmm diag_speech_ubm;

    BaseFloat frame_shift_, frame_overlap_, chunk_time_;
    
    kaldi::segmenter::SegmentationPostProcessingOptions seg_opts_;
    std::string use_gpu_log_;
    
    int pad_length_, post_pad_length_;
    
    Vector<BaseFloat> Main_vec_log_;    
    KALDI_DISALLOW_COPY_AND_ASSIGN(Vad);
  };
}  // namespace kaldi

#endif  // KALDI_SNT5_ONLINE_ONLINE_VAD_BUFFER_ALGO_H_
