// SNT5/feat.h

#ifndef KALDI_VAD_GMM_C_VAD_GMM_H_
#define KALDI_VAD_GMM_C_VAD_GMM_H_


#include <iomanip>
#include <string>
#include <map>
#include <iostream>


#include "feat.h"
#include "segmenter.h"
#include "online-gmm-decodable-vad.h"
#include "online-gmm-decoding-vad.h"

#include "online/online-feat-input.h"
#include "online/online-decodable.h"
#include "online/onlinebin-util.h"
#include "online2/online-feature-pipeline.h"
#include "ivector/voice-activity-detection.h"
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

#define SNT_KALDI_SMP_FREQ 8000

using namespace std;
namespace kaldi {

  struct  VadOptions {
    int32 num_frames_skipped, segment_buffer_len;
    BaseFloat frame_shift, frame_overlap, chunk_time, speech_offset, energy_threshold; 
    OnlineGmmDecodingConfig gmm_decoding_opts;
    OnlineFeaturePipelineCommandLineConfig vad_feature_config;
    std::string decoding_graph, decoding_model, use_gpu_log, vad_segments_filename;
    kaldi::segmenter::SegmentationPostProcessingOptions seg_opts;
    int pad_length, post_pad_length;
    float speech_to_sil_ratio;
 
   VadOptions() 
   : use_gpu_log("no")
    {}

    void Register(OptionsItf *po) {
      (*po).Register("speech-offset", &speech_offset, "Conditional speech likelihood offset");
      (*po).Register("VAD-energy-threshold", &energy_threshold, "threshold energy for speech prob offset");
      (*po).Register("frame-shift", &frame_shift, "audio frame shift");
      (*po).Register("frame-overlap", &frame_overlap, "audio frame overlap");
      (*po).Register("chunk-time", &chunk_time, "audio chunk input time");
      (*po).Register("seg-buffer-len", &segment_buffer_len, "segmentation processing input frame buffer length");
      (*po).Register("num-frames-skipped", &num_frames_skipped, "number of frames"
			"to be skipped from the end of chunk while taking VAD alignments");
      (*po).Register("use-gpu-log", &use_gpu_log,
                     "yes|no|optional|wait, only has effect if compiled with CUDA");
      (*po).Register("decoding-model", &decoding_model,
                     "transition model in segmentation");
      (*po).Register("decoding-graph", &decoding_graph,
                     "Decoding graph in segmentation");
      (*po).Register("save-vad-segments-filename", &vad_segments_filename,
		     "Output vad segments file");
      (*po).Register("speech-to-sil-ratio", &speech_to_sil_ratio, "sp_sil_ratio");
      (*po).Register("pad-length", &pad_length,"first stage pad length");
      (*po).Register("post-pad-length",&post_pad_length, "2nd stage pad length");

      seg_opts.Register(po);
      gmm_decoding_opts.Register(po);
      vad_feature_config.Register(po);
    }
  };

  using fst::SymbolTable;
  using fst::VectorFst;
  using fst::StdArc;

  class Vad {
 public:
    ofstream out_segfile;
    int32 num_frames_total, num_chunks_lat, old_frames_decoded_, new_frames_decoded_;
    BaseFloat prev_start, prev_end, curr_start, curr_end;
    bool isrecognize_cont;
    explicit Vad(const VadOptions &vad_opts);
    void reinitiate();
    bool Compute_online(const VectorBase<BaseFloat> &waveform,
                        std::vector<std::vector<BaseFloat> > *seg_times, 
			std::string wav_id,
			bool isrecordingcontinue) ;

 private:
    BaseFloat frame_shift_, frame_overlap_, chunk_time_, speech_offset_, energy_threshold_;
    OnlineFeaturePipelineConfig *feat_config_;
    OnlineFeaturePipeline *feat_pipeline_;
    Vector<BaseFloat> wave_buffer_;
    Vector<BaseFloat> wave_memory;    
    
    OnlineGmmDecodingModels *models_;
    SingleUtteranceGmmDecoder *gmm_decoder_;
    OnlineGmmDecodingConfig gmm_decoding_opts_;
    OnlineGmmAdaptationState adaptation_state_;
 
    kaldi::segmenter::SegmentationPostProcessingOptions seg_opts_;
    std::string decoding_graph_, decoding_model_, use_gpu_log_;
    float speech_to_sil_ratio_;
    int pad_length_, post_pad_length_, num_frames_skipped_, segment_buffer_len_, chunk_count_buf_, past_memory_size_, seg_no_;
    TransitionModel trans_model_;
    VectorFst<StdArc> *decode_fst_;
 
    kaldi::segmenter::SegmentationPostProcessor post_processor;
    KALDI_DISALLOW_COPY_AND_ASSIGN(Vad);
  };
}  // namespace kaldi

#endif  // KALDI_SNT5_FEAT_H_
