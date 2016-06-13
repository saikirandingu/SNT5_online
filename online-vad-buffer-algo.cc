// SNT5/vad.cc
// Coice Activity detector using NNET3 framework

#include "online-vad-buffer-algo.h"

namespace kaldi {

Vad::Vad(const VadOptions &opts) : feat_mfcc_(opts.mfcc_opts) {
  isrecognize_cont = true;
  frame_shift_ = opts.frame_shift;
  frame_overlap_ = opts.frame_overlap;
  chunk_time_ = opts.chunk_time;
  speech_threshold_ = opts.speech_threshold;
  sil_threshold_ = opts.sil_threshold;
  speech_buffer_ = opts.speech_buffer;

  total_frames_ = 0;  
  prev_start = 0;
  prev_end = 0;
  curr_start = 0;
  curr_end = 0; 

  num_sil_ = 0;
  speech_flag_ = 0;
  sil_flag_ = 0;  
  
  use_gpu_log_ = opts.use_gpu_log;
  pad_length_ = opts.pad_length;
  post_pad_length_ = opts.post_pad_length;
  delta_opts = opts.delta_opts;
  cmn_opts = opts.cmn_opts;        
  ns_delta_opts = opts.ns_delta_opts;
  ns_cmn_opts = opts.ns_cmn_opts;
  num_gselect = opts.num_gselect;
  ns_num_gselect = opts.ns_num_gselect;
  //    vad_opts = opts.vad_opts; 

  bool binary;
  Input nonspeech_input(opts.nonspeech_ubm_filename.c_str(), &binary);
  nonspeech_ubm.Read(nonspeech_input.Stream(), binary);
  Input speech_input(opts.speech_ubm_filename.c_str(), &binary);
  speech_ubm.Read(speech_input.Stream(), binary);
  diag_nonspeech_ubm.CopyFromFullGmm(nonspeech_ubm);
  diag_speech_ubm.CopyFromFullGmm(speech_ubm);
  
}

void Vad::reinitiate()
{
  isrecognize_cont = true;
  prev_start = 0;
  prev_end = 0;
  curr_start = 0;
  curr_end = 0;
}

bool Vad::Compute_online(const VectorBase<BaseFloat> &waveform, std::vector<std::vector<BaseFloat> > *seg_times) {
  try{
    //      KALDI_LOG << "chunk";
    isrecognize_cont = true;    	
    Matrix<BaseFloat> hires_features;
    feat_mfcc_.Compute(waveform, &hires_features, 1.0);
    //    Vector<BaseFloat> *vad;
    //    ComputeVadEnergy(vad_opts, hires_features, vad);
    
    Matrix<BaseFloat> delta_feats;
    ComputeDeltas(ns_delta_opts, hires_features, &delta_feats);

    std::vector<std::vector<int32> > gselect_speech(delta_feats.NumRows());
    std::vector<std::vector<int32> > gselect_nonspeech(delta_feats.NumRows());
    diag_speech_ubm.GaussianSelection(delta_feats, ns_num_gselect,
                                      &gselect_speech);
    diag_nonspeech_ubm.GaussianSelection(delta_feats, ns_num_gselect,
                                         &gselect_nonspeech);
    Vector<BaseFloat> vec_log;
    vec_log.Resize(delta_feats.NumRows());

    for (int32 i = 0; i < delta_feats.NumRows(); i++ ) {
      
      Vector<BaseFloat> speech_loglikes;
      Vector<BaseFloat> nonspeech_loglikes;
      speech_ubm.LogLikelihoodsPreselect(delta_feats.Row(i),
                                         gselect_speech[i], &speech_loglikes);
      nonspeech_ubm.LogLikelihoodsPreselect(delta_feats.Row(i),
                                            gselect_nonspeech[i], &nonspeech_loglikes);
      double speech_prob = speech_loglikes.LogSumExp();
      double nonspeech_prob = nonspeech_loglikes.LogSumExp();
      if (speech_prob >= nonspeech_prob) {
	  vec_log(i) = 1.0;} 
      else {
	  vec_log(i) = 0.0;  }
	}
	  
    //FasterDecoder decoder(*decode_fst_, decoder_opts_);
    if (vec_log.Dim() == 0) {
      KALDI_WARN << "Zero-length utterance: ";
    }

    Main_vec_log_.Resize(Main_vec_log_.Dim()+vec_log.Dim(), kCopyData);
    for (int32 i = 0; i < vec_log.Dim(); i++) {
	if (vec_log(i) == 0.0) { 
	    num_sil_++;
	    if (num_sil_ >= sil_threshold_) {sil_flag_ = 1;}
	} 
	else {num_sil_ = 0;
		sil_flag_ = 0;}
	
	Main_vec_log_(Main_vec_log_.Dim() - vec_log.Dim() + i) = vec_log(i);
	if ((Main_vec_log_.Dim() - vec_log.Dim() + i) > (speech_buffer_ - 1)) {
	    BaseFloat sum = 0;
	    for (int32 j = 0; j < speech_buffer_; j++) {
		sum = sum + Main_vec_log_(Main_vec_log_.Dim() - vec_log.Dim() + i - j);}
	    if (sum > speech_threshold_ && speech_flag_ == 0) {
	      curr_start = (std::max(0.0f, (total_frames_ + i - (speech_buffer_ - 1))*frame_shift_));
		speech_flag_ = 1;}
	    if ((speech_flag_ == 1) && (sil_flag_ == 1)) {
		curr_end = (std::max(0.0f, (total_frames_ + i - (speech_buffer_ - 1))*frame_shift_ + frame_overlap_));
		speech_flag_ = 0;
		KALDI_LOG << "PRESENT SEG : " << curr_start << " : : " << curr_end;
		
          curr_start -= pad_length_*frame_shift_;
          if (prev_end > 0) prev_end += pad_length_*frame_shift_;
          //	    if (curr_start <= prev_end) { curr_start = prev_start; }
          if (((curr_start - prev_end) < (seg_opts_.max_intersegment_length*frame_shift_))&&(prev_end>0)) {
            curr_start = prev_start;
          }
          curr_start -= post_pad_length_*frame_shift_;	    
          if (prev_end > 0) prev_end += post_pad_length_*frame_shift_;
          if ((curr_start <= prev_end)&&(prev_end>0)) { curr_start = prev_start; }

          if (((curr_start - prev_end) > 0) && ((prev_end - prev_start)>0)) {
            //   return true;
	        isrecognize_cont = false;
		std::vector<BaseFloat> row;
                row.push_back(prev_start);
                row.push_back(prev_end);
                (*seg_times).push_back(row);

          } else {
            isrecognize_cont = true;}

          prev_start = curr_start;
          prev_end = curr_end;
                
      }
    }
   }
        if (isrecognize_cont == false) {
	  Vector<BaseFloat> vec_dummy;
          vec_dummy.Resize(speech_buffer_, kSetZero);
          for ( int i = 0; i < vec_dummy.Dim(); i++) {
            vec_dummy(i) = Main_vec_log_(Main_vec_log_.Dim() - vec_dummy.Dim() + i);}
	}
        if ((prev_end - prev_start)>0) {
            std::vector<BaseFloat> row;
            row.push_back(prev_start);
            row.push_back(prev_end);
            (*seg_times).push_back(row);
        }

	total_frames_ = total_frames_ + vec_log.Dim();    
 
    return isrecognize_cont;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return true;
  }


}
}
