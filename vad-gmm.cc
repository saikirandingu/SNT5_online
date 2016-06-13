// SNT5/vad.cc
// Coice Activity detector using NNET3 framework

#include "vad-gmm.h"

namespace kaldi {

// returns true if successfully appended.
bool AppendFeats(const std::vector<Matrix<BaseFloat> > &in,
                 std::string utt,
                 int32 tolerance,
                 Matrix<BaseFloat> *out) {
  // Check the lengths
  int32 min_len = in[0].NumRows(),
      max_len = in[0].NumRows(),
      tot_dim = in[0].NumCols();
  for (int32 i = 1; i < in.size(); i++) {
    int32 len = in[i].NumRows(), dim = in[i].NumCols();
    tot_dim += dim;
    if(len < min_len) min_len = len;
    if(len > max_len) max_len = len;
  }
  if (max_len - min_len > tolerance || min_len == 0) {
    KALDI_WARN << "Length mismatch " << max_len << " vs. " << min_len
               << (utt.empty() ? "" : " for utt ") << utt
               << " exceeds tolerance " << tolerance;
    out->Resize(0, 0);
    return false;
  }
  if (max_len - min_len > 0) {
    KALDI_VLOG(2) << "Length mismatch " << max_len << " vs. " << min_len
                  << (utt.empty() ? "" : " for utt ") << utt
                  << " within tolerance " << tolerance;
  }

  out->Resize(min_len, tot_dim);
  int32 dim_offset = 0;
  for (int32 i = 0; i < in.size(); i++) {
    int32 this_dim = in[i].NumCols();
    out->Range(0, min_len, dim_offset, this_dim).CopyFromMat(
        in[i].Range(0, min_len, 0, this_dim));
    dim_offset += this_dim;
  }
  return true;
}

void ComputeFrameSnrsUsingCorruptedFbank(const Matrix<BaseFloat> &clean_fbank,
                                         const Matrix<BaseFloat> &fbank,
                                         Vector<BaseFloat> *frame_snrs,
                                         BaseFloat ceiling = 100) {
  int32 min_len = frame_snrs->Dim();

  for (size_t t = 0; t < min_len; t++) {
    Vector<BaseFloat> clean_fbank_t(clean_fbank.Row(t));
    Vector<BaseFloat> fbank_t(fbank.Row(t));

    BaseFloat clean_energy_t = clean_fbank_t.LogSumExp();
    BaseFloat total_energy_t = fbank_t.LogSumExp();

    if (kaldi::ApproxEqual(total_energy_t, clean_energy_t, 1e-10)) {
      (*frame_snrs)(t) = ceiling;
    } else {
      BaseFloat noise_energy_t = (total_energy_t > clean_energy_t ?
                                  LogSub(total_energy_t, clean_energy_t) :
                                  LogSub(clean_energy_t, total_energy_t) );

      (*frame_snrs)(t) = clean_energy_t - noise_energy_t;
    }
  }
}

Vad::Vad(const VadOptions &opts) : feat_mfcc_(opts.mfcc_opts), post_processor(opts.seg_opts) {
  isrecognize_cont = true;
  frame_shift_ = opts.frame_shift;
  frame_overlap_ = opts.frame_overlap;
  chunk_time_ = opts.chunk_time;
  num_chunks = 0;
  prev_start = 0;
  prev_end = 0;
  curr_start = 0;
  curr_end = 0; 
  use_gpu_log_ = opts.use_gpu_log;
  allow_partial_ = opts.allow_partial;
  decoding_model_ = opts.decoding_model;
  decoding_graph_ = opts.decoding_graph;
  decoding_method_ = opts.decoding_method;
  seg_opts_ = opts.seg_opts;
  speech_to_sil_ratio_ = opts.speech_to_sil_ratio;
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
  decoder_opts_ = opts.decoder_opts;  // true == include obscure settings.
  acoustic_scale_ = opts.acoustic_wt;
  Input nonspeech_input(opts.nonspeech_ubm_filename.c_str(), &binary);
  nonspeech_ubm.Read(nonspeech_input.Stream(), binary);
  Input speech_input(opts.speech_ubm_filename.c_str(), &binary);
  speech_ubm.Read(speech_input.Stream(), binary);
  diag_nonspeech_ubm.CopyFromFullGmm(nonspeech_ubm);
  diag_speech_ubm.CopyFromFullGmm(speech_ubm);
  
  ReadKaldiObject(decoding_model_, &trans_model_);

  decode_fst_ = fst::ReadFstKaldi(decoding_graph_);
 
}

void Vad::reinitiate()
{
  isrecognize_cont = true;
  num_chunks = 0;
  prev_start = 0;
  prev_end = 0;
  curr_start = 0;
  curr_end = 0;
}
bool Vad::Compute_online(const VectorBase<BaseFloat> &waveform, std::vector<std::vector<BaseFloat> > *seg_times, bool isrecording_continue) {
  try{
    //      KALDI_LOG << "chunk";
    Matrix<BaseFloat> hires_features;
    feat_mfcc_.Compute(waveform, &hires_features, 1.0);
    //    Vector<BaseFloat> *vad;
    //    ComputeVadEnergy(vad_opts, hires_features, vad);
    
    Matrix<BaseFloat> delta_feats;
    ComputeDeltas(ns_delta_opts, hires_features, &delta_feats);
    Matrix<BaseFloat> cmvn_feats(delta_feats.NumRows(),
                                 delta_feats.NumCols());
    SlidingWindowCmn(ns_cmn_opts, delta_feats, &cmvn_feats);
    vector<std::vector<int32> > gselect_speech(cmvn_feats.NumRows());
    vector<std::vector<int32> > gselect_nonspeech(cmvn_feats.NumRows());
    diag_speech_ubm.GaussianSelection(cmvn_feats, ns_num_gselect,
                                      &gselect_speech);
    diag_nonspeech_ubm.GaussianSelection(cmvn_feats, ns_num_gselect,
                                         &gselect_nonspeech);
    Matrix<BaseFloat> matrix_log;
    matrix_log.Resize(cmvn_feats.NumRows(), 2);

    for (int32 i = 0; i < cmvn_feats.NumRows(); i++ ) {
      
      Vector<BaseFloat> speech_loglikes;
      Vector<BaseFloat> nonspeech_loglikes;
      speech_ubm.LogLikelihoodsPreselect(cmvn_feats.Row(i),
                                         gselect_speech[i], &speech_loglikes);
      nonspeech_ubm.LogLikelihoodsPreselect(cmvn_feats.Row(i),
                                            gselect_nonspeech[i], &nonspeech_loglikes);
      double speech_prob = speech_loglikes.LogSumExp();
      double nonspeech_prob = nonspeech_loglikes.LogSumExp();
	  matrix_log(i,0) = nonspeech_prob;
	  matrix_log(i,1) = speech_prob;  
	}
	  
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    FasterDecoder decoder(*decode_fst_, decoder_opts_);
    if (matrix_log.NumRows() == 0) {
      KALDI_WARN << "Zero-length utterance: ";
    }

    DecodableMatrixScaledMapped decodable(trans_model_, matrix_log, acoustic_scale_);
    decoder.Decode(&decodable);

    VectorFst<LatticeArc> decoded;  // linear FST.

    if ( (allow_partial_ || decoder.ReachedFinal())
         && decoder.GetBestPath(&decoded) ) {
      if (!decoder.ReachedFinal())
        KALDI_WARN << "Decoder did not reach end-state, outputting partial traceback.";

      std::vector<int32> alignment;
      std::vector<int32> words;
      LatticeWeight weight;

      GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
      //delete decode_fst;

      std::vector<std::vector<int32> > split_phones;
      SplitToPhones(trans_model_, alignment, &split_phones);

      std::vector<int32> phones;
      for (size_t i = 0; i < split_phones.size(); i++) {
        KALDI_ASSERT(!split_phones[i].empty());
        int32 phone = trans_model_.TransitionIdToPhone(split_phones[i][0]);
        int32 num_repeats = split_phones[i].size();
        for(int32 j = 0; j < num_repeats; j++)
          phones.push_back(phone);
      }
      int64 num_segments = 0;
      std::vector<int64> frame_counts_per_class;

      kaldi::segmenter::Segmentation speech_seg_online;
      num_segments = (speech_seg_online).InsertFromAlignment(phones, 0, &frame_counts_per_class);
      KALDI_LOG << "Segmentation initiated from alignment : " << num_segments;

//      BaseFloat frame_shift_ = 0.01, frame_overlap_ = 0.01, chunk_time_ = 1.015;
    
      post_processor.RemoveSegments(&speech_seg_online);
      post_processor.MergeLabels(&speech_seg_online);

      if ((speech_seg_online).Begin()->Length() > 1 || !isrecording_continue) {
        for (kaldi::segmenter::SegmentList::const_iterator it = (speech_seg_online).Begin();
             it != (speech_seg_online).End(); ++it)  {
          curr_start = (num_chunks*chunk_time_) + ((it->start_frame)*frame_shift_);
          curr_end = (num_chunks*chunk_time_) + ((it->end_frame + 1)*frame_shift_ + frame_overlap_);
          //	    KALDI_LOG << "yes";
          curr_start -= pad_length_*frame_shift_;
          if (prev_end > 0) prev_end += pad_length_*frame_shift_;
          //	    if (curr_start <= prev_end) { curr_start = prev_start; }
          if (((curr_start - prev_end) < (seg_opts_.max_intersegment_length*frame_shift_))&&(prev_end>0)) {
            curr_start = prev_start;
          }
          curr_start -= post_pad_length_*frame_shift_;	    
          if (prev_end > 0) prev_end += post_pad_length_*frame_shift_;
          if ((curr_start <= prev_end)&&(prev_end>0)) { curr_start = prev_start; }

          if ((((curr_start - prev_end) > 0) || !isrecording_continue) && ((prev_end - prev_start)>0)) {
            //   return true;
            while ((prev_end - prev_start)>(seg_opts_.max_segment_length*frame_shift_)) {  
              std::vector<BaseFloat> row;
              row.push_back(prev_start);
              row.push_back(std::min(prev_end, prev_start+((seg_opts_.max_segment_length+1)*frame_shift_)+frame_overlap_));
              (*seg_times).push_back(row);
              prev_start += (seg_opts_.max_segment_length - seg_opts_.overlap_length)*frame_shift_;
            }
            //		KALDI_LOG << "PREVIOUS SEG START" << prev_start << " END" << prev_end;
            std::vector<BaseFloat> row;
            row.push_back(prev_start);
            row.push_back(std::min(prev_end, prev_start+((seg_opts_.max_segment_length+1)*frame_shift_)+frame_overlap_));
            (*seg_times).push_back(row);
            isrecognize_cont = false;
				
          }   //else isreco_cont = true;
          prev_start = curr_start; 
          prev_end = curr_end;
        }
        if (!isrecording_continue) {
          while ((prev_end - prev_start)>(seg_opts_.max_segment_length*frame_shift_)) {
            std::vector<BaseFloat> row;
            row.push_back(prev_start);
            row.push_back(std::min(prev_end, prev_start+((seg_opts_.max_segment_length+1)*frame_shift_)+frame_overlap_));
            (*seg_times).push_back(row);
            prev_start += (seg_opts_.max_segment_length - seg_opts_.overlap_length)*frame_shift_;
          }
          //          KALDI_LOG << "PREVIOUS SEG START" << prev_start << " END" << prev_end;
          std::vector<BaseFloat> row;
          row.push_back(prev_start);
          row.push_back(std::min(prev_end, prev_start+((seg_opts_.max_segment_length+1)*frame_shift_)+frame_overlap_));
          (*seg_times).push_back(row);
          isrecognize_cont = false;
        }
      }
    }
    num_chunks++;
    return isrecognize_cont;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return true;
  }


}
}
