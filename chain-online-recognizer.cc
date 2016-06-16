
//      Feature Extraction functions
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"

//      Lattice analysis and decoder functions
#include "fstext/fstext-lib.h"
#include "decoder/lattice-faster-decoder.h"
#include "lat/sausages.h"
#include "lat/confidence.h"
#include "lat/lattice-functions.h"
#include "lm/const-arpa-lm.h"

//      YCYC -Gilad,Kevin need to resolve this
#define PORT_AUDIO_FIX

//      online (DNN Version) functions
#include "nnet3/am-nnet-simple.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"

// Speaker Verification Includes
#include "ivector/voice-activity-detection.h"
#include "ivector/ivector-extractor.h"
#include "ivector/plda.h"

// Speech/Nonspeech functions
#include "ivector/logistic-regression.h"

#include "SNT5_online/KaldiUtils.h"
#include "SNT5_online/chain-recognize.h"

#include "SNT5_online/online-vad.h"

using namespace kaldi;
using namespace fst;
using namespace nnet3;

#define SNT_KALDI_CONF_OUTPUT

#define SNT_KALDI_SMP_FREQ 8000


int main(int argc, char *argv[]) {
  try {
    string use_gpu = "yes";
    VadOptions vad_opts;
    std::string results, chain_model, chain_graph, word_syms_filename_;
    const char *usage =
                "chain-model-ivec-recognizer --config=<conf-file>"
                   "<wav-rspecifier> <output filename> \n"
        "Convert audio to text using chain models";
    // construct all the global objects
    ParseOptions po(usage);
    ParseOptions vadparse("vad", &po);
    ParseOptions chainparse("chaindec", &po);
    (po).Register("use-gpu", &use_gpu,
                  "yes|no|optional|wait,"
                  "only has effect if compiled with CUDA");
    vad_opts.Register(&vadparse);

    OnlineNnet2FeaturePipelineConfig m_configFeature;
    OnlineNnet3DecodingConfig m_configNnet3Decoding;
    m_configFeature.Register(&chainparse);
    m_configNnet3Decoding.Register(&chainparse);
    (chainparse).Register("chain-model", &chain_model, "chain mode main");
    (chainparse).Register("chain-graph", &chain_graph, "decoding chain graph");
    (chainparse).Register("word-symbol-table", &word_syms_filename_, "words syms table for decoding");

    po.Read(argc, argv);

    std::string wav_id = po.GetArg(1); 
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    char *StartingDir = getcwd(NULL, 0);
    // Get current directory so we can return to it at end of program
    char ExePath[PATH_MAX];
    strncpy(ExePath, argv[0], PATH_MAX);
    char *PathEnd = strrchr(ExePath, '/');   // Find end of path
    if (PathEnd) {
      *PathEnd = '\0';    // Terminate the string there
      chdir(ExePath);     // And change the working directory to that path.
    }

    Vad vad(vad_opts);
    int total_samples_proc = 0, prev_proc_time = 0, samp_dec_tracker = 0;
    int chunk_length_smp = ((int)(vad_opts.chunk_time*SNT_KALDI_SMP_FREQ));
    OnlineNnet2FeaturePipelineInfo *m_pFeatInfo        = new OnlineNnet2FeaturePipelineInfo(m_configFeature);
    OnlineIvectorExtractorAdaptationState *m_pAdaptatState    = new OnlineIvectorExtractorAdaptationState(m_pFeatInfo->ivector_extractor_info);
    OnlineNnet2FeaturePipeline *m_pFeaturePipeline = new OnlineNnet2FeaturePipeline(*m_pFeatInfo);

    KaldiUtils::Parameters kaldi_params;
    KaldiUtils *m_KaldiUtils = new KaldiUtils(kaldi_params);
    CompactLattice     clat;

    float last_seg_start, last_seg_end;
    bool isRecoContinue = true, m_bUseVAD = true;
    bool isRecordingContinue = true;
    Vector<BaseFloat>  dataChunk(chunk_length_smp);
    TransitionModel trans_model_;
    nnet3::AmNnetSimple am_nnet_;
    int  rcvSmpCount;
  
    fst::SymbolTable *word_syms_;
    word_syms_ = fst::SymbolTable::ReadText(word_syms_filename_);
    VectorFst<StdArc> *decode_fst_;
    decode_fst_ = fst::ReadFstKaldi(chain_graph);
    Vector<BaseFloat> wave_data;

    bool binary;
    Input ki(chain_model, &binary);
    trans_model_.Read(ki.Stream(), binary);
    am_nnet_.Read(ki.Stream(), binary);

    SingleUtteranceNnet3Decoder *m_pNnetDecoder     = new SingleUtteranceNnet3Decoder(m_configNnet3Decoding,
                                                                 trans_model_,
                                                                 am_nnet_,
                                                                 *decode_fst_,
                                                                 m_pFeaturePipeline);
    

    while(isRecordingContinue)
    {
      isRecordingContinue = Read(&dataChunk, rcvSmpCount, stdin);
      SubVector<BaseFloat> wave_part(dataChunk, 0, rcvSmpCount);
      std::vector<std::vector<BaseFloat> > seg_times;

      if(m_bUseVAD) {
        isRecoContinue = vad.Compute_online(wave_part, &seg_times, wav_id, isRecordingContinue);
        wave_data.Resize(wave_data.Dim()+wave_part.Dim(),kCopyData);
        for(int i = 0; i < wave_part.Dim(); i++)
        {wave_data(wave_data.Dim()-wave_part.Dim()+i) = wave_part(i);}

        if (isRecoContinue && isRecordingContinue) {
          for ( std::vector<std::vector<BaseFloat> >::iterator it
                    = (seg_times).begin();
                it!= (seg_times).end(); ++it) {
            int wav_proc_len = std::min(int(SNT_KALDI_SMP_FREQ*((*it)[1])) - total_samples_proc + int(6*SNT_KALDI_SMP_FREQ*vad_opts.frame_shift), wave_data.Dim()) -
                (std::max(samp_dec_tracker, int(SNT_KALDI_SMP_FREQ*(*it)[0] - total_samples_proc - 12*(SNT_KALDI_SMP_FREQ*vad_opts.frame_shift))));
            //    KALDI_LOG << "wav_proc_len " << wav_proc_len;
            if (wav_proc_len > 0) {
              SubVector<BaseFloat> wav_proc_in (wave_data, (std::max(samp_dec_tracker,
                                                                     int(SNT_KALDI_SMP_FREQ*(*it)[0] - total_samples_proc
						 - 12*(SNT_KALDI_SMP_FREQ*vad_opts.frame_shift)))),
                                                wav_proc_len);
    //          KALDI_LOG << "WAV_PROC FEATS IN : " << wav_proc_len/(SNT_KALDI_SMP_FREQ*vad_opts.frame_shift);
	      m_pFeaturePipeline->AcceptWaveform(SNT_KALDI_SMP_FREQ, wav_proc_in);
//	      KALDI_LOG << "FEAT PIPE FRAMES : " << m_pFeaturePipeline->NumFramesReady();
              samp_dec_tracker = std::min(int(SNT_KALDI_SMP_FREQ*(*it)[1]) -
                                          total_samples_proc + int(6*SNT_KALDI_SMP_FREQ*vad_opts.frame_shift), wave_data.Dim());
    //          KALDI_LOG << "samp_dec_trkr : " << samp_dec_tracker;
              m_pNnetDecoder->AdvanceDecoding();}
            last_seg_start = (*it)[0];
            last_seg_end = (*it)[1];

          }
        }
        else {
          int wav_proc_len = std::max(0, std::min(int(SNT_KALDI_SMP_FREQ*(last_seg_end)) - total_samples_proc + int(6*SNT_KALDI_SMP_FREQ*vad_opts.frame_shift), wave_data.Dim()) -
                                      (std::max(samp_dec_tracker, int(SNT_KALDI_SMP_FREQ*last_seg_start - total_samples_proc - 12*(SNT_KALDI_SMP_FREQ*vad_opts.frame_shift)))));
          //        KALDI_LOG << "wav_proc_len " << wav_proc_len;
          if (wav_proc_len > 0) {
            SubVector<BaseFloat> wav_proc_in (wave_data, (std::max(samp_dec_tracker,
                                                                   int(SNT_KALDI_SMP_FREQ*last_seg_start - total_samples_proc
								- 12*(SNT_KALDI_SMP_FREQ*vad_opts.frame_shift)))),
                                              wav_proc_len);
            m_pFeaturePipeline->AcceptWaveform(SNT_KALDI_SMP_FREQ, wav_proc_in);
            samp_dec_tracker = std::min(int(SNT_KALDI_SMP_FREQ*last_seg_end) -
                                        total_samples_proc + int(6*SNT_KALDI_SMP_FREQ*vad_opts.frame_shift), wave_data.Dim());
        //    KALDI_LOG << "samp_dec_trkr : " << samp_dec_tracker;
            m_pNnetDecoder->AdvanceDecoding();}

          for ( std::vector<std::vector<BaseFloat> >::iterator it
                    = (seg_times).begin();
                it!= (seg_times).end(); ++it) {
//	    KALDI_LOG << "SEG TIMING : " << (*it)[0] << " :: " << (*it)[1];
            CompactLattice clat;
            int wav_proc_len = std::min(int(SNT_KALDI_SMP_FREQ*((*it)[1])) - total_samples_proc + int(6*SNT_KALDI_SMP_FREQ*vad_opts.frame_shift), wave_data.Dim()) -
                (std::max(samp_dec_tracker, int(SNT_KALDI_SMP_FREQ*(*it)[0] - total_samples_proc - 12*(SNT_KALDI_SMP_FREQ*vad_opts.frame_shift))));

            if ((*it)[0] == last_seg_start) {
              if (wav_proc_len > 0) {
                SubVector<BaseFloat> wav_proc_in (wave_data, (std::max(samp_dec_tracker,
                                                                       int(SNT_KALDI_SMP_FREQ*(*it)[0] - total_samples_proc
							- 12*(SNT_KALDI_SMP_FREQ*vad_opts.frame_shift)))),
                                                  wav_proc_len);
                m_pFeaturePipeline->AcceptWaveform(SNT_KALDI_SMP_FREQ, wav_proc_in);
                samp_dec_tracker = std::min(int(SNT_KALDI_SMP_FREQ*(*it)[1]) -
                                            total_samples_proc + int(6*SNT_KALDI_SMP_FREQ*vad_opts.frame_shift), wave_data.Dim());
                m_pNnetDecoder->AdvanceDecoding();}
              last_seg_start = (*it)[0];
              last_seg_end = (*it)[1];
              continue;
            }
            m_pFeaturePipeline->GetAdaptationState(m_pAdaptatState);
//	    KALDI_LOG << m_pNnetDecoder->NumFramesDecoded() << " : DECODER NUMFRAMES";
            if (m_pNnetDecoder->NumFramesDecoded() > 0) {
              m_pNnetDecoder->GetLattice(true, &clat);
	      std::vector<int32> words;
              std::vector<BaseFloat> confidences;
            //  KALDI_LOG << m_pNnetDecoder->NumFramesDecoded() << "DECODER NUMFRAMES";
              MinimumBayesRisk mbr(clat);
              words = mbr.GetOneBest();
              confidences = mbr.GetOneBestConfidences();
              ivec_TransformConfidences(confidences, 0.0, 0.0, 1.0);
             //      m_KaldiUtils->PrintSausageStats(mbr.GetSausageStats(), m_pWordSyms);
              m_KaldiUtils->PrintSausageStatsTimes(mbr.GetSausageStats(),
                                                 mbr.GetSausageTimes(),
                                                 word_syms_, last_seg_start - 12*vad_opts.frame_shift, 3);
              results = m_KaldiUtils->PrintResult(words, confidences, word_syms_, true);

		}
            m_pFeaturePipeline->InputFinished();
            m_pFeaturePipeline = new OnlineNnet2FeaturePipeline(*m_pFeatInfo);
            m_pFeaturePipeline->SetAdaptationState(*m_pAdaptatState);
            delete m_pNnetDecoder;
            m_pNnetDecoder     = new SingleUtteranceNnet3Decoder(m_configNnet3Decoding,
                                                                 trans_model_,
                                                                 am_nnet_,
                                                                 *decode_fst_,
                                                                 m_pFeaturePipeline);

            if (wav_proc_len > 0) {
              SubVector<BaseFloat> wav_proc_in (wave_data, (std::max(samp_dec_tracker,
                                                                     int(SNT_KALDI_SMP_FREQ*(*it)[0] - total_samples_proc
							- 12*(SNT_KALDI_SMP_FREQ*vad_opts.frame_shift)))),
                                                wav_proc_len);

              m_pFeaturePipeline->AcceptWaveform(SNT_KALDI_SMP_FREQ, wav_proc_in);
              samp_dec_tracker = std::min(int(SNT_KALDI_SMP_FREQ*(*it)[1]) - total_samples_proc + int(6*SNT_KALDI_SMP_FREQ*vad_opts.frame_shift), wave_data.Dim());

              m_pNnetDecoder->AdvanceDecoding();}
            last_seg_start = (*it)[0];
            last_seg_end = (*it)[1];
          }
          samp_dec_tracker = 0;
          total_samples_proc = total_samples_proc + wave_data.Dim();
          wave_data.Resize(0);

        }
      }

      else
      {
        isRecoContinue = isRecordingContinue;
        m_pFeaturePipeline->AcceptWaveform(SNT_KALDI_SMP_FREQ, wave_part);
      }
      if (!isRecordingContinue) {
        if (m_pNnetDecoder->NumFramesDecoded() > 0) {
          m_pNnetDecoder->GetLattice(true, &clat);
        std::vector<int32> words;
        std::vector<BaseFloat> confidences;
        MinimumBayesRisk mbr(clat);
        words = mbr.GetOneBest();
        confidences = mbr.GetOneBestConfidences();
        ivec_TransformConfidences(confidences, 0.0, 0.0, 1.0);
        //      m_KaldiUtils->PrintSausageStats(mbr.GetSausageStats(), m_pWordSyms);

        m_KaldiUtils->PrintSausageStatsTimes(mbr.GetSausageStats(),
                                             mbr.GetSausageTimes(),
                                             word_syms_, last_seg_start - 12*vad_opts.frame_shift, 3);
        results = m_KaldiUtils->PrintResult(words, confidences, word_syms_, true);}

      }
    }
    
    vad.out_segfile.close();
    if (StartingDir) {
      chdir(StartingDir);
      free(StartingDir);
      StartingDir = NULL;
    }
  } catch (...) {
    KALDI_WARN << "Failed to compute Voice Activity ";
  }
}

