// Chain-decode mechanism test for Theory B (issue #5).
//
// Single context, single sequence. Issue M decodes of a guessed token back-to-back
// before a single llama_synchronize, and compare correctness + wall time against the
// serial reference (decode -> sync per step).
//
// Env: LLAMA_GRAPH_REUSE_DISABLE=0 (default) vs =1 toggles the graph-reuse fast path
// that synchronizes between decodes when pipeline_parallel is on.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// M: decodes issued back-to-back before one sync
static int g_chain = 2;

// outer steps (each step issues M decodes then samples once)
static int g_repeats = 64;

// number of prompt tokens to build when no -p prompt is given
static int g_prompt_len = 128;

static std::vector<char *> filter_argv(int argc, char ** argv) {
    std::vector<char *> out;
    out.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--chain") {
            if (i + 1 < argc) {
                g_chain = std::atoi(argv[++i]);
            }
        } else if (a == "--repeats") {
            if (i + 1 < argc) {
                g_repeats = std::atoi(argv[++i]);
            }
        } else if (a == "--prompt-len") {
            if (i + 1 < argc) {
                g_prompt_len = std::atoi(argv[++i]);
            }
        } else {
            out.push_back(argv[i]);
        }
    }
    return out;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n, true, true) < 0) {
        tokens.clear();
    }
    return tokens;
}

// build a prompt of at least n_tokens tokens by repeating a fixed sentence
static std::string make_prompt(const llama_vocab * vocab, int n_tokens) {
    const std::string sentence =
        "The quick brown fox jumps over the lazy dog while the sun rises over the "
        "quiet valley and the river flows gently beneath the old stone bridge "
        "past fields of wheat and groves of ancient oak trees standing tall. ";

    std::string prompt;
    for (;;) {
        const std::string candidate = prompt + sentence;
        const int n = -llama_tokenize(vocab, candidate.c_str(), candidate.size(), nullptr, 0, true, true);
        if (n >= n_tokens) {
            return candidate;
        }
        prompt = candidate;
        if (prompt.size() > 1024 * 1024) {
            return candidate;
        }
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    auto fargv = filter_argv(argc, argv);
    if (!common_params_parse((int) fargv.size(), fargv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const int M = std::max(1, g_chain);
    const int R = std::max(1, g_repeats);

    if (params.n_ctx == 0) {
        params.n_ctx = 1024;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params, /*model_only=*/true);
    llama_model * model = init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const std::string prompt = params.prompt.empty() ? make_prompt(vocab, g_prompt_len) : params.prompt;
    std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt);
    if (prompt_tokens.empty()) {
        fprintf(stderr, "%s: failed to tokenize prompt\n", __func__);
        return 1;
    }

    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;

    const auto mkctx = [&]() -> llama_context * {
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx == nullptr) {
            fprintf(stderr, "%s: failed to create context\n", __func__);
            return nullptr;
        }
        llama_batch b = llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size());
        if (llama_decode(ctx, b)) {
            fprintf(stderr, "%s: prefill failed\n", __func__);
            return nullptr;
        }
        llama_synchronize(ctx);
        return ctx;
    };

    const auto mk_smpl = []() -> llama_sampler * {
        const auto sp = llama_sampler_chain_default_params();
        llama_sampler * s = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(s, llama_sampler_init_greedy());
        return s;
    };

    llama_context * ctx_serial = mkctx();
    llama_context * ctx_chain  = mkctx();
    if (ctx_serial == nullptr || ctx_chain == nullptr) {
        return 1;
    }

    llama_sampler * smpl_serial = mk_smpl();
    llama_sampler * smpl_chain  = mk_smpl();

    const llama_token t0 = llama_sampler_sample(smpl_serial, ctx_serial, -1);

    // serial reference: decode -> sync per step, sample after M steps
    std::vector<llama_token> hist_serial(R);
    llama_token t_ser = t0;
    const int64_t ts0 = ggml_time_us();
    for (int r = 0; r < R; ++r) {
        for (int m = 0; m < M; ++m) {
            llama_batch b = llama_batch_get_one(&t_ser, 1);
            llama_decode(ctx_serial, b);
            llama_synchronize(ctx_serial);
        }
        t_ser = llama_sampler_sample(smpl_serial, ctx_serial, -1);
        hist_serial[r] = t_ser;
    }
    const int64_t ts1 = ggml_time_us();

    // chain: M decodes back-to-back, then one sync, then sample
    std::vector<llama_token> hist_chain(R);
    llama_token t_ch = t0;
    const int64_t tc0 = ggml_time_us();
    for (int r = 0; r < R; ++r) {
        for (int m = 0; m < M; ++m) {
            llama_batch b = llama_batch_get_one(&t_ch, 1);
            llama_decode(ctx_chain, b);
        }
        llama_synchronize(ctx_chain);
        t_ch = llama_sampler_sample(smpl_chain, ctx_chain, -1);
        hist_chain[r] = t_ch;
    }
    const int64_t tc1 = ggml_time_us();

    const double serial_s = (ts1 - ts0) / 1e6;
    const double chain_s  = (tc1 - tc0) / 1e6;

    int n_match = 0;
    for (int r = 0; r < R; ++r) {
        if (hist_serial[r] == hist_chain[r]) {
            ++n_match;
        }
    }

    fprintf(stderr, "%s: M=%d repeats=%d serial=%.3fs chain=%.3fs speedup=%.2fx samples_match=%d/%d\n",
            __func__, M, R, serial_s, chain_s,
            serial_s > 0 ? serial_s / chain_s : 0.0, n_match, R);

    printf("RESULT chain M=%d repeats=%d serial_wall_s=%.3f chain_wall_s=%.3f speedup=%.2f samples_match=%d/%d\n",
           M, R, serial_s, chain_s,
           serial_s > 0 ? serial_s / chain_s : 0.0, n_match, R);

    llama_sampler_free(smpl_serial);
    llama_sampler_free(smpl_chain);
    llama_free(ctx_serial);
    llama_free(ctx_chain);
    init.reset();
    llama_backend_free();

    return 0;
}
