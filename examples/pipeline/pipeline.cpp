// Proof of concept for true pipeline parallelism (layer split) across
// independent llama contexts. See issue #4.
//
// Loads a dense model layer-split across all GPUs, creates K independent
// llama_contexts with the same prompt, and decodes greedily. In "overlap"
// mode (default) the K llama_decode calls are issued back-to-back before the
// synchronizations, so the per-GPU layer pipeline of one context overlaps the
// still-running pipeline of another. In "--serial" mode each context is
// decoded and synchronized one at a time, matching the stock server's loop.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// number of independent contexts
static int g_n_contexts = 2;

// serial control mode (decode -> sync -> sample per context, no overlap)
static bool g_serial = false;

// number of prompt tokens to build when no -p prompt is given
static int g_prompt_len = 128;

// extract the custom flags (-k/--n-contexts, --serial, --prompt-len) and pass
// the rest of argv to common_params_parse untouched
static std::vector<char *> filter_argv(int argc, char ** argv) {
    std::vector<char *> out;
    out.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-k" || a == "--n-contexts") {
            if (i + 1 < argc) {
                g_n_contexts = std::atoi(argv[++i]);
            }
        } else if (a == "--serial") {
            g_serial = true;
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

static std::string tokens_to_str(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string out;
    for (const llama_token t : tokens) {
        char buf[128];
        const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        if (n >= 0) {
            out.append(buf, n);
        }
    }
    return out;
}

struct slot {
    llama_context * ctx  = nullptr;
    llama_sampler * smpl = nullptr;
    std::vector<llama_token> generated;
    bool eog = false;
};

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    auto fargv = filter_argv(argc, argv);
    if (!common_params_parse((int) fargv.size(), fargv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const int K = std::max(1, g_n_contexts);

    if (params.n_predict <= 0) {
        params.n_predict = 64;
    }
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

    // each context handles a single sequence
    llama_context_params cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;

    std::vector<slot> slots(K);
    for (int k = 0; k < K; ++k) {
        slots[k].ctx = llama_init_from_model(model, cparams);
        if (slots[k].ctx == nullptr) {
            fprintf(stderr, "%s: failed to create context %d\n", __func__, k);
            return 1;
        }

        const auto sparams = llama_sampler_chain_default_params();
        slots[k].smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(slots[k].smpl, llama_sampler_init_greedy());

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size());
        if (llama_decode(slots[k].ctx, batch)) {
            fprintf(stderr, "%s: prefill failed for context %d\n", __func__, k);
            return 1;
        }
        llama_synchronize(slots[k].ctx);
    }

    fprintf(stderr, "%s: %d contexts, prompt_tokens=%zu, n_predict=%d, mode=%s\n",
            __func__, K, prompt_tokens.size(), params.n_predict, g_serial ? "serial" : "overlap");

    const int64_t t0 = ggml_time_us();
    for (int step = 0; step < params.n_predict; ++step) {
        if (g_serial) {
            for (int k = 0; k < K; ++k) {
                llama_token t = llama_sampler_sample(slots[k].smpl, slots[k].ctx, -1);
                llama_batch b = llama_batch_get_one(&t, 1);
                llama_decode(slots[k].ctx, b);
                llama_synchronize(slots[k].ctx);
                slots[k].generated.push_back(t);
                if (llama_vocab_is_eog(vocab, t)) {
                    slots[k].eog = true;
                }
            }
        } else {
            std::vector<llama_token> toks(K);
            for (int k = 0; k < K; ++k) {
                toks[k] = llama_sampler_sample(slots[k].smpl, slots[k].ctx, -1);
            }
            for (int k = 0; k < K; ++k) {
                llama_batch b = llama_batch_get_one(&toks[k], 1);
                llama_decode(slots[k].ctx, b);
            }
            for (int k = 0; k < K; ++k) {
                llama_synchronize(slots[k].ctx);
            }
            for (int k = 0; k < K; ++k) {
                slots[k].generated.push_back(toks[k]);
                if (llama_vocab_is_eog(vocab, toks[k])) {
                    slots[k].eog = true;
                }
            }
        }
    }
    const int64_t t1 = ggml_time_us();

    const double wall_s = (t1 - t0) / 1e6;
    const int total_tokens = K * params.n_predict;
    const double tok_s = wall_s > 0 ? total_tokens / wall_s : 0.0;

    fprintf(stderr, "\n=== results (K=%d, mode=%s) ===\n", K, g_serial ? "serial" : "overlap");
    fprintf(stderr, "wall=%.3fs total_tokens=%d aggregate_tok_s=%.2f\n", wall_s, total_tokens, tok_s);

    for (int k = 0; k < K; ++k) {
        const std::string text = tokens_to_str(vocab, slots[k].generated);
        fprintf(stderr, "ctx[%d]: %zu tokens eog=%d\n%s\n",
                k, slots[k].generated.size(), slots[k].eog ? 1 : 0, text.c_str());
    }

    // machine-readable summary line
    printf("RESULT k=%d mode=%s wall_s=%.3f tokens=%d tok_s=%.2f\n",
           K, g_serial ? "serial" : "overlap", wall_s, total_tokens, tok_s);

    for (int k = 0; k < K; ++k) {
        llama_sampler_free(slots[k].smpl);
        llama_free(slots[k].ctx);
    }
    init.reset(); // free the model, owned by common_init_result
    llama_backend_free();

    return 0;
}
