#include "models.h"

llm_build_hyperclovax::llm_build_hyperclovax(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    int32_t       n_tokens    = this->n_tokens;
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    struct ggml_tensor * input_layer = build_inp_embd(model.tok_embd);

    struct ggml_tensor * cur       = input_layer;
    struct ggml_tensor * input_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale =
        hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * input_sa = input_layer;

        cur = build_norm(input_layer, model.layers[il].attn_norm, model.layers[il].attn_norm_b, LLM_NORM, il);
        cb(cur, "attn_norm", il);

        {
            struct ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }

            struct ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);
            }

            struct ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(ctx0, Qcur, input_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, input_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur", il);

            cur = build_attn(inp_attn, model.layers[il].wo, model.layers[il].bo, Qcur, Kcur, Vcur, nullptr, nullptr,
                             nullptr, kq_scale, il);
        }

        if (il == n_layer - 1) {
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            n_tokens                         = n_outputs;

            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            input_sa = ggml_get_rows(ctx0, input_sa, inp_out_ids);
        }

        if (model.layers[il].attn_post_norm) {
            cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM, il);
            cb(cur, "attn_post_norm", il);
        }

        struct ggml_tensor * ffn_input = ggml_add(ctx0, cur, input_sa);
        cb(ffn_input, "ffn_input", il);

        cur = build_norm(ffn_input, model.layers[il].ffn_norm, model.layers[il].ffn_norm_b, LLM_NORM, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur, model.layers[il].ffn_up, model.layers[il].ffn_up_b, nullptr, model.layers[il].ffn_gate,
                        model.layers[il].ffn_gate_b, nullptr, model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                        nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (model.layers[il].ffn_post_norm) {
            cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM, il);
            cb(cur, "ffn_post_norm", il);
        }

        cur = ggml_add(ctx0, cur, ffn_input);
        cb(cur, "ffn_out", il);

        input_layer = cur;
    }

    cur = build_norm(cur, model.output_norm, model.output_norm_b, LLM_NORM, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    if (hparams.f_logit_scale) {
        cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
    }
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
