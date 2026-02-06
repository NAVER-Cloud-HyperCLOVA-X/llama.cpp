#include "models.h"

ggml_cgraph * clip_graph_siglip::build() {
    ggml_tensor * inp = build_inp();

    ggml_tensor * learned_pos_embd = model.position_embeddings;
    if (proj_type == PROJECTOR_TYPE_LFM2) {
        learned_pos_embd = resize_position_embeddings();
    }

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            nullptr);

    if (proj_type == PROJECTOR_TYPE_GEMMA3) {
        const int batch_size = 1;
        GGML_ASSERT(n_patches_x == n_patches_y);
        const int patches_per_image = n_patches_x;
        const int kernel_size = hparams.n_merge;

        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont_4d(ctx0, cur, patches_per_image, patches_per_image, n_embd, batch_size);

        // doing a pool2d to reduce the number of output tokens
        cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0] * cur->ne[0], n_embd, batch_size);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        // apply norm before projection
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);

        // apply projection
        cur = ggml_mul_mat(ctx0,
            ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_input_proj_w)),
            cur);

    } else if (proj_type == PROJECTOR_TYPE_IDEFICS3) {
        // pixel_shuffle
        // https://github.com/huggingface/transformers/blob/0a950e0bbe1ed58d5401a6b547af19f15f0c195e/src/transformers/models/idefics3/modeling_idefics3.py#L578
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);
        cur = ggml_mul_mat(ctx0, model.projection, cur);

    } else if (proj_type == PROJECTOR_TYPE_LFM2) {
        // pixel unshuffle block
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        // projection, in LFM2-VL input norm is optional
        if (model.mm_input_norm_w) {
            cur = ggml_norm(ctx0, cur, 1e-5); // default nn.LayerNorm
            cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
        }

        if (model.mm_input_norm_b) {
            cur = ggml_add(ctx0, cur, model.mm_input_norm_b);
        }

        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_JANUS_PRO) {
        cur = build_ffn(cur,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr,
            model.mm_1_w, model.mm_1_b,
            hparams.ffn_op,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_CABSTRACTOR) {
        const int num_queries = hparams.num_queries;

        ggml_tensor * embeddings = cur;
        // ne: whcn
        // embeddings shape: 1024, 576, 1, 1 (1 x 576 x 1024)
        int hw = int(pow(embeddings->ne[1], 0.5));
        embeddings = ggml_add(ctx0, model.mm_model_pos_embd, embeddings);

        // preparing for conv
        // embeddings shape: 1024, 24, 24, 1
        embeddings = ggml_reshape_4d(ctx0, embeddings, embeddings->ne[0], hw, hw, embeddings->ne[3]);

        // embeddings shape: 24, 24, 1024, 1
        embeddings = ggml_cont(ctx0, ggml_permute(ctx0, embeddings, 2, 0, 1, 3));

        struct ggml_tensor * block_1 = nullptr;
        // 0
        {
            // block1
            {

                // conv1
                {
                    // note: ggml_conv_2d: s - stride, p - padding, d - dilation
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_0_block_1_conv_1_conv_w, embeddings, 1, 1, 0, 0, 1,
                                           1);
                    // LayerNormAct2d
                    // 24, 24, 1024, 1 -> 1024, 24, 24, 1
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_0_block_1_conv_1_bn_w),
                                       model.mm_model_0_block_1_conv_1_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // conv2
                {
                    block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_0_block_1_conv_2_conv_w, block_1, 1, 1, 1,
                                                     1, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_0_block_1_conv_2_bn_w),
                                       model.mm_model_0_block_1_conv_2_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // SE module
                {
                    struct ggml_tensor *block_1_se = ggml_pool_2d(ctx0, block_1, GGML_OP_POOL_AVG, block_1->ne[0],
                                                                  block_1->ne[1], block_1->ne[0], block_1->ne[1], 0,
                                                                  0);
                    // fc1
                    block_1_se = ggml_conv_2d(ctx0, model.mm_model_0_block_1_se_fc1_w, block_1_se, 1, 1, 0, 0, 1,
                                              1);
                    block_1_se = ggml_add(ctx0, block_1_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_0_block_1_se_fc1_b, 1, 1,
                                                          model.mm_model_0_block_1_se_fc1_b->ne[0]));
                    block_1_se = ggml_silu_inplace(ctx0, block_1_se);
                    // fc2
                    block_1_se = ggml_conv_2d(ctx0, model.mm_model_0_block_1_se_fc2_w, block_1_se, 1, 1, 0, 0, 1,
                                              1);
                    block_1_se = ggml_add(ctx0, block_1_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_0_block_1_se_fc2_b, 1, 1,
                                                          model.mm_model_0_block_1_se_fc2_b->ne[0]));
                    // gate
                    block_1_se = ggml_sigmoid(ctx0, block_1_se);

                    block_1 = ggml_mul(ctx0, block_1, block_1_se);
                }
                // conv3
                {
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_0_block_1_conv_3_conv_w, block_1, 1, 1, 0, 0, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_0_block_1_conv_3_bn_w),
                                       model.mm_model_0_block_1_conv_3_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    // residual connect
                    block_1 = ggml_add(ctx0, block_1, embeddings);
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                    embeddings = block_1;
                }
            } // block1
            // block2
            {

                // conv1
                {
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_0_block_2_conv_1_conv_w, block_1, 1, 1, 0, 0, 1,
                                           1);
                    // LayerNormAct2d
                    // 24, 24, 1024, 1 -> 1024, 24, 24, 1
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_0_block_2_conv_1_bn_w),
                                       model.mm_model_0_block_2_conv_1_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // conv2
                {
                    block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_0_block_2_conv_2_conv_w, block_1, 1, 1, 1,
                                                      1, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_0_block_2_conv_2_bn_w),
                                       model.mm_model_0_block_2_conv_2_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // SE module
                {
                    struct ggml_tensor *block_2_se = ggml_pool_2d(ctx0, block_1, GGML_OP_POOL_AVG, block_1->ne[0],
                                                                  block_1->ne[1], block_1->ne[0], block_1->ne[1], 0,
                                                                  0);
                    // fc1
                    block_2_se = ggml_conv_2d(ctx0, model.mm_model_0_block_2_se_fc1_w, block_2_se, 1, 1, 0, 0, 1,
                                              1);
                    block_2_se = ggml_add(ctx0, block_2_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_0_block_2_se_fc1_b, 1, 1,
                                                          model.mm_model_0_block_2_se_fc1_b->ne[0]));
                    block_2_se = ggml_silu_inplace(ctx0, block_2_se);
                    // fc2
                    block_2_se = ggml_conv_2d(ctx0, model.mm_model_0_block_2_se_fc2_w, block_2_se, 1, 1, 0, 0, 1,
                                              1);
                    block_2_se = ggml_add(ctx0, block_2_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_0_block_2_se_fc2_b, 1, 1,
                                                          model.mm_model_0_block_2_se_fc2_b->ne[0]));
                    // gate
                    block_2_se = ggml_sigmoid(ctx0, block_2_se);

                    block_1 = ggml_mul(ctx0, block_1, block_2_se);
                }
                // conv3
                {
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_0_block_2_conv_3_conv_w, block_1, 1, 1, 0, 0, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_0_block_2_conv_3_bn_w),
                                       model.mm_model_0_block_2_conv_3_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    // residual connect
                    block_1 = ggml_add(ctx0, block_1, embeddings);
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                    embeddings = block_1;
                }
            } // block2
            // block3
            {

                // conv1
                {
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_0_block_3_conv_1_conv_w, block_1, 1, 1, 0, 0, 1,
                                           1);
                    // LayerNormAct2d
                    // 24, 24, 1024, 1 -> 1024, 24, 24, 1
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_0_block_3_conv_1_bn_w),
                                       model.mm_model_0_block_3_conv_1_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // conv2
                {
                    block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_0_block_3_conv_2_conv_w, block_1, 1, 1, 1,
                                                     1, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_0_block_3_conv_2_bn_w),
                                       model.mm_model_0_block_3_conv_2_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // SE module
                {
                    struct ggml_tensor *block_3_se = ggml_pool_2d(ctx0, block_1, GGML_OP_POOL_AVG, block_1->ne[0],
                                                                  block_1->ne[1], block_1->ne[0], block_1->ne[1], 0,
                                                                  0);
                    // fc1
                    block_3_se = ggml_conv_2d(ctx0, model.mm_model_0_block_3_se_fc1_w, block_3_se, 1, 1, 0, 0, 1,
                                              1);
                    block_3_se = ggml_add(ctx0, block_3_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_0_block_3_se_fc1_b, 1, 1,
                                                          model.mm_model_0_block_3_se_fc1_b->ne[0]));
                    block_3_se = ggml_silu_inplace(ctx0, block_3_se);
                    // fc2
                    block_3_se = ggml_conv_2d(ctx0, model.mm_model_0_block_3_se_fc2_w, block_3_se, 1, 1, 0, 0, 1,
                                              1);
                    block_3_se = ggml_add(ctx0, block_3_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_0_block_3_se_fc2_b, 1, 1,
                                                          model.mm_model_0_block_3_se_fc2_b->ne[0]));
                    // gate
                    block_3_se = ggml_sigmoid(ctx0, block_3_se);

                    block_1 = ggml_mul(ctx0, block_1, block_3_se);
                }
                // conv3
                {
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_0_block_3_conv_3_conv_w, block_1, 1, 1, 0, 0, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_0_block_3_conv_3_bn_w),
                                       model.mm_model_0_block_3_conv_3_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    // residual connect
                    block_1 = ggml_add(ctx0, block_1, embeddings);
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                    embeddings = block_1;
                }
            }//block3
        }

        // 1 - Adaptive average pool 2d
        {
            int target_size = int(pow(num_queries, 0.5));
            int stride = int(block_1->ne[0] / target_size);
            int kernel_size = block_1->ne[0] - (target_size - 1) * stride;
            block_1 = ggml_pool_2d(ctx0, block_1, GGML_OP_POOL_AVG, kernel_size, kernel_size, stride, stride, 0, 0);
        }
        embeddings = block_1;

        // 2
        {// block1
            {

                // conv1
                {
                    // note: ggml_conv_2d: s - stride, p - padding, d - dilation
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_2_block_1_conv_1_conv_w, block_1, 1, 1, 0, 0, 1, 1);
                    // LayerNormAct2d
                    // 24, 24, 1024, 1 -> 1024, 24, 24, 1
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_2_block_1_conv_1_bn_w),
                                       model.mm_model_2_block_1_conv_1_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // conv2
                {
                    block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_2_block_1_conv_2_conv_w, block_1, 1, 1, 1,
                                                     1, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_2_block_1_conv_2_bn_w),
                                       model.mm_model_2_block_1_conv_2_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // SE module
                {
                    struct ggml_tensor *block_1_se = ggml_pool_2d(ctx0, block_1, GGML_OP_POOL_AVG, block_1->ne[0],
                                                                  block_1->ne[1], block_1->ne[0], block_1->ne[1], 0,
                                                                  0);
                    // fc1
                    block_1_se = ggml_conv_2d(ctx0, model.mm_model_2_block_1_se_fc1_w, block_1_se, 1, 1, 0, 0, 1,
                                              1);
                    block_1_se = ggml_add(ctx0, block_1_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_2_block_1_se_fc1_b, 1, 1,
                                                          model.mm_model_2_block_1_se_fc1_b->ne[0]));
                    block_1_se = ggml_silu_inplace(ctx0, block_1_se);
                    // fc2
                    block_1_se = ggml_conv_2d(ctx0, model.mm_model_2_block_1_se_fc2_w, block_1_se, 1, 1, 0, 0, 1,
                                              1);
                    block_1_se = ggml_add(ctx0, block_1_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_2_block_1_se_fc2_b, 1, 1,
                                                          model.mm_model_2_block_1_se_fc2_b->ne[0]));
                    // gate
                    block_1_se = ggml_sigmoid(ctx0, block_1_se);

                    block_1 = ggml_mul(ctx0, block_1, block_1_se);
                }
                // conv3
                {
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_2_block_1_conv_3_conv_w, block_1, 1, 1, 0, 0, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_2_block_1_conv_3_bn_w),
                                       model.mm_model_2_block_1_conv_3_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_add(ctx0, block_1, embeddings);
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                    embeddings = block_1;
                }
            } // block1
            // block2
            {

                // conv1
                {
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_2_block_2_conv_1_conv_w, block_1, 1, 1, 0, 0, 1,
                                           1);
                    // LayerNormAct2d
                    // 24, 24, 1024, 1 -> 1024, 24, 24, 1
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_2_block_2_conv_1_bn_w),
                                       model.mm_model_2_block_2_conv_1_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // conv2
                {
                    block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_2_block_2_conv_2_conv_w, block_1, 1, 1, 1,
                                                     1, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_2_block_2_conv_2_bn_w),
                                       model.mm_model_2_block_2_conv_2_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // SE module
                {
                    struct ggml_tensor *block_2_se = ggml_pool_2d(ctx0, block_1, GGML_OP_POOL_AVG, block_1->ne[0],
                                                                  block_1->ne[1], block_1->ne[0], block_1->ne[1], 0,
                                                                  0);
                    // fc1
                    block_2_se = ggml_conv_2d(ctx0, model.mm_model_2_block_2_se_fc1_w, block_2_se, 1, 1, 0, 0, 1,
                                              1);
                    block_2_se = ggml_add(ctx0, block_2_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_2_block_2_se_fc1_b, 1, 1,
                                                          model.mm_model_2_block_2_se_fc1_b->ne[0]));
                    block_2_se = ggml_silu_inplace(ctx0, block_2_se);
                    // fc2
                    block_2_se = ggml_conv_2d(ctx0, model.mm_model_2_block_2_se_fc2_w, block_2_se, 1, 1, 0, 0, 1,
                                              1);
                    block_2_se = ggml_add(ctx0, block_2_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_2_block_2_se_fc2_b, 1, 1,
                                                          model.mm_model_2_block_2_se_fc2_b->ne[0]));
                    // gate
                    block_2_se = ggml_sigmoid(ctx0, block_2_se);

                    block_1 = ggml_mul(ctx0, block_1, block_2_se);
                }
                // conv3
                {
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_2_block_2_conv_3_conv_w, block_1, 1, 1, 0, 0, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_2_block_2_conv_3_bn_w),
                                       model.mm_model_2_block_2_conv_3_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_add(ctx0, block_1, embeddings);
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                    embeddings = block_1;
                }
            } // block2
            // block3
            {

                // conv1
                {
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_2_block_3_conv_1_conv_w, block_1, 1, 1, 0, 0, 1,
                                           1);
                    // LayerNormAct2d
                    // 24, 24, 1024, 1 -> 1024, 24, 24, 1
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_2_block_3_conv_1_bn_w),
                                       model.mm_model_2_block_3_conv_1_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // conv2
                {
                    block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_2_block_3_conv_2_conv_w, block_1, 1, 1, 1,
                                                     1, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_2_block_3_conv_2_bn_w),
                                       model.mm_model_2_block_3_conv_2_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                }
                // SE module
                {
                    struct ggml_tensor *block_3_se = ggml_pool_2d(ctx0, block_1, GGML_OP_POOL_AVG, block_1->ne[0],
                                                                  block_1->ne[1], block_1->ne[0], block_1->ne[1], 0,
                                                                  0);
                    // fc1
                    block_3_se = ggml_conv_2d(ctx0, model.mm_model_2_block_3_se_fc1_w, block_3_se, 1, 1, 0, 0, 1,
                                              1);
                    block_3_se = ggml_add(ctx0, block_3_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_2_block_3_se_fc1_b, 1, 1,
                                                          model.mm_model_2_block_3_se_fc1_b->ne[0]));
                    block_3_se = ggml_silu_inplace(ctx0, block_3_se);
                    // fc2
                    block_3_se = ggml_conv_2d(ctx0, model.mm_model_2_block_3_se_fc2_w, block_3_se, 1, 1, 0, 0, 1,
                                              1);
                    block_3_se = ggml_add(ctx0, block_3_se,
                                          ggml_reshape_3d(ctx0, model.mm_model_2_block_3_se_fc2_b, 1, 1,
                                                          model.mm_model_2_block_3_se_fc2_b->ne[0]));
                    // gate
                    block_3_se = ggml_sigmoid(ctx0, block_3_se);

                    block_1 = ggml_mul(ctx0, block_1, block_3_se);
                }
                // conv3
                {
                    block_1 = ggml_conv_2d(ctx0, model.mm_model_2_block_3_conv_3_conv_w, block_1, 1, 1, 0, 0, 1, 1);
                    // LayerNormAct2d
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_2_block_3_conv_3_bn_w),
                                       model.mm_model_2_block_3_conv_3_bn_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    block_1 = ggml_add(ctx0, block_1, embeddings);
                    block_1 = ggml_silu_inplace(ctx0, block_1);
                    embeddings = block_1;
                }
            } // block3
        }  // 2

        // 12, 12, 1024, 1 (1, 1024, 12, 12) -> 1024, 144, 1
        block_1 = ggml_reshape_3d(ctx0, block_1, block_1->ne[0] * block_1->ne[1], block_1->ne[2], block_1->ne[3]);
        block_1 = ggml_cont(ctx0, ggml_transpose(ctx0, block_1));

        // readout
        embeddings = ggml_mul_mat(ctx0, model.mm_model_readout_0_w, block_1);
        embeddings = ggml_add(ctx0, embeddings, model.mm_model_readout_0_b);
        embeddings = ggml_silu(ctx0, embeddings);
        embeddings = ggml_mul_mat(ctx0, model.mm_model_readout_2_w, embeddings);
        embeddings = ggml_add(ctx0, embeddings, model.mm_model_readout_2_b);
        cur = embeddings;
    } else {
        GGML_ABORT("SigLIP: Unsupported projector type");
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
