# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense DSpark late-conditioning stages, with request-owned context K/V.

Batched projections and variable-length attention use no target workspace,
paged-cache writes, collectives, or forward-context mutation.
"""

from itertools import accumulate

import torch
import torch.nn.functional as F


def rms_norm(x, norm):
    value = x.float()
    value = value * torch.rsqrt(
        value.square().mean(-1, keepdim=True) + norm.variance_epsilon
    )
    return norm.weight * value.to(x.dtype)


def device_ints(values, device, dtype=torch.long):
    return torch.tensor(
        values, dtype=dtype, pin_memory=torch.device(device).type == "cuda"
    ).to(device, non_blocking=True)


def packed_rows(lengths, device):
    offsets = device_ints([0, *accumulate(lengths)], device, dtype=torch.int32)
    owners = torch.repeat_interleave(
        torch.arange(len(lengths), device=device),
        device_ints(lengths, device),
        output_size=sum(lengths),
    )
    local = torch.arange(sum(lengths), device=device) - offsets[owners]
    return owners, local, offsets


class LateConditioningStages:
    def __init__(
        self, model, max_requests, max_length, device, dtype, num_speculative_tokens=7
    ):
        self.num_query_per_req = num_speculative_tokens
        self.graphs = {}
        self.context_graphs = {}
        self.model = model
        self.backbone = model.model
        config = model.config
        late = config.dflash_config["late_conditioning"]
        self.bank_size = late.get("bank_size", 12)
        self.head_dim = config.head_dim
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.hidden = config.hidden_size
        self.max_length = max_length
        self.varlen_attention = None
        if torch.device(device).type == "cuda":
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            self.varlen_attention = flash_attn_varlen_func
        theta = config.rope_parameters.get("rope_theta", 1000000.0)
        self.inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim
            )
        )
        if torch.device(device).type == "cuda":
            required = (
                5
                * 2
                * max_requests
                * max_length
                * self.n_kv_heads
                * self.head_dim
                * torch.empty((), dtype=dtype).element_size()
            )
            free, _ = torch.accelerator.get_memory_info(device)
            if required > free:
                raise ValueError(
                    "4+1 context reservation exceeds free GPU memory; reduce "
                    "--max-num-seqs or --max-model-len"
                )
        # Allocated at model load, so vLLM's memory profiling accounts for it.
        self.kv = torch.empty(
            5,
            2,
            max_requests,
            max_length,
            self.n_kv_heads,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

    def init_context_projection(self):
        if hasattr(self, "context_weight"):
            return
        attns = [layer.self_attn for layer in self.backbone.layers]
        self.context_weight = torch.cat([a.qkv_proj.weight[a.q_size :] for a in attns])
        self.context_norm = torch.stack([a.k_norm.weight for a in attns])[
            None, :, None, :
        ]
        self.context_eps = torch.tensor(
            [a.k_norm.variance_epsilon for a in attns], device=self.kv.device
        )[None, :, None, None]

    def select_graph(self, batch, length, finish):
        if not self.graphs:
            return None
        batch = next(n for n in self.graph_batches if n >= batch)
        capacity = next(n for n in self.graph_lengths if n >= length)
        return self.graphs[finish, batch, capacity]

    def capture_graphs(self):
        if self.graphs:
            return
        self.graph_batches = graph_buckets(self.kv.shape[2])
        self.graph_lengths = graph_buckets(self.max_length, first=128, factor=4)
        stream = torch.cuda.Stream(device=self.kv.device)
        stream.wait_stream(torch.cuda.current_stream(self.kv.device))
        # Separate pools are essential: bank can run concurrently with target
        # and finish. Captures sharing a pool are only replayed serially.
        for finish in (False, True):
            pool = torch.cuda.graph_pool_handle()
            for batch in reversed(self.graph_batches):
                for capacity in reversed(self.graph_lengths):
                    graph = LateStageGraph(self, batch, capacity, finish)
                    stream.wait_stream(torch.cuda.current_stream(self.kv.device))
                    graph.capture(pool, stream)
                    self.graphs[finish, batch, capacity] = graph
        self.init_context_projection()
        pool = torch.cuda.graph_pool_handle()
        for batch in reversed(self.graph_batches):
            context_graph = LateContextGraph(self, batch)
            context_graph.capture(pool, stream)
            self.context_graphs[batch] = context_graph
        stream.synchronize()

    def rope(self, x, positions):
        angles = positions.float()[:, None] * self.inv_freq[None, :]
        angles = torch.cat((angles, angles), -1)[:, None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        left, right = x.chunk(2, -1)
        return x * cos + torch.cat((-right, left), -1) * sin

    def append_context(self, slot, start, features):
        self.append_context_batch([slot], [start], features, [features.shape[0]])

    def append_context_batch(self, slots, starts, features, lengths):
        if any(
            not 0 <= start <= start + n <= self.max_length
            for start, n in zip(starts, lengths)
        ):
            raise ValueError("4+1 context exceeds the reserved cache")
        if not sum(lengths):
            return
        owners, local, _ = packed_rows(lengths, features.device)
        rows = device_ints(slots, features.device)[owners]
        positions = device_ints(starts, features.device)[owners] + local
        context = rms_norm(
            F.linear(features, self.backbone.fc.weight), self.backbone.hidden_norm
        )
        for index, layer in enumerate(self.backbone.layers):
            attn = layer.self_attn
            kv = F.linear(context, attn.qkv_proj.weight[attn.q_size :])
            k, v = kv.chunk(2, -1)
            k = rms_norm(k.view(-1, self.n_kv_heads, self.head_dim), attn.k_norm)
            k = self.rope(k, positions)
            self.kv[index, 0, rows, positions] = k
            self.kv[index, 1, rows, positions] = v.view(
                -1, self.n_kv_heads, self.head_dim
            )

    def batch_layout(self, slots, lengths, width):
        device = self.kv.device
        owners, local, context_offsets = packed_rows(lengths, device)
        cu_q = torch.arange(len(slots) + 1, device=device, dtype=torch.int32) * width
        cu_k = context_offsets + cu_q
        context_dst = torch.arange(sum(lengths), device=device) + owners * width
        query_dst = (cu_k[1:] - width)[:, None] + torch.arange(width, device=device)
        return (
            device_ints(slots, device)[owners],
            local,
            context_dst,
            query_dst.flatten(),
            cu_q,
            cu_k,
            lengths,
            width,
        )

    def layer_batch(self, index, states, positions, layout):
        layer = self.backbone.layers[index]
        attn = layer.self_attn
        residual = states
        x = rms_norm(states, layer.input_layernorm)
        qkv = F.linear(x, attn.qkv_proj.weight)
        q, k, v = qkv.split((attn.q_size, attn.kv_size, attn.kv_size), -1)
        q = self.rope(
            rms_norm(q.view(-1, self.n_heads, self.head_dim), attn.q_norm), positions
        )
        k = self.rope(
            rms_norm(k.view(-1, self.n_kv_heads, self.head_dim), attn.k_norm), positions
        )
        rows, local, ctx_dst, query_dst, cu_q, cu_k, lengths, width = layout
        total = sum(lengths) + len(lengths) * width
        packed_k = k.new_empty(total, self.n_kv_heads, self.head_dim)
        packed_v = torch.empty_like(packed_k)
        v = v.view(-1, self.n_kv_heads, self.head_dim)
        if ctx_dst is None:
            # Graph layout stores per-request slots/lengths, not capacity-sized
            # gather indices. The packer never reads or writes unused capacity.
            if k.is_cuda:
                from vllm.model_executor.models.dspark_late_kernels import pack_kv

                pack_kv(
                    self.kv[index, 0],
                    self.kv[index, 1],
                    k,
                    v,
                    rows,
                    local,
                    cu_k,
                    packed_k,
                    packed_v,
                    width,
                    max(lengths),
                )
            else:
                # CPU reference for the same valid-only packing contract.
                for i, (slot, length, start) in enumerate(
                    zip(rows.tolist(), local.tolist(), cu_k.tolist())
                ):
                    packed_k[start : start + length] = self.kv[index, 0, slot, :length]
                    packed_v[start : start + length] = self.kv[index, 1, slot, :length]
                    packed_k[start + length : start + length + width] = k[
                        i * width : (i + 1) * width
                    ]
                    packed_v[start + length : start + length + width] = v[
                        i * width : (i + 1) * width
                    ]
        else:
            packed_k[ctx_dst] = self.kv[index, 0, rows, local]
            packed_v[ctx_dst] = self.kv[index, 1, rows, local]
            packed_k[query_dst] = k
            packed_v[query_dst] = v
        # Context is read-only; temporary bank K/V never enter the cache.
        if self.varlen_attention is not None:
            y = self.varlen_attention(
                q=q,
                k=packed_k,
                v=packed_v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=width,
                max_seqlen_k=max(lengths) + width,
                causal=False,
            )
        else:
            # CPU reference for tests; CUDA always uses one varlen call.
            outputs = []
            boundaries = cu_k.tolist()
            for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                outputs.append(
                    F.scaled_dot_product_attention(
                        q[i * width : (i + 1) * width].transpose(0, 1).unsqueeze(0),
                        packed_k[start:end].transpose(0, 1).unsqueeze(0),
                        packed_v[start:end].transpose(0, 1).unsqueeze(0),
                        is_causal=False,
                        enable_gqa=True,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
            y = torch.cat(outputs)
        y = y.reshape(states.shape[0], -1)
        x = residual + F.linear(y, attn.o_proj.weight)
        gate, up = F.linear(
            rms_norm(x, layer.post_attention_layernorm), layer.mlp.gate_up_proj.weight
        ).chunk(2, -1)
        return x + F.linear(F.silu(gate) * up, layer.mlp.down_proj.weight)

    def prepare_bank(self, slot, length, bank_start):
        return self.prepare_bank_batch([slot], [length], [bank_start])[0]

    def prepare_bank_batch(self, slots, lengths, bank_starts):
        graph = self.select_graph(len(slots), max(lengths), False)
        if graph is not None:
            return graph.replay(slots, lengths, bank_starts)
        states = self.backbone.future_mask.expand(len(slots) * self.bank_size, -1)
        positions = (
            device_ints(bank_starts, states.device)[:, None]
            + torch.arange(self.bank_size, device=states.device)
        ).flatten()
        layout = self.batch_layout(slots, lengths, self.bank_size)
        for index in range(4):
            states = self.layer_batch(index, states, positions, layout)
        return states.view(len(slots), self.bank_size, -1)

    def finish(self, slot, length, bank, bank_start, bonus_embedding):
        return self.finish_batch(
            [slot],
            [length],
            bank.unsqueeze(0),
            [bank_start],
            bonus_embedding.reshape(1, -1),
        )[0]

    def finish_batch(self, slots, lengths, banks, bank_starts, bonus_embeddings):
        width = self.num_query_per_req
        offsets = [length + 1 - start for length, start in zip(lengths, bank_starts)]
        if any(not 0 <= offset <= banks.shape[1] - (width - 1) for offset in offsets):
            raise ValueError("4+1 bank does not cover this acceptance boundary")
        graph = self.select_graph(len(slots), max(lengths), True)
        if graph is not None:
            return graph.replay(slots, lengths, bank_starts, banks, bonus_embeddings)
        indices = device_ints(offsets, banks.device)[:, None] + torch.arange(
            width - 1, device=banks.device
        )
        selected = banks.gather(1, indices[..., None].expand(-1, -1, self.hidden))
        anchor = F.linear(bonus_embeddings, self.backbone.anchor_adapter.weight)
        states = torch.cat((anchor[:, None], selected), dim=1).flatten(0, 1)
        positions = (
            device_ints(lengths, banks.device)[:, None]
            + torch.arange(width, device=banks.device)
        ).flatten()
        states = self.layer_batch(
            4, states, positions, self.batch_layout(slots, lengths, width)
        )
        return rms_norm(states, self.backbone.norm).view(len(slots), width, -1)


def graph_buckets(limit, first=1, factor=2):
    sizes = []
    size = first
    while size < limit:
        sizes.append(size)
        size *= factor
    return [*sizes, limit]


class LateStageGraph:
    """One bounded shape; only metadata changes between replays.

    Bank and finish use different pools. Outputs are copied out before another
    replay can overwrite the pool, including a legacy full-acceptance rebuild.
    """

    def __init__(self, stages, batch, context_capacity, finish):
        self.stages = stages
        self.batch = batch
        self.capacity = context_capacity
        self.finish = finish
        device, dtype = stages.kv.device, stages.kv.dtype
        self.slots = torch.zeros(batch, device=device, dtype=torch.long)
        self.lengths = torch.zeros_like(self.slots)
        self.starts = torch.ones_like(self.slots)
        self.banks = (
            torch.zeros(
                batch, stages.bank_size, stages.hidden, device=device, dtype=dtype
            )
            if finish
            else None
        )
        self.bonus = (
            torch.zeros(batch, stages.hidden, device=device, dtype=dtype)
            if finish
            else None
        )

    def forward(self):
        stages = self.stages
        batch, width = (
            self.batch,
            stages.num_query_per_req if self.finish else stages.bank_size,
        )
        device = self.slots.device
        context_offsets = torch.cat((self.lengths.new_zeros(1), self.lengths.cumsum(0)))
        cu_q = torch.arange(batch + 1, device=device, dtype=torch.int32) * width
        cu_k = context_offsets.to(torch.int32) + cu_q
        layout = (
            self.slots,
            self.lengths,
            None,
            None,
            cu_q,
            cu_k,
            [self.capacity] * batch,
            width,
        )
        if self.finish:
            assert self.banks is not None and self.bonus is not None
            offsets = self.lengths + 1 - self.starts
            indices = offsets[:, None] + torch.arange(width - 1, device=device)
            selected = self.banks.gather(
                1, indices[..., None].expand(-1, -1, stages.hidden)
            )
            anchor = F.linear(self.bonus, stages.backbone.anchor_adapter.weight)
            states = torch.cat((anchor[:, None], selected), 1).flatten(0, 1)
            positions = (
                self.lengths[:, None] + torch.arange(width, device=device)
            ).flatten()
            states = stages.layer_batch(4, states, positions, layout)
            return rms_norm(states, stages.backbone.norm).view(batch, width, -1)
        states = stages.backbone.future_mask.expand(batch * width, -1)
        positions = (
            self.starts[:, None] + torch.arange(width, device=device)
        ).flatten()
        for index in range(4):
            states = stages.layer_batch(index, states, positions, layout)
        return states.view(batch, width, -1)

    def capture(self, pool, stream):
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.forward()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool, stream=stream):
            self.output = self.forward()

    def replay(self, slots, lengths, starts, banks=None, bonus=None):
        return self.replay_tensors(
            device_ints(slots, self.slots.device),
            device_ints(lengths, self.slots.device),
            device_ints(starts, self.slots.device),
            banks,
            bonus,
        )

    def replay_tensors(self, slots, lengths, starts, banks=None, bonus=None):
        count = slots.numel()
        self.slots.zero_()
        self.lengths.zero_()
        self.starts.fill_(1)
        self.slots[:count].copy_(slots)
        self.lengths[:count].copy_(lengths)
        self.starts[:count].copy_(starts)
        if self.finish:
            assert self.banks is not None and self.bonus is not None
            self.banks[:count].copy_(banks)
            self.bonus[:count].copy_(bonus)
        self.graph.replay()
        return self.output[:count].clone()


class LateSamplingGraph:
    """Capture the vocabulary head and configured Markov sampling steps."""

    def __init__(
        self,
        sample,
        batch,
        hidden_size,
        device,
        dtype,
        pool,
        stream,
        num_speculative_tokens=7,
    ):
        self.hidden = torch.zeros(
            batch, num_speculative_tokens, hidden_size, device=device, dtype=dtype
        )
        self.mapping = torch.full((batch,), -1, device=device, dtype=torch.int32)
        self.lengths = torch.zeros(batch, device=device, dtype=torch.long)
        self.bonus = torch.zeros_like(self.lengths)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                sample(self.hidden, self.mapping, self.lengths, self.bonus)
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool, stream=stream):
            self.output = sample(self.hidden, self.mapping, self.lengths, self.bonus)

    def replay(self, hidden, mapping, lengths, bonus):
        count = hidden.shape[0]
        self.hidden[:count].copy_(hidden)
        self.mapping.fill_(-1)
        self.mapping[:count].copy_(mapping)
        self.lengths[:count].copy_(lengths)
        self.bonus[:count].copy_(bonus)
        self.graph.replay()
        return self.output[:count].clone()


class LateContextGraph:
    """Decode-only context update. Rejected and padded rows never write K/V."""

    def __init__(self, stages, batch):
        self.stages = stages
        self.width = stages.num_query_per_req + 1
        device, dtype = stages.kv.device, stages.kv.dtype
        self.features = torch.zeros(
            batch * self.width,
            stages.backbone.fc.weight.shape[1],
            device=device,
            dtype=dtype,
        )
        self.slots = torch.zeros(batch, device=device, dtype=torch.long)
        self.starts = torch.zeros_like(self.slots)
        self.counts = torch.zeros_like(self.slots)

    def forward(self):
        s = self.stages
        context = rms_norm(
            F.linear(self.features, s.backbone.fc.weight), s.backbone.hidden_norm
        )
        kv = F.linear(context, s.context_weight).view(
            -1, 5, 2, s.n_kv_heads, s.head_dim
        )
        k, v = kv[:, :, 0], kv[:, :, 1]
        value = k.float()
        k = (
            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + s.context_eps)
        ).to(k.dtype) * s.context_norm
        positions = (
            self.starts[:, None] + torch.arange(self.width, device=k.device)
        ).flatten()
        k = s.rope(
            k.reshape(-1, s.n_kv_heads, s.head_dim), positions.repeat_interleave(5)
        ).view_as(k)
        if k.is_cuda:
            from vllm.model_executor.models.dspark_late_kernels import store_context

            store_context(k, v, self.slots, self.starts, self.counts, s.kv, self.width)
        else:
            for i, (slot, start, count) in enumerate(
                zip(self.slots.tolist(), self.starts.tolist(), self.counts.tolist())
            ):
                s.kv[:, 0, slot, start : start + count] = k[
                    i * self.width : i * self.width + count
                ].transpose(0, 1)
                s.kv[:, 1, slot, start : start + count] = v[
                    i * self.width : i * self.width + count
                ].transpose(0, 1)

    def capture(self, pool, stream):
        stream.wait_stream(torch.cuda.current_stream(self.slots.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.forward()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool, stream=stream):
            self.forward()

    def replay(self, features, slots, starts, counts):
        n = slots.numel()
        self.features[: features.shape[0]].copy_(features)
        self.slots[:n].copy_(slots)
        self.starts[:n].copy_(starts)
        self.counts.zero_()
        self.counts[:n].copy_(counts)
        self.graph.replay()
