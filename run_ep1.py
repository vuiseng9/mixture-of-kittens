"""Run the EP1 test directly under torchrun."""

import gc
import os
import time

import torch
import torch.distributed as dist

from mok import functional, ops
from tests.test_misc import _run_e2e_case


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    CUDADBG_ATTACH = False
    debug_rank = 0 # fix at rank 0 for now
    if int(os.environ.get("CUDADBG_ATTACH", "0")) == 1:
        CUDADBG_ATTACH = True
    if CUDADBG_ATTACH and rank == int(debug_rank):
        go_file = f"/tmp/cudagdb_go_{os.getpid()}"

        print(
            f"CUDA-GDB target: rank={rank} "
            f"local_rank={local_rank} pid= {os.getpid()} "
            f"go_file= {go_file}",
            flush=True,
        )

        while not os.path.exists(go_file):
            time.sleep(0.2) # poll frequency, every 200ms

    singleton_groups: list[dist.ProcessGroup] = []
    try:
        singleton_groups = [
            dist.new_group(ranks=[group_rank])
            for group_rank in range(world_size)
        ]
        ep_group = singleton_groups[rank]

        config = functional.MoKConfig(
            fwd_num_comm_sms=2,
            bwd_num_comm_sms=2,
            minibatch_size=256,
            macrobatch_size=512,
            all_gather_top_experts_chunk_bytes=16,
        )
        functional.clear_workspace_cache()
        workspace = functional.get_workspace(
            config,
            ep_group,
            device=device,
            num_local_tokens=512,
            hidden_size=256,
            topk=1,
        )

        assert workspace.ep_rank == 0
        assert workspace.ep_size == 1
        for buffer, handle, pointers in (
            (workspace.x_buffer, workspace.x_buffer_handle, workspace.x_buffer_ptrs),
            (
                workspace.combine_buffer,
                workspace.combine_buffer_handle,
                workspace.combine_buffer_ptrs,
            ),
            (workspace.d_y_buffer, workspace.d_y_buffer_handle, workspace.d_y_buffer_ptrs),
            (
                workspace.d_x_routed_buffer,
                workspace.d_x_routed_buffer_handle,
                workspace.d_x_routed_buffer_ptrs,
            ),
            (
                workspace.router_weight_buffer,
                workspace.router_weight_buffer_handle,
                workspace.router_weight_buffer_ptrs,
            ),
            (
                workspace.d_router_weight_buffer,
                workspace.d_router_weight_buffer_handle,
                workspace.d_router_weight_buffer_ptrs,
            ),
        ):
            assert handle is None
            assert pointers == [buffer.data_ptr()]

        assert workspace.all_gather_top_experts_buffer_handle is None
        assert (
            workspace.all_gather_top_experts_buffer_multicast_ptr
            == workspace.all_gather_top_experts_buffer.data_ptr()
        )
        assert workspace.barrier_buffer_handle is None
        assert workspace.barrier_buffer_ptrs == [workspace.barrier_buffer.data_ptr()]
        assert (
            workspace.barrier_buffer_multicast_ptr
            == workspace.barrier_buffer.data_ptr()
        )

        top_experts = torch.zeros(512, 1, dtype=torch.int32, device=device)
        ops.all_gather_top_experts(
            top_experts,
            workspace.all_gather_top_experts_buffer,
            workspace.all_gather_top_experts_buffer_multicast_ptr,
            0,
            16,
        )
        assert torch.equal(workspace.all_gather_top_experts_buffer[0], top_experts)

        ops.barrier_all(
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
        )
        torch.cuda.synchronize(device)
        assert int(workspace.barrier_buffer.item()) == 0
        assert int(workspace.barrier_target.item()) == 0

        _run_e2e_case(
            (rank, world_size, device),
            name="ep1-minimum",
            hidden_size=256,
            intermediate_size=256,
            num_local_experts=1,
            topk=1,
            config=config,
            precisions=("bf16", "mxfp8"),
            group=ep_group,
        )
        if rank == 0:
            print("EP1 passed")
    finally:
        functional.clear_workspace_cache()
        dist.barrier()
        for group in singleton_groups:
            dist.destroy_process_group(group)
        gc.collect()
        dist.destroy_process_group()


if __name__ == "__main__":
    PYDBG_ATTACH = False
    if int(os.environ.get("PYDBG_ATTACH", "0")) == 1:
        PYDBG_ATTACH = True
        
    if PYDBG_ATTACH and int(os.environ.get("RANK", "0")) == 0:
        import debugpy
        debugpy.listen(("127.0.0.1", 5678))
        # optional (only when you want to pause immediately):
        print('\n\n\n\n\n#### Waiting for debugger attach...', flush=True)
        debugpy.wait_for_client()
    main()
