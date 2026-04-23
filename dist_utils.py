import torch
import torch.distributed as dist



def is_dist_avail_and_initialized():
    if not dist.is_available(): return False
    if not dist.is_initialized(): return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized(): return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized(): return 1
    return dist.get_world_size()



class DiffAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        world_size = get_world_size()
        if world_size == 1: return tensor


        tensors_gather = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors_gather, tensor)


        return torch.cat(tensors_gather, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        world_size = get_world_size()
        if world_size == 1: return grad_output


        rank = get_rank()
        input_count = grad_output.shape[0] // world_size
        return grad_output[rank * input_count: (rank + 1) * input_count]


def gather_tensor(tensor):
    return DiffAllGather.apply(tensor)


def reduce_mean(tensor):

    if not is_dist_avail_and_initialized(): return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt