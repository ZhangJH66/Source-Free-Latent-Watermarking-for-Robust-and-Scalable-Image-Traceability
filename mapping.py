"""Unchanged native local transforms and patch permutations."""
import random
import torch

class LatentShuffler:
    def __init__(self, splits, shuffle_indices, transform_indices=None, position_indices=None):
        self.splits = splits
        self.shuffle_indices = shuffle_indices
        self.transform_indices = transform_indices
        self.position_indices = position_indices
        assert len(shuffle_indices) == splits * splits, "Invalid shuffle indices length"
        if transform_indices is not None:
            assert len(transform_indices) == splits * splits, "Invalid transform indices length"
        if position_indices is not None:
            assert len(position_indices) == splits * splits, "Invalid position indices length"

    def random_exchange(self, arr, num_flips):
        # 检查 num_flips 的有效性
        if num_flips > len(arr) // 2:
            raise ValueError("num_flips cannot be greater than half the length of the array")

        used_indices = set()

        for _ in range(num_flips):
            # 随机选择两个不同的索引
            while True:
                idx1, idx2 = random.sample(range(len(arr)), 2)
                if idx1 not in used_indices and idx2 not in used_indices:
                    break
            # 交换这些索引对应的元素
            arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
            used_indices.add(idx1)
            used_indices.add(idx2)

        return arr

    def transform_square(self, square, index, reverse=False):
        if index == 0:
            return torch.rot90(square, k=1 if not reverse else 3, dims=[1, 2])
        elif index == 1:
            return torch.rot90(square, k=2, dims=[1, 2])
        elif index == 2:
            return torch.rot90(square, k=3 if not reverse else 1, dims=[1, 2])
        elif index == 3:
            return torch.flip(square, dims=(2,))
        elif index == 4:
            return torch.flip(square, dims=(1,))
        else:
            raise ValueError(f'Invalid index: {index}')

    def shuffle(self, latent_tensor, update_shuffle_indices=False, mismatch_number=None):
        batch = latent_tensor  # 直接操作传入的 latent tensor
        assert batch.size(2) == batch.size(3), "Latents must be square"

        # 使用局部变量处理索引，避免在 update_shuffle_indices=False 时修改实例属性
        shuffle_indices = self.shuffle_indices
        position_indices = self.position_indices
        transform_indices = self.transform_indices

        if update_shuffle_indices:
            if mismatch_number is None:
                shuffle_indices = [random.sample(range(self.splits ** 2), self.splits ** 2) for _ in
                                        range(batch.size(0))]
                position_indices = [random.sample(range(self.splits ** 2), self.splits ** 2) for _ in
                                         range(batch.size(0))]
            else:
                shuffle_indices = [self.random_exchange(shuffle_indices[i], mismatch_number) for i in
                                        range(batch.size(0))]
                position_indices = [self.random_exchange(position_indices[i], mismatch_number) for i in
                                         range(batch.size(0))]
            transform_indices = [[random.choice(range(5)) for _ in range(self.splits ** 2)] for _ in
                                      range(batch.size(0))]
            # 更新实例属性
            self.shuffle_indices = shuffle_indices
            self.position_indices = position_indices
            self.transform_indices = transform_indices
        elif isinstance(shuffle_indices[0], int):
            # 转换为二维列表格式（仅用于本次处理，不修改实例属性）
            shuffle_indices = [shuffle_indices for _ in range(batch.size(0))]
            position_indices = [position_indices for _ in range(batch.size(0))]
            # 如果 transform_indices 是一维列表，也需要转换为二维列表
            if transform_indices is not None and isinstance(transform_indices[0], int):
                transform_indices = [transform_indices for _ in range(batch.size(0))]

        batch_size = batch.size(0)
        side_length = batch.size(2) // self.splits
        multiplier = batch_size // len(shuffle_indices)

        recombined_latents = []

        for k in range(multiplier):
            for b in range(len(shuffle_indices)):
                latent = batch[b + k * len(shuffle_indices)]

                squares = [latent[:, i * side_length:(i + 1) * side_length, j * side_length:(j + 1) * side_length]
                           for i in range(self.splits) for j in range(self.splits)]

                transformed_squares = [self.transform_square(sq, transform_indices[b][i]) for i, sq in
                                       enumerate(squares)]
                shuffled_squares = [transformed_squares[i] for i in position_indices[b]]

                recombined_rows = [torch.cat(shuffled_squares[i:i + self.splits], dim=2) for i in
                                   range(0, self.splits * self.splits, self.splits)]
                recombined_latent = torch.cat(recombined_rows, dim=1)

                recombined_latents.append(recombined_latent.unsqueeze(0))

        recombined_batch = torch.cat(recombined_latents, dim=0)

        # 返回处理后的 latent tensor
        return recombined_batch

    def unshuffle(self, recombined_latent_tensor):
        recombined_batch = recombined_latent_tensor
        assert recombined_batch.size(2) == recombined_batch.size(3), "Latents must be square"

        # 使用局部变量处理索引，避免修改实例属性
        shuffle_indices = self.shuffle_indices
        position_indices = self.position_indices
        transform_indices = self.transform_indices

        # 如果索引是一维列表，转换为二维列表（仅用于本次处理，不修改实例属性）
        if isinstance(shuffle_indices[0], int):
            shuffle_indices = [shuffle_indices for _ in range(recombined_batch.size(0))]
            position_indices = [position_indices for _ in range(recombined_batch.size(0))]
            # 如果 transform_indices 是一维列表，也需要转换为二维列表
            if transform_indices is not None and isinstance(transform_indices[0], int):
                transform_indices = [transform_indices for _ in range(recombined_batch.size(0))]

        side_length = recombined_batch.size(2) // self.splits

        original_latents = []
        for b in range(len(shuffle_indices)):
            recombined_latent = recombined_batch[b]
            position_inverse_indices = torch.argsort(torch.tensor(position_indices[b]))
            inverse_indices = torch.argsort(torch.tensor(shuffle_indices[b]))

            squares = [
                recombined_latent[:, i * side_length:(i + 1) * side_length, j * side_length:(j + 1) * side_length]
                for i in range(self.splits) for j in range(self.splits)]

            unshuffled_squares = [squares[i] for i in position_inverse_indices]
            inverse_transformed_squares = [self.transform_square(sq, transform_indices[b][i], reverse=True) for
                                           i, sq in enumerate(unshuffled_squares)]

            original_rows = [torch.cat(inverse_transformed_squares[i:i + self.splits], dim=2) for i in
                             range(0, self.splits * self.splits, self.splits)]
            original_latent = torch.cat(original_rows, dim=1)

            original_latents.append(original_latent.unsqueeze(0))

        original_batch = torch.cat(original_latents, dim=0)

        # 返回还原后的 latent tensor
        return original_batch
