import torch
from dn3.transforms.channels import DEEP_1010_CHS_LISTING


row_lengths = [
    1, 3, 5, 11, 6, 5, 7, 6, 6, 5, 7, 6, 5, 3, 1, 4, 3, 1, 5
]


def generate_adjacency_matrix():
    curr_row = 1
    curr_entry = 1

    # a channel is always adjacent to itself
    adjacency_matrix = torch.eye(90, dtype=torch.int)
    adjacency_matrix[0] = torch.Tensor([1, 1, *([0] * 88)])
    for i, channel in enumerate(DEEP_1010_CHS_LISTING[1:]):
        i += 1

        # check above & below
        if row_lengths[curr_row - 1] >= row_lengths[curr_row] or curr_entry <= row_lengths[curr_row - 1]:
            adjacency_matrix[i, i - row_lengths[curr_row - 1]] = 1
        if curr_row < len(row_lengths) - 1 and \
                (row_lengths[curr_row + 1] >= row_lengths[curr_row] or curr_entry <= row_lengths[curr_row + 1]):
            adjacency_matrix[i, i + row_lengths[curr_row]] = 1

        # check left & right
        if curr_entry < row_lengths[curr_row]:
            adjacency_matrix[i, i + 1] = 1
        if curr_entry > 1:
            adjacency_matrix[i, i - 1] = 1

        if curr_entry == row_lengths[curr_row]:
            curr_entry = 1
            curr_row += 1
        else:
            curr_entry += 1

    return adjacency_matrix


def generate_adjacency_matrix_d2():
    d1_mat = generate_adjacency_matrix()

    d2_mat = d1_mat.clone()

    for i in range(d1_mat.size()[0]):
        for j in range(d1_mat.size()[1]):
            if d1_mat[i, j]:
                for k in range(len(d1_mat[j])):
                    if d1_mat[j, k]:
                        d2_mat[i, k] = 1
    return d2_mat


def get_laplacian_filter(adj_matrix: torch.Tensor) -> torch.Tensor:
    above = torch.roll(adj_matrix, 1, 0)
    bottom = torch.roll(adj_matrix, -1, 0)
    left = torch.roll(adj_matrix, 1, 1)
    right = torch.roll(adj_matrix, -1, 1)

    return (above + bottom + left - 4 * right) / above.size()[0] ** 2


if __name__ == "__main__":
    adj_mat = generate_adjacency_matrix_d2()

