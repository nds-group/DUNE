def __is_valid_prefix(sequence):
    max_val = 1
    for val in sequence:
        if val > max_val:
            return False
        max_val = max(max_val, val + 1)
    return True

def __next_partition(sequence):
    n = len(sequence)
    for i in range(n - 1, -1, -1):
        max_val = max(sequence[:i + 1]) + 1
        for new_val in range(sequence[i] + 1, max_val + 1):
            new_sequence = sequence[:i] + [new_val] + [1] * (n - i - 1)
            if __is_valid_prefix(new_sequence):
                return new_sequence
    return None

def __generate_partitions(n):
    if n == 0:
        return []
    partitions = []
    sequence = [1] * n
    while sequence:
        partitions.append(sequence[:])
        sequence = __next_partition(sequence)
    return partitions

def __convert_to_sets(partition_sequence, elements):
    from collections import defaultdict
    partition_dict = defaultdict(list)
    for idx, part in enumerate(partition_sequence):
        partition_dict[part].append(elements[idx])
    return list(partition_dict.values())

def generate_partitions_of_set_lexicographically(elements):
    n = len(elements)
    partition_sequences = __generate_partitions(n)
    partitions = [__convert_to_sets(seq, elements) for seq in partition_sequences]
    return partitions


def generate_partitions_of_set(set_elements):
    if not set_elements:
        return [[]]

    first_element = set_elements[0]
    rest_partitions = generate_partitions_of_set(set_elements[1:])

    new_partitions = []
    for partition in rest_partitions:
        for i in range(len(partition)):
            new_partition = partition[:i] + [[first_element] + partition[i]] + partition[i + 1:]
            new_partitions.append(new_partition)
        new_partitions.append([[first_element]] + partition)

    return new_partitions