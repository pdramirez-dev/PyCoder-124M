def binary_search(items, target):
    left, right = 0, len(items)
    while left < right:
        middle = (left + right) // 2
        if items[middle] < target:
            left = middle + 1
        else:
            right = middle
    return left if left < len(items) and items[left] == target else -1
