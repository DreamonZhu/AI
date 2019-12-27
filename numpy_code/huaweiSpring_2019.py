import numpy as np

arr = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
row, col = arr.shape
# print(arr.shape)

# print(row, col)
# print(arr[0][0])


def update_arr():
    for i in range(row):
        for j in range(1, col):
            if arr[i][j] == 1 & arr[i][j-1] != 0:
                arr[i][j] = arr[i][j-1] + 1


def pre_col(b):
    i = 0
    while i < row:
        if arr[i][b] == 0:
            continue
        j = i
        upper = 0
        while j < row & arr[j][b] > 0:
            upper += arr[j][b]
            j += 1
        # k = i
        # while k < j:
        #     arr[k][b] = upper

        i = j
        i += 1


def calc_upper():
    for i in range(col):
        pre_col(i)


def search_up_down(a, b):
    cnt = 1
    val = arr[a][b]
    for i in range(a):
        if arr[i][b] >= val:
            cnt += 1
        else:
            break

    for j in range(a+1, row):
        if arr[j][b] >= val:
            cnt += 1
        else:
            break

    if cnt >= val:
        return val**2
    else:
        return cnt**2


def get_max():
    max_square = 0
    for i in range(row):
        for j in range(col):
            if arr[i][j] != 0:
                current = search_up_down(i, j)
                if current > max_square:
                    max_square = current
    return max_square


update_arr()
print(arr)
square = get_max()
print(square)
