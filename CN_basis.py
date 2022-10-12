def generate_basis():
    N = int(input("Please input the dimension of the vector space: "))
    str_len = len(str(N - 1)) + 2
    for n in range(N):
        for i in range(N):
            exponent = n * i % N
            if exponent == 0:
                print("1" + ' ' * (str_len - 1), end='  ')
            elif exponent == 1:
                print("w" + ' ' * (str_len - 1), end='  ')
            else:
                str_to_print = 'w^' + str(exponent)
                l = len(str_to_print)
                print(str_to_print + ' ' * (str_len - l), end='  ')
        print()


if __name__ == "__main__":
    generate_basis()