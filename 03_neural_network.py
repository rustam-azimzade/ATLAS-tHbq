from sklearn.model_selection import train_test_split


def main():
    X = [1,2,3]
    Y = [10,24,64]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10, test_size=0.3)


if __name__ == '__main__':
    main()
