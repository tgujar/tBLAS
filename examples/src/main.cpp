#include <iostream>

#include <tBLAS/blas.hpp>
#include <tBLAS/matrix.hpp>

int main()
{
    using namespace std;
    using namespace tBLAS;
    MatrixX<double> A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    MatrixX<double> B{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    auto C = multiply(A, B);

    cout << "Transpose of A: \n"
         << transpose(A) << endl;
    cout << "Multiplication of A and B: \n"
         << C << endl;
    return 0;
}