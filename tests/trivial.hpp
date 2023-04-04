#include <vector>

namespace tBLAS_test
{

    template <typename T>
    /*
     * Trivial implementation of matrix transpose and matrix multiplication.
     */
    class Trivial
    {
    public:
        std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &A)
        {
            std::vector<std::vector<T>> B(A[0].size(), std::vector<int>(A.size()));
            for (int i = 0; i < A.size(); ++i)
            {
                for (int j = 0; j < A[0].size(); ++j)
                {
                    B[j][i] = A[i][j];
                }
            }
            return B;
        }
        std::vector<std::vector<T>> matmul(const std::vector<std::vector<T>> &A, const std::vector<std::vector<T>> &B)
        {
            std::vector<std::vector<T>> C(A.size(), std::vector<int>(B[0].size()));

            for (int i = 0; i < A.size(); ++i)
            {
                for (int j = 0; j < B[0].size(); ++j)
                {
                    for (int k = 0; k < B.size(); ++k)
                    {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            return C;
        }
    };

}
