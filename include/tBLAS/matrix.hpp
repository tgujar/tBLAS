#ifndef TBLAS_MATRIX_H
#define TBLAS_MATRIX_H

#include <cstddef>
#include <array>
namespace tBLAS
{
    template <typename T, size_t M, size_t N>
    class MatrixXd
    {
    private:
        alignas(64) std::vector<T> m_data;

    public:
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        MatrixXd() : m_data(M * N, 0) {} // allocate on construct, could be better
        ~MatrixXd() = default;
        MatrixXd(const MatrixXd &other) : m_data(other.get_data()) {}
        MatrixXd(const std::vector<std::vector<T>> &other) : m_data(M * N, 0) // kinda sketchy
        {
            for (int i = 0; i < other.size(); i++)
            {
                for (int j = 0; j < other[0].size(); j++)
                {
                    (*this)(i, j) = other[i][j];
                }
            }
        }
        MatrixXd(MatrixXd &&other) : m_data(std::move(other.get_data())) {}
        MatrixXd &operator=(const MatrixXd &other)
        {
            m_data = other.get_data();
            return *this;
        }
        MatrixXd &operator=(MatrixXd &&other)
        {
            m_data = std::move(other.get_data());
            return *this;
        }

        T &operator()(size_t i, size_t j)
        {
            return m_data.at(i * N + j);
        }
        const T &operator()(size_t i, size_t j) const
        {
            return m_data.at(i * N + j);
        }

        std::vector<T> &get_data()
        {
            return m_data;
        }

        const std::vector<T> &get_data() const
        {
            return m_data;
        }

        inline iterator begin() noexcept { return m_data.begin(); }
        inline const_iterator cbegin() const noexcept { return m_data.cbegin(); }
        inline iterator end() noexcept { return m_data.end(); }
        inline const_iterator cend() const noexcept { return m_data.cend(); }
    };

    template <typename T, size_t M, size_t N>
    std::ostream &operator<<(std::ostream &os, const MatrixXd<T, M, N> &A)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                os << A(i, j) << " ";
            }
            os << "\n";
        }
        return os;
    }

} // namespace tBLAS

#endif