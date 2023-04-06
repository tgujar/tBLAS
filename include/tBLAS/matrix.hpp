#ifndef TBLAS_MATRIX_H
#define TBLAS_MATRIX_H

#include <cstddef>
#include <array>
#include <algorithm>
#include <vector>
#include <initializer_list>
#include <cassert>
namespace tBLAS
{

    template <typename T, size_t M, size_t N>
    class Matrix
    {
    protected:
        alignas(64) std::vector<T> m_data;
        size_t m_rows;
        size_t m_cols;

    public:
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        Matrix() : m_data(M * N, 0), m_rows(M), m_cols(N) {} // allocate on construct, could be better
        ~Matrix() = default;
        Matrix(const Matrix &other) : m_data(other.get_data()) {}
        Matrix(std::initializer_list<std::initializer_list<T>> lst) : m_rows(M), m_cols(N) // kinda sketchy
        {
            if (lst.size() != M || lst.begin()->size() != N)
            {
                throw std::invalid_argument("Matrix initializer list has incorrect dimensions");
            }
            m_data.reserve(M * N);
            for (auto row : lst)
            {
                copy(row.begin(), row.end(), back_inserter(m_data));
            }
        }

        Matrix &operator=(const Matrix &other)
        {
            m_data = other.get_data();
            return *this;
        }
        Matrix &operator=(Matrix &&other)
        {
            m_data = std::move(other.get_data());
            return *this;
        }

        T &operator()(size_t i, size_t j)
        {
            return m_data.at(i * cols() + j);
        }
        const T &operator()(size_t i, size_t j) const
        {
            return m_data.at(i * cols() + j);
        }

        std::vector<T> &get_data()
        {
            return m_data;
        }

        const std::vector<T> &get_data() const
        {
            return m_data;
        }

        std::vector<std::vector<T>> to_vector() const
        {
            std::vector<std::vector<T>> A(M, std::vector<T>(N));
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; ++j)
                {
                    A[i][j] = (*this)(i, j);
                }
            }
            return A;
        }

        inline size_t rows() const noexcept { return m_rows; }
        inline size_t cols() const noexcept { return m_cols; }

        inline iterator begin() noexcept { return m_data.begin(); }
        inline const_iterator cbegin() const noexcept { return m_data.cbegin(); }
        inline iterator end() noexcept { return m_data.end(); }
        inline const_iterator cend() const noexcept { return m_data.cend(); }
    };

    template <typename T, size_t M, size_t N>
    std::ostream &operator<<(std::ostream &os, const Matrix<T, M, N> &A)
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