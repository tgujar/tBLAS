#ifndef TBLAS_Matrix_H
#define TBLAS_Matrix_H

#include <cstddef>
#include <array>
#include <algorithm>
#include <vector>
#include <initializer_list>
#include <cassert>

#include "constants.hpp"
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

        Matrix() : m_data(M * N, 0), m_rows(M), m_cols(N) {}
        virtual ~Matrix() = default;
        Matrix(const Matrix &other) : m_data(other.get_data()), m_rows(other.rows()), m_cols(other.cols()) {}
        Matrix(std::initializer_list<std::initializer_list<T>> lst)
        {
            if (lst.size() != M || lst.begin()->size() != N)
            {
                throw std::invalid_argument("Matrix_base initializer list has incorrect dimensions");
            }

            this->m_rows = M;
            this->m_cols = N;
            this->m_data.reserve(M * N);

            for (auto row : lst)
            {
                copy(row.begin(), row.end(), back_inserter(this->m_data));
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
            std::vector<std::vector<T>> A(rows(), std::vector<T>(cols()));
            for (int i = 0; i < rows(); i++)
            {
                for (int j = 0; j < cols(); ++j)
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

    template <typename T>
    class MatrixX : public Matrix<T, Dynamic, Dynamic>
    {
    public:
        MatrixX(size_t rows, size_t cols)
        {
            this->m_rows = rows;
            this->m_cols = cols;
            this->m_data.resize(rows * cols, 0);
        }
        MatrixX(std::initializer_list<std::initializer_list<T>> lst)
        {
            this->m_rows = lst.size();
            this->m_cols = lst.begin()->size();
            this->m_data.reserve(this->m_rows * this->m_cols);
            for (auto row : lst)
            {
                copy(row.begin(), row.end(), back_inserter(this->m_data));
            }
        }

        MatrixX(const std::vector<std::vector<T>> &vct)
        {
            this->m_rows = vct.size();
            this->m_cols = vct.begin()->size();
            this->m_data.reserve(this->m_rows * this->m_cols);
            for (auto row : vct)
            {
                copy(row.begin(), row.end(), back_inserter(this->m_data));
            }
        }

        void resize(size_t rows, size_t cols)
        {
            this->m_data.resize(rows * cols);
            if (rows * cols < this->m_rows * this->m_cols)
            {
                this->m_data.clear();
            }
            this->m_rows = rows;
            this->m_cols = cols;
        }
    };

} // namespace tBLAS

#endif