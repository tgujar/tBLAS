#ifndef TBLAS_Matrix_H
#define TBLAS_Matrix_H

#include <cstddef>
#include <array>
#include <algorithm>
#include <vector>
#include <initializer_list>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <array>

#include "constants.hpp"
namespace tBLAS
{
    /* ------------------------- Matrix CRTP Base class ------------------------- */
    template <typename T, typename Derived, typename Storage>
    class MatrixBase
    {
    private:
        MatrixBase() {}
        friend Derived;

    public:
        using iterator = typename Storage::iterator;
        using const_iterator = typename Storage::const_iterator;

        T &operator()(size_t i, size_t j)
        {
            return (*static_cast<Derived *>(this))(i, j);
        }
        const T &operator()(size_t i, size_t j) const
        {
            return (*static_cast<Derived *>(this))(i, j);
        }

        inline size_t rows() const noexcept { return static_cast<const Derived *>(this)->rows(); }
        inline size_t cols() const noexcept { return static_cast<const Derived *>(this)->cols(); }

        inline iterator begin() noexcept { return static_cast<Derived *>(this)->begin(); }
        inline const_iterator const cbegin() const noexcept { return static_cast<const Derived *>(this)->cbegin(); }
        inline iterator end() noexcept { return static_cast<Derived *>(this)->end(); }
        inline const_iterator cend() const noexcept { return static_cast<const Derived *>(this)->cend(); }
    };

    /* -------------------------------- Interface ------------------------------- */
    template <typename T, size_t M, size_t N>
    class Matrix : public MatrixBase<T, Matrix<T, M, N>, std::array<T, M * N>>
    {
    private:
        // alignas is somewhat broken in C++11 and might get ignored
        alignas(64) std::array<T, M * N> m_data;
        size_t m_rows;
        size_t m_cols;

        friend std::ostream &operator<<(std::ostream &os, const Matrix<T, M, N> &p);

    protected:
        bool is_bounded(size_t m, size_t n);

    public:
        using iterator = typename std::array<T, M * N>::iterator;
        using const_iterator = typename std::array<T, M * N>::const_iterator;

        Matrix();
        Matrix(std::initializer_list<std::initializer_list<T>> lst);
        Matrix(const std::array<std::array<T, N>, M> &arr);

        Matrix &operator=(const Matrix &other);
        Matrix &operator=(Matrix &&other);
        T &operator()(size_t i, size_t j);
        const T &operator()(size_t i, size_t j) const;

        virtual ~Matrix() = default;

        inline size_t rows() const noexcept;
        inline size_t cols() const noexcept;

        inline iterator begin() noexcept;
        inline const_iterator cbegin() const noexcept;
        inline iterator end() noexcept;
        inline const_iterator cend() const noexcept;
    };

    template <typename T>
    class MatrixX : public MatrixBase<T, MatrixX<T>, std::vector<T>>
    {
    private:
        // alignas is somewhat broken in C++11 and might get ignored
        alignas(64) std::vector<T> m_data;
        size_t m_rows;
        size_t m_cols;
        friend std::ostream &operator<<(std::ostream &os, const MatrixX<T> &p);

    protected:
        bool is_bounded(size_t m, size_t n);

    public:
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        MatrixX(size_t rows, size_t cols);
        MatrixX(std::initializer_list<std::initializer_list<T>> lst);
        MatrixX(const MatrixX &A) = default;
        MatrixX(const std::vector<std::vector<T>> &vct);

        virtual ~MatrixX() = default;

        MatrixX &operator=(const MatrixX &other);
        MatrixX &operator=(MatrixX &&other);
        T &operator()(size_t i, size_t j);
        const T &operator()(size_t i, size_t j) const;

        inline size_t rows() const noexcept;
        inline size_t cols() const noexcept;

        inline iterator begin() noexcept;
        inline const_iterator cbegin() const noexcept;
        inline iterator end() noexcept;
        inline const_iterator cend() const noexcept;

        std::vector<std::vector<T>> to_vector() const;

        void resize(size_t rows, size_t cols);
    };

    /* ----------------------------- Matrix Implementation ----------------------------- */

    template <typename T, size_t M, size_t N>
    bool Matrix<T, M, N>::is_bounded(size_t m, size_t n)
    {
        return m < std::numeric_limits<size_t>::max() / n;
    }

    template <typename T, size_t M, size_t N>
    Matrix<T, M, N>::Matrix() : m_data(), m_rows(M), m_cols(N) {}

    template <typename T, size_t M, size_t N>
    Matrix<T, M, N>::Matrix(std::initializer_list<std::initializer_list<T>> lst)
    {
        if (lst.size() != M || lst.begin()->size() != N)
        {
            throw std::invalid_argument("Matrix_base initializer list has incorrect dimensions");
        }

        int j = 0;
        for (auto row = lst.begin(); row != lst.end(); row++, j += N)
        {
            std::copy(row->begin(), row->end(), m_data.begin() + j);
        }
    }

    template <typename T, size_t M, size_t N>
    Matrix<T, M, N> &Matrix<T, M, N>::operator=(const Matrix<T, M, N> &other)
    {
        m_data = other.get_data();
        return *this;
    }

    template <typename T, size_t M, size_t N>
    Matrix<T, M, N> &Matrix<T, M, N>::operator=(Matrix<T, M, N> &&other)
    {
        m_data = std::move(other.get_data());
        return *this;
    }

    template <typename T, size_t M, size_t N>
    T &Matrix<T, M, N>::operator()(size_t i, size_t j)
    {
        return m_data.at(i * N + j);
    }

    template <typename T, size_t M, size_t N>
    const T &Matrix<T, M, N>::operator()(size_t i, size_t j) const
    {
        return m_data.at(i * N + j);
    }

    template <typename T, size_t M, size_t N>
    inline size_t Matrix<T, M, N>::rows() const noexcept { return m_rows; }
    template <typename T, size_t M, size_t N>
    inline size_t Matrix<T, M, N>::cols() const noexcept { return m_cols; }

    template <typename T, size_t M, size_t N>
    inline typename Matrix<T, M, N>::iterator Matrix<T, M, N>::begin() noexcept { return m_data.begin(); }

    template <typename T, size_t M, size_t N>
    inline typename Matrix<T, M, N>::iterator Matrix<T, M, N>::end() noexcept { return m_data.end(); }

    template <typename T, size_t M, size_t N>
    inline typename Matrix<T, M, N>::const_iterator Matrix<T, M, N>::cbegin() const noexcept { return m_data.cbegin(); }

    template <typename T, size_t M, size_t N>
    inline typename Matrix<T, M, N>::const_iterator Matrix<T, M, N>::cend() const noexcept { return m_data.cend(); }

    template <typename T, size_t M, size_t N>
    std::ostream &operator<<(std::ostream &os, const tBLAS::Matrix<T, M, N> &A)
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

    /* -------------------------- MatrixX Implementation ------------------------- */

    template <typename T>
    bool MatrixX<T>::is_bounded(size_t m, size_t n)
    {
        return m < std::numeric_limits<size_t>::max() / n;
    }

    template <typename T>
    MatrixX<T>::MatrixX(size_t m, size_t n) : m_rows(m), m_cols(n)
    {
        if (!is_bounded(m, n))
        {
            throw std::overflow_error("Matrix dimensions are too large");
        }
        m_data.resize(m * n, 0);
    }

    template <typename T>
    MatrixX<T>::MatrixX(std::initializer_list<std::initializer_list<T>> lst)
    {
        m_rows = lst.size();
        m_cols = lst.begin()->size();
        m_data.reserve(m_rows * m_cols);
        for (auto row : lst)
        {
            copy(row.begin(), row.end(), back_inserter(m_data));
        }
    }

    template <typename T>
    MatrixX<T>::MatrixX(const std::vector<std::vector<T>> &vct)
    {
        this->m_rows = vct.size();
        this->m_cols = vct.begin()->size();
        this->m_data.reserve(this->m_rows * this->m_cols);
        for (auto row : vct)
        {
            std::copy(row.begin(), row.end(), back_inserter(this->m_data));
        }
    }

    template <typename T>
    MatrixX<T> &MatrixX<T>::operator=(const MatrixX<T> &other)
    {
        m_data = other.get_data();
        return *this;
    }

    template <typename T>
    MatrixX<T> &MatrixX<T>::operator=(MatrixX<T> &&other)
    {
        m_data = std::move(other.get_data());
        return *this;
    }

    template <typename T>
    T &MatrixX<T>::operator()(size_t i, size_t j)
    {
        return m_data.at(i * cols() + j);
    }

    template <typename T>
    const T &MatrixX<T>::operator()(size_t i, size_t j) const
    {
        return m_data.at(i * cols() + j);
    }

    template <typename T>
    inline size_t MatrixX<T>::rows() const noexcept { return m_rows; }
    template <typename T>
    inline size_t MatrixX<T>::cols() const noexcept { return m_cols; }

    template <typename T>
    std::vector<std::vector<T>> MatrixX<T>::to_vector() const
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

    template <typename T>
    inline typename MatrixX<T>::iterator MatrixX<T>::begin() noexcept { return m_data.begin(); }

    template <typename T>
    inline typename MatrixX<T>::iterator MatrixX<T>::end() noexcept { return m_data.end(); }

    template <typename T>
    inline typename MatrixX<T>::const_iterator MatrixX<T>::cbegin() const noexcept { return m_data.cbegin(); }

    template <typename T>
    inline typename MatrixX<T>::const_iterator MatrixX<T>::cend() const noexcept { return m_data.cend(); }

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const MatrixX<T> &A)
    {
        for (int i = 0; i < A.rows(); i++)
        {
            for (int j = 0; j < A.cols(); j++)
            {
                os << A(i, j) << " ";
            }
            os << "\n";
        }
        return os;
    }
}; // namespace tBLAS

#endif