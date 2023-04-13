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

        /**
         * @brief Access the element at position (i,j) in the derived matrix class
         *
         * @param i Row index
         * @param j Column index
         * @return T& Reference to the element
         */
        T &operator()(size_t i, size_t j)
        {
            return (*static_cast<Derived *>(this))(i, j);
        }
        const T &operator()(size_t i, size_t j) const
        {
            return (*static_cast<const Derived *>(this))(i, j);
        }

        size_t rows() const noexcept { return static_cast<const Derived *>(this)->rows(); }
        size_t cols() const noexcept { return static_cast<const Derived *>(this)->cols(); }

        iterator begin() noexcept { return static_cast<Derived *>(this)->begin(); }
        const_iterator const cbegin() const noexcept { return static_cast<const Derived *>(this)->cbegin(); }
        iterator end() noexcept { return static_cast<Derived *>(this)->end(); }
        const_iterator cend() const noexcept { return static_cast<const Derived *>(this)->cend(); }
    };

    /* -------------------------------- Interface ------------------------------- */

    /**
     * @brief Matrix class with fixed size
     *
     * To be used for small matrices. Memory is allocated on the stack.
     *
     * @tparam T Type of the elements
     * @tparam M Number of rows
     * @tparam N Number of columns
     */
    template <typename T, size_t M, size_t N>
    class Matrix : public MatrixBase<T, Matrix<T, M, N>, std::array<T, M * N>>
    {
    private:
        // alignas is somewhat broken in C++11 and might get ignored
        alignas(64) std::array<T, M * N> m_data; /**< Storage for matrix elements*/
        size_t m_rows;                           /**< Number of rows*/
        size_t m_cols;                           /**< Number of cols*/

        friend std::ostream &operator<<(std::ostream &os, const Matrix<T, M, N> &p);

    protected:
        /**
         * @brief Check if the given indices don't cause overflow
         *
         * @param m Number of rows
         * @param n Number of columns
         * @return bool
         */
        bool is_bounded(size_t m, size_t n);

    public:
        using iterator = typename std::array<T, M * N>::iterator;
        using const_iterator = typename std::array<T, M * N>::const_iterator;

        Matrix();
        Matrix(std::initializer_list<std::initializer_list<T>> lst);
        Matrix(const std::array<std::array<T, N>, M> &arr);

        virtual ~Matrix() = default;

        Matrix &operator=(const Matrix &other);
        Matrix &operator=(Matrix &&other);

        /**
         * @brief Access the element at position (i,j)
         *
         * @param i row index
         * @param j column index
         * @return T& Reference to the element
         */
        T &operator()(size_t i, size_t j);
        const T &operator()(size_t i, size_t j) const;

        size_t rows() const noexcept;
        size_t cols() const noexcept;

        iterator begin() noexcept;
        const_iterator cbegin() const noexcept;
        iterator end() noexcept;
        const_iterator cend() const noexcept;
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
        /**
         * @brief Check if the given indices don't cause overflow
         *
         * @param m Number of rows
         * @param n Number of columns
         * @return bool
         */
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

        /**
         * @brief Access the element at position (i,j)
         *
         * @param i row index
         * @param j column index
         * @return T& Reference to the element
         */
        T &operator()(size_t i, size_t j);
        const T &operator()(size_t i, size_t j) const;

        size_t rows() const noexcept;
        size_t cols() const noexcept;

        iterator begin() noexcept;
        const_iterator cbegin() const noexcept;
        iterator end() noexcept;
        const_iterator cend() const noexcept;

        /**
         * @brief Resize the matrix to the given dimensions
         *
         * This simply increases the size of the matrix and does not gurantee that
         * new values follow the same layout as in the old matrix.
         *
         * @note If the new number of dimensions is smaller than the old one, the
         * values will be zeroed out.
         *
         * @param rows Number of rows
         * @param cols Number of columns
         */
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
    Matrix<T, M, N>::Matrix(const std::array<std::array<T, N>, M> &arr)
    {
        if (arr.size() != M || arr.begin()->size() != N)
        {
            throw std::invalid_argument("Matrix_base initializer list has incorrect dimensions");
        }

        int j = 0;
        for (auto row = arr.begin(); row != arr.end(); row++, j += N)
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
    size_t Matrix<T, M, N>::rows() const noexcept { return m_rows; }
    template <typename T, size_t M, size_t N>
    size_t Matrix<T, M, N>::cols() const noexcept { return m_cols; }

    template <typename T, size_t M, size_t N>
    typename Matrix<T, M, N>::iterator Matrix<T, M, N>::begin() noexcept { return m_data.begin(); }

    template <typename T, size_t M, size_t N>
    typename Matrix<T, M, N>::iterator Matrix<T, M, N>::end() noexcept { return m_data.end(); }

    template <typename T, size_t M, size_t N>
    typename Matrix<T, M, N>::const_iterator Matrix<T, M, N>::cbegin() const noexcept { return m_data.cbegin(); }

    template <typename T, size_t M, size_t N>
    typename Matrix<T, M, N>::const_iterator Matrix<T, M, N>::cend() const noexcept { return m_data.cend(); }

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
    size_t MatrixX<T>::rows() const noexcept { return m_rows; }
    template <typename T>
    size_t MatrixX<T>::cols() const noexcept { return m_cols; }

    template <typename T>
    void MatrixX<T>::resize(size_t rows, size_t cols)
    {
        if (!is_bounded(rows, cols))
        {
            throw std::overflow_error("Matrix dimensions are too large");
        }
        this->m_data.resize(rows * cols);
        if (rows * cols < this->m_rows * this->m_cols)
        {
            this->m_data.clear();
        }
        this->m_rows = rows;
        this->m_cols = cols;
    }

    template <typename T>
    typename MatrixX<T>::iterator MatrixX<T>::begin() noexcept { return m_data.begin(); }

    template <typename T>
    typename MatrixX<T>::iterator MatrixX<T>::end() noexcept { return m_data.end(); }

    template <typename T>
    typename MatrixX<T>::const_iterator MatrixX<T>::cbegin() const noexcept { return m_data.cbegin(); }

    template <typename T>
    typename MatrixX<T>::const_iterator MatrixX<T>::cend() const noexcept { return m_data.cend(); }

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