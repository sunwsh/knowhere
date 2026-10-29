#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <stdexcept>
#include <streambuf>
#include <utility>

namespace pag {

struct MemoryIOWriter {
  MemoryIOWriter() = default;
  MemoryIOWriter(const MemoryIOWriter &) = delete;
  MemoryIOWriter &operator=(const MemoryIOWriter &) = delete;
  MemoryIOWriter(MemoryIOWriter &&other) noexcept
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_),
        owned_(other.owned_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    other.owned_ = false;
  }
  MemoryIOWriter &operator=(MemoryIOWriter &&) = delete;

  ~MemoryIOWriter() {
    if (owned_) {
      delete[] data_;
    }
  }

  uint8_t *data() { return data_; }

  size_t tellg() const { return size_; }

  std::pair<uint8_t *, size_t> release() {
    owned_ = false;
    std::pair<uint8_t *, size_t> result(data_, size_);
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
    return result;
  }

  size_t operator()(const void *ptr, size_t size, size_t nitems) {
    if (size == 0 || nitems == 0) {
      return 0;
    }
    if (nitems > std::numeric_limits<size_t>::max() / size) {
      throw std::runtime_error("PAG write size overflow");
    }
    const size_t nbytes = size * nitems;
    const size_t need = size_ + nbytes;
    if (need > capacity_) {
      size_t next_capacity = capacity_ == 0 ? 4096 : capacity_;
      while (next_capacity < need) {
        if (next_capacity > std::numeric_limits<size_t>::max() / 2) {
          next_capacity = need;
          break;
        }
        next_capacity *= 2;
      }
      uint8_t *next = new uint8_t[next_capacity];
      if (data_ != nullptr && size_ > 0) {
        std::memcpy(next, data_, size_);
      }
      delete[] data_;
      data_ = next;
      capacity_ = next_capacity;
      owned_ = true;
    }
    if (ptr != nullptr && nbytes > 0) {
      std::memcpy(data_ + size_, ptr, nbytes);
    }
    size_ += nbytes;
    return nitems;
  }

  template <typename T>
  size_t write(T *ptr, size_t size, size_t nitems = 1) {
    return (*this)(ptr, size, nitems);
  }

private:
  uint8_t *data_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
  bool owned_ = true;
};

struct MemoryIOReader {
  MemoryIOReader(uint8_t *data, size_t size) : data_(data), total_(size) {}

  size_t operator()(void *ptr, size_t size, size_t nitems) {
    if (size == 0 || nitems == 0 || pos_ >= total_) {
      return 0;
    }
    if (nitems > std::numeric_limits<size_t>::max() / size) {
      nitems = (total_ - pos_) / size;
    }
    const size_t remain = total_ - pos_;
    const size_t want = size * nitems;
    const size_t nbytes = remain < want ? remain : want;
    const size_t got = nbytes / size;
    const size_t copy_n = got * size;
    if (ptr != nullptr && copy_n > 0) {
      std::memcpy(ptr, data_ + pos_, copy_n);
    }
    pos_ += copy_n;
    return got;
  }

  template <typename T>
  size_t read(T *ptr, size_t size, size_t nitems = 1) {
    return (*this)(ptr, size, nitems);
  }

  size_t tellg() const { return pos_; }

  void seekg(size_t pos) {
    if (pos > total_) {
      throw std::runtime_error("PAG seek past end");
    }
    pos_ = pos;
  }

  size_t size() const { return total_; }

private:
  uint8_t *data_ = nullptr;
  size_t total_ = 0;
  size_t pos_ = 0;
};

// istream view of a caller-owned buffer. tellg() works, so the existing
// info.bin parser can read a memory section without a temporary file.
class MemoryStreamBuf : public std::streambuf {
public:
  MemoryStreamBuf(uint8_t *data, size_t size) : begin_(reinterpret_cast<char *>(data)), size_(size) {
    setg(begin_, begin_, begin_ + size_);
  }

protected:
  std::streampos seekoff(std::streamoff off, std::ios_base::seekdir way,
                         std::ios_base::openmode which) override {
    if ((which & std::ios_base::in) == 0 || begin_ == nullptr) {
      return std::streampos(-1);
    }
    std::ptrdiff_t base = 0;
    if (way == std::ios_base::cur) {
      base = gptr() - begin_;
    } else if (way == std::ios_base::end) {
      base = static_cast<std::ptrdiff_t>(size_);
    }
    const std::ptrdiff_t next = base + static_cast<std::ptrdiff_t>(off);
    if (next < 0 || static_cast<size_t>(next) > size_) {
      return std::streampos(-1);
    }
    setg(begin_, begin_ + next, begin_ + size_);
    return std::streampos(next);
  }

  std::streampos seekpos(std::streampos pos, std::ios_base::openmode which) override {
    return seekoff(std::streamoff(pos), std::ios_base::beg, which);
  }

private:
  char *begin_ = nullptr;
  size_t size_ = 0;
};

} // namespace pag
