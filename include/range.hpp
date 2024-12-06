#ifndef RANGE_ITER_HPP
#define RANGE_ITER_HPP

#include <iterator>

class range_iter {

    //tags
    using input_iterator_tag = std::input_iterator_tag;
    using difference_type = size_t;
    using value_type = int;
    using reference = int&;
    using pointer = int*;

    public:
        range_iter(value_type curr, value_type step) : curr(curr), step(step) {}
        value_type operator*() const { return curr; }
        range_iter& operator++() { curr += step; return *this; }
        range_iter operator++(int) { range_iter tmp = *this; ++(*this); return tmp; }
        bool operator==(const range_iter& rhs) const { return curr == rhs.curr; }
        bool operator!=(const range_iter& rhs) const { return curr != rhs.curr; }

    private:
        value_type curr, step;

};

class range {
    public:
        range(int start, int end, int step = 1) : start(start), stop(end), step(step) {
            if(step == 0) throw std::invalid_argument("Step cannot be zero");
            if(step < 0 && start < end) throw std::invalid_argument("Step cannot be negative if start < end");
            if(step > 0 && start > end) throw std::invalid_argument("Step cannot be positive if start > end");
        }
        range(int end) : range(0, end) {}
        range_iter begin() { return range_iter(start, step); }
        range_iter end() { return range_iter(stop, step); }
    private:
        int start, stop, step;
};

#endif // RANGE_ITER_HPP