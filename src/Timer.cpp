#include "Timer.hpp"

namespace EllipticForest {

Timer::Timer()
    {}

void Timer::start() {
    startTime_ = Clock::now();
}

double Timer::stop() {
    endTime_ = Clock::now();
    std::chrono::duration<double> diff = endTime_ - startTime_;
    accumulatedTime_ += diff.count();
    return diff.count();
}

double Timer::time() { return accumulatedTime_; }

void Timer::restart() {
    accumulatedTime_ = 0;
}

} // NAMESPACE : EllipticForest