#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <string>
#include <thread>
#include <chrono>
#include <map>

namespace EllipticForest {

/**
 * @brief Class for timing different pieces of execution
 * 
 */
class Timer {

public:

    /**
     * @brief Alias for clock backend
     * 
     */
    using Clock = std::chrono::steady_clock;

    /**
     * @brief Construct a new Timer object
     * 
     */
    Timer();

    /**
     * @brief Starts the timer
     * 
     */
    void start();

    /**
     * @brief Stops the timer and returns the elapsed time
     * 
     * @return double 
     */
    double stop();

    /**
     * @brief Returns the total accumulated time of this timer
     * 
     * @return double 
     */
    double time();

    /**
     * @brief Restarts the timer accumulated time
     * 
     */
    void restart();

private:

    /**
     * @brief Storage for the accumulated time of this timer
     * 
     */
    double accumulatedTime_ = 0;

    /**
     * @brief Start time of trigger
     * 
     */
    std::chrono::time_point<Clock> startTime_;

    /**
     * @brief End time of trigger
     * 
     */
    std::chrono::time_point<Clock> endTime_;

};

} // NAMESPACE : EllipticForest

#endif // TIMER_HPP_