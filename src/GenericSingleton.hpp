#ifndef GENERIC_SINGLETON_HPP_
#define GENERIC_SINGLETON_HPP_

#include <mutex>
#include <thread>

namespace EllipticForest {

template<class ActualClass>
class GenericSingleton {

public:

    static ActualClass& getInstance() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (actualClassPointer_ == nullptr) {
            actualClassPointer_ = new ActualClass;
        }
        return *actualClassPointer_;
    }

    GenericSingleton(GenericSingleton& other) = delete;
    void operator=(const GenericSingleton&) = delete;

protected:

    GenericSingleton() {}
    ~GenericSingleton() {}

    static ActualClass* actualClassPointer_;
    static std::mutex mutex_;

};

template<class ActualClass>
ActualClass* GenericSingleton<ActualClass>::actualClassPointer_{nullptr};

template<class ActualClass>
std::mutex GenericSingleton<ActualClass>::mutex_;

} // NAMESPACE : EllipticForest

#endif // GENERIC_SINGLETON_HPP_