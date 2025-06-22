#ifndef __SINGLETON_HPP__
#define __SINGLETON_HPP__

#include <memory>

/// @brief 单例类
/// @tparam T 模板类
/// @tparam X
template <class T, class X = void, int N = 0>
class Singleton {
public:
    static T& GetInstance() {
        static T v;
        return v;
    }

protected:
    Singleton() = default;

    Singleton(const Singleton&) = delete;

    Singleton& operator=(const Singleton&) = delete;
};

template <class T, class X = void, int N = 0>
class SingletonPtr {
public:
    static std::shared_ptr<T> GetInstance() {

        static std::shared_ptr<T> v(new T);
        return v;
    }
};

#endif