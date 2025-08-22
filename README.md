# lcpp [LearningCurvePlusPlus]

**C++ Header-Only Learning Curve Generation Tool**  

Generate learning curves for supervised machine learning algorithms with just header files — no separate compilation needed!  

---

## Detailed Documentations
[algo](docs/algo.md)
[src](docs/src.md)
[utils](docs/utils.md)
[data](docs/data.md)

---

## About  
`lcpp` is designed to help you easily generate learning curves for supervised ML algorithms.  
It provides a clean C++ header-only implementation, making it easy to integrate into your own projects without heavy build setup.  

---

## Quick Start  

Too lazy to set everything up?  

Simply pull my pre-built image:  

```bash
docker pull taylanot/cxx-dev:latest
```

---

## Contributions

Any contributions are welcome. Please make sure you test what your contributions in the related test files.

- Feature Curves generation is on the roadmap of this project.
- New learning algorithms are always welcome.
- New sampling strategies can be useful for researchers.

## Dependencies

- [C++>= C++20](https://en.cppreference.com/w/cpp/20.html)
- [Armadillo 14.0.0>=](https://arma.sourceforge.net/docs.html)  
- [mlpack 4.4.1>=](https://github.com/shivamshivanshu/mlpack/tree/master)  
- [ensmallen 2.21.1>=](https://github.com/mlpack/ensmallen)  
- [libcurl 7.81.0>=](https://curl.se/libcurl/)

> **Note:** These libraries may have their own dependencies. Make sure they are properly installed before use.  


