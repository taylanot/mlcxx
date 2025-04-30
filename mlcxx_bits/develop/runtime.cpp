/**
 * @file runtime.cpp
 * @author Ozgur Taylan Turan
 *
 * Trying to take runtime types...
 */


#include "headers.h"

/* template<typename T> */
/* bool foo(const int& v) { */
/*     std::cout << __PRETTY_FUNCTION__ << " " << v << "\n"; */
/*     return sizeof(T) > 4; */
/* } */

/* enum class DataType: uint8_t { */ 
/*     kInt32, */
/*     kDouble, */
/*     kString */
/* }; */

/* using VType = std::variant<std::type_identity<int32_t>, */
/*                            std::type_identity<double>, */
/*                            std::type_identity<std::string>>; */

/* static const std::unordered_map<DataType, VType> dispatcher = { */
/*     {DataType::kInt32, std::type_identity<int32_t>{}}, */
/*     {DataType::kDouble, std::type_identity<double>{}}, */
/*     {DataType::kString, std::type_identity<std::string>{}} */
/* }; */


/* auto foo_wrapper(DataType type, int arg1) */
/* { */
/*     return std::visit([&](auto v){ */
/*         return foo<typename decltype(v)::type>(arg1); */
/*     }, dispatcher.at(type)); */
/* } */

/* int main (int argv, char** args) */
/* { */
/*   auto& conf = CLIStore::getInstance(); */
/*   conf.Register<size_t>("i",0); */

/*   foo_wrapper(conf.Get<size_t>("i"),conf.Get<size_t>("i")); */
/* } */


// Dummy class template
template<typename T, typename O>
struct A
{
  void run()
  {
    std::cout << "A<" << typeid(T).name() << ", " << typeid(O).name() << ">\n";
  }
};


template<typename T=int, typename O=double>
struct B
{
  void run()
  {
    std::cout << "B<" << typeid(T).name() << ", " << typeid(O).name() << ">\n";
  }
};

template<typename T=int, typename O=double>
struct C
{
  void run()
  {
    std::cout << "B<" << typeid(T).name() << ", " << typeid(O).name() << ">\n";
  }
};
// Available types
using TypeSet1 = std::variant<
  std::type_identity<int>,
  std::type_identity<double>,
  std::type_identity<std::string>,
  std::type_identity<B<>>
>;

// Available types
using TypeSet2 = std::variant<
  std::type_identity<int>,
  std::type_identity<double>,
  std::type_identity<std::string>,
  std::type_identity<C<>>
>;

// Dispatcher maps from string to type_identity
const std::unordered_map<std::string, TypeSet1> dispatcher1 = {
  {"int", std::type_identity<int>{}},
  {"double", std::type_identity<double>{}},
  {"string", std::type_identity<std::string>{}},
  {"b", std::type_identity<B<>>{}}
};

// Dispatcher maps from string to type_identity
const std::unordered_map<std::string, TypeSet2> dispatcher2 = {
  {"int", std::type_identity<int>{}},
  {"double", std::type_identity<double>{}},
  {"string", std::type_identity<std::string>{}},
  {"c", std::type_identity<C<>>{}}
};

int main(int argc, char* argv[])
{
  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0] << " <T-type> <O-type>\n";
    return 1;
  }

  std::string t_str = argv[1];
  std::string o_str = argv[2];

  try
  {
    TypeSet1 t_type = dispatcher1.at(t_str);
    TypeSet2 o_type = dispatcher2.at(o_str);

    std::visit([&](auto t) {
      using T = typename decltype(t)::type;

      std::visit([&](auto o) {
        using O = typename decltype(o)::type;

        A<T, O> obj;
        obj.run();

      }, o_type);

    }, t_type);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}

