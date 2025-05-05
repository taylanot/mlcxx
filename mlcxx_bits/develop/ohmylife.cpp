/**
 * @file ohmylife.cpp
 * @author Ozgur Taylan Turan
 *
 * Here tyring to find if I can do this optimization of hpt stuff correctly 
 * compared to mlpack. Since they are not retraining with the whole dataset...
 * Otherwise I have to use the BestModel for selection I think, which is not
 * what I would do, but for the sake of generality gotta give up somethings
 * I guess...
 *
 */

#include <headers.h>

using LOSS = mlpack::MSE;
using SAMPLE = data::RandomSelect<>;
/* using CV = template<class,class,class,class,class> mlpack::SimpleCV; */
using DATASET = data::Dataset<arma::Row<DTYPE>>;
using MODEL = mlpack::LinearRegression<>;

#include <iostream>
#include <type_traits>
#include <concepts>
#include <ranges>

/* template <typename T> */
/* concept Iterable = requires(T t) */
/* { */
/*   std::begin(t); */
/*   std::end(t); */
/* }; */

/* template <typename T> */
/* bool is_iterable(const T&) */
/* { */
/*   return Iterable<T>; */
/* } */

// Concept to check if a type is iterable
template <typename T>
concept Iterable = requires(T t) {
    std::begin(t);
    std::end(t);
};

// Helper to check if a type is iterable
template <typename T>
struct is_iterable {
    static constexpr bool value = Iterable<T>;
};

template <typename T>
auto wrap(T&& val)
{
  using RawT = std::remove_cvref_t<T>;

  if constexpr (!Iterable<RawT>)
  {
    // Force move to satisfy Fixed(T&&)
    return mlpack::Fixed<RawT>(std::move(val));
  }
  else
  {
    return std::forward<T>(val);
  }
}

// Main function to adjust parameter pack
template <typename... Args>
auto create_fixedpack(Args&&... args)
{
  return std::make_tuple(wrap(std::forward<Args>(args))...);
}

// Internal implementation that uses index sequence
template <typename... Args, std::size_t... Is>
std::vector<std::size_t> get_ids_impl(std::index_sequence<Is...>, Args&&... args)
{
  std::vector<std::size_t> indices;
  ((is_iterable<std::remove_cvref_t<Args>>::value ? indices.push_back(Is) : void()), ...);
  return indices;
}

// Public-facing function
template <typename... Args>
std::vector<std::size_t> get_ids(Args&&... args)
{
  return get_ids_impl(std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);
}


// Helper to modify a tuple at runtime using its index
template <typename Tuple, typename T>
void set_tuple_element_at(std::size_t index, Tuple& tpl, T&& value)
{
  std::apply(
    [&](auto&... elements)
    {
      std::size_t i = 0;
      ((i++ == index ? (elements = std::forward<T>(value), void()) : void()), ...);
    },
    tpl
  );
}

// Main function to apply replacements at given indices
template <typename... Ts, typename... Replacements>
std::tuple<Ts...> replace_tuple_elements(
  const std::vector<std::size_t>& indices,
  const std::tuple<Ts...>& tpl,
  Replacements&&... replacements)
{
  constexpr std::size_t tuple_size = sizeof...(Ts);
  if (indices.size() != sizeof...(Replacements)) {
    throw std::invalid_argument("Mismatch between indices and replacement values.");
  }

  std::tuple<Ts...> copy = tpl;
  auto replacements_tuple = std::forward_as_tuple(std::forward<Replacements>(replacements)...);

  for (std::size_t i = 0; i < indices.size(); ++i) {
    std::size_t idx = indices[i];
    if (idx >= tuple_size) {
      throw std::out_of_range("Index out of bounds in tuple replacement.");
    }

    std::apply(
      [&](auto&&... vals)
      {
        std::size_t j = 0;
        ((j++ == i ? set_tuple_element_at(idx, copy, std::forward<decltype(vals)>(vals)) : void()), ...);
      },
      replacements_tuple
    );
  }
  return copy;
}

template<typename Tuple, std::size_t... Indices>
void printTupleImpl(const Tuple& tup, std::index_sequence<Indices...>)
{
  std::cout << "(";
  ((std::cout << (Indices == 0 ? "" : ", ") << std::get<Indices>(tup)), ...);
  std::cout << ")";
}

template<typename... Args>
void PRINT_TUPLE(const std::tuple<Args...>& tup)
{
  printTupleImpl(tup, std::index_sequence_for<Args...>{});
}

template<typename... Ts>
void printTupleTypes(std::tuple<Ts...> const&)
{
  ((std::cout << typeid(Ts).name() << "\n"), ...);
}

template<class... Args>
void TUNE(Args... args)
{
  DATASET dataset(2);
  dataset.Linear(5000);
  /* auto pack = create_fixedpack(args...); */
  /* PRINT_VAR(typeid(std::get<0>(pack)).name()); */
  /* PRINT_VAR(typeid(std::get<1>(pack)).name()); */
  auto pack_ = create_fixedpack(args...);
  /* PRINT_VAR(typeid(std::get<0>(pack)).name()); */
  /* PRINT_VAR(typeid(std::get<1>(pack)).name()); */

  /* auto best = std::make_tuple(0.01); */
  auto ids = get_ids(args...);
  for (int id : ids)
    PRINT(id);
  auto tuple = std::make_tuple(std::forward<Args>(args)...);

  /* auto new_tpl = std::apply( [&](auto&&... args) */ 
  /* { */
  /*   return replace_tuple_elements(ids,tuple,std::forward<decltype(args)>(args)...); */
  /* }, best ); */

  mlpack::HyperParameterTuner<mlpack::LinearRegression<>,LOSS,mlpack::SimpleCV> hpt(0.2,dataset.inputs_,dataset.labels_);

  auto best_tpl = std::apply(
  [&](auto&&... args) {
    return hpt.Optimize(std::forward<decltype(args)>(args)...);
  }, pack_ );


  auto new_tpl = std::apply( [&](auto&&... args) 
  {
    return replace_tuple_elements(ids,tuple,std::forward<decltype(args)>(args)...);
  }, best_tpl );

  const int i = 0;
  /* PRINT_TUPLE(new_tpl); */
  PRINT_VAR(std::get<i>(new_tpl));
  PRINT_VAR(std::get<0>(new_tpl));
  PRINT_VAR(std::get<1>(new_tpl));


  /* mlpack::LinearRegression<> model (dataset.inputs_,dataset.labels_,); */
  /* PRINT_VAR(model.ComputeError(dataset.inputs_,dataset.labels_)); */
  auto a = std::make_tuple(0.1,1);
  printTupleTypes(a);
  printTupleTypes(new_tpl);
  auto model = std::apply(
  [&](auto&&... args) {
    return mlpack::LinearRegression<>(
      dataset.inputs_, dataset.labels_,
      std::forward<decltype(args)>(args)...
    );
  }, a );

  PRINT(model.Lambda( ));
  PRINT(model.ComputeError(dataset.inputs_, dataset.labels_));

}

int main ( int argc, char** argv )
{
  arma::wall_clock timer;
  timer.tic();

  

  auto armalrs = arma::logspace<arma::Row<DTYPE>>(-5,-1,10);
  bool inters = true;
  auto tuple = std::make_tuple(armalrs,inters);

  TUNE(armalrs,inters);

  /* auto tpl = std::make_tuple(42, std::string("hello"), 3.14); */
  /* std::vector<std::size_t> indices = {0, 2}; */

  /* auto new_tpl = replace_tuple_elements(indices, tpl, 99, 2.718); */

  /* std::cout << std::get<0>(new_tpl) << ", " */
  /*           << std::get<1>(new_tpl) << ", " */
  /*           << std::get<2>(new_tpl) << '\n'; */

  /* auto indexes = get_ids(armalrs,inters); */
  /* auto indexes = get_ids(inters,armalrs); */
  

  /* TUNE(armalrs,inters); */

  /* DTYPE armalrs = 0.1; */ 
  /* std::vector<bool> inters = {true,false}; */

  /* auto pack_ = adjust_parameter_pack(armalrs,inters); */

  /* PRINT(typeid(pack).name()); */

  // Print the tuple types to verify the order
  /* std::apply([](auto&&... elements) { */
  /*   ((std::cout << typeid(elements).name() << '\n'), ...); */
  /* }, pack_); */
  /* PRINT_VAR(typeid(std::get<0>(pack)).name()); */
  /* PRINT_VAR(typeid(std::get<1>(pack)).name()); */

  /* auto wrapped_a = wrap_arg(armalrs); */
  /* auto wrapped_b = wrap_arg(inters); */

  /* PRINT_VAR(std::boolalpha<<std::is_same_v<decltype(wrapped_a), mlpack::FixedArg<arma::Row<DTYPE>,0>>) */
  /* PRINT_VAR(typeid(wrapped_a).name()); */
  /* PRINT_VAR(std::is_same_v<decltype(wrapped_b), bool>) */

  /* auto lrs = arma::conv_to<std::vector<DTYPE>>::from(armalrs); */

  /* auto pack = std::make_tuple(armalrs,mlpack::Fixed(inters)); */

   /* std::apply([](auto&&... elements) { */
   /*  ((std::cout << typeid(elements).name() << '\n'), ...); */
  /* }, pack); */

  /* mlpack::HyperParameterTuner<mlpack::LinearRegression<>,LOSS,mlpack::SimpleCV> hpt(0.2,dataset.inputs_,dataset.labels_); */

  /* auto a = std::apply([&](auto&&... args) { */
  /*   return hpt.Optimize(std::forward<decltype(args)>(args)...); */
  /* }, pack); */

  /* auto pack_fin = dynamic_replace(tuple, a); */



  /* PRINT_VAR(std::get<0>(pack)); */
  /* PRINT_VAR(std::get<1>(pack)); */

  /* PRINT_VAR(std::get<0>(pack_fin)); */
  /*   // Print the results */
  /*   std::apply([](auto&&... vals) { */
  /*       ((std::cout << vals << " "), ...); */
  /*   }, result); */

  /* auto a = std::apply(hpt.Optimize,pack); */
  /* PRINT_VAR(std::get<0>(a)); */

  /* std::vector<bool> ints = {true,false}; */
  /* auto a = hpt.Optimize(lrs,ints); */

  /* PRINT(std::get<0>(a)); */
  /* PRINT(std::get<1>(a)); */

  

  PRINT_TIME(timer.toc())
  return 0;
}




