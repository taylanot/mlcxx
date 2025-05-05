/**
 * @file tuple_manip.cpp
 * @author Ozgur Taylan Turan
 *
 * Manipulating tuples is harder than it seems :(
 *
 *
 */

#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <type_traits>

// Function to decompose a tuple into a new tuple
template <typename... Args>
auto decomposeTuple(const std::tuple<Args...>& t) 
{
  return std::apply([](const Args&... args) 
    {
      return std::make_tuple(args...);
    }, t);
}


// Helper function to change type of tuple elements
template <std::size_t Index, typename T, typename... Types>
auto change_type(std::tuple<Types...>& tup, T new_value) {
    if constexpr (Index < sizeof...(Types)) {
        std::get<Index>(tup) = new_value;
    }
}

// Utility function to display tuple
template <std::size_t Index, typename... Types>
void print_tuple(const std::tuple<Types...>& tup) {
    std::cout << std::get<Index>(tup) << std::endl;
}

template <typename... Tuples>
auto concatenate(const std::vector<std::size_t>& order, const Tuples&... tuples) {
    // Create a tuple with all provided tuples
    std::tuple<Tuples...> all_tuples = std::make_tuple(tuples...);
    
    // Create a new tuple to hold the ordered concatenation
    std::tuple<> result;
    
    // Concatenate tuples in the specified order
    for (auto idx : order) {
        result = std::tuple_cat(result, std::get<idx>(all_tuples));
    }
    
    return result;
}


/* int main() { */
/*     // Define a tuple */
/*     std::tuple<int, double, char> my_tuple(1, 2.5, 'a'); */
    
/*     // Change the type of the second element (double to int) */
/*     auto dec = decomposeTuple(my_tuple); */
/*     auto change = concatenate({1,3,2},dec); */

/*     std::cout << std::get<0>(dec) << std::endl; */
/*     std::cout << std::get<1>(dec) << std::endl; */
/*     std::cout << std::get<2>(dec) << std::endl; */

/*     kk */

    
    
/*     return 0; */
/* } */

#include <any>
#include <stdexcept>
#include <tuple>
#include <iostream>

// Variadic template to concatenate an arbitrary number of tuples
template<typename... Tuples>
auto concatenateTuples(Tuples&&... tuples)
{
  return std::tuple_cat(std::forward<Tuples>(tuples)...);
}

// Helper to print a tuple (for demonstration)
template<typename Tuple, std::size_t... Indices>
void printTupleImpl(const Tuple& t, std::index_sequence<Indices...>)
{
  ((std::cout << (Indices == 0 ? "" : ", ") << std::get<Indices>(t)), ...);
}

template<typename... Args>
void printTuple(const std::tuple<Args...>& t)
{
  std::cout << "(";
  printTupleImpl(t, std::index_sequence_for<Args...>{});
  std::cout << ")" << std::endl;
}

template<std::size_t N, class TupleT, class NewT>
constexpr auto replace_tuple_element( const TupleT& t, const NewT& n )
{
    constexpr auto tail_size = std::tuple_size<TupleT>::value - N - 1;

    return [&]<std::size_t... I_head, std::size_t... I_tail>
        ( std::index_sequence<I_head...>, std::index_sequence<I_tail...> )
        {
            return std::tuple{
                std::get<I_head>( t )...,
                n,
                std::get<I_tail + N + 1>( t )...
            };
        }(  
           std::make_index_sequence<N>{}, 
           std::make_index_sequence<tail_size>{} 
          );
}

template<size_t drop, size_t ...ixs>
constexpr auto calc_drop_sequence_dropper(std::index_sequence<ixs...>)
{
    return std::index_sequence<(ixs >= drop ? ixs + 1 : ixs)...>{};
}

//Creates a monotonically increasing sequence on the range [0, `count`), except
//that `drop` will not appear.
template<size_t count, size_t drop>
constexpr auto calc_drop_copy_sequence()
{
    static_assert(count > 0, "You cannot pass an empty sequence.");
    static_assert(drop < count, "The drop index must be less than the count.");
    constexpr auto count_sequence = std::make_index_sequence<count - 1>();
    return calc_drop_sequence_dropper<drop>(count_sequence);
}

template<typename Tuple, size_t ...ixs>
constexpr auto copy_move_tuple_by_sequence(Tuple &&tpl, std::index_sequence<ixs...>)
{
    using TplType = std::remove_reference_t<Tuple>;

    return std::tuple<std::tuple_element_t<ixs, TplType>...>(
        std::get<ixs>(std::forward<Tuple>(tpl))...);
}

template<size_t drop, typename Tuple>
constexpr auto drop_tuple_element(Tuple &&tpl)
{
    using TplType = std::remove_reference_t<Tuple>;

    constexpr size_t tpl_size = std::tuple_size<TplType>::value;

    constexpr auto copy_seq = calc_drop_copy_sequence<tpl_size, drop>();

    return copy_move_tuple_by_sequence(std::forward<Tuple>(tpl), copy_seq);
}

// Helper function: recursively access tuple elements
template<std::size_t I = 0, typename... Ts>
std::any get(const std::tuple<Ts...>& tup, std::size_t index)
{
  if constexpr (I < sizeof...(Ts))
  {
    if (I == index)
    {
      return std::any(std::get<I>(tup));
    }
    else
    {
      return get<I + 1>(tup, index);
    }
  }
  else
  {
    throw std::out_of_range("Tuple index out of range");
  }
}


template <typename TupleT, std::size_t... Is>
auto rebuild_with_replacement_impl(const TupleT& tpl, std::size_t index, const std::any& new_value, std::index_sequence<Is...>)
{
  return std::make_tuple((Is == index ? new_value : std::any(std::get<Is>(tpl)))...);
}

template <typename... Ts>
auto replace_tuple_element_rt(const std::tuple<Ts...>& tpl, std::size_t index, const std::any& new_value)
{
  if (index >= sizeof...(Ts))
  {
    throw std::out_of_range("Index out of range in replace_tuple_element_rt");
  }

  return rebuild_with_replacement_impl(tpl, index, new_value, std::index_sequence_for<Ts...>{});
}




template <typename TupleT, std::size_t... Is>
auto rebuild_with_dropped_impl(const TupleT& tpl, std::size_t index, std::index_sequence<Is...>)
{
  return std::make_tuple((std::any((Is < index) ? std::get<Is>(tpl) : std::get<Is + 1>(tpl)))...);
}

template <typename... Ts>
auto drop_tuple_element_rt(const std::tuple<Ts...>& tpl, std::size_t index)
{
  constexpr std::size_t N = sizeof...(Ts);
  if (index >= N)
  {
    throw std::out_of_range("Index out of range in drop_tuple_element_rt");
  }

  return rebuild_with_dropped_impl(tpl, index, std::make_index_sequence<N - 1>{});
}


/* int main() */
/* { */
/*   auto t = std::make_tuple(1, "a",3.14, "hello", true,false); */
/*   /1* auto other = std::make_tuple("ali","veli"); *1/ */
/*   printTuple(t); */

/*   auto tdrop = drop_tuple_element<0>(t); */
/*   auto tdrop2 = replace_tuple_element_rt(t,0,std::any(99)); */
/*   printTuple(tdrop); */
/*   printTuple(tdrop2); */

/*   auto t1 = replace_tuple_element<0> (tdrop, "ali"); */
/*   printTuple(t1); */

/*   auto t2 = replace_tuple_element<0> (t1, "veli"); */
/*   printTuple(t2); */

/*   auto t3 = replace_tuple_element<1> (t2, true); */
/*   printTuple(t3); */

/* } */


#include <tuple>
#include <iostream>
#include <utility>
#include <type_traits>

// Helper to generate a tuple with elements starting from index N
template<std::size_t N, typename Tuple, std::size_t... Is>
auto drop_first_impl(Tuple&& tpl, std::index_sequence<Is...>)
{
  return std::make_tuple(std::get<N + Is>(std::forward<Tuple>(tpl))...);
}

// Main interface
template<std::size_t N, typename... Ts>
auto drop_first(const std::tuple<Ts...>& tpl)
{
  static_assert(N <= sizeof...(Ts), "Cannot drop more elements than tuple size");
  return drop_first_impl<N>(
    tpl,
    std::make_index_sequence<sizeof...(Ts) - N>{}
  );
}


// Helper: Get type at index in tuple
template<std::size_t I, typename Tuple>
using tuple_element_t = typename std::tuple_element<I, Tuple>::type;

// Core filter: collects indices where type == Target
template<typename Tuple, typename Target, std::size_t... Is>
constexpr auto filter_tuple_indices_impl(std::index_sequence<Is...>)
{
  return std::index_sequence<
    Is... // include only if type at Is is Target
    >{
    ((std::is_same<tuple_element_t<Is, Tuple>, Target>::value ? Is : static_cast<std::size_t>(-1)) + 1)... // +1 trick to shift -1 to 0 and skip later
  };
}

// Final filter that drops `-1+1 = 0` fake indices
template<std::size_t... Is>
constexpr auto compress(std::index_sequence<Is...>)
{
  constexpr std::array<std::size_t, sizeof...(Is)> raw = {Is...};
  constexpr auto is_valid = [](std::size_t i) { return i != 0; };

  std::array<std::size_t, sizeof...(Is)> result{};
  std::size_t count = 0;

  for (std::size_t i = 0; i < raw.size(); ++i)
  {
    if (is_valid(raw[i])) result[count++] = raw[i] - 1; // fix back to original index
  }

  // Manual compile-time pack creation
  if constexpr (count == 0) return std::index_sequence<>{};
  else if constexpr (count == 1) return std::index_sequence<result[0]>{};
  else if constexpr (count == 2) return std::index_sequence<result[0], result[1]>{};
  else if constexpr (count == 3) return std::index_sequence<result[0], result[1], result[2]>{};
  else static_assert(count <= 3, "Expand this for more indices");
}

// Entry point: from tuple type
template<typename Target, typename Tuple>
constexpr auto filter_tuple_indices()
{
  constexpr std::size_t N = std::tuple_size<Tuple>::value;
  return compress(filter_tuple_indices_impl<Tuple, Target>(std::make_index_sequence<N>{}));
}

int main()
{
  auto tup = std::make_tuple(1, 2.5, 'a', "hello");

  auto tup2 = std::make_tuple(1, 2.5);

  auto dropped = std::tuple_cat(drop_first<std::tuple_size<decltype(tup2)>::value>(tup),tup2);

  // Output the resulting tuple
  std::apply([](auto&&... args) 
  {
    ((std::cout << args << " "), ...);
  }, dropped);
  // Output the resulting tuple
  std::cout << "\n#####" << std::endl;
  std::apply([](auto&&... args) 
  {
    ((std::cout << args << " "), ...);
  }, tup);

  std::cout << std::endl;

  return 0;
}

