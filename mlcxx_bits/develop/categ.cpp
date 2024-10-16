/**
 * @file catag.cpp
 * @author Ozgur Taylan Turan
 *
 * This file is for developing the catogirical cehcking
 *
 */

#define DTYPE double  

#include <headers.h>

template<class T=DTYPE>
bool isCategorical(const arma::Row<T>& row) 
{
  std::set<int> distinctValues;
  
  // Iterate over the array
  for (size_t i = 0; i < row.n_elem; ++i) 
  {
    double value = row(i);
    
    // Check if the value is a whole number
    if (value == static_cast<int>(value)) 
        distinctValues.insert(static_cast<int>(value));
    else 
        // If any value is not a whole number, it's not categorical
        return false;
  }

  // If we have only a few distinct values, consider it categorical
  return distinctValues.size() < row.n_elem;
}

template<class T=DTYPE>
// Function to convert real values into categorical values
arma::Row<size_t> convertToCategorical(const arma::Row<double>& row) 
{
  std::unordered_map<T, size_t> valueToIndex;
  size_t categoryIndex = 0;

  // Create a mapping of unique values to integer indices
  for (size_t i = 0; i < row.n_elem; ++i) 
  {
      T value = row(i);
      // If the value has not been seen before, assign a new category
      if (valueToIndex.find(value) == valueToIndex.end()) 
          valueToIndex[value] = categoryIndex++;
  }

  // Create a new Row<size_t> to store the mapped categorical values
  arma::Row<size_t> categoricalRow(row.n_elem);

  // Map each original value to its corresponding categorical index
  for (size_t i = 0; i < row.n_elem; ++i) {
      categoricalRow(i) = valueToIndex[row(i)];
  }

  return categoricalRow;
}

template<class T=DTYPE>
arma::Row<size_t> processRow(const arma::Row<T>& row) 
{
  if (isCategorical(row)) 
  {
    // Return the row as is if it's already categorical
    std::cout << "The row is already categorical." << std::endl;
    // Convert it to a size_t type (though it might not be necessary if it's already integers)
    arma::Row<size_t> categoricalRow(row.n_elem);
    for (size_t i = 0; i < row.n_elem; ++i) 
        categoricalRow(i) = static_cast<size_t>(row(i));
    return categoricalRow;
  } 
  else 
  {
    // Convert real values into categorical values
    std::cout << "The row is not categorical. Converting..." << std::endl;
    return convertToCategorical(row);
  }
}

int main() 
{
  // Example Row of real values (continuous)
  arma::Row<double> realValues = {1.5, 2.3, 3.7, 5.6, 2.3, 1.5};
  
  // Example Row of categorical values (discrete)
  arma::Row<double> categoricalValues = {0, 1, 2, 1, 0, 2};
  
  // Process the rows
  arma::Row<size_t> processedRealValues = processRow(realValues);
  arma::Row<size_t> processedCategoricalValues = processRow(categoricalValues);
  
  // Print the results
  std::cout << "Processed real values: ";
  processedRealValues.print();
  
  std::cout << "Processed categorical values: ";
  processedCategoricalValues.print();
  
  return 0;
}
