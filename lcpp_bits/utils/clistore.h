/**
 * @file clistore.h
 * @author Ozgur Taylan Turan
 *
 * A Simple holder for all your applications where you can write easier
 * command-line interfaces...
 */

#ifndef FLAG_H
#define FLAG_H

//-----------------------------------------------------------------------------
// CLIStore : This is a command line interface storing singleton.
// It has some nice functionality that might be of use while desining apps
// for HPC usage.
//-----------------------------------------------------------------------------
class CLIStore
{
public:
  // Define the supported value types
  using FlagValue = std::variant<int, float, double, std::string, bool, size_t>;

  // Singleton access
  static CLIStore& GetInstance()
  {
    static CLIStore instance;
    return instance;
  }
  /////////////////////////////////////////////////////////////////////////////
  // Register a flag using the long name
  template<typename T>
  void Register( const std::string& longName, const T& defaultValue )
  {
    if (flags_.count(longName))
      throw std::runtime_error("Flag already registered: " + longName);

    flags_[longName] = defaultValue; // Store the flag
  }
  /////////////////////////////////////////////////////////////////////////////
  template<typename T>
  void Register( const std::string& longName,
                 const T& defaultValue,
                 const std::vector<T>& options )
  {
    this->Register(longName, defaultValue);

    std::vector<FlagValue> convertedOptions;
    convertedOptions.reserve(options.size());

    for (const T& val : options)
      convertedOptions.push_back(val);  // Implicitly converts to FlagValue

    options_[longName] = std::move(convertedOptions);
  }
  /////////////////////////////////////////////////////////////////////////////
  // Get flag value by name
  template<typename T>
  T Get ( const std::string& name ) const
  {
    auto it = flags_.find(name);
    if (it == flags_.end())
      throw std::runtime_error("Flag not found: " + name);

    return std::get<T>(it->second);  // Return typed value
  }
  /////////////////////////////////////////////////////////////////////////////
  // Get flag options by name
  template<typename T>
  std::vector<T> GetOptions( const std::string& name ) const
  {
    auto it = options_.find(name);
    if (it == options_.end())
      throw std::runtime_error("Flag not found: " + name);

    return _ExtractTypedVector<T>(it->second, name);
  }
  /////////////////////////////////////////////////////////////////////////////
  // Set flag value by name
  template<typename T>
  void Set ( const std::string& name, const T& value )
  {
    auto it = flags_.find(name);
    if (it == flags_.end())
      throw std::runtime_error("Flag not found: " + name);

    it->second = value;  // Update the flag
  }
  /////////////////////////////////////////////////////////////////////////////
  // Parse command-line arguments  
  void Parse ( int argc, char** argv )
  {
    for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];

      // Handle long flag: --flag
      if (arg.size() >= 3 && arg.substr(0, 2) == "--")
      {
        std::string flag = arg.substr(2);

        if (flags_.count(flag) == 0)
          throw std::runtime_error("Error: Unknown flag '--" + flag + "'");

        std::string value;
        if (i + 1 < argc && argv[i + 1][0] != '-')
          value = argv[++i];
        else
        {
          // Value is required unless type is bool
          if (!std::holds_alternative<bool>(flags_[flag]))
            throw std::runtime_error("Error: Missing value for flag '--" +
                                                                    flag + "'");
          value = "true";  
        }

        Set(flag, ParseValue(flag, value));
      }

      // Unexpected argument format
      else
        throw std::runtime_error("Error: Invalid argument format '" + arg + "'");
    }
  }
  /////////////////////////////////////////////////////////////////////////////
  // Print what you are holding
  void Print ( std::ostream& out = std::cout ) 
  {
    out << std::string(50, '-') << "\n"
        << std::left  
        << std::setw(25) << "Flag Name"
        << std::setw(25) << "Flag Value" << std::endl
        << std::string(50, '-') << "\n";

    for (const auto &entry : flags_)
    {
      const auto &key = entry.first;
      const auto &value = entry.second;

      std::string value_str;
      std::string type_str;

      std::visit([&](const auto &val)
      {
        std::ostringstream oss;
        oss << val;
        value_str = oss.str();

      }, value);

      out << std::left
          << std::setw(25) << key
          << std::setw(25) << value_str
          << "\n";
    }
    out << std::string(50, '-') << "\n";
  }
  /////////////////////////////////////////////////////////////////////////////
  // Sanatize the name
  std::string Sanitize ( const std::string &input ) const
  {
    std::string output = input;

    // Replace '.' with 'p' (e.g., 0.01 → 0p01)
    std::replace(output.begin(), output.end(), '.', 'p');

    // Remove or replace other non-alphanumerics as needed
    for (char &c : output)
      if (!std::isalnum(static_cast<unsigned char>(c)))
        c = '_';

    return output;
  }
  /////////////////////////////////////////////////////////////////////////////
  // Generate a unique name with the keys you are holding
  std::string GenName ( ) const 
  {
    std::vector<std::string> keys;
    for (const auto &pair : flags_)
    {
      keys.push_back(pair.first);
    }
    std::sort(keys.begin(), keys.end()); // deterministic order

    std::ostringstream oss;

    for (const auto &key : keys)
    {
      const auto &value = flags_.at(key);

      std::string value_str;
      std::visit([&](const auto &val)
      {
        std::ostringstream val_stream;

        if constexpr (std::is_same_v<decltype(val), bool>)
          val_stream << (val ? "true" : "false");
        else
          val_stream << val;

        value_str = val_stream.str();
      }, value);

      oss << Sanitize(key) << "_" << Sanitize(value_str) << "-";
    }

    std::string result = oss.str();
    if (!result.empty())
      result.erase(result.size() - 1); // remove last "__"

    return result;
  }
  /////////////////////////////////////////////////////////////////////////////
  // Generate a unique name with the specified keys
  std::string GenName(const std::vector<std::string>& include_keys) const
  {
    std::vector<std::string> keys;
    for (const auto &key : include_keys)
    {
      if (flags_.find(key) != flags_.end())
      {
        keys.push_back(key);
      }
    }
    std::sort(keys.begin(), keys.end()); // deterministic order

    std::ostringstream oss;

    for (const auto &key : keys)
    {
      const auto &value = flags_.at(key);

      std::string value_str;
      std::visit([&](const auto &val)
      {
        std::ostringstream val_stream;

        if constexpr (std::is_same_v<decltype(val), bool>)
          val_stream << (val ? "true" : "false");
        else
          val_stream << val;

        value_str = val_stream.str();
      }, value);

      oss << Sanitize(key) << "_" << Sanitize(value_str) << "-";
    }

    std::string result = oss.str();
    if (!result.empty())
      result.erase(result.size() - 1); // remove last '+'
    return result;
  }

private:

  template<typename T>
  std::vector<T> _ExtractTypedVector(const std::vector<FlagValue>& input,
                                     const std::string& name) const 
  {
    std::vector<T> output;
    for (const auto& val : input)
    {
      if (!std::holds_alternative<T>(val))
        throw std::runtime_error("Type mismatch in options for flag: " + name);

      output.push_back(std::get<T>(val));
    }
    return output;
  }
  /////////////////////////////////////////////////////////////////////////////

  CLIStore() = default;  // Private constructor
  // Store flag values
  std::unordered_map<std::string, FlagValue> flags_;       
  // options for the flags
  std::unordered_map<std::string, std::vector<FlagValue>> options_;       

  // Convert string value to proper type based on registered default
  FlagValue ParseValue(const std::string& name, const std::string& valueStr)
  {
    const auto& val = flags_.at(name);
    if (std::holds_alternative<int>(val))
      return std::stoi(valueStr);
    if (std::holds_alternative<size_t>(val))
      return static_cast<size_t>(std::stoi(valueStr));
    if (std::holds_alternative<float>(val))
      return std::stof(valueStr);
    if (std::holds_alternative<double>(val))
      return std::stod(valueStr);
    if (std::holds_alternative<std::string>(val))
      return valueStr;
    if (std::holds_alternative<bool>(val))
      return valueStr == "true" || valueStr == "1";
    throw std::runtime_error("Unsupported flag type for: " + name);
  }
};
#endif




