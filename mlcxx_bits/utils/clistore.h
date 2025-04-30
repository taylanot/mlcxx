/**
 * @file clistore.h
 * @author Ozgur Taylan Turan
 *
 * A Simple holder for all your applications where you can write easier
 * command-line interfaces...
 */

#ifndef FLAG_H
#define FLAG_H

class CLIStore
{
public:
  // Define the supported value types
  using FlagValue = std::variant<int, DTYPE, std::string, bool, size_t>;

  // Singleton access
  static CLIStore& getInstance()
  {
    static CLIStore instance;
    return instance;
  }

  // Register a flag using the long name; generate short name automatically
  template<typename T>
  void Register(const std::string& longName, const T& defaultValue)
  {
    std::string shortName = GenerateShortName(longName);  // Generate short flag

    if (flags_.count(longName) || flags_.count(shortName))
      throw std::runtime_error("Flag already registered: " + longName);

    flags_[longName] = defaultValue;          // Store the long flag
    shortFlags_[shortName] = longName;        // Map short to long flag
    usedShortNames_.insert(shortName);        // Track used short flags
  }

  // Get flag value by name (long or short)
  template<typename T>
  T Get(const std::string& name) const
  {
    auto it = flags_.find(name);
    if (it == flags_.end())
    {
      auto shortIt = shortFlags_.find(name);
      if (shortIt != shortFlags_.end())
        it = flags_.find(shortIt->second);
      else
        throw std::runtime_error("Flag not found: " + name);
    }

    return std::get<T>(it->second);  // Return typed value
  }

  // Set flag value by name (long or short)
  template<typename T>
  void Set(const std::string& name, const T& value)
  {
    auto it = flags_.find(name);
    if (it == flags_.end())
    {
      auto shortIt = shortFlags_.find(name);
      if (shortIt != shortFlags_.end())
        it = flags_.find(shortIt->second);
      else
        throw std::runtime_error("Flag not found: " + name);
    }

    it->second = value;  // Update the flag
  }

  // Parse command-line arguments  
  void Parse(int argc, char** argv)
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
            throw std::runtime_error("Error: Missing value for flag '--" + flag + "'");
          value = "true";  // Allow boolean flags to be set without explicit value
        }

        Set(flag, ParseValue(flag, value));
      }

      // Handle short flag: -f
      else if (arg.size() >= 2 && arg[0] == '-' && arg[1] != '-')
      {
        std::string shortFlag = arg.substr(1);

        if (shortFlags_.count(shortFlag) == 0)
          throw std::runtime_error("Error: Unknown short flag '-" + shortFlag + "'");

        std::string longName = shortFlags_[shortFlag];
        std::string value;
        if (i + 1 < argc && argv[i + 1][0] != '-')
          value = argv[++i];
        else
        {
          if (!std::holds_alternative<bool>(flags_[longName]))
            throw std::runtime_error("Error: Missing value for flag '-" + shortFlag + "'");
          value = "true";
        }

        Set(longName, ParseValue(longName, value));
      }

      // Unexpected argument format
      else
      {
        throw std::runtime_error("Error: Invalid argument format '" + arg + "'");
      }
    }
  }

  // Print what you are holding
  void Print ( std::ostream& out = std::cout) 
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

  std::string Sanitize(const std::string &input)
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

  std::string GenName ( )
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

      oss << Sanitize(key) << "_" << Sanitize(value_str) << "+";
    }

    std::string result = oss.str();
    if (!result.empty())
      result.erase(result.size() - 1); // remove last "__"

    return result;
  }

private:
  CLIStore() = default;  // Private constructor
  // Store flag values
  std::unordered_map<std::string, FlagValue> flags_;       
  // Short-to-long mapping
  std::unordered_map<std::string, std::string> shortFlags_; 
  // Track used short flags
  std::set<std::string> usedShortNames_;                   

  // Try to generate a short name from longName (1 to 3 characters)
  std::string GenerateShortName(const std::string& longName)
  {
    for (size_t len = 1; len <= 3 && len <= longName.size(); ++len)
    {
      std::string candidate = longName.substr(0, len);
      if (usedShortNames_.count(candidate) == 0)
        return candidate;
    }
    throw std::runtime_error("Unable to generate unique short name for: "
        + longName);
  }

  // Convert string value to proper type based on registered default
  FlagValue ParseValue(const std::string& name, const std::string& valueStr)
  {
    const auto& val = flags_.at(name);
    if (std::holds_alternative<int>(val))
      return std::stoi(valueStr);
    if (std::holds_alternative<size_t>(val))
      return static_cast<size_t>(std::stoi(valueStr));
    if (std::holds_alternative<DTYPE>(val))
      return std::stod(valueStr);
    if (std::holds_alternative<std::string>(val))
      return valueStr;
    if (std::holds_alternative<bool>(val))
      return valueStr == "true" || valueStr == "1";
    throw std::runtime_error("Unsupported flag type for: " + name);
  }
};

#endif




