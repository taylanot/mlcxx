/**
 * @file xval.cpp
 * @author Ozgur Taylan Turan
 *
 * Let's check xval results
 */
#include <headers.h>

const static std::string smallLetters = "abcdefghijklmnopqrstuvwxyz";
const static std::string specialCharacters = "!@#$%^&*()_+-=[]{}|;:'.<>?/";
const static std::string capitalLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const static std::string numbers = "0123456789";
extern std::mt19937 gen(SEED);

// Function to calculate password strength based on length and options used
std::string getStrength(int length, int opt) 
{
  if (length < 8) 
  {
    switch (opt) 
    {
      case 1:
        return "Weakest";
      case 2:
        return "Weak";
      default:
        return "Invalid";
    }
  } 
  else 
  { // length >= 8
    switch (opt) 
    {
      case 3:
        return "Ok";
      case 4:
        return "Strong";
      case 5:
        return "Strongest";
      default:
        return "Invalid";
    }
  }
}

// Function to select random elements from a vector and return their sum
std::string getcPool ( const std::vector<std::string>& vec, size_t k ) 
{
  size_t vectorSize = vec.size();

  // Step 1: Generate a list of indices
  std::vector<size_t> indices(vectorSize);
  for (size_t i = 0; i < vectorSize; ++i) 
    indices[i] = i;
  // Step 2: Shuffle the indices
  std::shuffle(indices.begin(), indices.end(), gen);

  // Step 3: Select the first k indices
  std::vector<size_t> randomIndices(indices.begin(), indices.begin() + k);
  for (int i = 0; i<randomIndices.size();i++)
    PRINT_VAR(i)

  // Step 4: Sum the elements at the selected indices
  std::string sum;
  for (size_t index : randomIndices) 
    sum += vec[index];

  return sum;
}

// Function to generate a random password
std::string genPass( int length, int opt ) 
{
  std::string cPool;
  std::string pw;
  std::vector<std::string> aPool = { smallLetters,specialCharacters,
                                    capitalLetters,numbers };
  switch (opt) 
  {
    case 1:
      cPool = getcPool(aPool,1);
      break;
    case 2:
      cPool = getcPool(aPool,2);
      break;
    case 3:
      cPool = getcPool(aPool,1);
      break;
    case 4:
      cPool = getcPool(aPool,2);
      break;
    case 5:
      cPool = getcPool(aPool,3);
      break;
    default:
      ERR("You cannot have this strength!");
      exit(0);
  }
  
  // Generate the password
  for (int i = 0; i < length; ++i) 
  {
    int randomIndex = rand() % cPool.size();
    pw+= cPool[randomIndex];
  }

  return pw;
}

int main(int argc, char* argv[]) 
{
  // Fix the seeds
  arma::arma_rng::set_seed(SEED); 
  std::filesystem::create_directories("../.exp/passdb");

  // Set default values for arguments
  int npass = 10;  // Default number of passwords
  int opt = 5;  // Default strength of password
  int verbosity = 1;  // Default verbosity level

  // Check if arguments are provided and override defaults
  if (argc >= 2) npass = std::stoi(argv[1]);
  if (argc >= 3) opt = std::stoi(argv[2]);
  if (argc >= 4) verbosity = std::stoi(argv[3]);

  if (argc < 3) 
  {
    ERR("Usage: " << argv[0] << " <number_of_passwords>" << argv[1] << " <strength_of_password>" << argv[2] << " <verbosity>");
    return 1;
  }


  if (opt < 1 || opt > 5) 
  {
    ERR("Invalid number of options!\n");
    return 1;
  }

  std::string csvFileName = std::to_string(opt)+".csv";
  std::ofstream csvFile(csvFileName, std::ios::app);

  if (!csvFile.is_open()) 
  {
    ERR("Failed to open the file for writing!\n");
    return 1;
  }

  for (int i = 0; i < npass; ++i) 
  {
    int length;
    if (opt <= 2)
    {
      std::uniform_int_distribution<> dist(4, 7);
      length = dist(gen);
    }
    else if (opt>2 && opt<8)
    {
      std::uniform_int_distribution<> dist(8, 12);
      length = dist(gen);
    }

    std::string password = genPass(length, opt);

    if (verbosity == 0) 
      LOG("Generated password: " << password << "Strength: " << getStrength(length,opt) << "\n");
    
    csvFile << password << "," << getStrength(length,opt) << "\n";
  }
  csvFile.close();
  LOG("Passwords and strengths saved to " << csvFileName);
  return 0;
}
