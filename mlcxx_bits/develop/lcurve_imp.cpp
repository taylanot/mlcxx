/**
 * @file lcurve_imp.cpp
 * @author Ozgur Taylan Turan
 *
 * 
 * I will try to imporove the Learning Curve creation here
 *
 * - Timer 
 * - Seed
 * - hyper-parameter tuning
 *
 */

#define DTYPE double
#include <headers.h>

int main ( int argc, char** argv )
{
  std::filesystem::path path = EXP_PATH/"21_02_25/lcurve";
  std::filesystem::create_directories(path);
  std::filesystem::current_path(path);

  /* auto loaded = src::LCurve<mlpack::LinearRegression<>,mlpack::MSE>::Load("LCurve.bin"); */
  /* src::LCurve<mlpack::LinearRegression<>,mlpack::MSE> task = std::move(*loaded); */
  /* std::cout << task.GetResults() << std::endl; */


  /* arma::wall_clock timer; */
  /* timer.tic(); */
  /* alarm(1); */

  /* size_t rep = 10000; */

  /* data::regression::Dataset dataset(3,100); */
  /* dataset.Generate("Linear"); */

  /* arma::irowvec Ns = arma::regspace<arma::irowvec>(1,1,30); */

  /* src::LCurve<mlpack::LinearRegression<>,mlpack::MSE> lc(Ns,rep,true,true); */
  /* lc.Bootstrap(dataset.inputs_,dataset.labels_,true); */

  /* PRINT_TIME(timer.toc()); */

  return 0;
}

/* #include <iostream> */
/* #include <csignal> */
/* #include <unistd.h> */
/* #include <functional> */
/* #include <fstream> */
/* #include <armadillo> */

/* #include <mlpack.hpp> */

/* std::function<void()> globalSafeFailFunc;  // Global function pointer */
/*                                            // */

/* class TaskHandler */ 
/* { */

/* private: */
/*     friend class cereal::access; */ 

/*     int taskId;  // Example parameter to store */
/*     std::string name; */
/*     arma::rowvec data; */

/*     template <class Archive> */
/*     void serialize(Archive& ar) */ 
/*     { */
/*       ar( CEREAL_NVP(taskId), */
/*           CEREAL_NVP(name), */
/*           CEREAL_NVP(data)); */
/*     } */



/* public: */
/*     TaskHandler( ) { }; */
    
/*     int GetId( ) {return taskId;}; */
/*     arma::rowvec GetData( ) {return data;}; */

/*     TaskHandler(int id, bool safe_fail=true,std::string name="name") : taskId(id), name(name) */
/*     { */
/*       data = arma::zeros<arma::rowvec>(10); */
/*       if (safe_fail) */
/*         registerSignalHandler(); */
/*       // Bind cleanup function with access to taskId */
/*       globalSafeFailFunc = [this]() { this->cleanup(); }; */
/*     } */

/*     void runTask() */ 
/*     { */
/*       std::cout << "Task " << taskId << " started...\n"; */
/*       data[0] = 1.; */
/*       #pragma omp parallel for collapse(2) */
/*       for (int i=0;i<100;i++) */
/*         for (int j=0;j<100;j++) */
/*           sleep(1); // Simulating a long-running task */
                     
/*       std::cout << "Task " << taskId << " completed!\n"; */
/*     } */

/*     void cleanup() */ 
/*     { */
/*       saveToBinary("name.bin"); */
/*       std::cout << "\nPerforming cleanup for Task " << taskId << " before exit...\n"; */
/*     } */

/*     static void signalHandler(int) */ 
/*     { */
/*         if (globalSafeFailFunc) globalSafeFailFunc(); */  
/*           std::cout << "Time limit exceeded! Exiting...\n"; */
/*         std::exit(0); */
/*     } */

/*     void registerSignalHandler() */ 
/*     { */
/*       signal(SIGALRM, TaskHandler::signalHandler); */
/*     } */

/*     void saveToBinary(const std::string& filename) { */
/*         std::ofstream file(filename, std::ios::binary); */
/*         if (!file) { */
/*             std::cerr << "Error: Cannot open file for writing: " << filename << "\n"; */
/*             return; */
/*         } */
/*         cereal::BinaryOutputArchive archive(file); */
/*         archive(cereal::make_nvp("TaskHandler", *this));  // Serialize the current object */
/*         std::cout << "TaskHandler saved to " << filename << "\n"; */
/*     } */
/*     static std::shared_ptr<TaskHandler> loadFromBinary(const std::string& filename) { */
/*         std::ifstream file(filename, std::ios::binary); */
/*         if (!file) { */
/*             std::cerr << "Error: Cannot open file for reading: " << filename << "\n"; */
/*             return nullptr; */
/*         } */
/*         cereal::BinaryInputArchive archive(file); */
/*         auto task = std::make_shared<TaskHandler>(); */
/*         archive(cereal::make_nvp("TaskHandler", *task));  // Deserialize into a new object */
/*         std::cout << "TaskHandler loaded from " << filename << "\n"; */
/*         return task; */
/*     } */
/* }; */



/* int main() */ 
/* { */
/*   TaskHandler task(42); // Assigning a task ID */
/*   alarm(2); // Set a 5-second timeout */
/*   task.runTask(); // Start long-running task */

/*   /1* auto loaded = TaskHandler::loadFromBinary("name.bin"); *1/ */
/*   /1* TaskHandler task = std::move(*loaded); *1/ */
/*   /1* std::cout << "taskId: " << task.GetId() << std::endl; *1/ */
/*   /1* std::cout << "data: " << task.GetData() << std::endl; *1/ */


/*   return 0; */
/* } */


