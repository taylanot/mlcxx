# utils 

This folder contains general-purpose utilities that are used in **LCPP**.

Together, these utilities form a lightweight toolkit for handling common tasks such as data serialization, HTTP requests, CLI argument management, and progress reporting.

Each utility is self-contained and can be used independently, but together they simplify common development patterns and improve code readability and reusability.

---

## 1. `cereal`

Overall, we use the [cereal](https://uscilab.github.io/cereal/) library for serialization and de-serialization purposes. However, not every object is directly available for serialization. If you want to contribute to **LCPP** and your contribution requires the serialization and de-serialization for a C++ object then please add it here and test it. For now, only `std::optinal<T>` and `std::filesystem::path` additions are present. 

## 2. `curl`

The `curl` utility provides a modern C++ wrapper around [libcurl](https://curl.se/libcurl/) for performing HTTP and HTTPS requests. In this project it is used to download datasets from [OpenML](https://www.openml.org/). Here, I have added a utility function to be able to convert the downloaded data to a string. 

## 3. `clistore`

This is a lightweight configuration class. It enables one to register some variables that might come in handy for relatively large projects. It is by design a singleton, hence we need to use the `GetInstance()` method to initialize it in our program. Then one can register variables as fundamental types with default values and can also choose to add a range of values that the variable can take as `std::vector<>`. You can access the variable from this singleton object and access its variables. Moreover, you can also print the used configuration to make sure all your variables are initialized as you wished. You can also generate unique names from the variables of your choice to make sure your output files are nicely tagged. 

```cpp
...
    // Get singleton instance
    auto conf = CLIStore::GetInstance();
    // Register variables
    conf.Register<size_t>("id",11,{11,15,24});
    conf.Register<bool>("hpt",false,{false,true});
    conf.Register<std::string>("state","init");
    conf.Register<size_t>("seed",24);
... 
    // Call the variables for 
    auto id = conf.Get<size_t>("id");
...
    // Print the configuration of the variables
    conf.Print();
``` 

Now, you can run your program with `./program --id <someid> ...` and change the variables you defined externally.

## 4. `progress`

It is a lightweight thread sage progress bar for learning curve creation tracking, but you can use it for any loop that you want. You just need to name it give the ultimate number of the variables that you are iterating over and update it in your loop. A simple example is shown below.

```cpp
...

  utils::ProgressBar pb(Loop, 100);

  for (size_t i=0; i < 100; ++i)
  {
    ...
    pb.Update()
  }

...
``` 



