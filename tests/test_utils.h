/**
 * @file test_utils.h
 * @author Ozgur Taylan Turan
 *
 */

#ifndef TEST_UTILS_H 
#define TEST_UTILS_H

TEST_SUITE("PROGRESS") 
{
  TEST_CASE("ProgressBar shows correct progress") 
  {
    std::ostringstream buffer;
    std::streambuf* oldCout = std::cout.rdbuf(buffer.rdbuf()); // redirect cout
    {
      utils::ProgressBar pb("Test", 5);
      for (int i = 0; i < 5; ++i) 
        pb.Update();
    }

    std::cout.rdbuf(oldCout); // restore cout

    std::string output = buffer.str();

    // Final percentage and label should be present
    CHECK(output.find("100 %") != std::string::npos);
    CHECK(output.find("Test") != std::string::npos);

    // Optional: check intermediate percentages
    CHECK(output.find(" 20 %") != std::string::npos);
    CHECK(output.find(" 40 %") != std::string::npos);
  }
} // SUITE -> PROGRESS
///////////////////////////////////////////////////////////////////////////////
TEST_SUITE("CEREAL")
{
  TEST_CASE("cereal optional<T> serialization") 
  {
    std::optional<int> original = 42;
    std::optional<int> restored;

    std::stringstream ss;

    // Serialize
    {
      cereal::JSONOutputArchive oarchive(ss);
      oarchive(cereal::make_nvp("opt", original));
    }

    // Deserialize
    {
      cereal::JSONInputArchive iarchive(ss);
      iarchive(cereal::make_nvp("opt", restored));
    }

    CHECK(original == restored); // They should match exactly
  }

  TEST_CASE("cereal optional<T> with nullopt") 
  {
    std::optional<int> original = std::nullopt;
    std::optional<int> restored = 123; // deliberately different

    std::stringstream ss;

    // Serialize
    {
      cereal::JSONOutputArchive oarchive(ss);
      oarchive(cereal::make_nvp("opt", original));
    }

    // Deserialize
    {
      cereal::JSONInputArchive iarchive(ss);
      iarchive(cereal::make_nvp("opt", restored));
    }

    CHECK(!restored.has_value()); // Should be nullopt after load
  }

  TEST_CASE("cereal filesystem::path serialization") 
  {
    std::filesystem::path original = "/tmp/myfile.txt";
    std::filesystem::path restored;

    std::stringstream ss;

    // Serialize
    {
      cereal::JSONOutputArchive oarchive(ss);
      oarchive(cereal::make_nvp("path", original));
    }

    // Deserialize
    {
      cereal::JSONInputArchive iarchive(ss);
      iarchive(cereal::make_nvp("path", restored));
    }

    CHECK(original == restored); // Exact match
  }
}  // SUITE -> CEREAL
///////////////////////////////////////////////////////////////////////////////
TEST_SUITE("CURL")
{
  TEST_CASE("WriteCallback appends data to string") 
  {
    std::string target;
    const char data[] = "Hello";

    size_t written = utils::WriteCallback(
        (void*)data,    // contents
        1,              // size of each element
        5,              // number of elements
        (void*)&target  // pointer to our string
    );

    CHECK(written == 5);           // Should return total bytes written
    CHECK(target == "Hello");      // String should now contain the data
  }

  TEST_CASE("WriteCallback appends multiple chunks") 
  {
    std::string target = "Start:";
    const char chunk1[] = "AAA";
    const char chunk2[] = "BBB";

    utils::WriteCallback((void*)chunk1, 1, 3, (void*)&target);
    utils::WriteCallback((void*)chunk2, 1, 3, (void*)&target);

    CHECK(target == "Start:AAABBB"); // Chunks should append sequentially
  }
} // SUITE -> CURL
///////////////////////////////////////////////////////////////////////////////
TEST_SUITE("CLISTORE")
{
  TEST_CASE("Register and Get basic types")
  {
    auto &cli = CLIStore::GetInstance();
    cli.Register<int>("intFlag", 10);
    cli.Register<std::string>("stringFlag", "hello");
    cli.Register<bool>("boolFlag", false);

    CHECK(cli.Get<int>("intFlag") == 10);
    CHECK(cli.Get<std::string>("stringFlag") == "hello");
    CHECK(cli.Get<bool>("boolFlag") == false);
  }

  TEST_CASE("Register with options and retrieve them")
  {
    auto &cli = CLIStore::GetInstance();
    cli.Register<int>("mode", 1, std::vector<int>{1, 2, 3});

    auto opts = cli.GetOptions<int>("mode");
    CHECK(opts.size() == 3);
    CHECK(opts[0] == 1);
    CHECK(opts[1] == 2);
    CHECK(opts[2] == 3);
  }

  TEST_CASE("Set changes value")
  {
    auto &cli = CLIStore::GetInstance();
    cli.Register<double>("alpha", 0.5);
    cli.Set<double>("alpha", 1.5);

    CHECK(cli.Get<double>("alpha") == doctest::Approx(1.5));
  }

  TEST_CASE("Parse updates flags from argv")
  {
    auto &cli = CLIStore::GetInstance();
    cli.Register<int>("epochs", 5);
    cli.Register<bool>("verbose", false);

    const char *argv[] = {
      "program",
      "--epochs", "20",
      "--verbose"
    };
    cli.Parse(4, const_cast<char**>(argv));

    CHECK(cli.Get<int>("epochs") == 20);
    CHECK(cli.Get<bool>("verbose") == true);
  }

  TEST_CASE("Parse throws on unknown flag")
  {
    auto &cli = CLIStore::GetInstance();
    cli.Register<int>("known", 1);

    const char *argv[] = {"program", "--unknown", "5"};
    CHECK_THROWS(cli.Parse(3, const_cast<char**>(argv)));
  }

  TEST_CASE("Sanitize replaces invalid chars")
  {
    auto &cli = CLIStore::GetInstance();
    CHECK(cli.Sanitize("0.01/val") == "0p01_val");
  }

  TEST_CASE("GenName returns correct concatenation")
  {
    auto &cli = CLIStore::GetInstance();
    cli.Register<int>("a", 1);
    cli.Register<std::string>("b", "x");

    auto name = cli.GenName();
    CHECK(name.find("a_1") != std::string::npos);
    CHECK(name.find("b_x") != std::string::npos);
  }

  TEST_CASE("GenName with subset of keys")
  {
    auto &cli = CLIStore::GetInstance();
    cli.Register<int>("x", 2);
    cli.Register<int>("y", 3);

    auto name = cli.GenName({"y"});
    CHECK(name == "y_3");
  }

  TEST_CASE("Print outputs table")
  {
    auto &cli = CLIStore::GetInstance();
    cli.Register<int>("num", 42);

    std::ostringstream oss;
    cli.Print(oss);

    auto out = oss.str();
    CHECK(out.find("num") != std::string::npos);
    CHECK(out.find("42") != std::string::npos);
  }
}
#endif
