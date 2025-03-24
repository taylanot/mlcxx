/**
 * @file down_dataset.cpp
 * @author Ozgur Taylan Turan
 *
 * Download datasets with given ids
 *
 */
#define DTYPE float

#include <headers.h>

using OpenML = data::oml::Dataset<DTYPE>;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        ERR("Provide at least one ID...");
        return 0;
    }

    arma::wall_clock total_timer;
    total_timer.tic();

    for (int i = 1; i < argc; ++i)
    {
        size_t id;
        try
        {
            id = std::stoul(argv[i]); // Convert string to unsigned long
        }
        catch (const std::invalid_argument& e)
        {
            ERR("Invalid ID provided: " + std::string(argv[i]));
            continue; // Skip invalid input and proceed with the next
        }
        catch (const std::out_of_range& e)
        {
            ERR("ID out of range: " + std::string(argv[i]));
            continue;
        }

        LOG("Processing ID: " << id );

        data::oml::Dataset<DTYPE> dataset(id,"down_datasets");

        data::report(dataset);

    }

    PRINT_TIME(total_timer.toc());

    return 0;
}
