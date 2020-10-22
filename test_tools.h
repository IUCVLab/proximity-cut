#include <cassert>
#include <string>
#include <map>
#include <cstring>
#include "hnswlib/hnswlib.h"


typedef std::pair<float*, hnswlib::labeltype*> dataset_t;
typedef std::pair<dataset_t, dataset_t> train_test_t;


dataset_t generate_dataset(const size_t vecdim, const size_t count);

dataset_t load_dataset(const std::string& filename, const size_t vecdim, const size_t count);

dataset_t load_libsvm(const std::string& filename, const size_t vecdim, const size_t count);

train_test_t split_dataset(dataset_t& dataset, const size_t vecdim, const size_t count, const size_t train_size);

void run_test_general(dataset_t& dataset, const size_t vecdim, const size_t count, const float train_part, const size_t M);