#include <cassert>
#include <string>
#include <map>
#include <cstring>
#include "hnswlib/hnswlib.h"


typedef std::pair<float*, hnswlib::labeltype*> dataset_t;
typedef std::pair<dataset_t, dataset_t> train_test_t;


dataset_t generateDataset(const size_t vecdim, const size_t count);

dataset_t loadDataset(const std::string& filename, const size_t vecdim, const size_t count);