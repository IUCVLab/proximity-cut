#include <iostream>

#include "test_tools.h"

dataset_t generateDataset(const size_t vecdim, const size_t count) {
	float* data = new float[count * vecdim];
	hnswlib::labeltype* labels = new hnswlib::labeltype[count];
	float border = .79 * sqrt(vecdim / 2);
	border *= border;

	for (size_t i = 0; i < count; ++i) {
		float r2 = 0;
		for (size_t j = 0; j < vecdim; j++) {
			data[i * vecdim + j] = (rand() / float(RAND_MAX));
			r2 += data[i * vecdim + j] * data[i * vecdim + j];
		}
		hnswlib::labeltype cls = r2 > border;
		labels[i] = cls;
	}
	return { data, labels };
}


dataset_t loadDataset(const std::string& filename, const size_t vecdim, const size_t count) {
	float* data = new float[count * vecdim];
	hnswlib::labeltype* labels = new hnswlib::labeltype[count];

	std::ifstream file(filename);
	for (size_t i = 0; i < count; ++i) {
		file >> labels[i];
		for (size_t j = 0; j < vecdim; ++j) {
			file >> data[i * vecdim + j];
		}
	}
	return { data, labels };
}