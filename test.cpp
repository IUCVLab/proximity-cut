
#include <cassert>
#include <ctime>
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"

void run_test(const ssize_t vecdim) {
	std::cout << "Dim: " << vecdim << std::endl;
	size_t count = 100000;
	hnswlib::L2Space l2space(vecdim);
	float data[count * vecdim];
	hnswlib::labeltype classes[count];
	float border = .79 * sqrt(vecdim / 2);
	border *= border;

	hnswlib::HierarchicalNSW<float> index = hnswlib::HierarchicalNSW<float>(&l2space, count * 10, vecdim * 3);

	std::cout << "Starting index building" << std::endl;

	size_t pos_cls = 0;
	for (size_t i = 0; i < count; ++i) {
		
		float r2 = 0;
		for (size_t j = 0; j < vecdim; j++) {
			data[i * vecdim + j] = (rand() / float(RAND_MAX));
			r2 += data[i * vecdim + j] * data[i * vecdim + j];
		}

		hnswlib::labeltype cls = r2 > border;
		pos_cls += cls == 1;
		index.addPoint((void*)(data + i * vecdim), i, cls);
	}

	std::cout << "Index is created. " << pos_cls << " * '1' / " << count << " total" << std::endl;
	std::cout << "Test started" << std::endl;

	size_t testsize = 5000, 
					positive_path = 0, 
					positive_nsw_path = 0,
					positive_1nn = 0;
	float testdata[vecdim * testsize];
	hnswlib::labeltype testclasses[testsize];

	for (size_t i = 0; i < testsize; ++i) {
		float r2 = 0.0;
		for (size_t j = 0; j < vecdim; j++) {
			testdata[j + vecdim * i] = (rand() / float(RAND_MAX));
			r2 += testdata[j + vecdim * i] * testdata[j + vecdim * i];
		}
		testclasses[i] = r2 > border;
	}

	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < testsize; ++i) {
		positive_path += index.classifyByPath((void*)(testdata + i * vecdim)) == testclasses[i];
	}
	auto stop = std::chrono::high_resolution_clock::now();
	float path_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

	std::cout << "HNSW-PATH: " << 100. * positive_path / testsize << "% in " << path_time / testsize << "ms avg" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < testsize; ++i) {
		positive_1nn += index.getPointClass(index.searchKnn((void*)(testdata + i * vecdim), 1).top().second) == testclasses[i];
	}
	stop = std::chrono::high_resolution_clock::now();
	float knn_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

	std::cout << "     1-NN: " << 100. * positive_1nn / testsize << "% in " << knn_time/ testsize << " ms avg" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < testsize; ++i) {
		positive_nsw_path += index.classifyByPathNSW((void*)(testdata + i * vecdim)) == testclasses[i];
	}
	stop = std::chrono::high_resolution_clock::now();
	float nsw_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();


	std::cout << " NSW-PATH: " << 100. * positive_nsw_path / testsize << "% in " << nsw_time / testsize << " ms avg" << std::endl;

}

int main() {
	srand(time(NULL));
	run_test(2);
	run_test(4);
	run_test(8);
	run_test(10);
	run_test(16);
}