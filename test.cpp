
#include <ctime>
#include <fstream>
#include <queue>
#include <chrono>
#include <utility>

#include "test_tools.h"



void run_test(const size_t count, const size_t vecdim, const size_t M) {
	std::cout << "|V|=" << count << "; Dim=" << vecdim << "; M=" << M << ". Starting index building. " << std::flush;
	hnswlib::L2Space l2space(vecdim);
	float *data, *testdata;
	hnswlib::labeltype *classes, *testclasses;
	hnswlib::HierarchicalNSW<float> index = hnswlib::HierarchicalNSW<float>(&l2space, count * 10, M);
	size_t pos_cls = 0;
	auto generated = generate_dataset(vecdim, count);
	data = generated.first;
	classes = generated.second;
	for (size_t i = 0; i < count; ++i) {
		pos_cls += 1 == classes[i];
		index.addPoint((void*)(data + i * vecdim), i, classes[i]);
	}

	std::cout << "Index is created. " << pos_cls << " * '1' / " << count << " total. Starting a test" << std::endl;

	size_t testsize = count / 20,
					positive_path = 0,
					positive_nsw_path = 0,
					positive_1nn = 0,
				  positive_5nn = 0;
	float nsw_time = 0, knn_time = 0, knn5_time = 0, path_time = 0;

	auto generated_test = generate_dataset(vecdim, testsize);
	testdata = generated_test.first;
	testclasses = generated_test.second;

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_1nn += index.getPointClass(index.searchKnn((void*)(testdata + i * vecdim), 1).top().second) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		knn_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << "     1-NN: " << 100. * positive_1nn / testsize << "% in " << knn_time / testsize << " ms avg" << std::endl;
	}

	{
		size_t k = 5;
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			std::map<hnswlib::labeltype, size_t> m;
			auto q = index.searchKnn((void*)(testdata + i * vecdim), k);
			while (!q.empty()) {
				auto p = q.top();
				q.pop();
				m[index.getPointClass(p.second)]++;
			}
			hnswlib::labeltype best;
			size_t best_count = 0;
			for (const auto& pair : m) {
				if (pair.second > best_count) {
					best = pair.first;
					best_count = pair.second;
				}
			}
			positive_5nn += best == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		knn5_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << "     5-NN: " << 100. * positive_5nn / testsize << "% in " << knn5_time / testsize << " ms avg" << std::endl;
	}

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_nsw_path += index.classifyByPathNSW((void*)(testdata + i * vecdim)) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		nsw_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << " NSW-PATH: " << 100. * positive_nsw_path / testsize << "% in " << nsw_time / testsize << " ms avg" << std::endl;
	}

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_path += index.classifyByPath((void*)(testdata + i * vecdim)) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		path_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << "HNSW-PATH: " << 100. * positive_path / testsize << "% in " << path_time / testsize << " ms avg" << std::endl;
	}

	std::cout << " NSWvs1NN: " << "speedup " << knn_time / nsw_time << " times avg" << std::endl;
	std::cout << "HNSWvs1NN: " << "speedup " << knn_time / path_time << " times avg" << std::endl;
	std::cout << " NSWvs5NN: " << "speedup " << knn5_time / nsw_time << " times avg" << std::endl;
	std::cout << "HNSWvs5NN: " << "speedup " << knn5_time / path_time << " times avg" << std::endl;

	delete data,
	delete testdata;
	delete classes;
	delete testclasses;
}

int main() {
	srand(time(NULL));
	int Ms[] = {4, 8, 16, 32, 64};

	for (size_t i = 0; i < 5; ++i) {

		std::cout << "Inspecting M=" << Ms[i] << std::endl;

		std::cout << "'=== 100 leaves margin ===" << std::endl;
		dataset_t leaves100mar = load_dataset("./data/100leaves/mar.txt", 64, 1600);
		run_test_general(leaves100mar, 64, 1600, .9, Ms[i]);

		std::cout << "'=== CovType ===" << std::endl;
		dataset_t covertype = load_libsvm("./data/covtype/covtype.binary.scale", 54, 581012);
		run_test_general(covertype, 54, 581012, .9, Ms[i]);

		std::cout << "'=== SUSY ===" << std::endl;
		dataset_t susy = load_libsvm("./data/susy/SUSY", 18, 5000000);
		run_test_general(susy, 18, 5000000, .95, Ms[i]);

		std::cout << "'=== HIGGS ===" << std::endl;
		dataset_t higgs = load_libsvm("./data/higgs/HIGGS", 28, 11000000);
		run_test_general(higgs, 28, 11000000, .95, Ms[i]);
	}

//==============================================================
/*
	// run_test_real("100leavesMar.txt", 64, 1600, .9, 50);
	//run_test_real("100leavesMar.txt", 64, 1600, .9, 192);
	//run_test_real("100leavesMar.txt", 64, 1600, .9, 16);
	//run_test_real("100leavesMar.txt", 64, 1600, .9, 8);
	//run_test_real("100leavesMar.txt", 64, 1600, .9, 4);
	run_test_real("road.txt", 1024, 10100, .9, 32);
	run_test_real("road.txt", 1024, 10100, .9, 64);
	run_test_real("road.txt", 1024, 10100, .9, 128);
	run_test_real("road.txt", 1024, 10100, .9, 256);
	//run_test_real("100leavesMar.txt", 64, 1600, .9, 193);
	run_test(3000, 256, 768);
	return 0;

	run_test_real("road.txt", 1024, 10100, .9, 8);
	run_test_real("mnist.txt", 64, 1798, .9, 8);
	run_test_real("mnist.txt", 64, 1798, .9, 64 * 3);
	run_test_real("100leavesMar.txt", 64, 1600, .9, 8);
	run_test_real("100leavesSha.txt", 64, 1600, .9, 8);
	run_test_real("100leavesTex.txt", 64, 1600, .9, 8);
	//run_test_real("road.txt", 1024, 10100, .99, 16);
	//run_test_real("road.txt", 1024, 10100, .9, 16);
	//run_test_real("road.txt", 1024, 10100, .99, 32);
	//run_test_real("road.txt", 1024, 10100, .99, 48);

	run_test(2000, 2, 2 * 3);
	run_test(5000, 2, 2 * 3);
	run_test(3000, 16, 16 * 3);
	run_test(3000, 64, 64 * 3);
	run_test(3000, 256, 256 * 3);

	run_test(2000, 2, 8);
	run_test(5000, 2, 8);
	run_test(3000, 16, 8);
	run_test(3000, 64, 8);
	run_test(3000, 256, 8);

	run_test(100000, 2, 8);
	run_test(100000, 4, 8);
	run_test(100000, 8, 8);
	run_test(100000, 10, 8);
	run_test(100000, 16, 8);

	run_test_real("road.txt", 1024, 10100, .99, 8);
	run_test_real("road.txt", 1024, 10100, .9, 8);
*/
}