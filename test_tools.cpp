#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#include "test_tools.h"

dataset_t generate_dataset(const size_t vecdim, const size_t count) {
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

dataset_t load_dataset(const std::string& filename, const size_t vecdim, const size_t count) {
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

dataset_t load_libsvm(const std::string& filename, const size_t vecdim, const size_t count) {
	float* data = new float[count * vecdim];
	std::fill(data, data + (count * vecdim), 0);
	hnswlib::labeltype* labels = new hnswlib::labeltype[count];

	std::ifstream file(filename);
	std::string str;

	size_t i = 0;
	while (std::getline(file, str)) {
		std::istringstream ss(str);
		ss >> labels[i];
		size_t feature;
		char c;
		while (ss >> feature) {
			feature -= 1;
			if (feature >= vecdim) {
				std::cout << "Feature Overflow! " << i << std::endl;
			}
			ss >> c >> data[i * vecdim + feature];
		}
		++i;
	}
	return { data, labels };
}

train_test_t split_dataset(dataset_t& dataset, const size_t vecdim, const size_t count, const size_t train_size) {
	dataset_t train, test;
	train.first = new float[train_size * vecdim];
	test.first = new float[(count - train_size) * vecdim];
	train.second = new hnswlib::labeltype[train_size];
	test.second = new hnswlib::labeltype[(count - train_size)];
	size_t j_test = 0, j_train = 0;
	float* where_write;
	hnswlib::labeltype* where_class;
	for (size_t i = 0; i < count; ++i) {
		if (
				(j_test * train_size < j_train * (count - train_size) && (j_test < count - train_size))
				||
				j_train >= train_size
				) {
			where_write = test.first + (j_test * vecdim);
			where_class = test.second + j_test;
			j_test++;
		}
		else {
			where_write = train.first + (j_train * vecdim);
			where_class = train.second + j_train;
			j_train++;
		}
		std::memcpy(where_write, dataset.first + i * vecdim, vecdim * sizeof(float));
		*where_class = dataset.second[i];
	}
	return { train, test };
}

void run_test_general(dataset_t& dataset, const size_t vecdim, const size_t count, const float train_part, const size_t M) {
	std::cout << " D=" << vecdim << "; |V|=" << count << "; train part=" << train_part << "; M=" << M << " ... " << std::flush;
	size_t trainsize = (size_t)(train_part * count);
	auto traintest = split_dataset(dataset, vecdim, count, trainsize);
	std::cout << "loaded and splitted " << trainsize << " train by " << (count - trainsize) << " test." << std::endl;
	delete dataset.first;
	delete dataset.second;
	auto train = traintest.first;
	auto test = traintest.second;

	float* data, * testdata;
	hnswlib::labeltype* classes, * testclasses;
	size_t testsize = count - trainsize,
		positive_path = 0,
		positive_nsw_path = 0,
		positive_1nn = 0,
		positive_5nn = 0,
		positive_nsw_5nn = 0,
		positive_nsw_9nn = 0,
		positive_nsw_11nn = 0,
		positive_nsw_5nn_w = 0,
		positive_nsw_9nn_w = 0,
		positive_nsw_11nn_w = 0;
	float nsw_time = 0, knn_time = 0, knn5_time = 0, path_time = 0,
			nsw_5nn_time = 0,nsw_9nn_time = 0, nsw_11nn_time = 0,
			nsw_5nn_w_time = 0,nsw_9nn_w_time = 0, nsw_11nn_w_time = 0;

	data = train.first;
	classes = train.second;
	testdata = test.first;
	testclasses = test.second;

	hnswlib::L2Space l2space(vecdim);
	hnswlib::HierarchicalNSW<float> index = hnswlib::HierarchicalNSW<float>(&l2space, trainsize * 2, M);

	size_t every_np = 1;

	for (size_t i = 0; i < trainsize; ++i) {
		if (i % 5000 == 0) {
			std::cout << "\r" << i << "/" << trainsize << " : index construction                   " << std::flush;
		}
		index.addPoint((void*)(data + i * vecdim), i, classes[i]);
	}
	std::cout << "\rIndex constructed!                              " << std::endl;
	std::cout << "1NN/A;1NN/T;5NN/A;5NN/T;NSW-P/A;NSW-P/T;HNSWP/A;HNSWP/T;NSW5/A;NSW5/T;NSW9/A;NSW9/T;NSW11/A;NSW11/T" << std::endl;

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_1nn += index.getPointClass(index.searchKnn((void*)(testdata + i * vecdim), 1).top().second) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		knn_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << 100. * positive_1nn / testsize << "%;" << knn_time / testsize << ";" << std::flush;
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
		std::cout << 100. * positive_5nn / testsize << "%;" << knn5_time / testsize << ";" << std::flush;

	}

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_nsw_path += index.classifyByPathNSW((void*)(testdata + i * vecdim)) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		nsw_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << 100. * positive_nsw_path / testsize << "%;" << nsw_time / testsize << ";" << std::flush;
	}

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_path += index.classifyByPath((void*)(testdata + i * vecdim)) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		path_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << 100. * positive_path / testsize << "%;" << path_time / testsize << ";" << std::flush;
	}

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_nsw_5nn += index.classifyByPathNSW_kNN((void*)(testdata + i * vecdim), 5) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		nsw_5nn_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << 100. * positive_nsw_5nn / testsize << "%;" << nsw_5nn_time / testsize << ";" << std::flush;
	}

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_nsw_9nn += index.classifyByPathNSW_kNN((void*)(testdata + i * vecdim), 9) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		nsw_9nn_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << 100. * positive_nsw_9nn / testsize << "%;" << nsw_9nn_time / testsize << ";" << std::flush;
	}

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_nsw_11nn += index.classifyByPathNSW_kNN((void*)(testdata + i * vecdim), 11) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		nsw_11nn_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << 100. * positive_nsw_11nn / testsize << "%;" << nsw_11nn_time / testsize << ";" << std::flush;
	}

		{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_nsw_5nn_w += index.classifyByPathNSW_kNN_weighted((void*)(testdata + i * vecdim), 5) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		nsw_5nn_w_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << 100. * positive_nsw_5nn_w / testsize << "%;" << nsw_5nn_w_time / testsize << ";" << std::flush;
	}

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_nsw_9nn_w += index.classifyByPathNSW_kNN_weighted((void*)(testdata + i * vecdim), 9) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		nsw_9nn_w_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << 100. * positive_nsw_9nn_w / testsize << "%;" << nsw_9nn_w_time / testsize << ";" << std::flush;
	}

	{
		auto start = std::chrono::high_resolution_clock::now();
		for (size_t i = 0; i < testsize; ++i) {
			positive_nsw_11nn_w += index.classifyByPathNSW_kNN_weighted((void*)(testdata + i * vecdim), 11) == testclasses[i];
		}
		auto stop = std::chrono::high_resolution_clock::now();
		nsw_11nn_w_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
		std::cout << 100. * positive_nsw_11nn_w / testsize << "%;" << nsw_11nn_w_time / testsize << ";" << std::flush;
	}

	std::cout << "\nNSWvs1NN;HNSWvs1NN;NSW5vs1NN;NSW9vs1NN;NSW11vs1NN;NSW5vs5NN;NSW9vs5NN;NSW11vs5NN\n"
				<< knn_time / nsw_time
				<< ";"
				<< knn_time / path_time
				<< ";"
				<< knn_time / nsw_5nn_time
				<< ";"
				<< knn_time / nsw_9nn_time
				<< ";"
				<< knn_time / nsw_11nn_time
				<< ";"
				<< knn5_time / nsw_5nn_time
				<< ";"
				<< knn5_time / nsw_9nn_time
				<< ";"
				<< knn5_time / nsw_11nn_time
				<< std::endl;


	delete data,
	delete testdata;
	delete classes;
	delete testclasses;
}