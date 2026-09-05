using namespace std;
#define _USE_MATH_DEFINES
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <stdint.h>
#include <sstream>
#include <chrono>
#include <ctime>
#include <string>
#include <fstream>
#include <cfloat>
#include <random>
#include <thread>
#include <vector>
#ifdef __linux__
#elif _WIN32
#include <conio.h>
#include <windows.h>
#pragma comment(lib, "user32.lib")
#endif
void init();
void predict();
void forward();
void apprendi();
void back_propagate();
float calculate_max_output_error();
void evaluate_model(float* error_list, float& max_error, float& average_error, int& max_error_file_index_line, uint16_t* hidden_activation_count);
void read_weights_from_file();
bool write_weights_on_file();
void read_samples_from_file_diagram_battery();
void print_hidden_activation_status(const uint16_t* hidden_activation_count);
//float overallMean(const float* arr1, const float* arr2, int size);
void normalizeArray(float* array, float* normalized_array, int size);
float mean_square_error(const float* arr1, const float* arr2, int size);
float calculateVariance(const float* data, int size);
float mean_value(const float* data, int size);
float calculateErrorPercentage(float mse, float reference_mean);
float calculate_cosine_shape_similarity_percentage(const float* arr1, const float* arr2, int size);
int count_training_samples(int linesPerSample);
int get_random_sample_first_line();
bool get_sample_for_test(int sampleIndex);
bool is_csv_header(const std::string& line);
void setTime();
float _err_epoca;
float _err_rete = 0.00f;
float _err_amm = 0.009f;
float _epsilon = 0.05f;
uint8_t const lines_per_training_sample = 8;
uint16_t const training_samples = 323;
const int training_report_epoch_interval = 10000;
const uint8_t numberOf_X = 2;
const uint8_t numberOf_H = 25;
const uint8_t numberOf_Y = 6;
const float leaky_relu_alpha = 0.01f;
float output_bias[numberOf_Y] = { 0.00 };
float hidden_bias[numberOf_H] = { 0.00 };
float W1[numberOf_X][numberOf_H] = { 0.00 };
float W2[numberOf_H][numberOf_Y] = { 0.00 };
float x[numberOf_X] = { 0.00 };
float h[numberOf_H] = { 0.00 };
float y[numberOf_Y] = { 0.00 };
float d[numberOf_Y] = { 0.00 };
float amps_training[training_samples]{};
float watts_hour_training[training_samples]{};
float battery_out_training[training_samples][numberOf_Y]{};
char _global_time[9] = { 0 };
float observed_data[6] = { 0.00f };
const string _relative_files_path = "72V-Battery-S11";
const string _files_name = "72V_Battery.csv";
//const string _files_name = "72V_Battery_Subset.csv";
float err_min_rete = FLT_MAX;
float _max_single_traning_output_error_average = 0.00f;
float _err_epoca_min_value = FLT_MAX;
float leaky_relu(float value) {
	return (value > 0.0f) ? value : leaky_relu_alpha * value;
}
int main() {

#ifdef __linux__

#elif _WIN32
	//// Sposta e massimizza la finestra
	//HWND consoleWindow = GetConsoleWindow();
	//SetWindowPos(consoleWindow, nullptr, -1920, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
	//ShowWindow(consoleWindow, SW_MAXIMIZE);
#endif
	int number_of_training_samples = count_training_samples(lines_per_training_sample);
	if (number_of_training_samples != training_samples) {
		cout << "\nError on samples number!!!!!! current value is set to " << training_samples << " but should be " << number_of_training_samples << "\n";
		return 0;
	}
	init();
#ifdef __linux__
	cout << "\a";
#elif _WIN32
	Beep(3000, 200);
#endif
	char response;
	cout << "\n'n' for new learning, 'c' for continue learning, 'e' for execute.\n";
	//cout << "\n Do you want load the weights file\n";
#ifdef __linux__
	response = std::cin.get();
	std::cin.ignore();
#elif _WIN32
	response = _getch();
#endif
	if (response == 'e' || response == 'E') {
		read_weights_from_file();
		predict();
	}
	else if (response == 'c') {
		cout << "\nmodel loaded.\n";
		read_weights_from_file();
		cout << "\nlast inserted epsilon is : " << _epsilon <<
			" do you wanna change it? y or n\n";
#ifdef __linux__
		response = std::cin.get();
		std::cin.ignore();
#elif _WIN32
		response = _getch();
#endif
		if (response == 'y') {
			cout << "\ninsert new epsilon value : ";
			cin >> _epsilon;
			cout << "\n epsilon changed\n";
		}
		else {
			cout << "\n epsilon not changed\n";
		}
		apprendi();
	}
	else if (response == 'n') {
		cout << "ATTENTION !!!!!!!!!!!!! are you sure to restart learning? press y to continue !!!!!!!!!!!!!!!!!!!\n\n";
#ifdef __linux__
		response = std::cin.get();
		std::cin.ignore();
#elif _WIN32
		response = _getch();
#endif
		if (response == 'y') {
			cout << "\n model overwritten\n";
			apprendi();
		}
		else {
			cout << "\nprocess blocked.\n";
		}
	}
}
void init() {
	random_device rd;
	mt19937 gen = mt19937(rd());
	double init_scale_input = sqrt(2.0 / numberOf_X);
	double init_scale_hidden = sqrt(2.0 / numberOf_H);
	normal_distribution<double> dist(0.0, 1.0);
	//-----------------------------------	bias initialization
	for (int i = 0; i < numberOf_Y; i++) {
		output_bias[i] = 0.1f;
	}
	for (int i = 0; i < numberOf_H; i++) {
		hidden_bias[i] = 0.1f;
	}
	//-----------------------------------	console input values + Hidden bias values
	//cout << "input elements initialization:\n\n";
	for (int i = 0; i < (numberOf_X); i++) {
		//x[i] = 0.00f;
		cout << "x[" << i << "]" << "=" << x[i] << "\n";
	}
	for (int i = 0; i < numberOf_H; i++) {
		cout << "hidden_bias[" << i << "]" << "=" << hidden_bias[i] << "-BIAS" << "\n";
	}
	//-----------------------------------	console hidden values + output bias values
	for (int i = 0; i < (numberOf_H); i++) {
		//h[i] = 0.00f;
		cout << "h[" << i << "]" << "=" << h[i] << "\n";
	}
	for (int i = 0; i < numberOf_Y; i++) {
		cout << "output_bias[" << i << "]" << "=" << output_bias[i] << "-BIAS" << "\n";
	}
	//cout << "output elements initialization:\n\n";
	//-----------------------------------	console output values
	for (int i = 0; i < numberOf_Y; i++) {
		//y[i] = 0.00f;
		cout << "y[" << i << "]=" << y[i] << "\n";
	}
	//-----------------------------------	console W1 values
	cout << "W1 elements initialization:\n\n";
	for (int i = 0; i < numberOf_X; i++) {
		for (int k = 0; k < numberOf_H; k++) {
			W1[i][k] = dist(gen) * init_scale_input;
			cout << "W1[" << i << "]" << "[" << k << "]" << "=" << W1[i][k] << "\n";
		}
	}
	//-----------------------------------	console W2 values
	for (int k = 0; k < numberOf_H; k++) {
		for (int j = 0; j < numberOf_Y; j++) {
			W2[k][j] = dist(gen) * init_scale_hidden;
			cout << "W2[" << k << "]" << "[" << j << "]" << "=" << W2[k][j] << "\n";
		}
	}
}
void predict() {
	float normalized_observed_output[numberOf_Y] = { 0.00 };
	//float normalized_predicted_output[numberOf_Y] = { 0.00 };
	while (true) {
		std::cout << "\nPress 'e' for next random sample or 'q' to exit:\n";
		char execution_command = '\0';
#ifdef __linux__
		if (!(std::cin >> execution_command)) {
			return;
		}
#elif _WIN32
		execution_command = _getch();
#endif
		if (execution_command == 'q' || execution_command == 'Q') {
#ifdef __linux__
#elif _WIN32
			for (uint8_t beep_index = 0; beep_index < 5; beep_index++) {
				Beep(3000, 200);
			}
#endif
			return;
		}
		if (execution_command != 'e' && execution_command != 'E') {
			continue;
		}
		int sample_first_line = get_random_sample_first_line();
		std::cout << "\nRandom sample first line: " << sample_first_line << "\n";
		if (!get_sample_for_test(sample_first_line)) {
			continue;
		}
		normalizeArray(observed_data, normalized_observed_output, numberOf_Y);
		x[0] = log(x[0] + 1.00f) / 10.00f;
		x[1] = log(x[1] + 1.00f) / 10.00f;
		forward();
		for (int i = 0; i < 6; i++) {
			y[i] = y[i] * 10.00f;
		}
		//normalizeArray(y, normalized_predicted_output, numberOf_Y);
		float mse = mean_square_error(observed_data, y, numberOf_Y);
		//float overall_mean = overallMean(normalized_observed_output, normalized_predicted_output, numberOf_Y);
		float observed_mean = mean_value(observed_data, numberOf_Y);
		float percentage = calculateErrorPercentage(mse, observed_mean);
		float varianza = calculateVariance(normalized_observed_output, numberOf_Y);
		float cosine_percentage = calculate_cosine_shape_similarity_percentage(observed_data, y, numberOf_Y);
		std::cout << "percentage = :" << percentage << "%\n";
		std::cout << "varianza = :" << varianza << "\n";
		std::cout << "cosine similarity percentage = :" << cosine_percentage << "%\n";
		// Stampa dei risultati
		std::cout << "\n Input X1 - Ampere (A) [x[0]] = " << (exp(x[0] * 10.00f) - 1.00f)
			<< "\n Input X2 - Wattora (Wh) [x[1]] = " << (exp(x[1] * 10.00f) - 1.00f) << "\n";
		std::ios::fmtflags original_output_flags = std::cout.flags();
		std::streamsize original_output_precision = std::cout.precision();
		std::cout << "\n+---------+---------------+---------------+\n"
			<< "| " << std::left << std::setw(7) << "Battery"
			<< " | " << std::setw(13) << "Predict (V)"
			<< " | " << std::setw(13) << "Observed (V)" << " |\n"
			<< "+---------+---------------+---------------+\n";
		for (int i = 0; i < numberOf_Y; i++) {
			std::cout << "| B" << std::left << std::setw(6) << i
				<< " | " << std::right << std::fixed << std::setprecision(5) << std::setw(13) << y[i]
				<< " | " << std::setw(13) << observed_data[i] << " |\n";
		}
		std::cout << "+---------+---------------+---------------+\n";
		std::cout.flags(original_output_flags);
		std::cout.precision(original_output_precision);
	}
}
void apprendi() {
	int max_error_file_index_line = 0;
	int _epoca_index = 0;
	int cout_counter = 0;
	auto start = std::chrono::system_clock::now();
	read_samples_from_file_diagram_battery();
	float average_err_rete = 0.00f;
	float varianza_err_rete = 0.00f;
	float deviazione_std_err_rete = 0.00f;
	float listOfErr_rete[training_samples] = { 0.00f };
	uint16_t hidden_activation_count[numberOf_H] = { 0 };
	do {
		_err_epoca = 0.00f;
		average_err_rete = 0.00f;
		varianza_err_rete = 0.00f;
		_max_single_traning_output_error_average = 0.00f;
		bool should_track_hidden_activation = (cout_counter == training_report_epoch_interval - 1);
		if (should_track_hidden_activation) {
			for (int k = 0; k < numberOf_H; k++) {
				hidden_activation_count[k] = 0;
			}
		}
		// Training pass: each sample updates weights and biases.
		for (unsigned long p = 0; p < training_samples; p++) {
			x[0] = log(amps_training[p] + 1.0f) / 10.0f;
			x[1] = log(watts_hour_training[p] + 1.0f) / 10.0f;
			for (int i = 0; i < numberOf_Y; i++) {
				d[i] = battery_out_training[p][i] / 10.00f;
			}
			forward();
			back_propagate();
		}
		// Evaluation pass: all statistics refer to the same final weights.
		evaluate_model(listOfErr_rete, _err_epoca, average_err_rete, max_error_file_index_line, should_track_hidden_activation ? hidden_activation_count : nullptr);
		_epoca_index++;
		for (unsigned long p = 0; p < training_samples; p++) {
			varianza_err_rete += pow(listOfErr_rete[p] - average_err_rete, 2);
		}
		varianza_err_rete /= training_samples;
		deviazione_std_err_rete = sqrt(varianza_err_rete);
		cout_counter++;
		if (_err_epoca_min_value > _err_epoca) {
			float previous_min_error = _err_epoca_min_value;
			_err_epoca_min_value = _err_epoca;
			if (!write_weights_on_file()) {
				_err_epoca_min_value = previous_min_error;
			}
			else {
				setTime();
			}
		}
		if (cout_counter == training_report_epoch_interval) {
			std::cout << "\nepoca:" << _epoca_index <<
				"\nerr_epoca=" << _err_epoca <<
				"\nmin. err_epoca=" << _err_epoca_min_value <<
				"\nlast time write on file = " << _global_time <<
				"\nvarianza di errore di rete = " << varianza_err_rete <<
				"\nmedia di errore di rete = " << average_err_rete <<
				"\ndeviazione standard errore di rete = " << deviazione_std_err_rete <<
				"\nmax err_epoca is on sample line = " << max_error_file_index_line <<
				"\npercentage dev.standard err_rete / media err_rete = " << (average_err_rete > 0.0f ? (deviazione_std_err_rete / average_err_rete) * 100.0f : 0.0f) << "%" <<
				"\nepsilon=" << _epsilon << "\n";
			print_hidden_activation_status(hidden_activation_count);
			cout_counter = 0;
		}
	} while (_err_epoca > _err_amm);
	setTime();
	std::cout << "learning stopped at : " << _global_time;
#ifdef __linux__
	// linux code goes here
#elif _WIN32
	Beep(3000, 200);
	Beep(3000, 200);
	Beep(3000, 200);
	Beep(3000, 200);
	Beep(3000, 200);
#endif
#ifdef __linux__
	getchar();
#elif _WIN32
	int response = _getch();
#endif
}
void forward() {
	for (int k = 0; k < (numberOf_H); k++) {
		float Zk = 0.00f;
		for (int i = 0; i < numberOf_X; i++) {
			Zk += (W1[i][k] * x[i]);
		}
		//insert X bias
		Zk += hidden_bias[k];
		h[k] = leaky_relu(Zk);
	}
	for (int j = 0; j < numberOf_Y; j++) {
		float Zj = 0.00f;
		for (int k = 0; k < numberOf_H; k++) {
			Zj += (W2[k][j] * h[k]);
		}
		//insert H bias
		Zj += output_bias[j];
		y[j] = Zj;
	}
}
void back_propagate() {
	float err_H[numberOf_H] = { 0.00f };
	float delta = 0.00f;
	// Calcolo del delta per il layer di output (attivazione lineare -> derivata = 1)
	for (int j = 0; j < numberOf_Y; j++) {
		delta = (d[j] - y[j]);  // Derivata del layer output lineare è 1
		// Aggiornamento dei pesi del layer di output e accumulo dell'errore per il layer nascosto
		for (int k = 0; k < numberOf_H; k++) {
			err_H[k] += delta * W2[k][j];
			W2[k][j] += (_epsilon * delta * h[k]);
		}
		// Aggiornamento del bias per il layer di output
		output_bias[j] += _epsilon * delta;
	}
	// Calcolo del delta per il layer nascosto usando la derivata della Leaky ReLU
	for (int k = 0; k < numberOf_H; k++) {
		float leaky_relu_derivative = (h[k] > 0.0f) ? 1.0f : leaky_relu_alpha;
		delta = err_H[k] * leaky_relu_derivative;
		// Aggiornamento dei pesi del layer nascosto
		for (int i = 0; i < numberOf_X; i++) {
			W1[i][k] += (_epsilon * delta * x[i]);
		}
		// Aggiornamento del bias per il layer nascosto
		hidden_bias[k] += _epsilon * delta;
	}
}
float calculate_max_output_error() {
	float max_output_error = 0.00f;
	for (int j = 0; j < numberOf_Y; j++) {
		float output_error = fabs(d[j] - y[j]);
		if (output_error > max_output_error) {
			max_output_error = output_error;
		}
	}
	return max_output_error;
}
void evaluate_model(float* error_list, float& max_error, float& average_error, int& max_error_file_index_line, uint16_t* hidden_activation_count) {
	max_error = 0.00f;
	average_error = 0.00f;
	max_error_file_index_line = 1;
	for (unsigned long p = 0; p < training_samples; p++) {
		x[0] = log(amps_training[p] + 1.0f) / 10.0f;
		x[1] = log(watts_hour_training[p] + 1.0f) / 10.0f;
		for (int i = 0; i < numberOf_Y; i++) {
			d[i] = battery_out_training[p][i] / 10.00f;
		}
		forward();
		if (hidden_activation_count != nullptr) {
			for (int k = 0; k < numberOf_H; k++) {
				if (h[k] > 0.0f) {
					hidden_activation_count[k]++;
				}
			}
		}
		_err_rete = calculate_max_output_error();
		if (_err_rete > max_error) {
			max_error = _err_rete;
			max_error_file_index_line = ((p)*lines_per_training_sample) + 1;
		}
		error_list[p] = _err_rete;
		average_error += _err_rete;
	}
	average_error /= training_samples;
}
void print_hidden_activation_status(const uint16_t* hidden_activation_count) {
	std::cout << "\nhidden activation count per epoch:\n";
	for (int k = 0; k < numberOf_H; k++) {
		std::cout << "h[" << k << "]=" << hidden_activation_count[k] << "/" << training_samples;
		if (hidden_activation_count[k] == 0) {
			std::cout << " LEAKY_ONLY";
		}
		std::cout << "\n";
	}
}
void read_samples_from_file_diagram_battery() {
	//std::cout << "Directory corrente: " << std::filesystem::current_path() << std::endl;
	std::string filename = _relative_files_path + "/" + _files_name;
	// Apertura del file
	std::ifstream file(filename);
	// Verifica se il file è stato aperto correttamente
	if (!file.is_open()) {
		std::cerr << "Errore nell'apertura del file " << filename << std::endl;
		return;
	}
	std::string line;
	if (!std::getline(file, line) || !is_csv_header(line)) {
		std::cerr << "Intestazione CSV mancante o non valida nel file " << filename << std::endl;
		return;
	}
	int training_block_index = 0;
	int training_row_index = 0;
	int training_row_pre_index = 0;
	std::string item;
	stringstream ss1;
	while (!file.eof()) {
		training_row_pre_index = training_row_index++;
		switch (training_row_pre_index) {
		case 0:
		case 1:
		case 2:
		case 3:
		case 4:
		case 5:
			std::getline(file, line);
			ss1.str(line);
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			battery_out_training[training_block_index][training_row_pre_index] = std::stod(item);
			cout << "battery[" << training_block_index << "]" << "[" << training_row_pre_index << "] = " << battery_out_training[training_block_index][training_row_pre_index] << "\n";
			break;
		case 6:
			std::getline(file, line);
			ss1.str(line);
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			watts_hour_training[training_block_index] = std::stod(item);
			cout << "Watts/hour[" << training_block_index << "] = " << watts_hour_training[training_block_index] << "\n";
			break;
		case 7:

			std::getline(file, line);
			ss1.str(line);
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			std::getline(ss1, item, ';');
			amps_training[training_block_index] = std::stod(item);
			cout << "Ampere[" << training_block_index << "] = " << amps_training[training_block_index] << "\n";
			break;
		default:
			training_block_index++;
			training_row_index = 0;
			break;
		}
	}
#ifdef __linux__
#elif _WIN32
#else
#endif
	if ((training_block_index)+1 != training_samples) {
		cout << "\n\nALLERT!!!!!!! training sample different to index = \t" << training_block_index << "\n";
#ifdef __linux__
#elif _WIN32
		system("pause");
#else
#endif
	}
	else {
		//cout << "\n\nTraining sample index is " << training_block_index << " and seems to have been loaded correctly.";
#ifdef __linux__

#elif _WIN32
		//system("pause");
#else

#endif
	}
	file.close();
}
void read_weights_from_file() {
	std::ifstream in(_relative_files_path + "/" + "model.hex", std::ios_base::binary);
	if (in.good()) {
		for (int k = 0; k < numberOf_H; k++) {
			for (int i = 0; i < numberOf_X; i++) {
				in.read((char*)&W1[i][k], sizeof(float));
			}
			in.read((char*)&hidden_bias[k], sizeof(float));
			cout << hidden_bias[k] << "\n";
		}
		for (int j = 0; j < numberOf_Y; j++) {
			for (int k = 0; k < numberOf_H; k++) {
				in.read((char*)&W2[k][j], sizeof(float));
			}
			in.read((char*)&output_bias[j], sizeof(float));
		}
		/*	for (int k = 0; k < numberOf_H; k++) {
				in.read((char*)&hidden_bias[k], sizeof(float));
			}
			for (int j = 0; j < numberOf_Y; j++) {
				in.read((char*)&output_bias[j], sizeof(float));
			}*/
		in.read((char*)&_err_epoca_min_value, sizeof(float));
		in.read((char*)&_epsilon, sizeof(float));
		//in.read((char*)&x[numberOf_X - 1], sizeof(float));
		//in.read((char*)&h[numberOf_H - 1], sizeof(float));
	}
}
bool write_weights_on_file() {
	const std::string model_path = _relative_files_path + "/" + "model.hex";
	const uint8_t max_write_attempts = 10;
	for (uint8_t attempt = 0; attempt < max_write_attempts; attempt++) {
		std::ofstream fw(model_path, std::ios_base::binary);
		if (!fw.good()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			continue;
		}
		for (int k = 0; k < numberOf_H; k++) {
			for (int i = 0; i < numberOf_X; i++) {
				fw.write((char*)&W1[i][k], sizeof(float));
			}
			fw.write((char*)&hidden_bias[k], sizeof(float));
		}
		for (int j = 0; j < numberOf_Y; j++) {
			for (int k = 0; k < numberOf_H; k++) {
				fw.write((char*)&W2[k][j], sizeof(float));
			}
			fw.write((char*)&output_bias[j], sizeof(float));
		}
		/*for (int k = 0; k < numberOf_H; k++) {
			fw.write((char*)&hidden_bias[k], sizeof(float));
		}
		for (int j = 0; j < numberOf_Y; j++) {
			fw.write((char*)&output_bias[j], sizeof(float));
		}*/
		fw.write((char*)&_err_epoca_min_value, sizeof(float));
		fw.write((char*)&_epsilon, sizeof(float));
		//fw.write((char*)&x[numberOf_X - 1], sizeof(float));
		//fw.write((char*)&h[numberOf_H - 1], sizeof(float));
		fw.close();
		if (fw.good()) {
			return true;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	std::cerr << "\nUnable to write model file: " << model_path << "\n";
	return false;
}
void normalizeArray(float* arr, float* normArr, int size) {
	float minVal = arr[0];
	float maxVal = arr[0];
	// Trova il minimo e il massimo
	for (int i = 1; i < size; i++) {
		if (arr[i] < minVal) minVal = arr[i];
		if (arr[i] > maxVal) maxVal = arr[i];
	}
	// Normalizza i valori
	for (int i = 0; i < size; i++) {
		if (maxVal != minVal) {
			normArr[i] = (arr[i] - minVal) / (maxVal - minVal);
		}
		else {
			normArr[i] = 0; // Evita divisione per zero nel caso di valori uguali
		}
	}
}
float mean_square_error(const float* arr1, const float* arr2, int size) {
	float sum = 0.00f;
	for (int i = 0; i < size; ++i) {
		float diff = arr1[i] - arr2[i];
		sum += (diff * diff);  // quadrato della differenza
	}
	// MSE = (1 / N) * Σ (diff^2)
	return sum / size;
}
/*
float overallMean(const float* arr1, const float* arr2, int size) {
	float sum = 0.0f;
	// Sommiamo tutti gli elementi di entrambi gli array
	for (int i = 0; i < size; ++i) {
		sum += arr1[i] + arr2[i];
	}
	// La media complessiva è la somma divisa per il numero totale di elementi (2*size)
	return sum / (2 * size);
}
*/
float mean_value(const float* data, int size) {
	float sum = 0.0f;
	for (int i = 0; i < size; ++i) {
		sum += data[i];
	}
	return sum / size;
}
float calculateErrorPercentage(float mse, float reference_mean) {
	// Calcola il Root Mean Squared Error (RMSE)
	float rms = sqrt(mse);
	// Calcola la percentuale: (RMSE / media reale osservata) * 100
	float errorPercentage = (rms / reference_mean) * 100.00f;
	return errorPercentage;
}
float calculate_cosine_shape_similarity_percentage(const float* arr1, const float* arr2, int size) {
	if (size < 2) {
		return 0.00f;
	}
	float dot_product = 0.00f;
	float arr1_square_sum = 0.00f;
	float arr2_square_sum = 0.00f;
	float slope_similarity_sum = 0.00f;
	for (int i = 1; i < size; i++) {
		float arr1_slope = arr1[i] - arr1[i - 1];
		float arr2_slope = arr2[i] - arr2[i - 1];
		if ((arr1_slope > 0.00f && arr2_slope < 0.00f) || (arr1_slope < 0.00f && arr2_slope > 0.00f)) {
			return 0.00f;
		}
		dot_product += arr1_slope * arr2_slope;
		arr1_square_sum += arr1_slope * arr1_slope;
		arr2_square_sum += arr2_slope * arr2_slope;
		if (arr1_slope == 0.00f && arr2_slope == 0.00f) {
			slope_similarity_sum += 1.00f;
		}
		else if ((arr1_slope > 0.00f && arr2_slope > 0.00f) || (arr1_slope < 0.00f && arr2_slope < 0.00f)) {
			float arr1_abs_slope = fabs(arr1_slope);
			float arr2_abs_slope = fabs(arr2_slope);
			float max_slope = (arr1_abs_slope > arr2_abs_slope) ? arr1_abs_slope : arr2_abs_slope;
			float min_slope = (arr1_abs_slope < arr2_abs_slope) ? arr1_abs_slope : arr2_abs_slope;
			slope_similarity_sum += min_slope / max_slope;
		}
	}
	if (arr1_square_sum == 0.00f && arr2_square_sum == 0.00f) {
		return 100.00f;
	}
	if (arr1_square_sum == 0.00f || arr2_square_sum == 0.00f) {
		return 0.00f;
	}
	float cosine_similarity = dot_product / sqrt(arr1_square_sum * arr2_square_sum);
	if (cosine_similarity <= 0.00f) {
		return 0.00f;
	}
	if (cosine_similarity > 1.00f) {
		cosine_similarity = 1.00f;
	}
	float slope_similarity = slope_similarity_sum / (size - 1);
	return cosine_similarity * slope_similarity * 100.00f;
}
float calculateVariance(const float* data, int size) {
	// Calcolo della media
	float sum = 0.0f;
	for (int i = 0; i < size; ++i) {
		sum += data[i];
	}
	float mean = sum / size;
	// Calcolo della somma dei quadrati delle differenze
	float sumSquaredDifferences = 0.0f;
	for (int i = 0; i < size; ++i) {
		float diff = data[i] - mean;
		sumSquaredDifferences += diff * diff;
	}
	// La varianza (per popolazione) è la media dei quadrati delle differenze
	return sumSquaredDifferences / size;
}
void setTime() {
	std::time_t now = std::time(nullptr);
	std::tm local_time;
#ifdef __linux__
	// Usa localtime_r per Linux
	struct tm timeinfo;
	localtime_r(&now, &local_time);
#elif _WIN32
	localtime_s(&local_time, &now);
#endif
	std::strftime(_global_time, sizeof(_global_time), "%H:%M:%S", &local_time);
}
int get_random_sample_first_line() {
	static std::mt19937 random_generator(std::random_device{}());
	std::uniform_int_distribution<int> sample_distribution(0, training_samples - 1);
	return sample_distribution(random_generator) * lines_per_training_sample + 1;
}
bool get_sample_for_test(int sampleIndex) {
	// Composizione del path completo del file
	std::string filename = _relative_files_path + "/" + _files_name;
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Errore nell'apertura del file: " << filename << std::endl;
		return false;
	}
	std::string line;
	if (!std::getline(file, line) || !is_csv_header(line)) {
		std::cerr << "Intestazione CSV mancante o non valida nel file " << filename << std::endl;
		return false;
	}
	// Ogni campione è composto da 8 righe; calcoliamo la riga iniziale del campione.
	//const int linesPerSample = 8;
	int startLine = sampleIndex - 1;// *linesPerSample;
	// Salta le righe fino al campione desiderato
	std::string dummy;
	for (int i = 0; i < startLine; ++i) {
		if (!std::getline(file, dummy)) {
			std::cerr << "Errore: file terminato prematuramente durante lo skip fino al campione "
				<< sampleIndex << std::endl;
			return false;
		}
	}
	std::istringstream ss;
	std::string token;
	// Legge le prime 6 righe per aggiornare observed_data
	for (int i = 0; i < 6; ++i) {
		if (!std::getline(file, line)) {
			std::cerr << "Errore: file terminato prematuramente nella lettura di observed_data, campione "
				<< sampleIndex << std::endl;
			return false;
		}
		if (line.empty()) {
			--i;
			continue;
		}
		ss.clear();
		ss.str(line);
		// Legge il primo token (es. "0")
		std::getline(ss, token, ';');
		// Legge il secondo token (es. "B0", "B1", ecc.) e lo scarta
		std::getline(ss, token, ';');
		// Legge il terzo token (il valore da utilizzare)
		std::getline(ss, token, ';');
		try {
			observed_data[i] = std::stof(token);
		}
		catch (const std::exception& e) {
			std::cerr << "Errore nella conversione del valore nella riga " << (startLine + i + 1)
				<< ": \"" << token << "\"." << std::endl;
			return false;
		}
	}

	// Legge la settima riga (wattora) per aggiornare x[1]
	if (!std::getline(file, line)) {
		std::cerr << "Errore: file terminato prematuramente nella lettura di x[1] (riga "
			<< (startLine + 7) << ")." << std::endl;
		return false;
	}
	while (line.empty() && std::getline(file, line)) {}
	ss.clear();
	ss.str(line);
	// Legge il primo token (es. "watts")
	std::getline(ss, token, ';');
	// Salta il secondo token
	std::getline(ss, token, ';');
	// Legge il terzo token (il valore per x[1])
	std::getline(ss, token, ';');
	try {
		x[1] = std::stof(token);
	}
	catch (const std::exception& e) {
		std::cerr << "Errore nella conversione del valore nella riga " << (startLine + 7)
			<< " per x[1]: \"" << token << "\"." << std::endl;
		return false;
	}

	// Legge l'ottava riga (ampere) per aggiornare x[0]
	if (!std::getline(file, line)) {
		std::cerr << "Errore: file terminato prematuramente nella lettura di x[0] (riga "
			<< (startLine + 8) << ")." << std::endl;
		return false;
	}
	while (line.empty() && std::getline(file, line)) {}
	ss.clear();
	ss.str(line);
	// Legge il primo token (es. "amps")
	std::getline(ss, token, ';');
	// Salta il secondo token
	std::getline(ss, token, ';');
	// Legge il terzo token (il valore per x[0])
	std::getline(ss, token, ';');
	try {
		x[0] = std::stof(token);
	}
	catch (const std::exception& e) {
		std::cerr << "Errore nella conversione del valore nella riga " << (startLine + 8)
			<< " per x[0]: \"" << token << "\"." << std::endl;
		return false;
	}
	file.close();
	return true;
}
int count_training_samples(int linesPerSample) {
	// Componi il percorso completo del file
	std::string filename = _relative_files_path + "/" + _files_name;
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Errore nell'apertura del file: " << filename << std::endl;
		return -1;
	}
	std::string line;
	if (!std::getline(file, line) || !is_csv_header(line)) {
		std::cerr << "Intestazione CSV mancante o non valida nel file " << filename << std::endl;
		return -1;
	}
	int totalLines = 0;
	// Conta solo le righe dati non vuote dopo l'intestazione.
	while (std::getline(file, line)) {
		if (!line.empty())
			++totalLines;
	}
	file.close();
	// Verifica che il numero totale di righe sia divisibile per linesPerSample
	if (totalLines % linesPerSample != 0) {
		std::cerr << "Il numero totale di righe (" << totalLines
			<< ") non è divisibile per " << linesPerSample << std::endl;
		return -1;
	}
	return totalLines / linesPerSample;
}
bool is_csv_header(const std::string& line) {
	return line.rfind("IDMessage;Battery;Value;", 0) == 0;
}
