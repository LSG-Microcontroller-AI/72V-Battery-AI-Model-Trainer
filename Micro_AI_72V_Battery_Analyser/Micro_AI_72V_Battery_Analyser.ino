/*
 Name:		AI_battery_analyser.ino
 Created:	2/18/2025 12:52:20 PM
 Author:	luigi.santagada
*/
#include <Arduino.h>
#include <SoftwareSerial.h>
//#ifndef RAMSTART
//extern int __data_start;
//#endif
//extern int __bss_end;
//extern void* __brkval;
extern char __heap_start;
extern void* __brkval;
#include <EEPROM.h>
const uint8_t numberOf_X = 2;
const uint8_t numberOf_H = 25;
const uint8_t numberOf_Y = 6;
const float leaky_relu_alpha = 0.01f;
float normalized_observed_output[numberOf_Y] = { 0.00 };
float normalized_predicted_output[numberOf_Y] = { 0.00 };
float output_bias[numberOf_Y] = { 0.00 };
float hidden_bias[numberOf_H] = { 0.00 };
//float W1[numberOf_X][numberOf_H] = { 0.00 };
//float W2[numberOf_H][numberOf_Y] = { 0.00 };
float x[numberOf_X] = { 0.00 };
float h[numberOf_H] = { 0.00 };
float y[numberOf_Y] = { 0.00f };
float observed_data[6] = { 0.00 };
const int headerSize = 2;
const uint8_t numFloats = 8;
const uint8_t dataSize = numFloats * sizeof(float);
const uint8_t checksumSize = sizeof(float);
const uint8_t footerSize = 1;
const uint8_t packetSize = headerSize + dataSize + checksumSize + footerSize; // 39 bytes
//void read_weights_from_eeprom();
// the setup function runs once when you press reset or power the board
SoftwareSerial mySerial(10, 11); // Definizione: RX, TX
void setup() {
	Serial.begin(9600);
	//read_weights_from_eeprom();
	observed_data[0] = 1.91f;
	observed_data[1] = 1.84f;
	observed_data[2] = 1.94f;
	observed_data[3] = 1.81f;
	observed_data[4] = 1.93f;
	observed_data[5] = 1.81f;
}
// the loop function runs over and over again until power down or reset
void loop() {
	//simulateTransmission();
	int ram_libera = freeMemory();
	Serial.print(F("RAM libera: "));
	Serial.print(ram_libera);
	Serial.println(F(" byte"));
	x[0] = 29.00f;
	x[1] = 479.00f;
	x[0] = log(x[0] + 1.0f) / 10.0f;
	x[1] = log(x[1] + 1.0f) / 10.0f;
	forward();
	for (int i = 0; i < 6; i++) {
		y[i] = y[i] * 10.00f;
	}
	//print_model_data();
	normalizeArray(observed_data, normalized_observed_output, numberOf_Y);
	normalizeArray(y, normalized_predicted_output, numberOf_Y);
	float mse = meanSquaredError(observed_data, y, numberOf_Y);
	// La percentuale usa le tensioni osservate, prima della normalizzazione min-max.
	float observed_mean = mean_value(observed_data, numberOf_Y);
	float percentage = calculateErrorPercentage(mse, observed_mean);
	Serial.print(F("percentage : ")); Serial.print(percentage); Serial.println(F("%"));
	float varianza = calculateVariance(normalized_observed_output, numberOf_Y);
	Serial.print(F("varianza tensioni normalizzate : ")); Serial.println(varianza);
	float cosine_percentage = calculate_cosine_shape_similarity_percentage(observed_data, y, numberOf_Y);
	Serial.print(F("cosine similarity percentage : ")); Serial.print(cosine_percentage); Serial.println(F("%"));
	//print_normalizer_processed_data();
	//print_normalizer_model_data();
	delay(2000);
}
void print_model_data() {
	Serial.println();
	Serial.print(F("Ampere (A) [x[0]] = ")); Serial.print(exp(x[0] * 10.00f) - 1.00f);
	Serial.print(F("    Wattora (Wh) [x[1]] = ")); Serial.println(exp(x[1] * 10.00f) - 1.00f);
	Serial.print(F("\n\ny[0] = ")); Serial.println(y[0]);
	Serial.print(F("y[1] = ")); Serial.println(y[1]);
	Serial.print(F("y[2] = ")); Serial.println(y[2]);
	Serial.print(F("y[3] = ")); Serial.println(y[3]);
	Serial.print(F("y[4] = ")); Serial.println(y[4]);
	Serial.print(F("y[5] = ")); Serial.println(y[5]);
	delay(2000);
}
void print_normalizer_processed_data() {
	Serial.println();
	Serial.print(F("y_normalized[0] = ")); Serial.println(normalized_observed_output[0]);
	Serial.print(F("y_normalized[1] = "));  Serial.println(normalized_observed_output[1]);
	Serial.print(F("y_normalized[2] = "));  Serial.println(normalized_observed_output[2]);
	Serial.print(F("y_normalized[3] = "));  Serial.println(normalized_observed_output[3]);
	Serial.print(F("y_normalized[4] = "));  Serial.println(normalized_observed_output[4]);
	Serial.print(F("y_normalized[5] = "));  Serial.println(normalized_observed_output[5]);
	delay(2000);
}
void print_normalizer_model_data() {
	Serial.println();
	Serial.print(F("y_normalized_model_data[0] = ")); Serial.println(normalized_predicted_output[0]);
	Serial.print(F("y_normalized_model_data[1] = "));  Serial.println(normalized_predicted_output[1]);
	Serial.print(F("y_normalized_model_data[2] = "));  Serial.println(normalized_predicted_output[2]);
	Serial.print(F("y_normalized_model_data[3] = "));  Serial.println(normalized_predicted_output[3]);
	Serial.print(F("y_normalized_model_data[4] = "));  Serial.println(normalized_predicted_output[4]);
	Serial.print(F("y_normalized_model_data[5] = "));  Serial.println(normalized_predicted_output[5]);
	delay(2000);
}
float leaky_relu(float value) {
	return (value > 0.0f) ? value : leaky_relu_alpha * value;
}
void forward() {
	int addr = 0;
	float Zk = 0.00f;
	float Zj = 0.00f;
	float data_from_eeprom = 0.00f;
	for (int k = 0; k < (numberOf_H); k++) {
		Zk = 0.00f;
		for (int i = 0; i < numberOf_X; i++) {
			EEPROM.get(addr, data_from_eeprom);
			Zk += (data_from_eeprom * x[i]);
			addr += sizeof(float);
		}
		//insert X bias
		EEPROM.get(addr, data_from_eeprom);
		Zk += data_from_eeprom;
		h[k] = leaky_relu(Zk);
		addr += sizeof(float);
	}
	for (int j = 0; j < numberOf_Y; j++) {
		Zj = 0.00f;
		for (int k = 0; k < numberOf_H; k++) {
			EEPROM.get(addr, data_from_eeprom);
			Zj += (data_from_eeprom * h[k]);
			addr += sizeof(float);
		}
		EEPROM.get(addr, data_from_eeprom);
		//insert H bias
		Zj += data_from_eeprom;
		y[j] = Zj;
		addr += sizeof(float);
	}
}
//void read_weights_from_eeprom() {
//	int addr = 0;  // partiamo dall'indirizzo 0 della EEPROM
//	// Legge la matrice W1: dimensione [numberOf_X][numberOf_H]
//		for (int k = 0; k < numberOf_H; k++) {
//			for (int i = 0; i < numberOf_X; i++) {
//			EEPROM.get(addr, W1[i][k]);
//			addr += sizeof(float);
//		}
//	}
//	// Legge la matrice W2: dimensione [numberOf_H][numberOf_Y]
//	for (int j = 0; j < numberOf_Y; j++) {
//		for (int k = 0; k < numberOf_H; k++) {
//			EEPROM.get(addr, W2[k][j]);
//			addr += sizeof(float);
//		}
//	}
//	// Legge il vettore dei bias per il layer nascosto
//	for (int k = 0; k < numberOf_H; k++) {
//		EEPROM.get(addr, hidden_bias[k]);
//		addr += sizeof(float);
//	}
//	// Legge il vettore dei bias per l'output
//	for (int j = 0; j < numberOf_Y; j++) {
//		EEPROM.get(addr, output_bias[j]);
//		addr += sizeof(float);
//	}
//}
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
float meanSquaredError(const float* arr1, const float* arr2, int size) {
	float sum = 0.0f;
	for (int i = 0; i < size; ++i) {
		float diff = arr1[i] - arr2[i];
		sum += (diff * diff);  // quadrato della differenza
	}
	// MSE = (1 / N) * Σ (diff^2)
	return sum / size;
}
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
// Funzione per calcolare il checksum (somma dei float)
float computeChecksum(const float* data, int count) {
	float sum = 0.0f;
	for (int i = 0; i < count; i++) {
		sum += data[i];
	}
	return sum;
}
// Processa un flusso di byte alla ricerca di pacchetti validi.
// Per ogni pacchetto trovato, controlla marker, checksum e stampa i dati.
// Costruisce un pacchetto nel buffer "packet" con i float forniti.
// Se "correctChecksum" è false, il checksum viene volutamente alterato.
// Scansiona un flusso di byte alla ricerca di pacchetti validi.
// Per ogni pacchetto trovato, controlla marker, footer e checksum.
void processDataStream(const byte* stream, int streamSize) {
	int i = 0;
	while (i <= streamSize - packetSize) {
		if (stream[i] == 0xAA && stream[i + 1] == 0xBB && stream[i + packetSize - 1] == 0xCC) {
			float values[numFloats];
			memcpy(values, stream + i + headerSize, dataSize);
			float transmittedChecksum;
			memcpy(&transmittedChecksum, stream + i + headerSize + dataSize, checksumSize);
			float computedChecksum = computeChecksum(values, numFloats);
			if (fabs(transmittedChecksum - computedChecksum) < 0.0001f) {
				Serial.print("Packet valido trovato a indice ");
				Serial.println(i);
				for (int j = 0; j < numFloats; j++) {
					Serial.print(F("Value "));
					Serial.print(j);
					Serial.print(F(": "));
					Serial.println(values[j]);
				}
			}
			else {
				Serial.print(F("Packet NON valido (checksum errato) a indice "));
				Serial.println(i);
			}
			i += packetSize; // Salta il pacchetto trovato
		}
		else {
			i++;
		}
	}
}
void simulateTransmission() {
	const int streamSize = 150;
	byte dataStream[streamSize];
	memset(dataStream, 0x00, streamSize);
	float packetValues1[numFloats] = { 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f };
	buildPacket(dataStream + 10, packetValues1, true);
	//float packetValues2[numFloats] = { 9.1f, 10.2f, 11.3f, 12.4f, 13.5f, 14.6f, 15.7f, 16.8f };
	//buildPacket(dataStream + 60, packetValues2, false);
	//float packetValues3[numFloats] = { 17.1f, 18.2f, 19.3f, 20.4f, 21.5f, 22.6f, 23.7f, 24.8f };
	//buildPacket(dataStream + 110, packetValues3, true);
	processDataStream(dataStream, streamSize);
}
void buildPacket(byte* packet, const float* values, bool correctChecksum) {
	packet[0] = 0xAA;
	packet[1] = 0xBB;
	memcpy(packet + headerSize, values, dataSize);
	float checksum = computeChecksum(values, numFloats);
	if (!correctChecksum) {
		checksum += 1.0f;  // Corrompe il checksum
	}
	memcpy(packet + headerSize + dataSize, &checksum, checksumSize);
	packet[packetSize - 1] = 0xCC;
}
//int freeMemory2() {
//	int free_memory;
//	if ((int)__brkval == 0) {
//		free_memory = ((int)&free_memory) - ((int)&__bss_end);
//	}
//	else {
//		free_memory = ((int)&free_memory) - ((int)__brkval);
//	}
//	return free_memory;
//}
unsigned int freeMemory() {
	char top;
	return (unsigned int)&top - (unsigned int)(__brkval == 0 ? &__heap_start : __brkval);
}
