#pragma once
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace battery_csv {
struct Sample {
    std::array<float, 6> batteries{};
    float watt_hours = 0;
    float amps = 0;
    int first_line = 0; // One-based physical CSV line, including any header.
};
inline std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? "" : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
inline float number(std::string s) {
    s = trim(s);
    for (char& c : s) if (c == ',') c = '.';
    size_t used = 0;
    const float value = std::stof(s, &used);
    if (used != s.size() || !std::isfinite(value) || value < 0)
        throw std::runtime_error("valore numerico non valido");
    return value;
}
inline std::vector<Sample> read(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Impossibile aprire " + filename);
    std::vector<Sample> result;
    Sample sample;
    int state = 0, line_number = 0;
    std::string line;
    while (std::getline(file, line)) {
        ++line_number;
        if (line_number == 1 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);

        if (line_number == 1) {
            if (trim(line) != "IDMessage;Battery;Value;W/h;amps")
                throw std::runtime_error(filename + ": intestazione CSV non valida");
            continue;
        }
        if (trim(line).empty()) continue;
        try {
            std::vector<std::string> fields;
            line = trim(line);
            std::stringstream stream(line);
            std::string field;
            while (std::getline(stream, field, ';')) fields.push_back(trim(field));
            if (!line.empty() && line.back() == ';') fields.push_back("");
            if (state < 6) {
                if (fields.size() != 5 || fields[1] != "B" + std::to_string(state) || !fields[3].empty() || !fields[4].empty())
                    throw std::runtime_error("attesa batteria B" + std::to_string(state));
                if (state == 0) sample.first_line = line_number;
                sample.batteries[state++] = number(fields[2]);
            } else if (state == 6 && fields.size() == 5 && fields[0].empty() && fields[1].empty() && fields[2].empty()) {
                sample.watt_hours = number(fields[3]);
                sample.amps = number(fields[4]);
                result.push_back(sample);
                state = 0;
            } else throw std::runtime_error("attese misure Wh e ampere");
        } catch (const std::exception& e) {
            throw std::runtime_error(filename + ": riga " + std::to_string(line_number) + ": " + e.what());
        }
    }
    if (line_number == 0) throw std::runtime_error(filename + ": intestazione CSV mancante");
    if (state != 0) throw std::runtime_error(filename + ": campione finale incompleto");
    return result;
}
}
