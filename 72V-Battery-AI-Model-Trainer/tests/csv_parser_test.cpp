#define main trainer_main
#include "../72V-Battery-AI-Model-Trainer.cpp"
#undef main
#include <cassert>
#include <cstdio>

int main(int argc, char** argv) {
    const std::string path = "battery_csv_test_input.tmp";
    const std::string batteries = "0;B0;12.1;;\n0;B1;12.2;;\n0;B2;12.3;;\n0;B3;12.4;;\n0;B4;12.5;;\n0;B5;12.6;;\n";
    auto parse = [&](const std::string& text) {
        { std::ofstream out(path); out << text; }
        return battery_csv::read(path);
    };
    auto samples = parse("\xEF\xBB\xBFIDMessage;Battery;Value;W/h;amps\r\n" + batteries + ";;;150.25;20.50\r\n\r\n" + batteries + ";;;151.25;21.50\n");
    assert(samples.size() == 2 && samples[0].first_line == 2 && samples[1].first_line == 10);
    assert(samples[0].watt_hours == 150.25f && samples[0].amps == 20.5f);
    assert(samples[1].amps == 21.5f && samples[1].batteries[5] == 12.6f);
    assert(parse("IDMessage;Battery;Value;W/h;amps\n" + batteries + ";;;150,25;20,50").size() == 1);

    for (const auto& bad : std::vector<std::string>{batteries, batteries + ";;;NaN;2\n", batteries + ";;;12oops;2\n", batteries + ";;;12;\n", batteries + "amps;;2;;\n", "0;B1;12;;\n", batteries + ";;;12;-1\n", batteries + "watts;;150.25;;\namps;;20.50;;\n"}) {
        bool failed = false;
        try { parse("IDMessage;Battery;Value;W/h;amps\n" + bad); } catch (const std::exception&) { failed = true; }
        assert(failed);
    }
    csv_samples.assign(training_samples, samples[0]);
    csv_samples[1] = samples[1];
    read_samples_from_file_diagram_battery();
    assert(amps_training[1] == 21.5f && watts_hour_training[0] == 150.25f);
    assert(battery_out_training[0][5] == 12.6f);
    assert(get_sample_for_test(10) && x[0] == 21.5f && x[1] == 151.25f);
    assert(!get_sample_for_test(-1));
    for (int i = 1; i < argc; ++i)
        std::cout << argv[i] << ": " << battery_csv::read(argv[i]).size() << " samples\n";
    std::remove(path.c_str());
    std::cout << "CSV parser and training/inference mapping tests passed\n";
}
