#ifndef PLOT_RENDERER_H
#define PLOT_RENDERER_H

#include <vector>
#include <algorithm>
//#include "imgui.h"
//#include "implot.h"

class PlotRenderer {
public:
    PlotRenderer();
    PlotRenderer(const char* window_name,
        const std::vector<double>& ascissa1,
        const std::vector<double>& ordinata,
        const char* x_label = "Epoca",
        const char* y_label = "Errore",
        const char* plot_title = "Andamento Errore");

    void Begin();

private:
    const char* window_name;
    std::vector<double> ascissa1;
    std::vector<double> ordinata;
    const char* x_label;
    const char* y_label;
    const char* plot_title;
    bool is_init = false;
};

#endif // PLOT_RENDERER_H
