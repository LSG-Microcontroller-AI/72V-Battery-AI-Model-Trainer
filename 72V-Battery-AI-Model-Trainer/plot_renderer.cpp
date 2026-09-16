#include "plot_renderer.h"
PlotRenderer::PlotRenderer() {}
PlotRenderer::PlotRenderer(const char* window_name,
    const std::vector<double>& ascissa1,
    const std::vector<double>& ordinata,
    const char* x_label,
    const char* y_label,
    const char* plot_title)
    : window_name(window_name), ascissa1(ascissa1), ordinata(ordinata),
    x_label(x_label), y_label(y_label), plot_title(plot_title) {
    is_init = true;
}
void PlotRenderer::Begin() {
    if (!is_init) return;
    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);
    ImGui::Begin(window_name, nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);

    ImVec2 plot_size = ImVec2(ImGui::GetWindowSize().x - 20, ImGui::GetWindowSize().y - 60);

    if (ImPlot::BeginPlot(plot_title, plot_size)) {
        ImPlot::PushStyleColor(ImPlotCol_AxisGrid, ImVec4(1.0f, 1.0f, 0.3f, 0.7f));
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.3f, 0.7f, 1.0f, 1.0f));
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1.0f, ImVec4(0.3f, 0.8f, 1.0f, 1.0f), 1.0f);

        ImPlot::SetupAxes(x_label, y_label, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, 100, ImGuiCond_Once);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.0, ImGuiCond_Once);

        if (!ascissa1.empty() && !ordinata.empty()) {
            ImPlot::PlotLine(y_label, ascissa1.data(), ordinata.data(), ascissa1.size());
        }

        ImPlot::PopStyleColor(2);
        ImPlot::EndPlot();
    }

    ImGui::End();
}
