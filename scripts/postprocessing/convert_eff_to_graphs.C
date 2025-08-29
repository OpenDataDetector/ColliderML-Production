#include <TFile.h>
#include <TKey.h>
#include <TClass.h>
#include <TString.h>
#include <TList.h>
#include <TEfficiency.h>
#include <TGraphAsymmErrors.h>
#include <TTree.h>
#include <vector>
#include <string>

namespace {
  std::vector<TString> split_csv(const char* csv) {
    std::vector<TString> tokens;
    if (!csv) return tokens;
    TString s(csv);
    s.ReplaceAll(" ", "");
    Ssiz_t from = 0;
    while (true) {
      Ssiz_t comma = s.Index(',', from);
      if (comma == kNPOS) {
        TString t = s(from, s.Length() - from);
        if (t.Length() > 0) tokens.push_back(t);
        break;
      }
      TString t = s(from, comma - from);
      if (t.Length() > 0) tokens.push_back(t);
      from = comma + 1;
    }
    return tokens;
  }

  void write_graph_and_tree(TGraphAsymmErrors* graph, const TString& baseName) {
    if (!graph) return;
    graph->SetName(baseName + "_gr");
    graph->Write();

    const int numPoints = graph->GetN();
    double x = 0.0, y = 0.0;
    double exLow = 0.0, exHigh = 0.0;
    double eyLow = 0.0, eyHigh = 0.0;

    TTree pointsTree(baseName + "_tree", baseName + " points");
    pointsTree.Branch("x", &x);
    pointsTree.Branch("y", &y);
    pointsTree.Branch("ex_low", &exLow);
    pointsTree.Branch("ex_high", &exHigh);
    pointsTree.Branch("ey_low", &eyLow);
    pointsTree.Branch("ey_high", &eyHigh);

    for (int i = 0; i < numPoints; i++) {
      x = graph->GetX()[i];
      y = graph->GetY()[i];
      exLow = graph->GetErrorXlow(i);
      exHigh = graph->GetErrorXhigh(i);
      eyLow = graph->GetErrorYlow(i);
      eyHigh = graph->GetErrorYhigh(i);
      pointsTree.Fill();
    }
    pointsTree.Write();
  }
}

// Convert selected TEfficiency objects in a ROOT file to TGraphAsymmErrors and an accompanying TTree.
// By default, converts efficiency vs pT and vs eta: trackeff_vs_pT, trackeff_vs_eta.
// Usage from shell:
//   root -l -b -q '/path/to/convert_eff_to_graphs.C+("/in.root","/out.root","trackeff_vs_pT,trackeff_vs_eta")'
void convert_eff_to_graphs(const char* input_path,
                           const char* output_path,
                           const char* key_list_csv = "trackeff_vs_pT,trackeff_vs_eta",
                           bool skip_missing = true) {
  if (!input_path || !output_path) {
    printf("[convert_eff_to_graphs] ERROR: input_path or output_path is null.\n");
    return;
  }

  TFile inputFile(input_path, "READ");
  if (inputFile.IsZombie()) {
    printf("[convert_eff_to_graphs] ERROR: Failed to open input file: %s\n", input_path);
    return;
  }

  TFile outputFile(output_path, "RECREATE");
  if (outputFile.IsZombie()) {
    printf("[convert_eff_to_graphs] ERROR: Failed to open output file: %s\n", output_path);
    return;
  }

  std::vector<TString> keysToConvert = split_csv(key_list_csv);
  if (keysToConvert.empty()) {
    printf("[convert_eff_to_graphs] WARNING: key list empty; nothing to do.\n");
    return;
  }

  printf("[convert_eff_to_graphs] Input: %s\n", input_path);
  printf("[convert_eff_to_graphs] Output: %s\n", output_path);
  printf("[convert_eff_to_graphs] Keys to convert (%zu): ", keysToConvert.size());
  for (size_t i = 0; i < keysToConvert.size(); ++i) {
    printf("%s%s", keysToConvert[i].Data(), (i + 1 < keysToConvert.size() ? ", " : "\n"));
  }

  for (const TString& keyName : keysToConvert) {
    TObject* obj = inputFile.Get(keyName);
    if (!obj) {
      printf("[convert_eff_to_graphs] %s: NOT FOUND%s\n", keyName.Data(), skip_missing ? " (skip)" : "");
      if (!skip_missing) {
        printf("[convert_eff_to_graphs] ERROR: required key missing. Aborting.\n");
        return;
      }
      continue;
    }

    printf("[convert_eff_to_graphs] %s: class %s\n", keyName.Data(), obj->ClassName());

    TGraphAsymmErrors* graph = nullptr;
    if (obj->InheritsFrom(TEfficiency::Class())) {
      TEfficiency* eff = static_cast<TEfficiency*>(obj);
      graph = eff->CreateGraph();
      if (!graph) {
        printf("[convert_eff_to_graphs] %s: CreateGraph() returned null, skipping.\n", keyName.Data());
        continue;
      }
      write_graph_and_tree(graph, keyName);
      delete graph; // CreateGraph returns a new object
    } else if (obj->InheritsFrom(TGraphAsymmErrors::Class())) {
      // Already a graph: write a normalized copy and its tree
      graph = static_cast<TGraphAsymmErrors*>(obj);
      TGraphAsymmErrors* clone = static_cast<TGraphAsymmErrors*>(graph->Clone());
      write_graph_and_tree(clone, keyName);
      delete clone;
    } else {
      printf("[convert_eff_to_graphs] %s: Unsupported class (expected TEfficiency or TGraphAsymmErrors), skipping.\n", keyName.Data());
      continue;
    }
  }

  printf("[convert_eff_to_graphs] Wrote objects:\n");
  outputFile.GetListOfKeys()->Print();

  outputFile.Close();
  inputFile.Close();
}


