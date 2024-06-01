// root
// .L /home/rustam/root_graphs/build_histograms_root.C
// main()
#include <TH1F.h>
#include <TCanvas.h>


int main() {
    std::vector<TString> branch_names = {"nnonbjets", "foxWolfram_2_momentum", "chi2_min_higgs_m", "nbjets",
    "njets", "nfwdjets", "njets_CBT5", "njets_CBT4", "sphericity", "chi2_min_Imvmass_tH", "chi2_min_Whad_m_ttAll",
    "chi2_min_tophad_m_ttAll", "chi2_min_deltaRq1q2", "chi2_min_bbnonbjet_m", "chi2_min_toplep_pt",
    "chi2_min_tophad_pt_ttAll", "bbs_top_m"};

    TFile *file = new TFile("single_t_weighted.root", "READ");
    TTree *tree = (TTree*)file->Get("nominal_Loose");
    TCanvas *canvas = new TCanvas();

    for (const auto& branch_name : branch_names) {
        tree->Draw(branch_name);
        TString file_name = branch_name + ".png";
        canvas->SaveAs(file_name);
    }

    delete canvas;
    file->Close();
    delete file;
    
    return 0;
}
