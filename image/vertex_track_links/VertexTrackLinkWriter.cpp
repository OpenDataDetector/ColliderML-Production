// VertexTrackLinkWriter: for every reco vertex of an AMVF vertex collection, write the indices of its tracks in the
// input track container (= reco/tracks track_id, which is ConstTrackContainer::index()) and the AMVF track weights.
//
// AdaptiveMultiVertexFinderAlgorithm stores, per TrackAtVertex, an InputTrack pointing into the
// TrackParametersContainer it read; TracksToParameters filled that container from the track container in order,
// skipping tracks without a reference surface. The writer rebuilds that order (track.index() of every track with a
// reference surface) and resolves each pointer to its position in the parameter container. A pointer outside the
// container, or a container whose size differs from the rebuilt order, is an error (FAILURE), never a silent skip.
//
// Output: CSV with header "event,vertex,track_id,weight,vx,vy,vz", one line per (vertex, track); vertex is the
// position in the vertex container (= reco/vertices vertex_id, the order RootVertexNTupleWriter uses); vx, vy, vz
// the fitted vertex position in mm (a join key check against the ntuple); weights and positions printed with 17
// significant digits (exact doubles). A vertex without tracks gets exactly one line with track_id -1. Lines of one event are contiguous; events may appear in any order (multi-threaded sequencer).
#include "Acts/Definitions/Units.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Vertexing/TrackAtVertex.hpp"
#include "Acts/Vertexing/Vertex.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/EventData/Vertex.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/WriterT.hpp"

#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <any>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace colliderml {

using namespace ActsExamples;

class VertexTrackLinkWriter final : public WriterT<VertexContainer> {
 public:
  struct Config {
    std::string inputVertices = "vertices";
    std::string inputTrackParameters = "selectedTracksParametersVertexing";
    std::string inputTracks = "tracks";
    std::string filePath = "vertex_track_links.csv";
  };

  VertexTrackLinkWriter(const Config& cfg, Acts::Logging::Level lvl)
      : WriterT<VertexContainer>(cfg.inputVertices, "VertexTrackLinkWriter", lvl), m_cfg(cfg) {
    m_inputParams.initialize(m_cfg.inputTrackParameters);
    m_inputTracks.initialize(m_cfg.inputTracks);
    m_out.open(m_cfg.filePath, std::ios::out | std::ios::trunc);
    if (!m_out) {
      throw std::runtime_error("cannot open " + m_cfg.filePath);
    }
    m_out << "event,vertex,track_id,weight,vx,vy,vz\n";
  }

  ProcessCode finalize() override {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_out.close();
    return m_out ? ProcessCode::SUCCESS : ProcessCode::ABORT;
  }

  const Config& config() const { return m_cfg; }

 protected:
  ProcessCode writeT(const AlgorithmContext& ctx, const VertexContainer& vertices) override {
    const auto& params = m_inputParams(ctx);
    const auto& tracks = m_inputTracks(ctx);
    std::vector<long> order;  // parameter position -> track index
    order.reserve(tracks.size());
    for (const auto& track : tracks) {
      if (track.hasReferenceSurface()) {
        order.push_back(static_cast<long>(track.index()));
      }
    }
    if (order.size() != params.size()) {
      ACTS_ERROR("event " << ctx.eventNumber << ": " << params.size() << " track parameters but " << order.size()
                          << " tracks with a reference surface");
      return ProcessCode::ABORT;
    }
    // address -> track index lookup (no pointer arithmetic on possibly unrelated pointers)
    std::unordered_map<const TrackParameters*, long> index_of;
    index_of.reserve(params.size());
    for (std::size_t k = 0; k < params.size(); ++k) {
      index_of.emplace(&params[k], order[k]);
    }
    std::ostringstream os;
    os << std::setprecision(17);
    for (std::size_t iv = 0; iv < vertices.size(); ++iv) {
      const auto& v = vertices[iv];
      const auto pos = v.position();
      if (v.tracks().empty()) {
        os << ctx.eventNumber << ',' << iv << ",-1,0," << pos.x() << ',' << pos.y() << ',' << pos.z() << '\n';
      }
      for (const auto& trk : v.tracks()) {
        const TrackParameters* p = nullptr;
        try {
          p = trk.originalParams.as<TrackParameters>();
        } catch (const std::bad_any_cast&) {
          ACTS_ERROR("event " << ctx.eventNumber << " vertex " << iv << ": track of unexpected type");
          return ProcessCode::ABORT;
        }
        const auto it = index_of.find(p);
        if (p == nullptr || it == index_of.end()) {
          ACTS_ERROR("event " << ctx.eventNumber << " vertex " << iv << ": track pointer outside the input container");
          return ProcessCode::ABORT;
        }
        os << ctx.eventNumber << ',' << iv << ',' << it->second << ',' << trk.trackWeight << ',' << pos.x() << ','
           << pos.y() << ',' << pos.z() << '\n';
      }
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_out << os.str();
    if (!m_out) {
      ACTS_ERROR("write failed");
      return ProcessCode::ABORT;
    }
    return ProcessCode::SUCCESS;
  }

 private:
  Config m_cfg;
  ReadDataHandle<TrackParametersContainer> m_inputParams{this, "InputTrackParameters"};
  ReadDataHandle<ConstTrackContainer> m_inputTracks{this, "InputTracks"};
  std::mutex m_mutex;
  std::ofstream m_out;
};

}  // namespace colliderml

namespace py = pybind11;

PYBIND11_MODULE(colliderml_vertex_links, m) {
  py::module_::import("acts.examples");  // registers IWriter and the logging level enum
  using W = colliderml::VertexTrackLinkWriter;
  auto w = py::class_<W, ActsExamples::IWriter, std::shared_ptr<W>>(m, "VertexTrackLinkWriter")
               .def(py::init<const W::Config&, Acts::Logging::Level>(), py::arg("config"), py::arg("level"))
               .def_property_readonly("config", &W::config);
  py::class_<W::Config>(w, "Config")
      .def(py::init<>())
      .def_readwrite("inputVertices", &W::Config::inputVertices)
      .def_readwrite("inputTrackParameters", &W::Config::inputTrackParameters)
      .def_readwrite("inputTracks", &W::Config::inputTracks)
      .def_readwrite("filePath", &W::Config::filePath);
}
