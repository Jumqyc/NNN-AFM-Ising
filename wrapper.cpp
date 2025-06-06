#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Ising.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ising, m)
{
    py::class_<Ising>(m, "Ising")

        // expose the functions to python

        // constructor
        .def(py::init<int, int>(), py::arg("a"), py::arg("b"))

        // random seed
        // note: this function reseeds the random number generator
        .def("random_seed", &Ising::random_seed)

        // getters for parameters
        .def("temperature", &Ising::temperature)
        .def("J1", &Ising::J1)
        .def("J2", &Ising::J2)
        .def("J3", &Ising::J3)

        .def("size_x", &Ising::size_x)
        .def("size_y", &Ising::size_y)

        // set parameters
        .def("set_parameters", &Ising::set_parameters,
             py::arg("temperature"), py::arg("J1"), py::arg("J2"), py::arg("J3"))

        // run the simulation
        .def("run_local", &Ising::run_local, py::arg("Nsample"), py::arg("spacing"))
        .def("run_cluster", &Ising::run_cluster, py::arg("Nsample"), py::arg("spacing"))

        .def("get_energy", [](const Ising &model)
             {
            auto vec = model.get_energy();
            return py::array_t<float>(vec.size(), vec.data()); })
        .def("get_magnetization", [](const Ising &model)
             {
            auto vec = model.get_magnetization();
            return py::array_t<int>(vec.size(), vec.data()); })
        .def("get_afm", [](const Ising &model)
             {
            auto vec = model.get_afm();
            return py::array_t<int>(vec.size(), vec.data()); })
        .def("get_spin", [](const Ising &model)
             {
            auto spin = model.get_spin();
            int rows = spin.size();
            int cols = rows > 0 ? spin[0].size() : 0;
            
            // 2d array creation
            auto arr = py::array_t<int>({rows, cols});
            auto buf = arr.mutable_unchecked<2>();
            
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    buf(i, j) = spin[i][j];
                }
            }
            return arr; })

        // these functions are only used for pickling, do not use them in python
        .def("set_spin", &Ising::set_spin)
        .def("set_collected_data", &Ising::set_collected_data)

        // pickle support
        .def(py::pickle(
            [](const Ising &ising)
            {
                return py::make_tuple(
                    ising.size_x(),
                    ising.size_y(),
                    ising.get_spin(),
                    ising.temperature(),
                    ising.J1(),
                    ising.J2(),
                    ising.J3(),
                    ising.get_energy(),
                    ising.get_magnetization(),
                    ising.get_afm());
            },
            // pickle will store the object in a tuple
            [](py::tuple t) -> Ising
            {
                if (t.size() != 10)
                    throw std::runtime_error("Invalid state!");
                // first reconstruct the object
                Ising ising(t[0].cast<int>(), t[1].cast<int>());
                // reset the spin
                ising.set_spin(t[2].cast<std::vector<std::vector<int>>>());
                // reset parameters
                ising.set_parameters(
                    t[3].cast<float>(),
                    t[4].cast<float>(),
                    t[5].cast<float>(),
                    t[6].cast<float>());
                // reset collected data
                ising.set_collected_data(
                    t[7].cast<std::vector<float>>(),
                    t[8].cast<std::vector<int>>(),
                    t[9].cast<std::vector<int>>());
                return ising;
            }));
}