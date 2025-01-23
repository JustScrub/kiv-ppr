#include "conf_loader.hpp"

size_t Conf::CHUNK_SIZE = 1024;
size_t Conf::SIZE_INCR_STEPS = 10;
unsigned Conf::NUM_REPS = 10;
size_t Conf::CHUNK_CAPACITY = 1024/sizeof(float);
unsigned Conf::MODES = 31;
int Conf::NUM_THREADS = -1;
std::string Conf::plots_dir = "plots\\";
std::string Conf::kernels_file = "kernels.cl";

const size_t Conf::VEC_REG_CAP = 8;
std::string Conf::mode_names[] = {"seq_ser", "seq_par", "vec_ser", "vec_par", "GPU", "GPU_nosort"};

int Conf::load_conf(const std::string& conf_path){
        std::ifstream conf_file(conf_path);
        if(!conf_file.is_open()){
            std::cerr << "Failed to open config file: " << conf_path << std::endl;
            return 0;
        }
        std::string line;
        while(std::getline(conf_file, line)){
            if(line[0] == '#') continue;
            size_t pos = line.find('=');
            if(pos == std::string::npos) continue;
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos+1);
            if(key == "CHUNK_SIZE"){
                Conf::CHUNK_SIZE = std::stoul(value);
                Conf::CHUNK_CAPACITY = Conf::CHUNK_SIZE / sizeof(float);
            } else if(key == "DATA_LEN_STEPS"){
                Conf::SIZE_INCR_STEPS = std::stoul(value);
            } else if(key == "MED_REPS"){
                Conf::NUM_REPS = std::stoul(value);
            } else if(key == "MODES"){
                Conf::MODES = std::stoul(value);
            } else if(key == "NUM_THREADS"){
                Conf::NUM_THREADS = std::stoi(value);
            } else if(key == "PLOTS_OUTPUT"){
                Conf::plots_dir = value;
                //trailing backslash
                if (Conf::plots_dir.back() != '\\')
                    Conf::plots_dir += '\\';
            } else if(key == "CL_KERNELS"){
                Conf::kernels_file = value;
            }
        }
        if(Conf::NUM_THREADS <= 0) {
            Conf::NUM_THREADS = omp_get_max_threads();
        }
        return 0;
    }