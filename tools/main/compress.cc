#include <iostream>
#include <fstream>
#include <zlib.h>
#include <dirent.h>
#include <vector>
#include <cstring>

#define CHUNK_SIZE 16384

int main() {
    DIR *dir;
    struct dirent *ent;
    std::vector<std::string> filenames;
    if ((dir = opendir("./tools/main/trace")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (std::string(ent->d_name).find("bin_trace_") == 0 &&
                std::string(ent->d_name).find(".raw") == std::string(ent->d_name).size() - 4) {
                filenames.push_back(ent->d_name);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error opening directory\n";
        return 1;
    }
    for (auto filename : filenames) {
        std::ifstream input_file("./tools/main/trace/" + filename, std::ios::binary);
        if (!input_file) {
            std::cerr << "Error opening input file: " << filename << "\n";
            continue;
        }

        filename.erase(0, 4); // remove first 4 characters ("bin_")
        std::string output_filepath = "./tools/main/trace/" + filename;
        gzFile output_file = gzopen(output_filepath.c_str(), "wb");
        if (output_file == NULL) {
            std::cerr << "Error opening output file: " << output_filepath << "\n";
            continue;
        }

        unsigned char buffer[CHUNK_SIZE];
        int bytes_read;
        while ((bytes_read = input_file.read(reinterpret_cast<char*>(buffer), CHUNK_SIZE).gcount()) > 0) {
            int bytes_written = gzwrite(output_file, buffer, bytes_read);
            if (bytes_written == 0) {
                std::cerr << "Error writing to output file: " << output_filepath << "\n";
                break;
            }
        }

        gzclose(output_file);
        input_file.close();
    }
    return 0;
}
