//
// This code compresses bin_trace_i.raw files which are uncompressed binary files, into compressed trace_i.raw files
//

#include <iostream>
#include <fstream>
#include <zlib.h>
#include <dirent.h>
#include <vector>
#include <cstring>

#define CHUNK_SIZE 16384

std::vector<std::string> listDirectories(const std::string& path)
{
    std::vector<std::string> directories;
    DIR* dir = opendir(path.c_str());
    if (dir != nullptr)
    {
        dirent* entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            // Check if the entry is a directory (DT_DIR)
            if (entry->d_type == DT_DIR)
            {
                // Ignore "." and ".." directories
                if (std::string(entry->d_name) != "." && std::string(entry->d_name) != "..")
                {
                    directories.push_back(std::string(entry->d_name));
                }
            }
        }
        closedir(dir);
    }
    return directories;
}

int main() {
    DIR *dir;
    struct dirent *ent;
    std::string trace_path = "/fast_data/trace/nvbit/vectormultadd/";
    std::vector<std::string> filenames;
    std::vector<std::string> dirnames = listDirectories(trace_path);
    for (const std::string& ker : dirnames) { // the name of a directory is the same as the name of a kernel
        // std::cout << trace_path + ker + "/" << std::endl;
        if ((dir = opendir((trace_path + ker + "/").c_str())) != NULL) {
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
            std::ifstream input_file(trace_path + ker + "/" + filename, std::ios::binary);
            if (!input_file) {
                std::cerr << "Error opening input file: " << filename << "\n";
                continue;
            }
            filename.erase(0, 4); // remove first 4 characters ("bin_")
            std::string output_filepath = trace_path + ker + "/" + filename;
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
    }
    return 0;
}
