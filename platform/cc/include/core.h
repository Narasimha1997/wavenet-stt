#ifdef __CORE_H
#define __CORE_H

#include <iostream>
#include <vector>
#include <string>

class IndexMapper {
    private :
        char * english_index;
    public:
        IndexMapper();
        std::vector<std::string> map_indexes(std::vector<std::vector<int>> index_array);
}

#endif