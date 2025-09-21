#include "utils.hpp"

#include <sstream>

std::string gpaw::to_json_entry(const std::unordered_map<std::string, std::string>& dict)
{
    // Quite hacky to do this manually, should prefer an existing lib.
    // But this will suffice for now

    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (auto &kv : dict)
    {
        if (!first)
        {
            oss << ", ";
        }
        first = false;
        oss << "\"" << kv.first << "\": ";

        // Leave unquoted if the value looks numeric. Very crude check
        const bool numeric = !kv.second.empty() &&
                       (isdigit(kv.second[0]) || kv.second[0] == '-' || kv.second[0] == '.');
        if (numeric)
        {
            oss << kv.second;
        }
        else
        {
            oss << "\"" << kv.second << "\"";
        }
    }
    oss << "}";
    return oss.str();
}
