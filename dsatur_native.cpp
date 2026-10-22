// Optional exact graph-colouring backend; no third-party dependencies.
// Build: c++ -O3 -std=c++17 -shared -fPIC dsatur_native.cpp -o dsatur_native.so
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace {
struct Search {
    int n, k;
    std::vector<std::vector<int>> adj;
    std::vector<int> col, count, degree;
    std::vector<uint64_t> mask;
    std::chrono::steady_clock::time_point end;
    uint64_t nodes = 0;
    bool timed_out = false;

    Search(int nn, int kk, double seconds)
        : n(nn), k(kk), adj(n), col(n, -1), count(n*k, 0),
          degree(n, 0), mask(n, 0) {
        end = seconds > 0 ? std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(seconds))
            : std::chrono::steady_clock::time_point::max();
    }
    bool dfs(int remaining, int used) {
        if (!remaining) return true;
        if ((nodes++ & 1023) == 0 && std::chrono::steady_clock::now() >= end) {
            timed_out = true;
            return false;
        }
        int v = -1, bs = -1, bd = -1;
        for (int u = 0; u < n; ++u) if (col[u] == -1) {
            int s = __builtin_popcountll(mask[u]);
            if (s >= k) return false;
            if (s > bs || (s == bs && degree[u] > bd)) {
                bs = s; bd = degree[u]; v = u;
            }
        }
        // Colours not yet used are equivalent: try only the first one.
        int nc = std::min(k, used + 1);
        for (int c = 0; c < nc; ++c) if (!(mask[v] & (uint64_t(1) << c))) {
            col[v] = c;
            for (int u : adj[v]) if (col[u] == -1) {
                if (++count[u*k+c] == 1) mask[u] |= uint64_t(1) << c;
                --degree[u];
            }
            if (dfs(remaining-1, std::max(used, c+1))) return true;
            for (int u : adj[v]) if (col[u] == -1) {
                if (--count[u*k+c] == 0) mask[u] &= ~(uint64_t(1) << c);
                ++degree[u];
            }
            col[v] = -1;
            if (timed_out) return false;
        }
        return false;
    }
};
}

// Returns 1=colourable, 0=not colourable, -1=timeout, -2=invalid input.
extern "C" int graph_color(int n, int k, int ne, const int* edges,
                           double seconds, int* colors) {
    if (n < 0 || k < 1 || k > 63 || ne < 0 || n > 10000) return -2;
    try {
        Search s(n, k, seconds);
        for (int i = 0; i < ne; ++i) {
            int u = edges[2*i], v = edges[2*i+1];
            if (u < 0 || v < 0 || u >= n || v >= n || u == v) return -2;
            s.adj[u].push_back(v); s.adj[v].push_back(u);
        }
        for (int u = 0; u < n; ++u) {
            std::sort(s.adj[u].begin(), s.adj[u].end());
            s.adj[u].erase(std::unique(s.adj[u].begin(), s.adj[u].end()), s.adj[u].end());
            s.degree[u] = int(s.adj[u].size());
        }
        // Vertices peeled outside the k-core can always be coloured later.
        std::vector<int> queue, removed;
        for (int v=0; v<n; ++v) if (s.degree[v] < k) queue.push_back(v);
        for (size_t i=0; i<queue.size(); ++i) {
            int v=queue[i];
            if (s.col[v] == -2) continue;
            s.col[v]=-2; removed.push_back(v);
            for (int u:s.adj[v]) if (s.col[u] != -2) {
                if (--s.degree[u] < k) queue.push_back(u);
            }
        }
        bool sat=s.dfs(n-int(removed.size()), 0);
        if (s.timed_out) return -1;
        if (!sat) return 0;
        for (auto it=removed.rbegin(); it!=removed.rend(); ++it) {
            int v=*it;
            uint64_t forbidden=0;
            for (int u:s.adj[v]) if (s.col[u]>=0) forbidden |= uint64_t(1)<<s.col[u];
            for (int c=0;c<k;++c) if (!(forbidden & (uint64_t(1)<<c))) {
                s.col[v]=c; break;
            }
            if (s.col[v]<0) return -2;
        }
        std::copy(s.col.begin(),s.col.end(),colors);
        return 1;
    } catch (...) { return -2; }
}
