本文应该在 VSCode 上浏览与编辑，其它平台可能会炸 Markdown 或 LaTeX（对此），转载自 [GitHub](https://github.com/Wang-Yile/OI-Notes/blob/main/DP%E7%AC%94%E8%AE%B0.md)。

# 链接

[动态规划初步·各种子序列问题](https://www.luogu.com.cn/blog/pks-LOVING/junior-dynamic-programming-dong-tai-gui-hua-chu-bu-ge-zhong-zi-xu-lie)

[题解 P1352 【没有上司的舞会】](https://www.luogu.com.cn/blog/xky-666/solution-p1352)

# 写在前面

判断性继承思想：下一状态最优值 = 最优比较函数（已经记录的最优值，可以由先前状态得出的最优值）

全部设状态思想：有不确定性（后效性）的状态全部设进 DP 方程。

~~时间复杂度越高的算法越全能，就像 DFS，它什么都能干。~~

**当发现题目变数很多但只需要最优结果时，大胆去动归。（[P2986 [USACO10MAR] Great Cow Gathering G](https://www.luogu.com.cn/problem/P2986)）**

# DP 类别

## 1. 最长上升子序列

### $n^2$ 做法

设 $dp_i$ 表示匹配到第 $i$ 位的最长上子序列长度。

$$dp_i=\max\limits_{1\le j<i}{\begin{cases}dp_j+1&a_i\ge a_j\\1&\text{overwise}\end{cases}}$$

$O(n^2)$ 求解 [B3637 最长上升子序列](https://www.luogu.com.cn/problem/B3637)代码：

```cpp
#include <string.h>

#include <algorithm>
#include <iostream>

using namespace std;

int n;
int a[100005];
int dp[100005];

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    for (int i = 1; i <= n; ++i) cin >> a[i];
    for (int i = 1; i <= n; ++i) {
        dp[i] = 1;
        for (int j = 1; j < i; ++j) {
            if (a[j] < a[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }
    int ans = 0;
    for (int i = 1; i <= n; ++i) {
        ans = max(ans, dp[i]);
    }
    cout << ans << endl;
    return 0;
}
```

### $n\log n$ 做法

设 $dp_i$ 表示长度为 $i$ 的上升子序列的最小末尾数值。

**原理 1**：当前的上升子序列长度如果已确定，那么如果这种长度的子序列的结尾元素越小，后面的元素就可以更方便地加入到这条上升子序列中。

**原理 2**：随着上升子序列长度的增加，其结尾的最小值一定是单调递增的$^{1.1}$。

考虑加入元素 $a_i$（不严格递增）$^{1.2}$：

1. $a_i\ge dp_{len}$，直接将该元素插入到 $dp$ 的末尾。
2. $a_i\le dp_{len}$，找到第一个大于它的元素，用 $a_i$ 替换它。

可以通过初始化极大值避免第一类讨论。

$O(n\log n)$ 求解 [B3637 最长上升子序列](https://www.luogu.com.cn/problem/B3637)代码：

```cpp
#include <string.h>

#include <algorithm>
#include <iostream>

using namespace std;

int n;
int a[100005];
int dp[100005];

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    for (int i = 1; i <= n; ++i) cin >> a[i];
    memset(dp, 0x3f, sizeof dp);  // 初始化成极大值 避免分类讨论
    int mx = dp[0];
    for (int i = 1; i <= n; ++i) {
        *upper_bound(dp + 1, dp + n + 1, a[i]) = a[i];
    }
    int ans = 1;
    while (dp[ans] != mx) ++ans;  // 统计答案
    cout << ans - 1 << endl;
    return 0;
}
```

### 输出最长上升子序列

在转移时维护从什么状态转移而来，倒序输出即可。

### Refrences

[\[1.1\] 最长上升子序列 $O(n\log n)$ 解法](https://blog.csdn.net/shizheng_Li/article/details/105572886)

[\[1.2\] 
动态规划基础 - OI Wiki](https://oi-wiki.org/dp/basic/#%E6%9C%80%E9%95%BF%E4%B8%8D%E4%B8%8B%E9%99%8D%E5%AD%90%E5%BA%8F%E5%88%97)

## 2. 最长公共子序列

### $nm$ 做法

设 $dp_{i,j}$ 表示匹配到 $a$ 的第 $i$ 位，第 $b$ 的第 $j$ 位的最长公共子序列的长度。

$$dp_{i,j}=\max{\begin{cases}dp_{i-1,j-1}+1&a_i=b_i\\dp_{i-1,j}\\dp_{i,j-1}\end{cases}}$$

**初始化**：全部为 $0$。

**答案**：$dp_{n,m}$。

$O(n^2)$ 求解 [P1439 【模板】最长公共子序列](https://www.luogu.com.cn/problem/P1439)代码（朴素 $50$ 分，滚动数组优化有 $60$ 分）：

```cpp
#include <iostream>

using namespace std;

int n;
int a[100005];
int b[100005];
int dp[2][100005];

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    for (int i = 1; i <= n; ++i) cin >> a[i];
    for (int i = 1; i <= n; ++i) cin >> b[i];
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (a[i] == b[j]) {
                dp[i & 1][j] = max(dp[i & 1][j], dp[(i - 1) & 1][j - 1] + 1);
            }
            dp[i & 1][j] = max(dp[i & 1][j], max(dp[(i - 1) & 1][j], dp[i & 1][j - 1]));
        }
    }
    cout << dp[n & 1][n] << endl;
    return 0;
}
```

### $n\log n$ 做法

**前提：两个序列都是 $1,2,3,\dots,n$ 的一个排列。**

**原理**：将最长公共子序列转化为最长上升子序列，用最长上升子序列的 $O(n\log n)$ 算法解。

统计 $a$ 中每个元素在 $b$ 中出现的位置映射到 $mapping$ 数组，对 $mapping$ 数组求最长上升子序列。

$O(n\log n)$ 求解 [P1439 【模板】最长公共子序列](https://www.luogu.com.cn/problem/P1439)代码（朴素 $50$ 分，滚动数组优化有 $60$ 分）：

```cpp
#include <string.h>

#include <iostream>

using namespace std;

int n;
int a[100005];
int b[100005];
int mapping[100005];
int dp[100005];

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    for (int i = 1; i <= n; ++i) {
        cin >> a[i];
        mapping[a[i]] = i;
    }
    for (int i = 1; i <= n; ++i) {
        cin >> b[i];
    }
    memset(dp, 0x3f, sizeof dp);
    int ans = 0;
    dp[0] = 0;
    for (int i = 1; i <= n; ++i) {
        int l = 0, r = ans, mid;
        if (mapping[b[i]] > dp[ans])
            dp[++ans] = mapping[b[i]];
        else {
            while (l < r) {
                mid = (l + r) >> 1;
                if (dp[mid] > mapping[b[i]])
                    r = mid;
                else
                    l = mid + 1;
            }
            dp[l] = min(mapping[b[i]], dp[l]);
        }
    }
    cout << ans;
    return 0;
}
```

### $\dfrac{nm}{w}$ 做法

用位运算优化，在 OI-Wiki 上有引用，自己去看。

## 3. 树形 DP

一般用递归实现。

### 选与不选影响其它

[P1352 没有上司的舞会](https://www.luogu.com.cn/problem/P1352)

**题意**：一棵树，每个点有权值 $a_i$，不能同时选相邻的两个点，最大化选的点的权值和。

因为有负贡献，所以需要 DP。

设 $dp_{u,0/1}$ 表示以 $u$ 为根的子树，$u$ 是否参加舞会时的最优解，$v$ 是 $u$ 的一个儿子。

$$dp_{u,0}=\max{\begin{cases}dp_{u,0}\\dp_{u,0}+dp_{v,1}\\dp_{v,0}\\dp_{v,1}\end{cases}}$$

$$dp_{u,1}=\max{\begin{cases}dp_{u,1}\\dp_{u,1}+dp_{v,0}\\dp_{v,0}\end{cases}}$$

**初始化**：$dp_{u,1}=a_u$

**答案**：所有状态取 $\max$。

求解 [P1352 没有上司的舞会](https://www.luogu.com.cn/problem/P1352)代码：

```cpp
#include <iostream>
#include <vector>

using namespace std;

int n;
vector<int> vec[6005];
int dp[6005][2];

static inline void dfs(int u, int fa) {
    for (auto v : vec[u]) {
        if (v == fa) continue;
        dfs(v, u);
        dp[u][1] = max(dp[u][1], max(dp[u][1]+dp[v][0], dp[v][0]));
        dp[u][0] = max(max(dp[u][0], dp[u][0] + dp[v][1]), max(dp[v][0], dp[v][1]));
    }
}

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    for (int i = 1; i <= n; ++i) cin >> dp[i][1];
    for (int i = 1; i < n; ++i) {
        int u, v;
        cin >> u >> v;
        vec[u].push_back(v);
        vec[v].push_back(u);
    }
    dfs(1, 0);
    int ans = -1e9;
    for (int i = 1; i <= n; ++i) {
        ans = max(ans, max(dp[i][0], dp[i][1]));
    }
    cout << ans << endl;
    return 0;
}
```

### 选与不选不影响其它

是上一种题型的弱化版。

[P1122 最大子树和](https://www.luogu.com.cn/problem/P1122)

**题意**：一棵树，每个点有权值 $a_i$，最大化选的点的权值和。

设 $dp_u$ 表示在 $u$ 的子树中选若干个点的最大权值和，$v$ 是 $u$ 的一个儿子。

$$dp_u=dp_u+\max{\begin{cases}dp_v\\0\end{cases}}$$

**初始化**：$dp_u=a_u$。

**答案**：所有状态取 $\max$。

求解 [P1122 最大子树和](https://www.luogu.com.cn/problem/P1122)代码：

```cpp
#include <iostream>
#include <vector>

using namespace std;

int n;
int a[16005];
vector<int> vec[16005];
int dp[16005];

static inline void dfs(int u, int fa) {
    for (auto v : vec[u]) {
        if (v == fa) continue;
        dfs(v, u);
        dp[u] += max(dp[v], 0);
    }
}

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    for (int i = 1; i <= n; ++i) {
        cin >> dp[i];
    }
    for (int i = 1; i < n; ++i) {
        int u, v;
        cin >> u >> v;
        vec[u].push_back(v);
        vec[v].push_back(u);
    }
    dfs(1, 0);
    int ans = -1e9;  // 出题人给你脸了: ans = 0 只有 90 分
    for (int i = 1; i <= n; ++i) {
        ans = max(ans, dp[i]);
    }
    cout << ans << endl;
    return 0;
}
```

### 换根 DP

在第一次 DFS 时预处理一些信息，基于这些信息，在第二次 DFS 时选择合适的根节点进行 DP。

[P2986 [USACO10MAR] Great Cow Gathering G](https://www.luogu.com.cn/problem/P2986)

**题意**：一棵树，每个点有权值 $a_i$ 和数量 $c_i$，选择一个节点，最小化其它点到它的 $a_i\times c_i$ 之和。

假设每个点都到了节点 $1$，此时节点 $i$ 的贡献为 $w_i$，若选择的节点为 $x$，可以通过加上或减去 $1$ 到 $x$ 的距离完成动态规划，在 $O(n)$ 的时间内求解。

求解 [P2986 [USACO10MAR] Great Cow Gathering G](https://www.luogu.com.cn/problem/P2986)代码：

```cpp
#include <iostream>
#include <vector>

#define int long long

using namespace std;

int n, sum;
int c[100005];
int cnt[100005];
int dis[100005];
int dp[100005];
vector<pair<int, int>> vec[100005];

static inline int dfs1(int u, int fa) {
    int count = 0;
    for (auto [v, w] : vec[u]) {
        if (v == fa) continue;
        int s = dfs1(v, u);
        dis[u] += dis[v] + w * s;
        count += s;
    }
    cnt[u] = c[u] + count;
    return cnt[u];
}

static inline void dfs2(int u, int fa) {
    for (auto [v, w] : vec[u]) {
        if (v == fa) continue;
        dp[v] = dp[u] - cnt[v] * w + (sum - cnt[v]) * w;  // 先处理过去 / 回来的牛, 再向下递归
        dfs2(v, u);
    }
}

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    for (int i = 1; i <= n; ++i) {
        cin >> c[i];
        sum += c[i];
    }
    for (int i = 1; i < n; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        vec[u].push_back(make_pair(v, w));
        vec[v].push_back(make_pair(u, w));
    }
    dfs1(1, 0);
    dfs2(1, 0);
    int ans = 1e9;
    for (int i = 1; i <= n; ++i) {
        ans = min(ans, dp[i]);
    }
    cout << ans + dis[1] << endl;  // 答案要加上最初的距离
    return 0;
}
```

树的直径（DP 解法）$^{3.1}$

设当 $1$ 为根时，每个节点向下所能延伸的最长路径长度 $d_1$ 与次长路径（与最长路径无公共边）长度 $d_2$，那么树的直径就是对于每一个点，该点 $d_1$ + $d_2$ 能取到的值中的最大值。

树形 DP 可以在存在负权边的情况下求解出树的直径（也可以边权加定值，用 4 次 DFS，两次求直径、两次求节点数）。

```cpp
#include <iostream>
#include <vector>

using namespace std;

int n, d = 0;
int d1[10005], d2[10005];
vector<int> vec[10005];

static inline void dfs(int u, int fa) {
    d1[u] = d2[u] = 0;
    for (int v : vec[u]) {
        if (v == fa) continue;
        dfs(v, u);
        int t = d1[v] + 1;
        if (t > d1[u])
            d2[u] = d1[u], d1[u] = t;
        else if (t > d2[u])
            d2[u] = t;
    }
    d = max(d, d1[u] + d2[u]);
}

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    for (int i = 1; i < n; ++i) {
        int u, v;
        cin >> u >> v;
        vec[u].push_back(v);
        vec[v].push_back(u);
    }
    dfs(1, 0);
    cout << d << endl;
    return 0;
}
```

### Refrences

[\[3.1\] 树形 DP 求树的直径 - OI-Wiki](https://oi-wiki.org/graph/tree-diameter/#%E5%81%9A%E6%B3%95-2-%E6%A0%91%E5%BD%A2-dp)

## 4. 背包 DP

### 01 背包 $^{4.1}$

[P2871 [USACO07DEC] Charm Bracelet S](https://www.luogu.com.cn/problem/P2871)

**题意**：有 $n$ 件物品和一个容量为 $m$ 的背包。第 $i$ 件物品的重量是 $w_i$，价值是 $v_i$。将哪些物品装入背包可使这些物品的重量总和不超过背包容量，且价值总和最大。

设 $dp_{i,j}$ 为在只能放前 $i$ 个物品的情况下，容量为 $j$ 的背包所能达到的最大总价值。

考虑转移。假设当前已经处理好了前 $i-1$ 个物品的所有状态，那么对于第 $i$ 个物品，当其不放入背包时，背包的剩余容量不变，背包中物品的总价值也不变，故这种情况的最大价值为 $dp_{i-1,j}$；当其放入背包时，背包的剩余容量会减小 $w_i$，背包中物品的总价值会增大 $v_i$，故这种情况的最大价值为 $dp_{i-1,j-w_i}+v_i$。

所以

$$dp_{i,j}=\max{\begin{cases}dp_{i-1,j}\\dp_{i-1,j-w_i}+d_i\end{cases}}$$

考虑滚动数组优化

$$dp_j=\max{\begin{cases}dp_j\\dp_{j-w_i}+d_i\end{cases}}$$

```cpp
#include <iostream>
using namespace std;

int n, m;
int w[3500];
int d[3500];
int dp[12880];

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> n >> m;
    for (int i = 1; i <= n; ++i) {
        cin >> w[i] >> d[i];
    }
    for (int i = 1; i <= n; ++i) {         // 枚举顺序不能错
        for (int j = m; j >= w[i]; --j) {  // 倒叙枚举不能错
            dp[j] = max(dp[j], dp[j - w[i]] + d[i]);
        }
    }
    cout << dp[m] << endl;
    return 0;
}
```

[P8803 [蓝桥杯 2022 国 B] 费用报销](https://www.luogu.com.cn/problem/P8803)

**题意**：有 $n$ 个物品，要求选出若干个物品使得任意两个物品之间的时间戳之差小于 $k$，和不超过 $m$，求最大价值。

设 $dp_{i,j}$ 表示前 $i$ 个物品占用 $j$ 的空间的最大价值。

本题有些特殊（带限制），$c_i$ 和 $w_i$ 相等，预处理 $g_i$ 表示离它最近的合法的物品的位置。

$$dp_{i,j}=\max(dp_{i-1,j},dp_{g_i,j-c_i}+c_i)$$

```cpp
#include <algorithm>
#include <iostream>
#include <utility>

using namespace std;

int n, m, k, ans;

pair<int, int> a[1005];
int f[1005][5005];
int g[1005];

const int days[]{0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};

static inline int date2day(int month, int day) { return days[month] + day; }

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    cin >> n >> m >> k;
    for (int i = 1; i <= n; ++i) {
        int mm, d;
        cin >> mm >> d >> a[i].first;
        a[i].second = date2day(mm, d);
    }
    sort(a + 1, a + n + 1, [](const pair<int, int> &x, const pair<int, int> &y) { return x.second < y.second; });
    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (a[i].second - a[j].second >= k) {
                g[i] = j;
            }
        }
    }
    for (int i = 1; i <= n; ++i) {
        for (int j = m; j >= a[i].first; --j) {
            f[i][j] = f[i - 1][j];
            f[i][j] = max(f[i][j], f[g[i]][j - a[i].first] + a[i].first);
        }
    }
    cout << f[n][m] << endl;
    return 0;
}
```

### 完全背包

### Refrences

[\[4.1\] 背包 DP - OI Wiki](https://oi-wiki.org/dp/knapsack/)

# 例题

[P9743 「KDOI-06-J」旅行](https://www.luogu.com.cn/problem/P9743)

**题意**：略。

[](https://www.luogu.com.cn/blog/_post/652280)

设 $dp_{x,y,c,i,j}$ 表示走到 $(x,y)$，花了 $c$ 元，还有 $(i,j)$ 张两公司的票。

容易写出这样的方程（记 $c'=c-a_{i,j}\times a-b_{i,j}\times b$）

$$dp_{x,y,c,i,j}=\sum\limits_{a=0}^{i}{\sum\limits_{b=0}^{j}{dp_{x-1,y,c',i-a+1,j-b}+dp_{x,y-1,c',i-a,j-b+1}}}$$

未完待续

# 版权声明

本文使用 GPL 协议，原文发布在 [GitHub](https://github.com/Wang-Yile/OI-Notes/blob/main/DP%E7%AC%94%E8%AE%B0.md)，以下是本文的版权声明，转载请依然使用 GPL 协议。

```
    Dynamic Programming Notes  动态规划笔记
    Copyright (C) 2023  王一乐

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
