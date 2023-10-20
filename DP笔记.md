提示：本文在本地 VSCode 上编辑，一些洛谷不支持的 Markdown 特性可能会炸（例如文中导航、内嵌 HTML 等，我是不会修的，你谷赶紧支持这些特性吧！）。

## 链接

[动态规划初步·各种子序列问题](https://www.luogu.com.cn/blog/pks-LOVING/junior-dynamic-programming-dong-tai-gui-hua-chu-bu-ge-zhong-zi-xu-lie)

## 思想

判断性继承思想：下一状态最优值 = 最优比较函数（已经记录的最优值，可以由先前状态得出的最优值）

全部设状态思想：有不确定性（后效性）的状态全部设进 DP 方程。

~~时间复杂度越高的算法越全能，就像 DFS，它什么都能干。~~

**当发现题目变数很多但只需要最优结果时，大胆去动归。（[P2986 [USACO10MAR] Great Cow Gathering G](https://www.luogu.com.cn/problem/P2986)）**

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

本文使用 GPL 协议，以下本文的版权声明，转载请依然使用 GPL 协议。

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
