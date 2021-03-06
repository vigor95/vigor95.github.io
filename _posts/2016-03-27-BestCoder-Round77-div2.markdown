---                                                                                                     
layout: post
title: "BestCoder-77-div2解题报告"
date: 2016-03-27
categories: solution
---
# 1001 so easy
排列组合题。每个数将出现2^(n-1)次。

{% highlight cpp %}
int main() {
    int t;
    scanf("%d", &t);
    while (t--) {
        int n;
        scanf("%d", &n);
        int a[1005];
        for (int i = 0; i < n; i++) {
            scanf("%d", &a[i]);
        }
        int res = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 1; j <= (int)pow((double)2, (double)n - 1); j++) {
                res ^= a[i];
            }
        }
        printf("%d\n", res);
    }
}
{% endhighlight %}

# 1002 xiaoxin juju needs help
如果串长度为奇数，中点只有一个，因此至多只能有一个字符出现奇数次。
如果串长为偶数，干脆不能有字符出现奇数次。
只考虑一半，问题转化为往n/2个格子里填Σa[i]/2(i=1...26)个字符。很简单的排列组合。

{% highlight cpp %}
long long c[1005][1005];

void com() {
    for (int i = 0; i <= 501; i++) c[i][0] = 1;
    for (int i = 1; i <= 501; i++)
        for (int j = 1; j <= i; j++)
            c[i][j] = (c[i-1][j-1] + c[i-1][j]) % mod;
}

int main() {
    com();
    int t;
    scanf("%d", &t);
    getchar();
    while (t--) {
        char s[1005];
        scanf("%s", s);
        int nums[1005];
        memset(nums, 0, sizeof(nums));
        int n = strlen(s);
        for (int i = 0; i < n; i++) {
            nums[s[i]-'a'+1]++;
        }
        int flag = 0;
        for (int i = 1; i <= 26; i++) {
            if (nums[i] & 1) {
                flag++;
                nums[i] = 0;
            }
            else nums[i] /= 2;
        }
        ll res = 1;
        for (int i = 1; i <= 26; i++) {
            res = (res * c[n/2][nums[i]]) % mod;
            n -= nums[i] * 2;
        }
        if (flag >= 2) res = 0;
        printf("%lld\n", res);
    }

    return 0;
}
{% endhighlight %}


# 1003 India and China Origins
二分答案，每次dfs，看能否到达对面。

{% highlight cpp %}
struct Point {
    int x, y;
} ptr[MAXN * MAXN];

int n, m, q;
char s[MAXN][MAXN], tmp[MAXN][MAXN];
int dx[] = {-1, 1, 0, 0}, dy[] = {0, 0, -1, 1};
bool flag[MAXN][MAXN];

void dfs(int x, int y) {
    flag[x][y] = 1;
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i], ny = y + dy[i];
        if (0 <= nx && nx < n && 0 <= ny && ny < m)
            if (s[nx][ny] == '0' && !flag[nx][ny])
                dfs(nx, ny);
    }
}

int f(int year) {
    if (year) s[ptr[year].x][ptr[year].y] = '1';
    memset(flag, 0, sizeof(flag));
    for (int i = 0; i < m; i++)
        if (s[0][i] == '0')
            dfs(0, i);

    int ok = 0;
    for (int i = 0; i < m; i++)
        if (s[n-1][i] == '0' && flag[n-1][i]) ok = 1;

    return ok;
}

int main() {
    int t;
    scanf("%d", &t);
    while (t--) {
        scanf("%d %d", &n, &m);
        for (int i = 0; i < n; i++) scanf("%s", s[i]);
        scanf("%d", &q);
        for (int i = 1; i <= q; i++) scanf("%d %d", &ptr[i].x, &ptr[i].y);
        int res = -1;
        for (int i = 0; i <= q; i++) {
            if (!f(i)) {
                res = i;
                break;
            }
        }
        printf("%d\n", res);
    }
    return 0;
}
{% endhighlight %}

# 1004 Bomber Man wants to bomb an Array.
设dp[i]为前i个格子的最大总破坏指数，可以枚举所有j...i间有恰有一个炸弹的j，有：dp[i] = max(dp[i] * (i - j))。
{% highlight cpp %}
int main() {
    int t;
    scanf("%d", &t);
    while (t--) {
        scanf("%d %d", &n, &m);
        int a[2005], sum[2005];
        memset(a, 0, sizeof(a));
        memset(sum, 0, sizeof(sum));
        for (int x, i = 1; i <= m; i++) {
            scanf("%d", &x);
            a[min(x, n - 1) + 1] = 1;
        }
        long double dp[2005];
        memset(dp, 0, sizeof(dp));
        for (int i = 1; i <= n; i++) {
            sum[i] = sum[i-1] + a[i];
            for (int j = 0; j < i; j++)
                if (sum[i] - sum[j] == 1)
                    dp[i] = max(dp[i], dp[j] + log(double(i-j)) / log(2.0));
        }
        printf("%lld\n", (ll)floor(dp[n] * 1e6));
    }
    return 0;
}
{% endhighlight %}
