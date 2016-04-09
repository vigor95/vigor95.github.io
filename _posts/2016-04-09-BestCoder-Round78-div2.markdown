---                                                                                                     
layout: post
title: "BestCoder-78-div2解题报告"
date: 2016-04-09
categories: solution
---
# 1001 jrMz and angles 
其实作出直线ax + by = c的图像就可以看出要枚举的范围了。

{% highlight cpp %}
int main() {
    int t;
    scanf("%d", &t);
    while (t--) {
        double m, n;
        scanf("%lf %lf", &m, &n);
        double a = 180 * (m - 2) / m, b = 180 * (n - 2) / n;
        int flag = 0;
        for (int i = 0; i <= 360 / a; i++) {
            for (int j = 0; j <= 360 / b; j++) {
                if (a * i + b * j == 360) {
                    flag = 1;
                    goto judge;
                }
            }
        }
judge:
        printf("%s\n", flag ? "Yes" : "No");
    }
    return 0;
}
Close

{% endhighlight %}

# 1002 Claris and XOR 
高位到低位，若x在这一位可取1，y可取0，好办；若y在这一位可取1，x可取0，同样；若x，y都只能取0或1，那也没得选...

{% highlight cpp %}
int main() {
    int t;
    cin >> t;
    while (t--) {
        ll a, b, c, d;
        cin >> a >> b >> c >> d;
        ll sumx = 0, sumy = 0;
        for (int i = 62; i >= 0; i--) {
            ll tmp = 1;
            tmp <<= i;
            if (tmp + sumx <= b && tmp + sumy > c)
                sumx += tmp;
            else if (tmp + sumy <= d && tmp + sumx > a)
                sumy += tmp;
            else if (tmp + sumx <= b && tmp + sumy <= d) {
                sumx += tmp;
                sumy += tmp;
            }
        }
        cout << (sumx ^ sumy) << endl;
    }
    return 0;
}
{% endhighlight %}
