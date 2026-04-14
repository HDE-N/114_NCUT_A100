# Other 訓練

### bak1 

震幅s 1.2，高斯雜訊n 0.5

### bak2

震幅s 1.2，高斯雜訊n 1.5

### bak3

使用 data_v54，bak1

震幅s 1.2，高斯雜訊n 1.5

測試參數較多

### bak4

使用 data_v54，bak1

與 bak3 相同，只是重新訓練

### bak5

使用 data_v54，bak3

震幅s 1.5，高斯雜訊n 2.5

原始數據小增強，NOISE_STD = 0.3、SCALE_FACTOR = 1.1

### bak6

使用 data_v54，bak3

震幅s 1.2，高斯雜訊n 1.5

原始數據小增強，NOISE_STD = 0.3、SCALE_FACTOR = 1.1

### bak7

使用 data_v54，bak5

震幅s 1.2，高斯雜訊n 2.5

### bak8(test已重跑完)

使用 data_v55，bak1

NOISE_STD = 2.0、SCALE_FACTOR = 1.2

原始數據小增強，NOISE_STD = 0.3、SCALE_FACTOR = 1.1

### bak9(test需要重新跑)

使用 data_v55，bak2

NOISE_STD = 1.5、SCALE_FACTOR = 1.2

原始數據小增強，NOISE_STD = 0.3、SCALE_FACTOR = 1.1


### bak10

bak8 重新訓練

所有參數均與bak8一致

### bak11

使用 data_v55，bak3

NOISE_STD = 1.8、SCALE_FACTOR = 1.2

原始數據小增強，NOISE_STD = 0.3、SCALE_FACTOR = 1.1

### bak12

bak11 重新訓練

所有參數均與bak11一致
