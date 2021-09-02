# Fluxなど必要なライブラリをインポート
using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated

# MNISTデータを取得

# 訓練データ
train_images = Flux.Data.MNIST.images(:train)
train_labels = Flux.Data.MNIST.labels(:train)

# テストデータ
test_images = Flux.Data.MNIST.images(:test)
test_labels = Flux.Data.MNIST.labels(:test)

# 28×28の二次元画像を784の一次元に変換
X_train = hcat(float.(reshape.(train_images, :))...) 

# ワンホットエンコーディング
Y_train = onehotbatch(train_labels, 0:9)

# モデル作成 784->32->relu->10->softmax
model = Chain(Dense(28^2, 32, relu), Dense(32, 10), softmax)

# 損失関数、最適化、データセット、精度計算
loss(x, y) = crossentropy(model(x), y) 
optim = ADAM()
dataset = repeated((X_train,Y_train),200) 
evalcb = () -> @show(loss(X_train, Y_train)) 
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# 訓練
# 結構時間かかる
Flux.train!(loss, params(model), dataset, optim, cb = throttle(evalcb, 10));

# 精度検証
test_X = hcat(float.(reshape.(test_images, :))...)
test_Y = onehotbatch(test_labels, 0:9)

@show accuracy(test_X, test_Y)