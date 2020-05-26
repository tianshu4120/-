let mnist;
let inputs = [];
let show = 1;
let train_image;

const model = tf.sequential();
model.add(tf.layers.dense({
    units: 64,
    inputShape: [784],
    activation: 'relu'
}));

model.add(tf.layers.dense({
    units: 10,
}));
const OPT = tf.train.adam(0.001)
const config = {
    optimizer: OPT,
    loss: tf.losses.softmaxCrossEntropy,
}
model.compile(config);


function setup() {
    createCanvas(200, 200);
    train_image = createImage(28, 28);
    loadMNIST(function (data) {
        mnist = data;
        console.log(mnist);
    })
}

train_index = 0;

function draw() {
    //train data
    inputs = tf.tensor2d(mnist.train_images);
    outputs_org = tf.tensor1d(mnist.train_labels);


}
