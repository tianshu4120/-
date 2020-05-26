xs = [];
ys = [];

/* const model = tf.sequential();
// First layer must have an input shape defined.
model.add(tf.layers.dense({
units: 1, //output size
inputShape: [2]// input size
})); */
//model.predict(tf.randomNormal([1,2])).print()
/**/
const model = tf.sequential();
model.add(tf.layers.dense({
    units: 10,
    inputShape: [2],
    activation: 'relu'
}));
model.add(tf.layers.dense({
    units: 10,
    activation: 'relu'
}));
model.add(tf.layers.dense({
    units: 1,
}));

const Optimizer = tf.train.adam(0.01);
const config = {
    optimizer: Optimizer,
    loss: tf.losses.meanSquaredError,
};
model.compile(config);

const inputs = tf.tensor2d([[1, 1],[2, 1],[3, 2], [1, 2],[2, 3],[1, 3],[3, 1]]);
const outputs = tf.tensor2d([[1], [1], [1], [0], [0], [0], [0]]);

function mousePressed() {
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 0, 1);
    xs.push([x, 1]);
    ys.push(y);
    console.log(xs);
}

function setup() {
    // put setup code here
    createCanvas(400, 400); // create a canvas with size of width*height
    background(0); //set background color as black
}
function draw() {

    background(100, 0, 100);
    stroke(1, 200, 100);
    strokeWeight(15);

    for (let i = 0; i < xs.length; i++) {
        let x = map(xs[i][0], 0, 1, 0, width);
        let y = map(ys[i], 0, 1, 0, height);
        point(x, y);
    }

    if (xs.length >= 10) {
        const inputs = tf.tensor2d(xs);
        const outputs = tf.tensor1d(ys);

        async function train() {
            for (let i = 1; i < 500; i++) {
                const h = await model.fit(inputs, outputs, {
                    batchSize: 100,
                    epochs: 1
                });
                console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
            }
        }

        train().then(() => {
            /* 		linex = [[0, 1], [1, 1]];
            tflinex = tf.tensor2d(linex);
            output_tem = model.predict(tflinex);
            output_tem = output_tem.dataSync();
            x1 = map(linex[0][0], 0, 1, 0, width);
            x2 = map(linex[1][0], 0, 1, 0, width);
            y1 = map(output_tem[0], 0, 1, 0, height);
            y2 = map(output_tem[1], 0, 1, 0, height);
            strokeWeight(5);
            line(x1, y1, x2, y2); */

           /* let linex = [];
            for (let x = 0; x < 1; x += 0.01) {
                linex.push([x, 1]);
            }
            tflinex = tf.tensor2d(linex);
            output_tem = model.predict(tflinex);
            output_tem = output_tem.dataSync();
            beginShape();
            noFill();
            stroke(255);
            strokeWeight(5);
            for (let i = 0; i < linex.length; i++) {
                let x = map(linex[i][0], 0, 1, 0, width);
                let y = map(output_tem[i], 0, 1, 0, height);
                vertex(x, y);
            }
            endShape();*/
            model.predict(inputs).print();


        });
        noLoop();
    }
}