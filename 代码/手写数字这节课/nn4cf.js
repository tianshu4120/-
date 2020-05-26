let xs = [];
let ys = [];

const model = tf.sequential();
// First layer must have an input shape defined.
model.add(tf.layers.dense({
		units: 10,
		inputShape: [2],
		activation: 'sigmoid'
	}));

model.add(tf.layers.dense({
		units: 10,
		activation: 'sigmoid'
	}));
model.add(tf.layers.dense({
		units: 2,
	}));

//model.predict(tf.randomNormal([1, 2])).print();


const Optimizer = tf.train.adam(0.1)
	const config = {
	optimizer: Optimizer,
	loss: tf.losses.softmaxCrossEntropy,
	//loss: tf.losses.meanSquaredError,
};
model.compile(config);

const inputs = tf.tensor2d([[1, 1], [2, 1], [3, 2], [1, 2], [2, 3], [1, 3], [3, 1]]);
//const outputs = tf.tensor2d([[1], [1], [1], [0], [0], [0], [0]]);
const outputs = tf.tensor2d([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1]]);
//const outputs = tf.tensor2d([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]);


function setup() {
	// put setup code here
	createCanvas(400, 400); // create a canvas with size of width*height
	background(0); //set background color as black

}

function draw() {
	model.predict(inputs).print();
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
		//model.predict(inputs).print();
		output_tem = model.predict(inputs);
		//output_tem.print();
		//tf.softmax(output_tem).print();
		tf.add(tf.argMax(output_tem, 1), 1).print();
	});
	noLoop();
}