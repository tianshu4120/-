let mnist;
let inputs = [];
let show = 1;
let train_image;
 
const model = tf.sequential();
/*model.add(tf.layers.dense({
		units: 128,
		inputShape: [784],
		activation: 'relu'
	}));
	
model.add(tf.layers.dense({
		units: 64,
		activation: 'relu'
	}));
	
model.add(tf.layers.dense({
		units: 64,
		activation: 'relu'
	}));
 
model.add(tf.layers.dense({
		units: 10,
	}));*/

model.add(tf.layers.conv2d({
	inputShape: [28, 28, 1],
	kernelSize: 5,
	filters: 16,
	strides: 1,
	activation: 'relu',
	kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({
	poolSize: [2, 2],
	strides: [2, 2]
}));
model.add(tf.layers.conv2d({
	kernelSize: 5,
	filters: 32,
	strides: 1,
	activation: 'relu',
	kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({
	poolSize: [2, 2],
	strides: [2, 2]
}));
model.add(tf.layers.dropout({
	rate:0.5,
}));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({
	units: 128,
	activation: 'relu'
}));
model.add(tf.layers.dense({
	units: 10,
	activation: 'relu'
}));

const OPT = tf.train.adam(0.002);
	const config = {
	optimizer: OPT,
	loss: tf.losses.softmaxCrossEntropy,
};
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

	if (mnist) {
		//train data
		inputs = tf.tensor2d(mnist.train_images.slice(0,60000));
		outputs_org = tf.tensor1d(mnist.train_labels.slice(0,60000));
		outputs = tf.oneHot((outputs_org), 10);

		//test data
		inputs=tf.div(inputs,tf.scalar(255.0));
		inputs=inputs.reshape([60000,28,28,1]);
		/*inputs_test = tf.tensor2d(mnist.test_images);
		outputs_test = tf.tensor1d(mnist.test_labels);*/

		outputs_org.dispose();

		inputs_test = tf.tensor2d(mnist.test_images.slice(0,10000));
		outputs_test = tf.tensor1d(mnist.test_labels.slice(0,10000));

		inputs_test = tf.div(inputs_test,tf.scalar(255.0));
		inputs_test = inputs_test.reshape([10000,28,28,1]);

		async function train() {
			for (let i = 1; i < 50; i++) {
				const h = await model.fit(inputs, outputs, {
						batchSize: 1000,
						epochs: 1
					});
				console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
			}
			const saveResults = await model.save('indexeddb://my-model-5');

		}
		
		train().then(() => {
			//model.predict(inputs).print();
			output_tem = model.predict(inputs_test);
			//output_tem.print();
			//tf.softmax(output_tem).print();
			label = tf.argMax(output_tem, 1);
			//tf.add(label, 1).print();
			tf.div(tf.sum(outputs_test.equal(label)), mnist.test_labels.length).print();
		});
		noLoop();
	}
	
}