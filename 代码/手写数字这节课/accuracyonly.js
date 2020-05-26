let mnist;
let inputs = [];
let train_image;
let train_index = 0;
// testing variables
let test_index = 0;
let total_tests = 0;
let total_correct = 0;
let nn;
let user_digit;
let user_has_drawing = false;
let user_guess_ele;
let percent_ele;
function setup() {
	createCanvas(400, 200);
	train_image = createImage(28, 28);
	loadMNIST(function (data) {
		mnist = data;
		//console.log('4444');
		console.log(mnist);
	})
}
function draw() {
	//console.log('4444');
	let inputs = [];
	if (mnist) {
		//test data
		inputs_test = tf.tensor2d(mnist.test_images.slice(0, 10000));
		//inputs_test = tf.div(inputs_test,tf.scalar(255.0));
		inputs_test = inputs_test.reshape([10000, 28, 28, 1]);
		outputs_test = tf.tensor1d(mnist.test_labels.slice(0, 10000));
		print(outputs_test.shape);


		async function test() {

			const model = await tf.loadLayersModel('indexeddb://my-model-5');
			//console.log('4444');
			console.log('Prediction from loaded model:');
			output_tem = model.predict(inputs_test);
			output_tem.print();
			tf.softmax(output_tem).print();
			label = tf.argMax(output_tem, 1);
			tf.add(label, 1).print();
			tf.div(tf.sum(outputs_test.equal(label)), mnist.test_labels.length).print();
		}
		test();
		noLoop();
	}


}

