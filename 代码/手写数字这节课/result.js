let mnist;
let inputs = [];
let outputs = [];
let show = 1;
let userinput;
let flag = false;
let usergass;

function setup(){
    createCanvas(200, 200).parent('container');
    userinput = createGraphics(200, 200);
    usergass = select('#user_guess');

    loadMNIST(function(data) {
        mnist = data;
    });
}



function draw() {
    background(0);

    if (mnist) {

        async function test() {
            const model = await tf.loadLayersModel('indexeddb://my-model-5');
            let img = userinput.get();
            if(!flag) {
                return img;
            }

            let inputs = [];
            img.resize(28, 28);
            img.loadPixels();

            for (let i = 0; i < 784; i++) {
                inputs[i] = img.pixels[i * 4];
            }
            inputs = tf.tensor2d([inputs]);
            inputs = inputs.reshape([1,28,28,1]);
            let prediction = model.predict(inputs);
            let guess = tf.argMax(prediction,1);
            usergass.html(guess.dataSync());
            return img;

        }
        test();
    }

    image(userinput, 0, 0);

    if (mouseIsPressed) {
        flag = true;
        userinput.stroke(255);
        userinput.strokeWeight(16);
        userinput.line(mouseX, mouseY, pmouseX, pmouseY);
    }
}

function keyPressed() {
    if (key == ' ') {
        flag = false;
        user_digit.background(0);
    }
}