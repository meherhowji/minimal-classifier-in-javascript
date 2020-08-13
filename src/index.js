import React from "react";
import ReactDOM from "react-dom";
import { Stage, Layer, Rect, Line, Group, Circle } from "react-konva";

import "./styles.css";

const find = [12, 4];
const epoch = [];

const trainingData = [
  [4, 8, 0],
  [4, 2, 0],
  [5, 7, 0],
  [7, 4, 0],
  [9, 9, 1],
  [7, 10, 1],
  [10, 12, 1],
  [3, 12, 1]
];

const getRandomWeights = () =>
  Array.from({ length: 2 }, () => Math.random() * 0.5 - 0.2);
const getRandomBias = () => Math.random() * 0.5 - 0.2;
const getRandomIndex = (till) => Math.floor(Math.random() * till);

const getInput = (data, index) => data[index];
const getTarget = (input) => [...input].pop();

const getWeightedInputSum = (input, weight, bias) =>
  weight[0] * input[0] + weight[1] * input[1] + bias;
const sigmoid = (sum) => 1 / (1 + Math.exp(-sum));

const compose = (f, g) => (arg) => f(g(arg));
const identity = (i) => i;

const getError = (pred, target) => (pred - target) ** 2;

const combineRateOfChange = (a, b, c) => a * b * c;
const withLearningRate = (weight) => combineRateOfChange(0.1, weight, 1);

const trainModel = (data) => {
  let weights = getRandomWeights();
  let bias = getRandomBias();

  for (let i = 0; i < 50000; i++) {
    const randomIndex = getRandomIndex(data.length);

    const input = getInput(data, randomIndex);
    const target = getTarget(input);

    const weightedSum = getWeightedInputSum(weights, input, bias);
    const weightedSumWRTweight0 = input[0];
    const weightedSumWRTweight1 = input[1];
    const weightedSumWRTbias = 1;

    const prediction = compose(sigmoid, identity)(weightedSum);
    const predictionWRTweightedSum = prediction * (1 - prediction);

    const error = getError(prediction, target);
    const errorWRTprediction = 2 * (prediction - target);

    // Δm = E / x
    let errorWRTweight0 = combineRateOfChange(
      errorWRTprediction,
      predictionWRTweightedSum,
      weightedSumWRTweight0
    );
    let errorWRTweight1 = combineRateOfChange(
      errorWRTprediction,
      predictionWRTweightedSum,
      weightedSumWRTweight1
    );
    let errorWRTbias = combineRateOfChange(
      errorWRTprediction,
      predictionWRTweightedSum,
      weightedSumWRTbias
    );

    weights[0] -= withLearningRate(errorWRTweight0);
    weights[1] -= withLearningRate(errorWRTweight1);
    bias -= withLearningRate(errorWRTbias);

    epoch.push([...weights, bias]);
  }

  return [...weights, bias];
};

const Graph = () => (
  <Rect
    x={0}
    y={0}
    width={300}
    height={300}
    fillLinearGradientStartPoint={{ x: 0, y: 300 }}
    fillLinearGradientEndPoint={{ x: 300, y: 300 }}
    fillLinearGradientColorStops={[0, "#343341", 0.5, "#2c2c38", 1, "#343341"]}
  />
);

const Grid = () => (
  <Group>
    <Line points={[150, 300, 150, 0]} stroke="#66677a30" strokeWidth="1" />
    <Line points={[0, 150, 300, 150]} stroke="#66677a30" strokeWidth="1" />
  </Group>
);

const DrawTrainingData = ({ data }) => {
  return (
    <Group>
      {data.map((coord, key) => (
        <Circle
          key={key}
          radius={5}
          x={coord[0] * 20}
          y={coord[1] * 20}
          fill="#32323e"
          stroke={coord[2] ? "#3aa4ff" : "#a36fff"}
          strokeWidth={3}
        />
      ))}
    </Group>
  );
};

const generateClassifier = (trueWeights = []) => {
  const arr = Array.from({ length: 61 }, (v, k) => k * 5);
  return arr.map((x, key) =>
    arr.map((y, key) => {
      let sig = sigmoid(
        (x / 20) * trueWeights[0] + (y / 20) * trueWeights[1] + trueWeights[2]
      );
      let sc = sig > 0.5 ? 1 - sig : sig;
      return (
        <Circle
          key={key}
          radius={0.9}
          x={x}
          y={y}
          opacity={sc}
          fill={"#f4f4f4"}
        />
      );
    })
  );
};

class App extends React.Component {
  constructor() {
    super();
    this.state = {
      p: 0
    };

    this.finalOutput = trainModel(trainingData);
  }

  componentDidMount() {
    this.animateClassier();
  }

  animateClassier() {
    if (this.state.p > 50000) return;
    let pp = this.state.p + 100;
    this.setState({ p: pp });
    this.myvar = setTimeout(() => {
      this.animateClassier();
    }, 50);
  }

  componentWillUnmount() {
    clearTimeout(this.myvar);
  }

  render() {
    let winW = window.innerWidth;
    let winH = window.innerHeight;
    let x = winW / 2 - 150;
    let y = winH / 2 - 150;

    return (
      <Stage x={x} y={y} width={winW} height={winH}>
        <Layer>
          <Graph />
          <Grid />
          <DrawTrainingData data={trainingData} />
          <Circle
            radius={5}
            x={find[0] * 20}
            y={find[1] * 20}
            fill="#555"
            stroke={"#ffd16f"}
            strokeWidth={4}
          />
        </Layer>

        <Layer>{generateClassifier(epoch[this.state.p])}</Layer>
      </Stage>
    );
  }
}

ReactDOM.render(<App />, document.getElementById("root"));
