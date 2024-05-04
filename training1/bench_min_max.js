// array-minmax-optimized.js

// Add min and max class variables and methods to Array prototype
Array.prototype.min = Infinity;
Array.prototype.max = -Infinity;

const originalPush = Array.prototype.push;
Array.prototype.push = function(...items) {
  for (const item of items) {
    if (typeof item === 'number') {
      if (item < this.min) {
        this.min = item;
      }
      if (item > this.max) {
        this.max = item;
      }
    }
  }
  return originalPush.apply(this, items);
};

// Generate a large array of random numbers
const generateRandomArray = (size) => {
  const arr = [];
  for (let i = 0; i < size; i++) {
    arr.push(Math.floor(Math.random() * 1000));
  }
  return arr;
};

// Benchmark function
const benchmark = (func, label) => {
  const startTime = process.hrtime.bigint();
  const result = func();
  const endTime = process.hrtime.bigint();
  const executionTime = (endTime - startTime) // BigInt(1000000); // Convert to milliseconds
  console.log(`${label} - Execution time: ${executionTime}ns`);
  return result;
};

// Example usage and benchmarking
const numbers = generateRandomArray(100000000); // Array with 10 million elements

const minValue = benchmark(() => {
  return numbers.min;
}, 'Min');
console.log(`Minimum value: ${minValue}`);

const maxValue = benchmark(() => {
  return numbers.max;
}, 'Max');
console.log(`Maximum value: ${maxValue}`);