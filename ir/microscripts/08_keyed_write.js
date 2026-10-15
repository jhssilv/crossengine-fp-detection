function keyedWrite(arr, i, val) {
  arr[i] = val;
  arr[0] = arr[0] + 1;
  return arr;
}
keyedWrite([0, 0, 0], 1, 42);
