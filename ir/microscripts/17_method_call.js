function methodCall(arr) {
  var len    = arr.length;
  var joined = arr.join(",");
  var sliced = arr.slice(0, 2);
  var pushed = arr.push(99);
  return joined;
}
methodCall([1, 2, 3]);
