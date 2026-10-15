function arrayLiteral(a, b) {
  var arr   = [a, b, a + b];
  var empty = [];
  return arr[2];
}
arrayLiteral(3, 4);
