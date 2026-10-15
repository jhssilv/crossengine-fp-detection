function objectLiteral(a, b) {
  var obj   = { x: a, y: b, sum: a + b };
  var empty = {};
  return obj.sum;
}
objectLiteral(3, 4);
