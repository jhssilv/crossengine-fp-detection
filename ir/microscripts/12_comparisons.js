function comparisons(a, b) {
  var eq  = a == b;
  var seq = a === b;
  var lt  = a < b;
  var gt  = a > b;
  var lte = a <= b;
  var gte = a >= b;
  return eq || seq || lt || gt || lte || gte;
}
comparisons(3, 5);
