function double_(x) {
  return x * 2;
}
function functionCall(a) {
  var x = double_(a);
  var y = double_(x);
  return y;
}
functionCall(5);
