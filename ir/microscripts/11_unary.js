function unary(a) {
  var neg  = -a;
  var bnot = ~a;
  var lnot = !a;
  var inc  = a + 1;
  var dec  = a - 1;
  return neg + bnot + inc + dec;
}
unary(5);
