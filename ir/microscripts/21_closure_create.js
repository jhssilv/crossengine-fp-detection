function closureCreate(factor) {
  var multiplier = function(n) {
    return n * factor;
  };
  return multiplier(3);
}
closureCreate(10);
