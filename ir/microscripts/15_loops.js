function loops(n) {
  var sum = 0;
  for (var i = 0; i < n; i++) {
    sum = sum + i;
  }
  var j = 0;
  while (j < n) {
    sum = sum + j;
    j++;
  }
  return sum;
}
loops(10);
