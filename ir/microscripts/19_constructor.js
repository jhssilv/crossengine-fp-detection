function Point(x, y) {
  this.x = x;
  this.y = y;
}
function constructor_(a, b) {
  var p   = new Point(a, b);
  var arr = new Array(3);
  return p.x + p.y;
}
constructor_(3, 4);
