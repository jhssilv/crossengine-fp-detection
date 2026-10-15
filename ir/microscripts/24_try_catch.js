function tryCatch(x) {
  var result;
  try {
    if (x < 0) throw new Error("negative");
    result = x * 2;
  } catch (e) {
    result = 0;
  }
  return result;
}
tryCatch(5);
tryCatch(-1);
