MathJax.Hub.Config({
  TeX: {
    Macros: {
      ket: ["\\left| #1 \\right\\rangle",1],
      bra: ["\\left\\langle #1 \\right|",1],
      braket: ["\\left\\langle #1 | #2 \\right\\rangle",2]
    }
  }
});
MathJax.Ajax.loadComplete("[MathJax]/config/local/local.js");
