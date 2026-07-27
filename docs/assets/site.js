// Shared top nav for vq-bench.com. Self-contained IIFE — leaks no globals, so
// the landing page's own inline scripts (COLORS/FIELDS/REFS/…) are unaffected.
// A standalone `vqb view` export never loads this file, so the nav is simply
// absent and the embedded dashboard stands on its own.
(function(){
  const LINKS = [
    {file:"index.html", label:"home"},
    {file:"docs.html",  label:"docs"},
  ];
  const GITHUB = "https://github.com/pinecone-io/vq-bench";

  function currentFile(){
    const f = location.pathname.split("/").pop();
    return f && f.length ? f : "index.html";
  }

  function buildNav(){
    const here = currentFile();
    const nav = document.createElement("nav");
    nav.id = "nav";
    for(const l of LINKS){
      const a = document.createElement("a");
      a.className = "lnk" + (l.file===here ? " active" : "");
      a.href = l.file === "index.html" ? "./" : l.file;
      a.textContent = l.label;
      nav.appendChild(a);
    }
    const spacer = document.createElement("span"); spacer.className = "spacer"; nav.appendChild(spacer);
    const gh = document.createElement("a");
    gh.className = "gh"; gh.href = GITHUB; gh.target = "_blank"; gh.rel = "noopener";
    gh.innerHTML = `<svg class="ghmark" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 012-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>vq-bench`;
    nav.appendChild(gh);
    document.body.insertAdjacentElement("afterbegin", nav);
  }

  if(document.readyState === "loading") document.addEventListener("DOMContentLoaded", buildNav);
  else buildNav();
})();
