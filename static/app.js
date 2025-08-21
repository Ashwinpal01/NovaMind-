const $ = (q) => document.querySelector(q);

const messagesEl = $("#messages");
const chatForm = $("#chatForm");
const chatInput = $("#chatInput");
const uploadForm = $("#uploadForm");
const fileInput = $("#fileInput");
const fileStatus = $("#fileStatus");
const resetBtn = $("#resetBtn");

function addMessage(role, content){
  const item = document.createElement("div");
  item.className = `message ${role}`;
  item.innerHTML = `
    <div class="avatar">${role === "user" ? "🧑" : "🤖"}</div>
    <div class="bubble">${escapeHtml(content)}</div>
  `;
  messagesEl.appendChild(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addLoader(){
  const item = document.createElement("div");
  item.className = "message ai";
  item.innerHTML = `<div class="avatar">🤖</div><div class="bubble"><span class="loader"></span></div>`;
  messagesEl.appendChild(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return item;
}

function escapeHtml(s){
  return s.replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));
}

// Upload
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = fileInput.files[0];
  if(!f){ alert("Choose a file first."); return; }

  const form = new FormData();
  form.append("file", f);

  fileStatus.textContent = "Uploading…";
  const res = await fetch("/upload", { method:"POST", body: form });
  const data = await res.json();

  if(data.ok){
    fileStatus.textContent = `Loaded: ${data.filename} (${data.chunk_count} chunks)`;
    addMessage("ai", `I loaded **${data.filename}**. Ask me anything about it.`);
  }else{
    fileStatus.textContent = "Upload failed";
    alert(data.error || "Upload error");
  }
});

// Chat
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if(!text) return;

  addMessage("user", text);
  chatInput.value = "";
  const loader = addLoader();

  try{
    const res = await fetch("/chat", {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    loader.remove();
    if(data.ok){
      addMessage("ai", data.reply);
    }else{
      addMessage("ai", "⚠️ " + (data.error || "Something went wrong"));
    }
  }catch(err){
    loader.remove();
    addMessage("ai", "⚠️ Network error");
  }
});

// Reset
resetBtn.addEventListener("click", async () => {
  await fetch("/reset", { method:"POST" });
  messagesEl.innerHTML = "";
  fileStatus.textContent = "No file uploaded";
  chatInput.value = "";
  addMessage("ai", "Session cleared. Upload a file to start again.");
});

// Greeting
addMessage("ai", "Welcome! Upload a PDF/DOCX/TXT, then ask questions. I’ll answer using your file.");
