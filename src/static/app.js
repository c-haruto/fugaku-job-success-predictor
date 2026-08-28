// シンプルなフォーム送信 + 結果表示のみ。フレームワークは使わない素のJS。

const form = document.getElementById("predict-form");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const probEl = document.getElementById("result-prob");
const barEl = document.getElementById("result-bar");
const messageEl = document.getElementById("result-message");

function fillHourOptions() {
  const select = document.getElementById("submit_hour");
  for (let h = 0; h < 24; h++) {
    const opt = document.createElement("option");
    opt.value = String(h);
    opt.textContent = String(h);
    select.appendChild(opt);
  }
  const now = new Date();
  select.value = String(now.getHours());
  document.getElementById("submit_dow").value = String((now.getDay() + 6) % 7); // JS: 0=日 -> 0=月に変換
}

async function loadOptions() {
  const res = await fetch("/api/options");
  if (!res.ok) {
    throw new Error("選択肢の取得に失敗しました");
  }
  const data = await res.json();

  const freqSelect = document.getElementById("freq_req");
  data.freq_req.forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v + " MHz";
    freqSelect.appendChild(opt);
  });
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
  resultBox.classList.add("hidden");
}

function showResult(data) {
  errorBox.classList.add("hidden");
  resultBox.classList.remove("hidden");

  const pct = (data.success_probability * 100).toFixed(1);
  probEl.textContent = `成功確率 ${pct}%`;
  probEl.className = "prob " + (data.label === "success" ? "success" : "failure");
  barEl.style.width = `${pct}%`;
  messageEl.textContent = data.message;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const submitBtn = form.querySelector("button[type=submit]");
  submitBtn.disabled = true;

  const memLimitRaw = document.getElementById("mem_limit_gib").value;

  const payload = {
    n_nodes: Number(document.getElementById("n_nodes").value),
    elapse_hours: Number(document.getElementById("elapse_hours").value),
    freq_req: document.getElementById("freq_req").value,
    submit_dow: Number(document.getElementById("submit_dow").value),
    submit_hour: Number(document.getElementById("submit_hour").value),
    mem_limit_gib: memLimitRaw === "" ? null : Number(memLimitRaw),
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `予測に失敗しました (HTTP ${res.status})`);
    }
    const data = await res.json();
    showResult(data);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    submitBtn.disabled = false;
  }
});

fillHourOptions();
loadOptions().catch((err) => showError(err.message || String(err)));
